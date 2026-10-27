# -*- coding: utf-8 -*-
"""
银行交易清洗 —— 基础可读版（clean_data_simple.py）
=================================================
这个文件做的事，和原版 clean_data.py 完全一样：
  1. 读入银行原始交易 CSV（data/raw/bank_transactions.csv，约 105 万笔）
  2. 把交易日期从印度格式 d/m/yy 解析成真正的日期，顺手检查解析失败数与日期范围
  3. 把 HHMMSS 形式的交易时间拆成"小时"，并标出深夜（0 点~5 点 59 分）交易
  4. 性别只保留 M/F，其余一律记为 unknown；用生日算出年龄（基准日 2016-09-15）
  5. 余额、金额原样保留；标出"余额不够却交易"的可疑记录
  6. 产出两张表：交易级 bank_clean.parquet、客户级 customer_level.parquet，
     外加一份清洗报告 cleaning_report.json

区别只是：本文件故意用"最朴素的写法"（for 循环、if 判断、普通函数），
每一步都写清楚了在做什么、为什么，方便从头读懂。

运行方式（和原版一样，没有命令行参数）：
    python scripts/clean_data_simple.py
"""

import json
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# 0. 路径配置
# ------------------------------------------------------------------
# 本文件在 scripts/ 下，上一级就是项目根目录
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "bank_transactions.csv"   # 原始交易 CSV
OUT = ROOT / "data" / "output"                          # 清洗产物放这里
OUT.mkdir(parents=True, exist_ok=True)                  # 没有就建目录

# 年龄的基准日：原数据交易发生在 2016 年 8~10 月，取中间这天算年龄
REF_DATE = pd.Timestamp("2016-09-15")


# ------------------------------------------------------------------
# 1. 把 HHMMSS 时间戳拆成小时
# ------------------------------------------------------------------
def hour_of(t):
    """143052 这种数字 → 14 点；转不出来给 -1（后面会被深夜判断排除掉）。"""
    try:
        return int(t // 10000)
    except Exception:
        return -1


# ------------------------------------------------------------------
# 2. 主流程
# ------------------------------------------------------------------
# ---- 2.1 读入原始 CSV ----
df = pd.read_csv(RAW)
report = {}
report["rows_in"] = len(df)

# ---- 2.2 交易日期：d/m/yy → 真正的日期 ----
# 印度写法，例如 21/10/16 表示 2016-10-21；解析失败的记成空值
df["TransactionDate"] = pd.to_datetime(
    df["TransactionDate"], format="%d/%m/%y", errors="coerce"
)
report["date_parse_fail"] = int(df["TransactionDate"].isna().sum())
report["date_min"] = str(df["TransactionDate"].min().date())
report["date_max"] = str(df["TransactionDate"].max().date())

# ---- 2.3 交易时间 → 小时 + 深夜标志 ----
# 原时间列是 143052 这种"时分秒挤在一起"的整数，取前两位就是小时
hour_list = []
for t in df["TransactionTime"]:
    hour_list.append(hour_of(t))
df["hour"] = hour_list

# 深夜 = 0 点到 5 点 59 分（hour 为 -1 的坏数据不算深夜）
late_list = []
for h in df["hour"]:
    if h >= 0 and h < 6:
        late_list.append(True)
    else:
        late_list.append(False)
df["is_late_night"] = late_list
report["late_night_pct"] = round(df["is_late_night"].mean() * 100, 3)

# ---- 2.4 性别：只认 M/F，其他（含空值）一律 unknown ----
gender_list = []
for g in df["CustGender"]:
    if g == "M" or g == "F":
        gender_list.append(g)
    else:
        gender_list.append("unknown")
df["CustGender"] = gender_list
report["gender_after"] = df["CustGender"].value_counts().to_dict()

# ---- 2.5 生日 → 年龄 ----
# 先把生日也按 d/m/yy 解析；再用基准日减生日，除以 365.25 得到岁数
dob = pd.to_datetime(df["CustomerDOB"], format="%d/%m/%y", errors="coerce")
age_raw = ((REF_DATE - dob).dt.days / 365.25).round(0)

# 年龄只保留 0~120 岁这个合理区间，离谱的（脏生日）记成空值
age_list = []
for a in age_raw:
    if pd.isna(a):
        age_list.append(None)
    elif 0 <= a <= 120:
        age_list.append(a)
    else:
        age_list.append(None)
df["age"] = age_list
report["age_null_pct"] = round(df["age"].isna().mean() * 100, 2)

# ---- 2.6 金额：原样保留，只统计负数和 0 的占比 ----
df["amount"] = df["TransactionAmount (INR)"]
report["neg_amount"] = int((df["amount"] < 0).sum())
report["zero_amount_pct"] = round((df["amount"] == 0).mean() * 100, 3)

# ---- 2.7 余额 + "超余额交易"标志 ----
# 业务含义：账户余额不为空、且余额 < 这笔交易金额，就是"钱不够还硬刷"，值得风控关注
df["balance"] = df["CustAccountBalance"]
over_list = []
for bal, amt in zip(df["balance"], df["amount"]):
    if pd.isna(bal):
        over_list.append(False)          # 余额缺失：无法判断，不算超余额
    elif bal < amt:
        over_list.append(True)
    else:
        over_list.append(False)
df["over_balance"] = over_list
report["over_balance_pct"] = round(df["over_balance"].mean() * 100, 3)
report["over_balance_amt_sum"] = round(
    float(df.loc[df["over_balance"], "amount"].sum()), 2
)

# ---- 2.8 拼成交易级清洗表 ----
# 先把三个长列名改成短名，丢掉不再需要的原始时间列和原始金额列
clean = df.rename(columns={
    "TransactionID": "txn_id",
    "CustomerID": "customer_id",
    "CustLocation": "city",
}).drop(columns=["TransactionTime", "TransactionAmount (INR)"])

# 按固定顺序挑列，性别列顺便改名 gender
clean = clean[
    ["txn_id", "customer_id", "age", "CustGender", "city", "balance",
     "TransactionDate", "hour", "is_late_night", "amount", "over_balance"]
].rename(columns={"CustGender": "gender"})

# 按客户、再按交易日期排序，方便后面看"同一个客户的交易流水"
clean = clean.sort_values(["customer_id", "TransactionDate"])
report["rows_out"] = len(clean)

# 统计每一列还剩多少空值，写进报告
nulls = {}
for c in clean.columns:
    nulls[c] = int(clean[c].isna().sum())
report["nulls_out"] = nulls

clean.to_parquet(OUT / "bank_clean.parquet", index=False)

# ---- 2.9 客户级汇总表 ----
# 每个客户一行：交易笔数、金额统计、首末交易日、深夜/超余额次数等
cust = clean.groupby("customer_id").agg(
    txn_count=("txn_id", "count"),
    total_amount=("amount", "sum"),
    max_amount=("amount", "max"),
    mean_amount=("amount", "mean"),
    balance=("balance", "first"),
    city=("city", "first"),
    gender=("gender", "first"),
    age=("age", "first"),
    late_night_count=("is_late_night", "sum"),
    over_balance_count=("over_balance", "sum"),
    first_date=("TransactionDate", "min"),
    last_date=("TransactionDate", "max"),
).reset_index()

# 这个客户有没有发生过至少一次"超余额交易"
has_over_list = []
for v in cust["over_balance_count"]:
    if v > 0:
        has_over_list.append(True)
    else:
        has_over_list.append(False)
cust["has_over_balance"] = has_over_list

report["customers"] = len(cust)
report["multi_txn_customers_pct"] = round(
    (cust["txn_count"] > 1).mean() * 100, 2
)
cust.to_parquet(OUT / "customer_level.parquet", index=False)

# ---- 2.10 写清洗报告并打印 ----
with open(OUT / "cleaning_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)

print("clean done", flush=True)
print(json.dumps(report, ensure_ascii=False), flush=True)
