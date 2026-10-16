# -*- coding: utf-8 -*-
"""银行交易清洗：
- 日期 %d/%m/%y → datetime（印度格式，范围 2016-08-01~2016-10-21）
- 时间 HHMMSS → hour + 深夜标志(00:00-05:59)
- 性别 NaN/T → unknown；DOB → 年龄
- 余额缺失保留 NaN；金额无负、0 值保留
输出：bank_clean.parquet（交易级）、customer_level.parquet（客户级）、cleaning_report.json
"""
import json
from pathlib import Path
import pandas as pd

RAW = Path(r"D:\B编程\powerBI\github\datasets\finance\bank_transactions.csv")
OUT = Path(__file__).resolve().parent.parent / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)
report = {}
report["rows_in"] = len(df)

# 日期：d/m/yy
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], format="%d/%m/%y", errors="coerce")
report["date_parse_fail"] = int(df["TransactionDate"].isna().sum())
report["date_min"] = str(df["TransactionDate"].min().date())
report["date_max"] = str(df["TransactionDate"].max().date())

# 时间 HHMMSS → hour
def _hour(t):
    try:
        return int(t // 10000)
    except Exception:
        return -1
df["hour"] = df["TransactionTime"].map(_hour)
df["is_late_night"] = (df["hour"] >= 0) & (df["hour"] < 6)
report["late_night_pct"] = round(df["is_late_night"].mean() * 100, 3)

# 性别
df["CustGender"] = df["CustGender"].map(lambda g: g if g in ("M", "F") else "unknown")
report["gender_after"] = df["CustGender"].value_counts().to_dict()

# DOB → 年龄（截至数据中位日期 2016-09）
ref = pd.Timestamp("2016-09-15")
dob = pd.to_datetime(df["CustomerDOB"], format="%d/%m/%y", errors="coerce")
df["age"] = ((ref - dob).dt.days / 365.25).round(0)
df["age"] = df["age"].where(df["age"].between(0, 120))
report["age_null_pct"] = round(df["age"].isna().mean() * 100, 2)

# 金额
df["amount"] = df["TransactionAmount (INR)"]
report["neg_amount"] = int((df["amount"] < 0).sum())
report["zero_amount_pct"] = round((df["amount"] == 0).mean() * 100, 3)

# 余额
df["balance"] = df["CustAccountBalance"]
# 超余额交易：余额非缺失且 余额 < 金额
has_bal = df["balance"].notna()
df["over_balance"] = has_bal & (df["balance"] < df["amount"])
report["over_balance_pct"] = round(df["over_balance"].mean() * 100, 3)
report["over_balance_amt_sum"] = round(float(df.loc[df["over_balance"], "amount"].sum()), 2)

# 交易级清洗表
clean = df.rename(columns={
    "TransactionID": "txn_id", "CustomerID": "customer_id", "CustLocation": "city",
}).drop(columns=["TransactionTime", "TransactionAmount (INR)"])
clean = clean[
    ["txn_id", "customer_id", "age", "CustGender", "city", "balance",
     "TransactionDate", "hour", "is_late_night", "amount", "over_balance"]
].rename(columns={"CustGender": "gender"})
clean = clean.sort_values(["customer_id", "TransactionDate"])
report["rows_out"] = len(clean)
report["nulls_out"] = {c: int(clean[c].isna().sum()) for c in clean.columns}
clean.to_parquet(OUT / "bank_clean.parquet", index=False)

# 客户级表
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
cust["has_over_balance"] = cust["over_balance_count"] > 0
report["customers"] = len(cust)
report["multi_txn_customers_pct"] = round((cust["txn_count"] > 1).mean() * 100, 2)
cust.to_parquet(OUT / "customer_level.parquet", index=False)

with open(OUT / "cleaning_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
print("clean done", flush=True)
print(json.dumps(report, ensure_ascii=False), flush=True)
