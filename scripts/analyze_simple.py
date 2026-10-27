# -*- coding: utf-8 -*-
"""
指标计算 —— 基础可读版（analyze_simple.py）
=================================================
这个文件做的事，和原版 analyze.py 完全一样：
  1. 读清洗产物 bank_clean.parquet（交易级）和 customer_level.parquet（客户级）
  2. 算一组业务指标，写进 data/output/metrics.json：
     · 金额分布（中位数/均值/99/99.9 分位/最大）
     · 超余额交易的笔数、占比、涉及金额
     · 深夜交易的笔数、占比、涉及金额
     · 交易笔数最多的前 10 个城市
     · 客户层面：单笔客户占比、多笔客户占比
     · 按客户总金额四分位分成 L1~L4 四层，每层的客户数/总金额/平均余额
     · 客户余额缺失率

运行方式（和原版一样，没有命令行参数；要先跑过 clean_data）：
    python scripts/analyze_simple.py
"""

import json
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# 0. 路径配置
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"

# 读上游产物：交易级清洗表 + 客户级汇总表
clean = pd.read_parquet(OUT / "bank_clean.parquet")
cust = pd.read_parquet(OUT / "customer_level.parquet")

m = {}
m["txn_rows"] = len(clean)
m["customers"] = len(cust)

# ------------------------------------------------------------------
# 1. 金额分布
# ------------------------------------------------------------------
am = clean["amount"]
m["amount_median"] = round(float(am.median()), 2)
m["amount_mean"] = round(float(am.mean()), 2)
m["amount_p99"] = round(float(am.quantile(0.99)), 2)
m["amount_p999"] = round(float(am.quantile(0.999)), 2)
m["amount_max"] = round(float(am.max()), 2)

# ------------------------------------------------------------------
# 2. 超余额交易
# ------------------------------------------------------------------
ob = clean[clean["over_balance"]]
m["over_balance_txns"] = len(ob)
m["over_balance_pct"] = round(len(ob) / len(clean) * 100, 2)
m["over_balance_amt"] = round(float(ob["amount"].sum()), 2)

# ------------------------------------------------------------------
# 3. 深夜交易
# ------------------------------------------------------------------
ln = clean[clean["is_late_night"]]
m["late_night_txns"] = len(ln)
m["late_night_pct"] = round(len(ln) / len(clean) * 100, 2)
m["late_night_amt"] = round(float(ln["amount"].sum()), 2)

# ------------------------------------------------------------------
# 4. 城市 Top10
# ------------------------------------------------------------------
m["city_top"] = clean["city"].value_counts().head(10).to_dict()

# ------------------------------------------------------------------
# 5. 客户交易笔数分布
# ------------------------------------------------------------------
m["single_txn_customers_pct"] = round((cust["txn_count"] == 1).mean() * 100, 2)
m["multi_txn_customers_pct"] = round((cust["txn_count"] > 1).mean() * 100, 2)

# ------------------------------------------------------------------
# 6. 客户资金价值分层（按总金额四分位）
# ------------------------------------------------------------------
# 先取 25/50/75 三个分位值，记录下来作为分层边界
q = cust["total_amount"].quantile([0.25, 0.5, 0.75]).to_dict()

# 再按总金额把客户切成 4 层：L4 最穷、L1 最富（qcut 的 labels 顺序如此）
cust["tier"] = pd.qcut(cust["total_amount"], 4, labels=["L4", "L3", "L2", "L1"])

# 把分位值字典转成"字符串键 → 保留两位小数"的报告格式
boundaries = {}
for k, v in q.items():
    boundaries[str(k)] = round(float(v), 2)
m["tier_boundaries"] = boundaries

# 每层汇总：客户数、总金额、平均余额
tier = cust.groupby("tier", observed=False).agg(
    customers=("customer_id", "count"),
    total_amount=("total_amount", "sum"),
    mean_balance=("balance", "mean"),
).reset_index()
m["tier_table"] = tier.to_dict("records")

# ------------------------------------------------------------------
# 7. 客户余额缺失率
# ------------------------------------------------------------------
m["customers_balance_missing_pct"] = round(cust["balance"].isna().mean() * 100, 2)

# ------------------------------------------------------------------
# 8. 写文件并打印
# ------------------------------------------------------------------
with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(m, f, ensure_ascii=False, indent=1)

print("metrics done", flush=True)
print(json.dumps(m, ensure_ascii=False), flush=True)
