# -*- coding: utf-8 -*-
"""银行交易指标计算 → metrics.json + 可视化所需聚合。"""
import json
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "output"
clean = pd.read_parquet(OUT / "bank_clean.parquet")
cust = pd.read_parquet(OUT / "customer_level.parquet")

m = {}
m["txn_rows"] = len(clean)
m["customers"] = len(cust)

# 金额分布
am = clean["amount"]
m["amount_median"] = round(float(am.median()), 2)
m["amount_mean"] = round(float(am.mean()), 2)
m["amount_p99"] = round(float(am.quantile(0.99)), 2)
m["amount_p999"] = round(float(am.quantile(0.999)), 2)
m["amount_max"] = round(float(am.max()), 2)

# 超余额
ob = clean[clean["over_balance"]]
m["over_balance_txns"] = len(ob)
m["over_balance_pct"] = round(len(ob) / len(clean) * 100, 2)
m["over_balance_amt"] = round(float(ob["amount"].sum()), 2)

# 深夜交易
ln = clean[clean["is_late_night"]]
m["late_night_txns"] = len(ln)
m["late_night_pct"] = round(len(ln) / len(clean) * 100, 2)
m["late_night_amt"] = round(float(ln["amount"].sum()), 2)

# 城市 Top
m["city_top"] = clean["city"].value_counts().head(10).to_dict()

# 客户：交易笔数分布
m["single_txn_customers_pct"] = round((cust["txn_count"] == 1).mean() * 100, 2)
m["multi_txn_customers_pct"] = round((cust["txn_count"] > 1).mean() * 100, 2)

# 客户资金价值分层（按 total_amount 四分位）
q = cust["total_amount"].quantile([0.25, 0.5, 0.75]).to_dict()
cust["tier"] = pd.qcut(cust["total_amount"], 4, labels=["L4", "L3", "L2", "L1"])
m["tier_boundaries"] = {str(k): round(float(v), 2) for k, v in q.items()}
tier = cust.groupby("tier", observed=False).agg(
    customers=("customer_id", "count"),
    total_amount=("total_amount", "sum"),
    mean_balance=("balance", "mean"),
).reset_index()
m["tier_table"] = tier.to_dict("records")

# 客户余额缺失率
m["customers_balance_missing_pct"] = round(cust["balance"].isna().mean() * 100, 2)

with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(m, f, ensure_ascii=False, indent=1)
print("metrics done", flush=True)
print(json.dumps(m, ensure_ascii=False), flush=True)
