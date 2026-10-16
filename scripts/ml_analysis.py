# -*- coding: utf-8 -*-
"""银行交易异常检测：
- 统计基线：Modified Z-score（MAD，可解释、易落地）
- 无监督：孤立森林 Isolation Forest（交易级特征，主方法）
- 业务规则：超余额 + 深夜大额（风控可解释基线）
- 诚实说明：不用 DL（异常检测需可解释"为什么标记这笔"，树集成/密度足够，且无标签无法有监督）
输出：ml_result.json + bank_scored.parquet
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

OUT = Path(__file__).resolve().parent.parent / "data" / "output"
clean = pd.read_parquet(OUT / "bank_clean.parquet")

r = {}
r["n"] = len(clean)

# ---------- 特征 ----------
f = pd.DataFrame(index=clean.index)
f["amount_log"] = np.log1p(clean["amount"])
f["balance_log"] = np.log1p(clean["balance"].fillna(0))
f["balance_missing"] = clean["balance"].isna().astype(int)
f["hour"] = clean["hour"]
f["is_late_night"] = clean["is_late_night"].astype(int)
f["over_balance"] = clean["over_balance"].astype(int)
f["age"] = clean["age"].fillna(clean["age"].median())
r["features"] = list(f.columns)

# ---------- 1. 统计基线：Modified Z-score（基于金额） ----------
med = clean["amount"].median()
mad = np.median(np.abs(clean["amount"] - med))
mad = mad if mad > 0 else 1e-9
clean["mad_z"] = 0.6745 * (clean["amount"] - med) / mad
flag_z = clean["mad_z"].abs() > 3.5
r["mad_flagged"] = int(flag_z.sum())
r["mad_flag_pct"] = round(flag_z.mean() * 100, 2)
r["mad_flagged_amt"] = round(float(clean.loc[flag_z, "amount"].sum()), 2)
r["mad_flagged_pct_of_total"] = round(float(clean.loc[flag_z, "amount"].sum() / clean["amount"].sum() * 100), 2)

# ---------- 2. 孤立森林（主方法） ----------
model = IsolationForest(n_estimators=200, max_samples=256, contamination=0.01,
                        random_state=42, n_jobs=-1)
clean["if_score"] = model.fit_predict(f)
# decision_function 越负越异常 → anomaly score = -decision（越大越异常）
clean["if_score"] = -model.decision_function(f)
clean["if_label"] = (clean["if_score"] >= np.quantile(clean["if_score"], 0.99)).astype(int)
r["if_flagged"] = int(clean["if_label"].sum())
r["if_flagged_amt"] = round(float(clean.loc[clean["if_label"] == 1, "amount"].sum()), 2)
r["if_flagged_pct_of_total"] = round(
    float(clean.loc[clean["if_label"] == 1, "amount"].sum() / clean["amount"].sum() * 100), 2)
r["if_top1pct_amt_share"] = r["if_flagged_pct_of_total"]

# ---------- 3. 业务规则（可解释基线） ----------
rule_large = clean["amount"] >= clean["amount"].quantile(0.99)
rule = (rule_large & (clean["over_balance"] | clean["is_late_night"]))
r["rule_flagged"] = int(rule.sum())
r["rule_flagged_amt"] = round(float(clean.loc[rule, "amount"].sum()), 2)

# 方法重叠
r["overlap_if_and_rule"] = int((rule & (clean["if_label"] == 1)).sum())

# ---------- Top 复核队列 ----------
top = clean.sort_values("if_score", ascending=False).head(50)
queue = top[["txn_id", "customer_id", "city", "amount", "hour",
             "is_late_night", "over_balance", "balance", "if_score"]]
r["queue_top50_amt"] = round(float(queue["amount"].sum()), 2)
r["queue_top50_customers"] = int(queue["customer_id"].nunique())
r["queue_top50_late_night"] = int(queue["is_late_night"].sum())
r["queue_top50_over_balance"] = int(queue["over_balance"].sum())

# 保存评分表
scored = clean[["txn_id", "customer_id", "age", "gender", "city", "balance",
                "TransactionDate", "hour", "is_late_night", "amount",
                "over_balance", "mad_z", "if_score", "if_label"]]
scored.to_parquet(OUT / "bank_scored.parquet", index=False)
queue.to_parquet(OUT / "queue_top50.parquet", index=False)

with open(OUT / "ml_result.json", "w", encoding="utf-8") as f:
    json.dump(r, f, ensure_ascii=False, indent=1)
print("ml done", flush=True)
print(json.dumps(r, ensure_ascii=False), flush=True)
