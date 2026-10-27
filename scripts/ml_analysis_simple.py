# -*- coding: utf-8 -*-
"""
异常检测 —— 基础可读版（ml_analysis_simple.py）
=================================================
这个文件做的事，和原版 ml_analysis.py 完全一样，用三种口径给交易打异常分：
  1. 统计基线：Modified Z-score（基于中位数和 MAD，金额离谱的标记出来）
  2. 主方法：孤立森林 Isolation Forest（用 7 个交易级特征，取分数最高的 1%）
  3. 业务规则：金额进前 1%、并且"超余额或深夜"，风控一眼能看懂

诚实说明（原版就有）：这里不用深度学习。异常检测需要能解释"为什么标记这笔"，
树集成/密度方法足够，而且没有标签没法做有监督。

产物：
  · data/output/bank_scored.parquet   每笔交易的评分明细
  · data/output/queue_top50.parquet   异常分最高的 50 笔复核队列
  · data/output/ml_result.json        各方法标记了多少笔、涉及多少金额

运行方式（和原版一样，没有命令行参数；要先跑过 clean_data）：
    python scripts/ml_analysis_simple.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ------------------------------------------------------------------
# 0. 路径与读数据
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"
clean = pd.read_parquet(OUT / "bank_clean.parquet")

r = {}
r["n"] = len(clean)

# ------------------------------------------------------------------
# 1. 构造特征表 f（和交易表一行对一行）
# ------------------------------------------------------------------
f = pd.DataFrame(index=clean.index)

# 金额取 log1p：金额极度右偏，取对数后模型更好学
f["amount_log"] = np.log1p(clean["amount"])

# 余额取 log1p；余额缺失的先按 0 填，再单独记一列"余额缺失"
f["balance_log"] = np.log1p(clean["balance"].fillna(0))

bal_missing_list = []
for v in clean["balance"]:
    if pd.isna(v):
        bal_missing_list.append(1)
    else:
        bal_missing_list.append(0)
f["balance_missing"] = bal_missing_list

f["hour"] = clean["hour"]

# 布尔标志转成 0/1，方便模型直接当数值用
late_int_list = []
for v in clean["is_late_night"]:
    if bool(v):
        late_int_list.append(1)
    else:
        late_int_list.append(0)
f["is_late_night"] = late_int_list

over_int_list = []
for v in clean["over_balance"]:
    if bool(v):
        over_int_list.append(1)
    else:
        over_int_list.append(0)
f["over_balance"] = over_int_list

# 年龄缺失的用中位数补上（无监督下常规做法）
f["age"] = clean["age"].fillna(clean["age"].median())

r["features"] = list(f.columns)

# ------------------------------------------------------------------
# 2. 统计基线：Modified Z-score（只看金额）
# ------------------------------------------------------------------
# 中位数 med 比均值抗异常；MAD = |金额-中位数| 的中位数，衡量正常波动有多大
med = clean["amount"].median()
mad = np.median(np.abs(clean["amount"] - med))
if mad <= 0:
    mad = 1e-9                       # 几乎所有金额都一样时，给个极小分母避免除零

# 标准化后的"偏离度"：|z| > 3.5 就算金额离谱
clean["mad_z"] = 0.6745 * (clean["amount"] - med) / mad
flag_z = clean["mad_z"].abs() > 3.5

r["mad_flagged"] = int(flag_z.sum())
r["mad_flag_pct"] = round(flag_z.mean() * 100, 2)
r["mad_flagged_amt"] = round(float(clean.loc[flag_z, "amount"].sum()), 2)
r["mad_flagged_pct_of_total"] = round(
    float(clean.loc[flag_z, "amount"].sum() / clean["amount"].sum() * 100), 2
)

# ------------------------------------------------------------------
# 3. 孤立森林（主方法）
# ------------------------------------------------------------------
# 参数固定：200 棵树、每次抽 256 个样本、预期异常占 1%、随机种子 42 保证可复现
model = IsolationForest(n_estimators=200, max_samples=256, contamination=0.01,
                        random_state=42, n_jobs=-1)
clean["if_score"] = model.fit_predict(f)

# decision_function 越负越异常，所以取负号：分数越大越异常
clean["if_score"] = -model.decision_function(f)

# 取分数前 1% 作为异常标记
cut = np.quantile(clean["if_score"], 0.99)
if_label_list = []
for s in clean["if_score"]:
    if s >= cut:
        if_label_list.append(1)
    else:
        if_label_list.append(0)
clean["if_label"] = if_label_list

r["if_flagged"] = int(clean["if_label"].sum())
r["if_flagged_amt"] = round(float(clean.loc[clean["if_label"] == 1, "amount"].sum()), 2)
r["if_flagged_pct_of_total"] = round(
    float(clean.loc[clean["if_label"] == 1, "amount"].sum() / clean["amount"].sum() * 100), 2
)
r["if_top1pct_amt_share"] = r["if_flagged_pct_of_total"]

# ------------------------------------------------------------------
# 4. 业务规则（可解释基线）
# ------------------------------------------------------------------
# 金额进前 1%，并且（超余额 或 深夜）—— 两个条件叠加才标记
large_cut = clean["amount"].quantile(0.99)
rule_list = []
for amt, ob_v, ln_v in zip(clean["amount"], clean["over_balance"], clean["is_late_night"]):
    if amt >= large_cut and (bool(ob_v) or bool(ln_v)):
        rule_list.append(True)
    else:
        rule_list.append(False)

r["rule_flagged"] = int(sum(rule_list))
r["rule_flagged_amt"] = round(float(clean.loc[rule_list, "amount"].sum()), 2)

# 业务规则和孤立森林同时命中的有多少笔（两种方法互相印证）
overlap = 0
for r_v, l_v in zip(rule_list, clean["if_label"]):
    if r_v and l_v == 1:
        overlap += 1
r["overlap_if_and_rule"] = overlap

# ------------------------------------------------------------------
# 5. Top50 复核队列
# ------------------------------------------------------------------
# 按异常分从高到低取前 50 笔，交给风控人工复核
top = clean.sort_values("if_score", ascending=False).head(50)
queue = top[["txn_id", "customer_id", "city", "amount", "hour",
             "is_late_night", "over_balance", "balance", "if_score"]]

r["queue_top50_amt"] = round(float(queue["amount"].sum()), 2)
r["queue_top50_customers"] = int(queue["customer_id"].nunique())
r["queue_top50_late_night"] = int(queue["is_late_night"].sum())
r["queue_top50_over_balance"] = int(queue["over_balance"].sum())

# ------------------------------------------------------------------
# 6. 保存评分表和队列
# ------------------------------------------------------------------
scored = clean[["txn_id", "customer_id", "age", "gender", "city", "balance",
                "TransactionDate", "hour", "is_late_night", "amount",
                "over_balance", "mad_z", "if_score", "if_label"]]
scored.to_parquet(OUT / "bank_scored.parquet", index=False)
queue.to_parquet(OUT / "queue_top50.parquet", index=False)

# ------------------------------------------------------------------
# 7. 写报告并打印
# ------------------------------------------------------------------
with open(OUT / "ml_result.json", "w", encoding="utf-8") as f:
    json.dump(r, f, ensure_ascii=False, indent=1)

print("ml done", flush=True)
print(json.dumps(r, ensure_ascii=False), flush=True)
