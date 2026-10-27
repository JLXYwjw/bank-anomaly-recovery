# -*- coding: utf-8 -*-
"""
数据探查 —— 基础可读版（explore_data_simple.py）
=================================================

  1. 直接读最原始的 bank_transactions.csv（不经过清洗）
  2. 算一组"摸底"统计：多少行、多少客户、人均几笔、日期范围、
     金额的均值/中位数/99 分位/最大/正负零占比、余额与金额的相关系数、
     每列缺失数、性别分布、只有一笔交易的客户占比
  3. 全部写进 data/output/explore_profile.json，并原样打印一份

它的用途是"开工前先看一眼长什么样"，不改任何数据。

运行方式（没有命令行参数）：
    python scripts/explore_data_simple.py


English:
Data exploration (plain, readable version).

Steps:
  1. Read the raw bank_transactions.csv directly (no cleaning)
  2. Compute a set of "first look" statistics: rows, customers,
     transactions per person, date range, amount mean / median / 99th
     percentile / max / share of positive-zero-negative, balance-amount
     correlation, missing count per column, gender distribution, share of
     single-transaction customers
  3. Write everything into data/output/explore_profile.json and also print
     it as-is

Purpose: "see what it looks like before starting"; it changes no data.

Run (no command-line arguments):
    python scripts/explore_data_simple.py
"""

import json
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# 0. 路径配置
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "bank_transactions.csv"
OUT = ROOT / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1. 读原始数据并逐段算统计
# ------------------------------------------------------------------
df = pd.read_csv(RAW)
am = df["TransactionAmount (INR)"]     # 把金额列取出来，后面反复用

d = {}
d["rows"] = len(df)
d["customers"] = int(df["CustomerID"].nunique())
d["tx_per_customer_mean"] = round(len(df) / df["CustomerID"].nunique(), 3)

# 原始 CSV 里的日期还是字符串，这里只看字符串的最小/最大值，不做解析
d["date_min_raw"] = str(df["TransactionDate"].min())
d["date_max_raw"] = str(df["TransactionDate"].max())

# 金额分布：均值、中位数、99% 分位、99.9% 分位、最大值
d["amt_mean"] = float(am.mean())
d["amt_median"] = float(am.median())
d["amt_p99"] = float(am.quantile(0.99))
d["amt_p999"] = float(am.quantile(0.999))
d["amt_max"] = float(am.max())

# 异常金额占比：负数、0
d["amt_neg_pct"] = round((am < 0).mean() * 100, 4)
d["amt_zero_pct"] = round((am == 0).mean() * 100, 4)

# 余额和金额的关系：相关系数；"余额 < 金额"的交易占比
d["balance_vs_amt_corr"] = round(df["CustAccountBalance"].corr(am), 3)
d["pct_balance_lt_amt"] = round((df["CustAccountBalance"] < am).mean() * 100, 3)

# 每一列有多少个缺失值
missing = {}
for c in df.columns:
    missing[c] = int(df[c].isna().sum())
d["missing"] = missing

# 性别分布（把空值也当成一类统计进来）
d["gender"] = df["CustGender"].value_counts(dropna=False).to_dict()

# 每个客户交易了几笔？其中"只交易 1 笔"的客户占多少
counts = df["CustomerID"].value_counts()
single = counts == 1
d["tx_per_customer_pct_1"] = round(single.mean() * 100, 2)

# ------------------------------------------------------------------
# 2. 写文件并打印
# ------------------------------------------------------------------
with open(OUT / "explore_profile.json", "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=1)

print(json.dumps(d, ensure_ascii=False, indent=1), flush=True)
