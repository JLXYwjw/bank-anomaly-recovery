# -*- coding: utf-8 -*-
"""探查 bank_transactions 全量，输出 profile 到 data/output/explore_profile.json。"""
import json
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "bank_transactions.csv"
OUT = Path(__file__).resolve().parent.parent / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)
am = df["TransactionAmount (INR)"]
d = {}
d["rows"] = len(df)
d["customers"] = int(df["CustomerID"].nunique())
d["tx_per_customer_mean"] = round(len(df) / df["CustomerID"].nunique(), 3)
d["date_min_raw"] = str(df["TransactionDate"].min())
d["date_max_raw"] = str(df["TransactionDate"].max())
d["amt_mean"] = float(am.mean())
d["amt_median"] = float(am.median())
d["amt_p99"] = float(am.quantile(0.99))
d["amt_p999"] = float(am.quantile(0.999))
d["amt_max"] = float(am.max())
d["amt_neg_pct"] = round((am < 0).mean() * 100, 4)
d["amt_zero_pct"] = round((am == 0).mean() * 100, 4)
d["balance_vs_amt_corr"] = round(df["CustAccountBalance"].corr(am), 3)
d["pct_balance_lt_amt"] = round((df["CustAccountBalance"] < am).mean() * 100, 3)
d["missing"] = {c: int(df[c].isna().sum()) for c in df.columns}
d["gender"] = df["CustGender"].value_counts(dropna=False).to_dict()
d["tx_per_customer_pct_1"] = round(
    (df["CustomerID"].value_counts() == 1).mean() * 100, 2)
with open(OUT / "explore_profile.json", "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=1)
print(json.dumps(d, ensure_ascii=False, indent=1), flush=True)
