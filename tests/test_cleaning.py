# -*- coding: utf-8 -*-
"""银行交易清洗口径测试。"""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "output"


def _clean():
    return pd.read_parquet(OUT / "bank_clean.parquet")


def test_clean_shape():
    df = _clean()
    assert len(df) == 1048567
    for c in ["amount", "hour", "is_late_night", "over_balance"]:
        assert c in df.columns


def test_date_range():
    df = _clean()
    assert df["TransactionDate"].min().date().isoformat() == "2016-08-01"
    assert df["TransactionDate"].max().date().isoformat() == "2016-10-21"


def test_no_negative_amount():
    df = _clean()
    assert (df["amount"] < 0).sum() == 0


def test_over_balance_count():
    df = _clean()
    assert df["over_balance"].sum() == 103549


def test_late_night_count():
    df = _clean()
    assert df["is_late_night"].sum() == 47031


def test_customer_level():
    cust = pd.read_parquet(OUT / "customer_level.parquet")
    assert len(cust) == 884265
    assert {"txn_count", "total_amount", "balance", "over_balance_count"} <= set(cust.columns)


def test_scored_parquet():
    s = pd.read_parquet(OUT / "bank_scored.parquet")
    assert len(s) == 1048567
