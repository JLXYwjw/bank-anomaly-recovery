# -*- coding: utf-8 -*-
"""报告数字一致性：从 metrics/ml 重算关键数字，断言 HTML 含同数字。"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"
HTML = ROOT / "报告" / "银行交易异常检测与客户资金价值分析.html"


def _html():
    return HTML.read_text(encoding="utf-8")


def _m():
    return json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))


def test_over_balance_match_html():
    m = _m()
    assert abs(m["over_balance_pct"] - 9.88) < 0.05
    assert "9.9%" in _html() or "9.88%" in _html()
    assert abs(m["over_balance_amt"] / 1e6 - 345.98) < 1
    assert "346M" in _html()


def test_late_night_match_html():
    m = _m()
    assert abs(m["late_night_pct"] - 4.49) < 0.1
    assert "4.5%" in _html() or "4.49%" in _html()


def test_concentration_match_html():
    m = _m()
    # L1 客户 25%，金额占比约 81%
    tier = m["tier_table"]
    l1 = next(t for t in tier if t["tier"] == "L1")
    total = sum(t["total_amount"] for t in tier)
    share = l1["total_amount"] / total * 100
    assert abs(share - 81.0) < 2
    assert "81%" in _html()


def test_if_flagged_match_html():
    ml = json.loads((OUT / "ml_result.json").read_text(encoding="utf-8"))
    assert ml["if_flagged"] == 10487
    assert "10,487" in _html() or "10487" in _html()


def test_queue_top50_match_html():
    ml = json.loads((OUT / "ml_result.json").read_text(encoding="utf-8"))
    assert ml["queue_top50_late_night"] == 50
    assert ml["queue_top50_over_balance"] == 45
    assert "45/50" in _html()


def test_overlap_match_html():
    ml = json.loads((OUT / "ml_result.json").read_text(encoding="utf-8"))
    assert ml["overlap_if_and_rule"] == 370
    assert "370" in _html()


def test_mad_vs_if_compare():
    ml = json.loads((OUT / "ml_result.json").read_text(encoding="utf-8"))
    assert ml["mad_flagged"] > ml["if_flagged"]  # MAD 噪声更大
