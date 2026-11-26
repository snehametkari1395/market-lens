"""Tests for checking answer figures against tool data."""

from __future__ import annotations

import pytest

from finance_research.factcheck import extract_figures, verify_answer
from finance_research.monitoring import summarize_events


def _sec_tools():
    rows = [
        ("AAPL", "revenue", "Revenue", 2023, 383_285_000_000.0),
        ("AAPL", "revenue", "Revenue", 2024, 391_035_000_000.0),
        ("AAPL", "net_income", "Net income", 2024, 93_736_000_000.0),
        ("AAPL", "eps_diluted", "Diluted earnings per share", 2024, 6.08),
        ("MSFT", "revenue", "Revenue", 2024, 245_122_000_000.0),
    ]
    return [
        {
            "name": "fetch_sec_financials",
            "content": {
                "records": [
                    {"ticker": t, "metric": m, "metric_label": label, "fiscal_year": y, "value_usd": v, "source_url": "https://www.sec.gov/x"}
                    for t, m, label, y, v in rows
                ]
            },
        }
    ]


def _status(result, text):
    return next(row["status"] for row in result["figures"] if row["text"] == text)


def test_extracts_figures_with_units_and_skips_noise():
    text = (
        "1. In FY2024 Apple (see [2] and Item 2.02 of its 8-K and 10-K) reported $391.0 billion, "
        "a 2.0% rise, EPS of $6.08, debt-to-equity of 0.70×, 3 segments, net income of 93,736 million "
        "and a loss of −$1.2B. https://www.sec.gov/cgi-bin/1234567"
    )
    figures = {figure.text: figure for figure in extract_figures(text)}
    assert set(figures) == {"$391.0 billion", "2.0%", "$6.08", "0.70×", "93,736 million", "−$1.2B"}
    assert figures["$391.0 billion"].value == pytest.approx(391.0e9) and figures["$391.0 billion"].kind == "money"
    assert figures["2.0%"].value == pytest.approx(0.02) and figures["2.0%"].kind == "percent"
    assert figures["0.70×"].kind == "multiple"
    assert figures["93,736 million"].value == pytest.approx(93_736e6)
    assert figures["−$1.2B"].value == pytest.approx(-1.2e9)


def test_correct_figures_match_with_rounding():
    answer = "Apple's revenue was $391.0 billion (FY2024), up from $383.3B. Diluted EPS was $6.08 and revenue was 391,035 million."
    result = verify_answer(answer, _sec_tools())
    assert result["checked"] == 4 and result["matched"] == 4 and result["unverified"] == 0
    assert "AAPL · FY2024 · Revenue (SEC)" in {row["source"] for row in result["figures"]}


def test_rounding_tolerance_follows_the_digits_shown():
    assert _status(verify_answer("Revenue was $391 billion.", _sec_tools()), "$391 billion") == "matched"
    # One decimal place allows ±$0.05B, so $391.2B is not the same as $391.035B.
    assert _status(verify_answer("Revenue was $391.2 billion.", _sec_tools()), "$391.2 billion") == "unverified"
    assert _status(verify_answer("Revenue was about $395 billion.", _sec_tools()), "$395 billion") == "matched"


def test_calculations_from_source_data_are_recognised():
    answer = "Revenue grew 2.0% and the net margin was 24.0%. Apple's revenue was 1.6× Microsoft's. Revenue rose by $7.8 billion."
    result = verify_answer(answer, _sec_tools())
    statuses = {row["text"]: (row["status"], row["source"]) for row in result["figures"]}
    assert statuses["2.0%"][0] == "calculated" and "growth" in statuses["2.0%"][1]
    assert statuses["24.0%"][0] == "calculated" and "net income ÷ revenue" in statuses["24.0%"][1]
    assert statuses["1.6×"][0] == "calculated"
    assert statuses["$7.8 billion"][0] == "calculated"


def test_made_up_figures_are_flagged():
    result = verify_answer("Apple's revenue was $412.7 billion and its margin was 31.4%.", _sec_tools())
    assert result["unverified"] == 2 and result["matched"] == 0


def test_numbers_in_filing_excerpts_and_text_fields_count():
    tools = [
        {
            "name": "search_filing_knowledge",
            "content": {"results": [{"ticker": "AAPL", "report_year": 2024, "text": "Total net sales were $391,035 million, and R&D rose 5%."}]},
        },
        {"name": "fetch_stock_quote", "content": {"ticker": "AAPL", "price": "229.8700", "change_percent": "1.25%"}},
    ]
    result = verify_answer("Net sales were $391.0 billion, R&D rose 5%, the price is $229.87 (up 1.25%).", tools)
    assert result["unverified"] == 0 and result["checked"] == 4
    assert any("10-K excerpt" in row["source"] for row in result["figures"])


def test_answers_without_tool_data_or_figures():
    no_tools = verify_answer("Revenue was $391.0 billion.", [])
    assert no_tools["checked"] == 1 and no_tools["has_tool_data"] is False and no_tools["unverified"] == 1
    assert verify_answer("Apple sells phones and services.", _sec_tools())["checked"] == 0


def test_monitoring_reports_the_share_of_verified_figures():
    events = [
        {"kind": "chat_answer", "figures_checked": 8, "figures_unverified": 2},
        {"kind": "chat_answer", "figures_checked": 2, "figures_unverified": 0},
        {"kind": "chat_answer"},
    ]
    stats = summarize_events(events)
    assert stats["figures_checked"] == 10 and stats["figure_check_rate"] == pytest.approx(0.8)
    assert summarize_events([])["figure_check_rate"] is None
