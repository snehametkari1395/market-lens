"""Tests for the factor scorecard and share-price history (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from finance_research.analytics import FACTORS, add_ratios, build_panel, factor_score, factor_scorecard, kpi_table
from finance_research.prices import (
    PriceDataError,
    fetch_alpha_vantage_monthly,
    fetch_price_history,
    fetch_yahoo_monthly,
    indexed_prices,
    price_growth,
)


def _records(ticker, rows):
    out = []
    for year, values in rows.items():
        for metric, value in values.items():
            out.append({"ticker": ticker, "company": ticker, "fiscal_year": year, "metric": metric, "value_usd": value})
    return out


def _ratios_and_kpis():
    records = _records(
        "AAA",
        {
            2022: {"revenue": 100, "net_income": 20, "total_assets": 200, "stockholders_equity": 100, "operating_cash_flow": 25, "research_and_development": 10, "operating_income": 25},
            2023: {"revenue": 120, "net_income": 30, "total_assets": 210, "stockholders_equity": 110, "operating_cash_flow": 36, "research_and_development": 12, "operating_income": 35},
            2024: {"revenue": 144, "net_income": 36, "total_assets": 220, "stockholders_equity": 132, "operating_cash_flow": 40, "research_and_development": 14, "operating_income": 40},
        },
    ) + _records(
        "BNK",  # a bank-like company with no R&D line and shrinking revenue
        {
            2022: {"revenue": 100, "net_income": 5, "total_assets": 1000, "operating_cash_flow": 4, "operating_income": 6},
            2023: {"revenue": 95, "net_income": 4, "total_assets": 1000, "operating_cash_flow": 4, "operating_income": 5},
            2024: {"revenue": 90, "net_income": 3, "total_assets": 1000, "operating_cash_flow": 3, "operating_income": 4},
        },
    )
    ratios = add_ratios(build_panel(records))
    return ratios, kpi_table(ratios)


def test_factor_score_is_linear_and_clamped():
    assert factor_score(0.10, 0.0, 0.20) == 50
    assert factor_score(-1, 0.0, 0.20) == 0
    assert factor_score(5, 0.0, 0.20) == 100
    assert factor_score(None, 0, 1) is None
    assert factor_score(float("nan"), 0, 1) is None


def test_scorecard_scores_and_ranks_companies():
    ratios, kpis = _ratios_and_kpis()
    card = factor_scorecard(ratios, kpis, annual_returns={"AAA": 0.10})
    assert list(card["ticker"]) == ["AAA", "BNK"]
    aaa = card.iloc[0]
    assert aaa["growth_value"] == pytest.approx(0.20)  # 100 -> 144 over 2 years
    assert aaa["growth_score"] == pytest.approx((0.20 + 0.05) / 0.30 * 100)
    assert aaa["consistency_value"] == 1.0
    assert aaa["shareholder_return_value"] == pytest.approx(0.10)
    assert aaa["factors_used"] == len(FACTORS)
    assert aaa["financial_health_value"] == pytest.approx(88 / 220)  # liabilities derived as assets − equity
    banks_skipped = factor_scorecard(ratios, kpis, annual_returns={"AAA": 0.10}, not_applicable={"financial_health": {"AAA"}})
    assert pd.isna(banks_skipped.set_index("ticker").loc["AAA", "financial_health_score"])


def test_missing_factors_are_skipped_not_zeroed():
    ratios, kpis = _ratios_and_kpis()
    bank = factor_scorecard(ratios, kpis).set_index("ticker").loc["BNK"]
    assert bank["innovation_score"] is None or pd.isna(bank["innovation_score"])
    assert bank["shareholder_return_score"] is None or pd.isna(bank["shareholder_return_score"])
    assert bank["financial_health_score"] is None or pd.isna(bank["financial_health_score"])  # no balance-sheet data
    assert bank["factors_used"] == len(FACTORS) - 3
    scored = [bank[f"{key}_score"] for key in FACTORS if bank[f"{key}_score"] is not None and not pd.isna(bank[f"{key}_score"])]
    assert bank["overall_score"] == pytest.approx(round(sum(scored) / len(scored), 1))


def test_weights_change_the_overall_score_and_zero_excludes():
    ratios, kpis = _ratios_and_kpis()
    only_growth = factor_scorecard(ratios, kpis, weights={key: 0 for key in FACTORS} | {"growth": 1}).set_index("ticker")
    assert only_growth.loc["AAA", "overall_score"] == pytest.approx(round(only_growth.loc["AAA", "growth_score"], 1))
    assert only_growth.loc["AAA", "factors_used"] == 1
    nothing = factor_scorecard(ratios, kpis, weights={key: 0 for key in FACTORS})
    assert nothing["overall_score"].isna().all()


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def _yahoo_payload(prices):
    start = pd.Timestamp("2015-01-15", tz="UTC")
    stamps = [int((start + pd.DateOffset(months=i)).timestamp()) for i in range(len(prices))]
    return {
        "chart": {
            "result": [
                {
                    "timestamp": stamps,
                    "indicators": {"quote": [{"close": prices}], "adjclose": [{"adjclose": [p * 0.9 for p in prices]}]},
                }
            ]
        }
    }


def test_yahoo_parsing_keeps_last_ten_years_monthly():
    prices = [10.0 + i for i in range(150)]  # 12.5 years of months
    frame = fetch_yahoo_monthly("BRK.B", 10, FakeSession([FakeResponse(_yahoo_payload(prices))]))
    assert list(frame.columns) == ["date", "close", "adj_close"]
    assert len(frame) == 121  # 10 years inclusive of both end months
    assert frame["date"].is_monotonic_increasing
    assert frame["adj_close"].iloc[-1] == pytest.approx(prices[-1] * 0.9)


def test_yahoo_symbol_and_errors():
    session = FakeSession([FakeResponse({}, 404)])
    with pytest.raises(PriceDataError, match="No price history"):
        fetch_yahoo_monthly("BRK.B", 10, session)
    assert session.calls[0][0].endswith("/BRK-B")
    with pytest.raises(PriceDataError, match="rate-limiting"):
        fetch_yahoo_monthly("AAPL", 10, FakeSession([FakeResponse({}, 429)]))


def test_alpha_vantage_parsing_and_fallback_to_yahoo():
    series = {
        f"{2016 + i // 12}-{i % 12 + 1:02d}-28": {"4. close": str(100 + i), "5. adjusted close": str(90 + i)} for i in range(24)
    }
    frame = fetch_alpha_vantage_monthly("IBM", "key", 10, FakeSession([FakeResponse({"Monthly Adjusted Time Series": series})]))
    assert len(frame) == 24 and frame["adj_close"].iloc[0] == 90

    limited = FakeResponse({"Information": "You have reached the daily limit."})
    session = FakeSession([limited, FakeResponse(_yahoo_payload([10.0, 11.0, 12.0]))])
    frame, source = fetch_price_history("IBM", 10, "key", session)
    assert source == "Yahoo Finance" and len(frame) == 3
    assert "apikey" in session.calls[0][1]["params"] and "apikey" not in session.calls[1][1].get("params", {})


def test_price_growth_and_indexing():
    dates = pd.date_range("2016-01-01", periods=121, freq="MS")
    up = pd.DataFrame({"date": dates, "close": 10.0, "adj_close": [10 * 1.01**i for i in range(121)]})
    dip = up.copy()
    dip.loc[60, "adj_close"] = dip.loc[59, "adj_close"] * 0.5
    stats = price_growth(up)
    assert stats["years"] == pytest.approx(10, abs=0.1)
    assert stats["total_return"] == pytest.approx(1.01**120 - 1)
    assert stats["value_of_1000"] == pytest.approx(1000 * 1.01**120)
    assert stats["max_drawdown"] == pytest.approx(0)
    assert price_growth(dip)["max_drawdown"] == pytest.approx(-0.5)
    assert price_growth(up.head(1))["total_return"] is None

    late = up[up["date"] >= "2018-01-01"]
    indexed = indexed_prices({"UP": up, "LATE": late})
    assert indexed.groupby("ticker")["date"].min().nunique() == 1  # common start
    assert indexed.groupby("ticker")["index"].first().tolist() == [100.0, 100.0]
