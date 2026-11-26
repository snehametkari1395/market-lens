"""European companies: 20-F filings, IFRS tags, currency conversion to USD and US-listed share ratios (offline)."""

from __future__ import annotations

import pytest

from finance_research.sec_client import (
    USD_PER_UNIT,
    extract_metric_records,
    shares_per_us_listing,
    usd_rate,
)

COMPANY = {"ticker": "SAP", "name": "SAP SE", "cik": 1000184}


def _entry(val, fy, end, start=None, form="20-F", filed=None, accn=None):
    return {
        "val": val, "fy": fy, "fp": "FY", "form": form, "end": end, "start": start or f"{int(end[:4])}-01-01",
        "filed": filed or f"{int(end[:4]) + 1}-03-01", "accn": accn or f"0001-{fy}",
    }


def _facts(revenue_units, eps_units=None, extra=None):
    ifrs = {"Revenue": {"units": revenue_units}}
    if eps_units:
        ifrs["DilutedEarningsLossPerShare"] = {"units": eps_units}
    facts = {"facts": {"ifrs-full": ifrs, **(extra or {})}}
    return facts


def test_ifrs_euro_figures_are_converted_to_dollars() -> None:
    facts = _facts({"EUR": [_entry(30e9, 2023, "2023-12-31"), _entry(34e9, 2024, "2024-12-31")]})
    records = extract_metric_records(COMPANY, facts, "revenue", 5)
    assert [r["fiscal_year"] for r in records] == [2023, 2024]
    latest = records[-1]
    assert latest["reported_currency"] == "EUR" and latest["reported_value"] == 34e9
    assert latest["value_usd"] == pytest.approx(34e9 * USD_PER_UNIT["EUR"][2024])
    assert latest["unit"] == "USD"


def test_main_reporting_currency_wins_over_a_few_convenience_dollar_figures() -> None:
    eur = [_entry(30e9 + year, year, f"{year}-12-31") for year in (2022, 2023, 2024)]
    usd = [_entry(37e9, 2024, "2024-12-31")]  # one convenience translation
    records = extract_metric_records(COMPANY, _facts({"EUR": eur, "USD": usd}), "revenue", 5)
    assert len(records) == 3 and {r["reported_currency"] for r in records} == {"EUR"}


def test_per_share_figures_follow_the_us_listing_ratio() -> None:
    company = {"ticker": "AZN", "name": "AstraZeneca", "cik": 901832}
    facts = _facts({"USD": [_entry(50e9, 2024, "2024-12-31")]}, {"USD/shares": [_entry(4.0, 2024, "2024-12-31")]})
    eps = extract_metric_records(company, facts, "eps_diluted", 3)
    assert eps[-1]["value_usd"] == pytest.approx(2.0)  # one AstraZeneca ADR is half an ordinary share
    assert shares_per_us_listing("azn") == 0.5 and shares_per_us_listing("AAPL") == 1.0


def test_newest_taxonomy_is_used_first() -> None:
    # Fresenius reported under US GAAP until 2017, then IFRS: the IFRS values must win for recent years.
    facts = _facts(
        {"EUR": [_entry(20e9, 2024, "2024-12-31")]},
        extra={"us-gaap": {"Revenues": {"units": {"USD": [_entry(18e9, 2017, "2017-12-31", form="20-F")]}}}},
    )
    records = extract_metric_records({"ticker": "FMS", "name": "FMC", "cik": 1}, facts, "revenue", 3)
    assert [r["fiscal_year"] for r in records] == [2024]


def test_dividends_paid_is_positive_and_unknown_currencies_are_skipped() -> None:
    facts = {"facts": {"ifrs-full": {
        "DividendsPaid": {"units": {"EUR": [_entry(-1.2e9, 2024, "2024-12-31")]}},
        "Revenue": {"units": {"JPY": [_entry(5e12, 2024, "2024-12-31")]}},
    }}}
    assert extract_metric_records(COMPANY, facts, "dividends_paid", 3)[-1]["reported_value"] == 1.2e9
    assert extract_metric_records(COMPANY, facts, "revenue", 3) == []  # no exchange rates for yen


def test_exchange_rates() -> None:
    assert usd_rate("USD", 2024) == 1.0
    assert usd_rate("EUR", 2030) == USD_PER_UNIT["EUR"][max(USD_PER_UNIT["EUR"])]
    assert usd_rate("DKK", 2024) == pytest.approx(USD_PER_UNIT["EUR"][2024] / 7.45)
    assert usd_rate("JPY", 2024) is None


def test_european_sector_cards() -> None:
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    sectors = next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SECTORS"
    )
    cards = {card["key"]: card for card in ast.literal_eval(sectors)}
    assert list(cards["germany"]["companies"]) == ["SAP", "DB", "FMS", "BNTX", "EVO"]
    assert len(cards["europe"]["companies"]) == 5
