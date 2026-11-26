"""Valuation ratios: today's share price (Yahoo Finance) combined with SEC-reported fundamentals.

Market cap = price x shares outstanding. Shares come from the cover page of the company's
latest 10-K/10-Q (dei:EntityCommonStockSharesOutstanding). Companies with several share
classes (e.g. Alphabet) don't report one total there, so the weighted-average diluted share
count from the latest filing is used instead. Ratios use the latest fiscal year's figures.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from finance_research.sec_client import extract_metric_records, get_company_facts, shares_per_us_listing

STALE_DAYS = 400  # a share count older than this (e.g. a company that stopped filing) isn't trusted
VALUATION_NOTE = (
    "Market cap uses SEC-reported shares outstanding times the latest price, so it can differ slightly from "
    "quote sites. P/E, P/S and P/B use the latest fiscal year's reported figures, not trailing-twelve-month numbers."
)


def _age_days(day: str, today: date) -> int:
    try:
        return (today - datetime.strptime(day, "%Y-%m-%d").date()).days
    except ValueError:
        return 10**6


def shares_outstanding(facts: dict[str, Any], today: date | None = None) -> dict[str, Any] | None:
    """Latest share count as {shares, as_of, basis}, or None when nothing recent is reported."""
    today = today or date.today()
    cover = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    if cover:
        latest = max(cover, key=lambda row: (row.get("end", ""), row.get("filed", "")))
        if _age_days(latest.get("end", ""), today) <= STALE_DAYS and latest.get("val"):
            return {"shares": float(latest["val"]), "as_of": latest["end"], "basis": "Shares outstanding (filing cover page)"}
    diluted = (
        facts.get("facts", {}).get("us-gaap", {}).get("WeightedAverageNumberOfDilutedSharesOutstanding", {})
        .get("units", {}).get("shares", [])
    ) or (  # IFRS reporters (European companies)
        facts.get("facts", {}).get("ifrs-full", {}).get("AdjustedWeightedAverageShares", {})
        .get("units", {}).get("shares", [])
    )
    if diluted:
        latest = max(diluted, key=lambda row: (row.get("end", ""), row.get("filed", "")))
        if _age_days(latest.get("end", ""), today) <= STALE_DAYS and latest.get("val"):
            return {"shares": float(latest["val"]), "as_of": latest["end"], "basis": "Weighted-average diluted shares"}
    return None


def _latest(company: dict[str, Any], facts: dict[str, Any], metric: str) -> dict[str, Any] | None:
    records = extract_metric_records(company, facts, metric, 3)
    return records[-1] if records else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator or denominator <= 0:
        return None  # negative earnings or equity make the ratio meaningless
    return numerator / denominator


def compute_valuation(company: dict[str, Any], facts: dict[str, Any], quote: dict[str, Any], today: date | None = None) -> dict[str, Any]:
    price = float(quote["price"])
    shares = shares_outstanding(facts, today)
    if shares and shares_per_us_listing(company["ticker"]) != 1.0:
        # The US price is per ADR, which stands for a different number of ordinary shares.
        ratio = shares_per_us_listing(company["ticker"])
        shares = {**shares, "shares": shares["shares"] / ratio, "basis": f"{shares['basis']}, as US-listed shares (1 = {ratio:g} ordinary)"}
    inputs = {
        metric: _latest(company, facts, metric)
        for metric in ("eps_diluted", "revenue", "stockholders_equity", "long_term_debt", "cash", "net_income")
    }
    value = {metric: (row["value_usd"] if row else None) for metric, row in inputs.items()}
    market_cap = price * shares["shares"] if shares else None
    enterprise_value = (
        market_cap + (value["long_term_debt"] or 0) - (value["cash"] or 0) if market_cap is not None else None
    )
    return {
        "ticker": company["ticker"],
        "company": company["name"],
        "price": price,
        "price_as_of": quote.get("as_of"),
        "market_cap": market_cap,
        "shares_outstanding": shares["shares"] if shares else None,
        "shares_basis": f"{shares['basis']}, {shares['as_of']}" if shares else "Not reported recently",
        "pe": _ratio(price, value["eps_diluted"]),
        "ps": _ratio(market_cap, value["revenue"]),
        "pb": _ratio(market_cap, value["stockholders_equity"]),
        "enterprise_value": enterprise_value,
        "ev_revenue": _ratio(enterprise_value, value["revenue"]),
        "earnings_yield_percent": (value["eps_diluted"] / price * 100) if value["eps_diluted"] is not None and price else None,
        "dividend_yield_percent": quote.get("dividend_yield"),
        "fiscal_year": max((row["fiscal_year"] for row in inputs.values() if row), default=None),
        "inputs": {
            metric: {"value": row["value_usd"], "fiscal_year": row["fiscal_year"], "source_url": row["source_url"]}
            for metric, row in inputs.items()
            if row
        },
        "sources": [row["source_url"] for row in inputs.values() if row and row.get("source_url")],
        "note": VALUATION_NOTE,
    }


def get_valuation(ticker: str, user_agent: str, quote: dict[str, Any]) -> dict[str, Any]:
    company, facts = get_company_facts(ticker, user_agent)
    return compute_valuation(company, facts, quote)
