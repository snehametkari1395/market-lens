"""Watchlist, portfolio holdings and price alerts: validation and calculations (no Streamlit)."""
from __future__ import annotations

import math
from typing import Any

from finance_research.market import normalize_ticker

ALERT_CONDITIONS = {"above": "Price rises above", "below": "Price falls below"}
MAX_ITEMS = 50


def clean_watchlist(tickers: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in tickers:
        ticker = normalize_ticker(str(value))
        if ticker not in cleaned:
            cleaned.append(ticker)
    return cleaned[:MAX_ITEMS]


def _blank(value: Any) -> bool:
    """Empty editor cells arrive as None, '', NaN or pandas' NA."""
    if value is None or type(value).__name__ == "NAType":
        return True
    return (isinstance(value, float) and math.isnan(value)) or not str(value).strip()


def _positive(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    if math.isnan(number):
        raise ValueError(f"{label} is missing.")
    if number <= 0:
        raise ValueError(f"{label} must be more than zero.")
    return number


def clean_holdings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows of {ticker, shares, cost_per_share}; blank rows are dropped, bad ones raise ValueError."""
    holdings = []
    for row in rows:
        if _blank(row.get("ticker")):
            continue
        ticker = normalize_ticker(str(row["ticker"]))
        holdings.append(
            {
                "ticker": ticker,
                "shares": _positive(row.get("shares"), f"Shares for {ticker}"),
                "cost_per_share": _positive(row.get("cost_per_share"), f"Cost per share for {ticker}"),
            }
        )
    return holdings[:MAX_ITEMS]


def clean_alerts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alerts = []
    for row in rows:
        if _blank(row.get("ticker")):
            continue
        ticker = normalize_ticker(str(row["ticker"]))
        condition = row.get("condition") if row.get("condition") in ALERT_CONDITIONS else "above"
        alerts.append({"ticker": ticker, "condition": condition, "price": _positive(row.get("price"), f"Alert price for {ticker}")})
    return alerts[:MAX_ITEMS]


def value_holdings(holdings: list[dict[str, Any]], prices: dict[str, float]) -> dict[str, Any]:
    """Market value and gain per holding plus totals. Holdings without a price are counted at cost and flagged."""
    rows, total_cost, total_value = [], 0.0, 0.0
    for holding in holdings:
        cost = holding["shares"] * holding["cost_per_share"]
        price = prices.get(holding["ticker"])
        value = holding["shares"] * price if price is not None else None
        rows.append(
            {
                **holding,
                "price": price,
                "cost": cost,
                "value": value,
                "gain": value - cost if value is not None else None,
                "gain_percent": (value / cost - 1) * 100 if value is not None and cost else None,
            }
        )
        total_cost += cost
        total_value += value if value is not None else cost
    for row in rows:
        row["weight_percent"] = ((row["value"] if row["value"] is not None else row["cost"]) / total_value * 100) if total_value else 0.0
    return {
        "rows": rows,
        "total_cost": total_cost,
        "total_value": total_value,
        "total_gain": total_value - total_cost,
        "total_gain_percent": (total_value / total_cost - 1) * 100 if total_cost else 0.0,
        "missing_prices": [row["ticker"] for row in rows if row["price"] is None],
    }


def triggered_alerts(alerts: list[dict[str, Any]], prices: dict[str, float]) -> list[dict[str, Any]]:
    hits = []
    for alert in alerts:
        price = prices.get(alert["ticker"])
        if price is None:
            continue
        if (alert["condition"] == "above" and price > alert["price"]) or (alert["condition"] == "below" and price < alert["price"]):
            hits.append({**alert, "current_price": price})
    return hits
