"""Latest share prices and the earnings / IPO calendars.

Quotes come from Yahoo Finance's public chart endpoint (no key; unofficial, so it can
rate-limit). The calendars come from Alpha Vantage: each is a single CSV download that
covers the next three months, so one request a day is enough even on the free tier.
"""
from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests

from finance_research.prices import BROWSER_HEADERS, YAHOO_URL, PriceDataError, _yahoo_symbol

ALPHA_URL = "https://www.alphavantage.co/query"
# Alpha Vantage serves both calendars with its public demo key; a user's own key is used when given.
ALPHA_DEMO_KEY = "demo"
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def normalize_ticker(value: str) -> str:
    ticker = value.strip().upper().lstrip("$")
    if not TICKER_PATTERN.match(ticker):
        raise ValueError(f"'{value.strip()}' doesn't look like a ticker symbol.")
    return ticker


def get_quote(ticker: str, session: Any = requests) -> dict[str, Any]:
    """Latest price, change since the previous close, 52-week range and trailing 12-month dividends."""
    ticker = normalize_ticker(ticker)
    response = session.get(
        YAHOO_URL.format(symbol=_yahoo_symbol(ticker)),
        params={"range": "1y", "interval": "1d", "events": "div"},
        headers=BROWSER_HEADERS,
        timeout=15,
    )
    if response.status_code == 404:
        raise PriceDataError(f"No price found for {ticker}.")
    if response.status_code == 429:
        raise PriceDataError("The price service is rate-limiting requests. Try again in a minute.")
    response.raise_for_status()
    result = ((response.json().get("chart") or {}).get("result") or [None])[0] or {}
    meta = result.get("meta") or {}
    price = meta.get("regularMarketPrice")
    if price is None:
        raise PriceDataError(f"No price found for {ticker}.")
    price = float(price)
    closes = [c for c in ((result.get("indicators", {}).get("quote") or [{}])[0].get("close") or []) if c is not None]
    previous = float(closes[-2]) if len(closes) >= 2 else None  # the last full session before the latest one
    dividends = sum(float(item.get("amount") or 0) for item in ((result.get("events") or {}).get("dividends") or {}).values())
    stamp = meta.get("regularMarketTime")
    return {
        "ticker": ticker,
        "name": meta.get("longName") or meta.get("shortName") or ticker,
        "price": price,
        "previous_close": previous,
        "change_percent": (price / previous - 1) * 100 if previous else None,
        "high_52w": meta.get("fiftyTwoWeekHigh"),
        "low_52w": meta.get("fiftyTwoWeekLow"),
        "dividends_12m": dividends,
        "dividend_yield": dividends / price * 100 if price else None,
        "currency": meta.get("currency") or "USD",
        "as_of": datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if stamp else None,
        "source": "Yahoo Finance",
    }


def _alpha_csv(function: str, api_key: str, session: Any, **params: str) -> list[dict[str, str]]:
    # Parameter order matters: the demo key only answers the exact URL shape from Alpha Vantage's docs.
    response = session.get(
        ALPHA_URL, params={"function": function, **params, "apikey": api_key.strip() or ALPHA_DEMO_KEY}, timeout=30
    )
    response.raise_for_status()
    text = response.text.strip()
    if text.startswith("{"):  # errors and rate-limit notes come back as JSON instead of CSV
        payload = response.json()
        raise PriceDataError(payload.get("Information") or payload.get("Note") or payload.get("Error Message") or "Calendar unavailable.")
    rows = list(csv.DictReader(io.StringIO(text)))
    date_field = "reportDate" if function == "EARNINGS_CALENDAR" else "ipoDate"
    valid = [row for row in rows if DATE_PATTERN.match(row.get(date_field) or "")]
    if rows and not valid:  # rate-limit messages can also arrive squeezed into the CSV columns
        raise PriceDataError("The calendar service is busy or rate-limited. Try again later, or add your own Alpha Vantage key.")
    return valid


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def earnings_calendar(api_key: str = "", session: Any = requests) -> list[dict[str, Any]]:
    """Every company expected to report in the next three months, soonest first."""
    rows = _alpha_csv("EARNINGS_CALENDAR", api_key, session, horizon="3month")
    events = [
        {
            "ticker": row.get("symbol", "").upper(),
            "name": (row.get("name") or "").title(),
            "date": row.get("reportDate", ""),
            "fiscal_period_end": row.get("fiscalDateEnding", ""),
            "eps_estimate": _float(row.get("estimate")),
            "time": {"pre-market": "Before open", "post-market": "After close"}.get(row.get("timeOfTheDay", ""), ""),
        }
        for row in rows
        if row.get("symbol") and row.get("reportDate")
    ]
    return sorted(events, key=lambda event: (event["date"], event["ticker"]))


def ipo_calendar(api_key: str = "", session: Any = requests) -> list[dict[str, Any]]:
    """IPOs expected in the next three months, soonest first."""
    rows = _alpha_csv("IPO_CALENDAR", api_key, session)
    events = [
        {
            "ticker": row.get("symbol", "").upper(),
            "name": row.get("name", ""),
            "date": row.get("ipoDate", ""),
            "price_low": _float(row.get("priceRangeLow")) or None,  # 0 means "not set yet"
            "price_high": _float(row.get("priceRangeHigh")) or None,
            "exchange": row.get("exchange", ""),
        }
        for row in rows
        if row.get("symbol") and row.get("ipoDate")
    ]
    return sorted(events, key=lambda event: (event["date"], event["ticker"]))


def upcoming_for(events: list[dict[str, Any]], tickers: list[str], days: int = 90, today: date | None = None) -> list[dict[str, Any]]:
    """Calendar events for the given tickers within the next `days` days."""
    start = today or date.today()
    end = (start + timedelta(days=days)).isoformat()
    wanted = {ticker.upper() for ticker in tickers}
    return [event for event in events if event["ticker"] in wanted and start.isoformat() <= event["date"] <= end]


# range key -> (Yahoo range, bar interval)
PRICE_RANGES = {
    "1mo": ("1mo", "1d"),
    "6mo": ("6mo", "1d"),
    "1y": ("1y", "1d"),
    "5y": ("5y", "1wk"),
    "max": ("max", "1mo"),
}


def price_series(ticker: str, range_key: str = "1y", session: Any = requests) -> list[dict[str, Any]]:
    """Closing prices as [{date, close}] for the chart (daily for up to a year, weekly or monthly beyond)."""
    ticker = normalize_ticker(ticker)
    yahoo_range, interval = PRICE_RANGES.get(range_key, PRICE_RANGES["1y"])
    response = session.get(
        YAHOO_URL.format(symbol=_yahoo_symbol(ticker)),
        params={"range": yahoo_range, "interval": interval},
        headers=BROWSER_HEADERS,
        timeout=20,
    )
    if response.status_code == 429:
        raise PriceDataError("The price service is rate-limiting requests. Try again in a minute.")
    if response.status_code == 404:
        raise PriceDataError(f"No price history found for {ticker}.")
    response.raise_for_status()
    result = ((response.json().get("chart") or {}).get("result") or [None])[0] or {}
    stamps = result.get("timestamp") or []
    closes = ((result.get("indicators", {}).get("quote") or [{}])[0].get("close")) or []
    rows = [
        {"date": datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%d"), "close": float(close)}
        for stamp, close in zip(stamps, closes)
        if close is not None
    ]
    if not rows:
        raise PriceDataError(f"No price history found for {ticker}.")
    return rows
