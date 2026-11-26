"""Monthly share-price history for the growth chart.

Alpha Vantage (official API, needs the user's key) is used when a key is available. Without a key the
app falls back to Yahoo Finance's public chart endpoint, which is unofficial and may change or be
rate-limited. Prices are split- and dividend-adjusted so growth reflects total shareholder return.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
ALPHA_URL = "https://www.alphavantage.co/query"
BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Market Lens research app)"}
PRICE_COLUMNS = ["date", "close", "adj_close"]


class PriceDataError(RuntimeError):
    pass


def _yahoo_symbol(ticker: str) -> str:
    return ticker.upper().replace(".", "-")  # e.g. BRK.B -> BRK-B


def _to_frame(rows: list[tuple[pd.Timestamp, float | None, float | None]], years: int) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=PRICE_COLUMNS).dropna(subset=["adj_close"])
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    cutoff = frame["date"].max() - pd.DateOffset(years=years)
    return frame[frame["date"] >= cutoff].reset_index(drop=True)


def fetch_yahoo_monthly(ticker: str, years: int = 10, session: Any = requests) -> pd.DataFrame:
    response = session.get(
        YAHOO_URL.format(symbol=_yahoo_symbol(ticker)),
        params={"range": f"{years + 1}y", "interval": "1mo", "events": "div,split"},
        headers=BROWSER_HEADERS,
        timeout=20,
    )
    if response.status_code == 404:
        raise PriceDataError(f"No price history found for {ticker}.")
    if response.status_code == 429:
        raise PriceDataError("The price service is rate-limiting requests. Try again in a minute.")
    response.raise_for_status()
    payload = response.json().get("chart") or {}
    if payload.get("error") or not payload.get("result"):
        raise PriceDataError(f"No price history found for {ticker}.")
    result = payload["result"][0]
    stamps = result.get("timestamp") or []
    closes = (result.get("indicators", {}).get("quote") or [{}])[0].get("close") or []
    adjusted = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or closes
    rows = [
        (pd.Timestamp(datetime.fromtimestamp(stamp, tz=timezone.utc)).tz_localize(None), close, adj)
        for stamp, close, adj in zip(stamps, closes, adjusted)
    ]
    return _to_frame(rows, years)


def fetch_alpha_vantage_monthly(ticker: str, api_key: str, years: int = 10, session: Any = requests) -> pd.DataFrame:
    response = session.get(
        ALPHA_URL,
        params={"function": "TIME_SERIES_MONTHLY_ADJUSTED", "symbol": ticker.upper(), "apikey": api_key},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    series = payload.get("Monthly Adjusted Time Series")
    if not series:
        message = payload.get("Note") or payload.get("Information") or payload.get("Error Message")
        raise PriceDataError(message or f"Alpha Vantage returned no prices for {ticker}.")
    rows = [
        (pd.Timestamp(day), float(values["4. close"]), float(values["5. adjusted close"]))
        for day, values in series.items()
    ]
    return _to_frame(rows, years)


def fetch_price_history(ticker: str, years: int = 10, alpha_key: str = "", session: Any = requests) -> tuple[pd.DataFrame, str]:
    """Return (monthly prices, source name). Tries Alpha Vantage first when a key is given."""
    alpha_error = None
    if alpha_key.strip():
        try:
            return fetch_alpha_vantage_monthly(ticker, alpha_key.strip(), years, session), "Alpha Vantage"
        except (PriceDataError, requests.RequestException, ValueError) as error:
            alpha_error = error  # e.g. free-tier daily limit; fall back below
    try:
        return fetch_yahoo_monthly(ticker, years, session), "Yahoo Finance"
    except (requests.RequestException, ValueError) as error:
        raise PriceDataError(f"Couldn't load prices for {ticker}: {alpha_error or error}") from error


def price_growth(prices: pd.DataFrame) -> dict[str, Any]:
    """Total return, annualised return (CAGR), worst drawdown and growth of $1,000 over the period."""
    if prices is None or len(prices) < 2:
        return {"years": 0, "total_return": None, "annual_return": None, "max_drawdown": None, "value_of_1000": None}
    first, last = prices.iloc[0], prices.iloc[-1]
    years = (last["date"] - first["date"]).days / 365.25
    ratio = last["adj_close"] / first["adj_close"]
    running_peak = prices["adj_close"].cummax()
    return {
        "start": first["date"],
        "end": last["date"],
        "years": round(years, 1),
        "start_price": float(first["close"]),
        "end_price": float(last["close"]),
        "total_return": ratio - 1,
        "annual_return": ratio ** (1 / years) - 1 if years > 0 else None,
        "max_drawdown": float((prices["adj_close"] / running_peak - 1).min()),
        "value_of_1000": 1000 * ratio,
    }


def indexed_prices(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every company's adjusted price set to 100 at the first month they all trade, for a fair comparison."""
    usable = {ticker: frame for ticker, frame in histories.items() if frame is not None and not frame.empty}
    if not usable:
        return pd.DataFrame(columns=["ticker", "date", "index", "close"])
    common_start = max(frame["date"].min() for frame in usable.values())
    parts = []
    for ticker, frame in usable.items():
        window = frame[frame["date"] >= common_start].copy()
        base = window["adj_close"].iloc[0]
        window["index"] = window["adj_close"] / base * 100
        window["ticker"] = ticker
        parts.append(window[["ticker", "date", "index", "close"]])
    return pd.concat(parts, ignore_index=True)
