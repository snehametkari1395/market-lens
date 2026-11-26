from datetime import date

import pytest

from finance_research import market
from finance_research.auth import UserStore
from finance_research.portfolio import clean_alerts, clean_holdings, clean_watchlist, triggered_alerts, value_holdings
from finance_research.prices import PriceDataError
from finance_research.sec_client import METRICS, RATIOS
from finance_research.tools import ToolKit, build_tools


class FakeResponse:
    def __init__(self, payload=None, text="", status=200):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, response):
        self.response, self.calls = response, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        return self.response


def test_quote_reads_price_change_and_trailing_dividends() -> None:
    payload = {"chart": {"result": [{
        "meta": {"regularMarketPrice": 102.0, "longName": "Coca-Cola", "fiftyTwoWeekHigh": 110, "fiftyTwoWeekLow": 90,
                 "regularMarketTime": 1795640402},
        "indicators": {"quote": [{"close": [98.0, None, 100.0, 102.0]}]},
        "events": {"dividends": {"1": {"amount": 0.5}, "2": {"amount": 0.5}}},
    }]}}
    quote = market.get_quote("ko", session=FakeSession(FakeResponse(payload)))
    assert quote["ticker"] == "KO" and quote["price"] == 102.0
    assert quote["previous_close"] == 100.0 and round(quote["change_percent"], 2) == 2.0
    assert quote["dividends_12m"] == 1.0 and round(quote["dividend_yield"], 3) == round(1 / 102 * 100, 3)


def test_quote_rejects_bad_tickers_and_missing_prices() -> None:
    with pytest.raises(ValueError):
        market.get_quote("not a ticker!")
    with pytest.raises(PriceDataError):
        market.get_quote("ZZZZ", session=FakeSession(FakeResponse({"chart": {"result": [{"meta": {}}]}})))


EARNINGS_CSV = (
    "symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\r\n"
    "CRM,SALESFORCE INC,2026-12-02,2026-10-31,2.41,USD,post-market\r\n"
    "COST,COSTCO WHOLESALE CORP,2026-12-10,2026-11-30,,USD,\r\n"
    "AAPL,APPLE INC,2027-03-30,2026-12-31,2.10,USD,pre-market\r\n"
)


def test_earnings_calendar_parses_and_filters() -> None:
    session = FakeSession(FakeResponse(text=EARNINGS_CSV))
    events = market.earnings_calendar("", session=session)
    assert [event["ticker"] for event in events] == ["CRM", "COST", "AAPL"]
    assert events[0]["time"] == "After close" and events[0]["eps_estimate"] == 2.41 and events[1]["eps_estimate"] is None
    params = session.calls[0][1]
    assert list(params) == ["function", "horizon", "apikey"] and params["apikey"] == "demo"  # demo key needs this order
    soon = market.upcoming_for(events, ["crm", "aapl"], days=30, today=date(2026, 11, 26))
    assert [event["ticker"] for event in soon] == ["CRM"]


def test_calendar_rate_limit_messages_are_errors() -> None:
    squeezed = "symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\r\nI,n,f,o,r,m,a\r\n"
    with pytest.raises(PriceDataError, match="rate-limited"):
        market.earnings_calendar("", session=FakeSession(FakeResponse(text=squeezed)))
    note = FakeResponse({"Note": "Thank you for using Alpha Vantage! Limit reached."}, text='{"Note": "limit"}')
    with pytest.raises(PriceDataError, match="Limit reached"):
        market.ipo_calendar("key", session=FakeSession(note))


def test_watchlist_holdings_and_alerts_validation() -> None:
    assert clean_watchlist(["aapl", "$msft", "AAPL"]) == ["AAPL", "MSFT"]
    with pytest.raises(ValueError):
        clean_watchlist(["bad ticker"])
    holdings = clean_holdings([{"ticker": "aapl", "shares": 10, "cost_per_share": 150}, {"ticker": "", "shares": None}])
    assert holdings == [{"ticker": "AAPL", "shares": 10.0, "cost_per_share": 150.0}]
    with pytest.raises(ValueError, match="Shares for MSFT"):
        clean_holdings([{"ticker": "MSFT", "shares": 0, "cost_per_share": 1}])
    assert clean_alerts([{"ticker": "ko", "condition": "weird", "price": "60"}]) == [{"ticker": "KO", "condition": "above", "price": 60.0}]


def test_portfolio_value_and_triggered_alerts() -> None:
    holdings = [
        {"ticker": "AAPL", "shares": 10, "cost_per_share": 100},
        {"ticker": "XYZ", "shares": 5, "cost_per_share": 20},  # no price: counted at cost
    ]
    result = value_holdings(holdings, {"AAPL": 150.0})
    assert result["total_cost"] == 1100 and result["total_value"] == 1600 and result["total_gain"] == 500
    assert result["missing_prices"] == ["XYZ"]
    assert round(sum(row["weight_percent"] for row in result["rows"]), 6) == 100
    alerts = [{"ticker": "AAPL", "condition": "above", "price": 140}, {"ticker": "AAPL", "condition": "below", "price": 100},
              {"ticker": "XYZ", "condition": "below", "price": 50}]
    assert triggered_alerts(alerts, {"AAPL": 150.0}) == [{**alerts[0], "current_price": 150.0}]


def test_user_store_keeps_lists_per_user(tmp_path) -> None:
    store = UserStore(tmp_path / "users.json")
    store.create_user("alice", "password123")
    store.create_user("bob", "password123")
    store.save_list("alice", "watchlist_items", ["AAPL"])
    store.save_list("Alice", "holdings", [{"ticker": "KO", "shares": 1.0, "cost_per_share": 50.0}])
    assert store.get_lists("alice")["watchlist_items"] == ["AAPL"] and store.get_lists("alice")["holdings"][0]["ticker"] == "KO"
    assert store.get_lists("bob") == {"watchlist_items": [], "holdings": [], "alerts": []}
    with pytest.raises(ValueError):
        store.save_list("alice", "password_hash", ["x"])


def test_dividend_metrics_ratio_and_earnings_tool(monkeypatch) -> None:
    assert METRICS["dividends_per_share"]["unit"] == "USD/shares" and RATIOS["payout_ratio"] == ("dividends_paid", "net_income")
    assert "get_upcoming_earnings" in [item.name for item in build_tools("Test test@example.com", None)]
    monkeypatch.setattr("finance_research.tools.earnings_calendar", lambda key: market.earnings_calendar(
        key, session=FakeSession(FakeResponse(text=EARNINGS_CSV))))
    monkeypatch.setattr("finance_research.tools.upcoming_for", lambda events, tickers, days: market.upcoming_for(
        events, tickers, days, today=date(2026, 11, 26)))
    result = ToolKit("Test test@example.com", None).get_upcoming_earnings(["crm", "nflx"])
    assert [event["ticker"] for event in result["earnings"]] == ["CRM"] and result["not_scheduled"] == ["NFLX"]


def test_blank_editor_cells_are_ignored_and_missing_numbers_rejected() -> None:
    nan = float("nan")
    assert clean_holdings([{"ticker": nan, "shares": nan, "cost_per_share": nan}]) == []
    with pytest.raises(ValueError, match="missing"):
        clean_holdings([{"ticker": "AAPL", "shares": nan, "cost_per_share": 10}])
    assert clean_alerts([{"ticker": None, "condition": None, "price": nan}]) == []
