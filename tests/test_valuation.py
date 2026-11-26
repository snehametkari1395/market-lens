from datetime import date

from finance_research import market
from finance_research.valuation import compute_valuation, shares_outstanding

COMPANY = {"ticker": "TEST", "cik": 1, "name": "Test Corp"}
TODAY = date(2026, 11, 26)


def annual(value, fy=2025, end="2025-12-31"):
    return {"val": value, "fy": fy, "fp": "FY", "form": "10-K", "start": f"{fy}-01-01", "end": end,
            "filed": "2026-02-01", "accn": "0000000001-26-000001"}


def instant(value, fy=2025, end="2025-12-31"):
    return {"val": value, "fy": fy, "fp": "FY", "form": "10-K", "end": end, "filed": "2026-02-01", "accn": "0000000001-26-000001"}


def facts(cover=None, diluted=None, eps=5.0, equity=200.0, revenue=1000.0):
    us_gaap = {
        "EarningsPerShareDiluted": {"units": {"USD/shares": [annual(eps)]}},
        "Revenues": {"units": {"USD": [annual(revenue)]}},
        "StockholdersEquity": {"units": {"USD": [instant(equity)]}},
        "LongTermDebt": {"units": {"USD": [instant(150.0)]}},
        "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [instant(50.0)]}},
        "NetIncomeLoss": {"units": {"USD": [annual(500.0)]}},
    }
    if diluted:
        us_gaap["WeightedAverageNumberOfDilutedSharesOutstanding"] = {"units": {"shares": diluted}}
    payload = {"facts": {"us-gaap": us_gaap}}
    if cover:
        payload["facts"]["dei"] = {"EntityCommonStockSharesOutstanding": {"units": {"shares": cover}}}
    return payload


def test_shares_prefer_recent_cover_page_then_diluted_and_ignore_stale() -> None:
    cover = [{"val": 100, "end": "2026-10-20", "filed": "2026-10-30"}, {"val": 90, "end": "2025-10-20", "filed": "2025-10-30"}]
    assert shares_outstanding(facts(cover=cover), TODAY)["shares"] == 100
    diluted = [{"val": 120, "end": "2026-09-30", "filed": "2026-10-30"}]
    result = shares_outstanding(facts(diluted=diluted), TODAY)
    assert result["shares"] == 120 and result["basis"].startswith("Weighted-average")
    stale = [{"val": 956153, "end": "2011-04-29", "filed": "2011-05-01"}]
    assert shares_outstanding(facts(cover=stale), TODAY) is None


def test_valuation_ratios() -> None:
    cover = [{"val": 10, "end": "2026-10-20", "filed": "2026-10-30"}]
    result = compute_valuation(COMPANY, facts(cover=cover), {"price": 50.0, "dividend_yield": 1.5}, TODAY)
    assert result["market_cap"] == 500
    assert result["pe"] == 10  # 50 / 5
    assert result["ps"] == 0.5 and result["pb"] == 2.5
    assert result["enterprise_value"] == 600 and result["ev_revenue"] == 0.6
    assert result["dividend_yield_percent"] == 1.5 and result["fiscal_year"] == 2025
    assert result["sources"] and result["inputs"]["eps_diluted"]["value"] == 5.0


def test_negative_earnings_and_missing_shares_leave_ratios_blank() -> None:
    result = compute_valuation(COMPANY, facts(eps=-2.0), {"price": 50.0}, TODAY)
    assert result["pe"] is None and result["market_cap"] is None and result["ps"] is None
    assert result["shares_basis"] == "Not reported recently"


def test_price_series_parses_closes() -> None:
    class Response:
        status_code = 200

        def json(self):
            return {"chart": {"result": [{"timestamp": [1764547200, 1764633600], "indicators": {"quote": [{"close": [10.0, None]}]}}]}}

        def raise_for_status(self):
            pass

    class Session:
        def get(self, url, params, headers, timeout):
            assert params == {"range": "5y", "interval": "1wk"}
            return Response()

    assert market.price_series("aapl", "5y", session=Session()) == [{"date": "2025-12-01", "close": 10.0}]
