from finance_research.sec_client import extract_metric_records
from finance_research.screening import build_research_shortlist
from finance_research.tools import _ratio_records


def _facts(revenue: list[tuple[int, float]], operating_income: list[tuple[int, float]]) -> dict:
    facts = {"facts": {"us-gaap": {}}}
    for tag, values in (
        ("RevenueFromContractWithCustomerExcludingAssessedTax", revenue),
        ("OperatingIncomeLoss", operating_income),
    ):
        facts["facts"]["us-gaap"][tag] = {
            "units": {
                "USD": [
                    {
                        "form": "10-K",
                        "fp": "FY",
                        "fy": year,
                        "val": amount,
                        "filed": f"{year + 1}-02-01",
                        "accn": f"0000000000-{year + 1:02d}-000001",
                    }
                    for year, amount in values
                ]
            }
        }
    return facts


def test_extract_metric_records_selects_latest_annual_records() -> None:
    facts = _facts([(2023, 100), (2024, 125)], [(2023, 10), (2024, 15)])
    company = {"ticker": "TEST", "cik": 1, "name": "Test Company"}

    records = extract_metric_records(company, facts, "revenue", 2)

    assert [row["fiscal_year"] for row in records] == [2023, 2024]
    assert records[-1]["value_usd"] == 125
    assert records[-1]["source_url"].startswith("https://www.sec.gov/Archives/")


def test_operating_margin_uses_sec_values(monkeypatch) -> None:
    def fake_records(ticker: str, metric: str, years: int, user_agent: str) -> list[dict]:
        values = {"revenue": 200, "operating_income": 30}
        return [
            {
                "fiscal_year": 2024,
                "value_usd": values[metric],
                "source_url": "https://www.sec.gov/example",
            }
        ]

    monkeypatch.setattr("finance_research.tools.get_metric_records", fake_records)

    result = _ratio_records("test", "operating_margin", 2024, "test@example.com")

    assert result["value_percent"] == 15
    assert result["formula"] == "(operating_income / revenue) × 100"


def test_revenue_growth_uses_prior_year_and_current_year(monkeypatch) -> None:
    def fake_records(ticker: str, metric: str, years: int, user_agent: str) -> list[dict]:
        return [
            {
                "fiscal_year": year,
                "value_usd": amount,
                "source_url": f"https://www.sec.gov/example/{year}",
            }
            for year, amount in [(2023, 100), (2024, 125)]
        ]

    monkeypatch.setattr("finance_research.tools.get_metric_records", fake_records)

    result = _ratio_records("test", "revenue_growth", 2024, "test@example.com")

    assert result["value_percent"] == 25
    assert len(result["sources"]) == 2


def test_extract_metric_records_rejects_unsupported_metric() -> None:
    company = {"ticker": "TEST", "cik": 1, "name": "Test Company"}

    try:
        extract_metric_records(company, {"facts": {"us-gaap": {}}}, "unknown", 3)
    except ValueError as error:
        assert "Unsupported metric" in str(error)
    else:
        raise AssertionError("Unsupported metrics must be rejected.")


def test_research_shortlist_explains_why_company_matches_selected_screen() -> None:
    records = {
        "TEST": {
            "revenue": [
                {"fiscal_year": 2023, "value_usd": 100, "source_url": "https://sec.test/2023"},
                {"fiscal_year": 2024, "value_usd": 120, "source_url": "https://sec.test/2024"},
            ],
            "net_income": [
                {
                    "company": "Test Company",
                    "fiscal_year": 2024,
                    "value_usd": 18,
                    "source_url": "https://sec.test/2024",
                }
            ],
        }
    }

    result = build_research_shortlist(records, 10, 10)[0]

    assert result["growth_match"] is True
    assert result["margin_match"] is True
    assert result["revenue_growth_percent"] == 20
    assert result["net_margin_percent"] == 15
    assert "meeting your 10.0% minimum" in result["explanation"]
    assert len(result["sources"]) == 2


def test_research_shortlist_marks_missing_consecutive_year_as_unavailable() -> None:
    records = {
        "TEST": {
            "revenue": [
                {"fiscal_year": 2022, "value_usd": 100, "source_url": "https://sec.test/2022"},
                {"fiscal_year": 2024, "value_usd": 120, "source_url": "https://sec.test/2024"},
            ],
            "net_income": [
                {
                    "company": "Test Company",
                    "fiscal_year": 2024,
                    "value_usd": 18,
                    "source_url": "https://sec.test/2024",
                }
            ],
        }
    }

    result = build_research_shortlist(records, 0, 5)[0]

    assert result["revenue_growth_percent"] is None
    assert result["growth_match"] is False
    assert "data was unavailable" in result["explanation"]


def _entry(fy, start, end, val, filed, form="10-K", fp="FY"):
    item = {"form": form, "fp": fp, "fy": fy, "end": end, "val": val, "filed": filed, "accn": f"0000000000-{filed[:4]}-000001"}
    if start:
        item["start"] = start
    return item


def test_extract_uses_each_periods_own_value_not_the_comparatives() -> None:
    """A 10-K repeats prior years, all tagged with the filing's fy. Regression test for a two-year shift."""
    entries = []
    for fy, values in {2023: [(2021, 300), (2022, 340), (2023, 380)], 2024: [(2022, 340), (2023, 381), (2024, 420)]}.items():
        for year, val in values:
            entries.append(_entry(fy, f"{year - 1}-10-01", f"{year}-09-30", val, f"{fy}-11-01"))
    entries.append(_entry(2024, "2024-07-01", "2024-09-30", 999, "2024-11-01"))  # a quarter inside the 10-K
    entries.append(_entry(2024, "2023-10-01", "2024-09-30", 777, "2024-08-01", form="10-Q", fp="Q3"))
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": entries}}}}}
    company = {"ticker": "TEST", "cik": 1, "name": "Test Company"}

    records = extract_metric_records(company, facts, "revenue", 3)

    assert [(row["fiscal_year"], row["value_usd"]) for row in records] == [(2022, 340), (2023, 381), (2024, 420)]
    assert records[1]["filed"] == "2024-11-01"  # restated 2023 value comes from the later filing
    assert records[-1]["period_end"] == "2024-09-30"


def test_extract_respects_companies_that_name_the_year_before_it_ends() -> None:
    """Retailers often call the year ending February 2025 'fiscal 2024'."""
    entries = [
        _entry(2024, "2024-02-04", "2025-02-01", 107, "2025-03-12"),
        _entry(2024, "2023-01-29", "2024-02-03", 106, "2025-03-12"),
        _entry(2023, "2023-01-29", "2024-02-03", 105, "2024-03-13"),
    ]
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": entries}}}}}
    records = extract_metric_records({"ticker": "T", "cik": 1, "name": "T"}, facts, "revenue", 2)
    assert [(row["fiscal_year"], row["value_usd"]) for row in records] == [(2023, 106), (2024, 107)]


def test_extract_fills_years_from_a_fallback_tag_and_handles_instants_and_per_share() -> None:
    facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [_entry(2024, "2024-01-01", "2024-12-31", 120, "2025-02-01")]}},
                "Revenues": {"units": {"USD": [_entry(2023, "2023-01-01", "2023-12-31", 100, "2024-02-01"), _entry(2024, "2024-01-01", "2024-12-31", 999, "2025-02-01")]}},
                "Assets": {"units": {"USD": [_entry(2024, None, "2023-12-31", 50, "2025-02-01"), _entry(2024, None, "2024-12-31", 60, "2025-02-01")]}},
                "EarningsPerShareDiluted": {"units": {"USD/shares": [_entry(2024, "2024-01-01", "2024-12-31", 2.5, "2025-02-01")]}},
            }
        }
    }
    company = {"ticker": "TEST", "cik": 1, "name": "Test Company"}
    revenue = extract_metric_records(company, facts, "revenue", 2)
    assert [(row["fiscal_year"], row["value_usd"]) for row in revenue] == [(2023, 100), (2024, 120)]  # preferred tag wins for 2024
    assets = extract_metric_records(company, facts, "total_assets", 2)
    assert [(row["fiscal_year"], row["value_usd"]) for row in assets] == [(2023, 50), (2024, 60)]
    eps = extract_metric_records(company, facts, "eps_diluted", 1)
    assert eps[0]["value_usd"] == 2.5 and eps[0]["unit"] == "USD/shares"
