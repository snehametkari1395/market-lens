from __future__ import annotations

import re
import threading
import time
from typing import Any
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


SEC_DATA = "https://data.sec.gov"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

METRICS: dict[str, dict[str, Any]] = {
    "revenue": {
        "label": "Revenue",
        "tags": (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenuesNetOfInterestExpense",  # banks such as Goldman Sachs report net revenue
        ),
        "ifrs_tags": ("Revenue", "RevenueFromContractsWithCustomers"),
    },
    "research_and_development": {
        "label": "Research and development",
        "tags": ("ResearchAndDevelopmentExpense",),
        "ifrs_tags": ("ResearchAndDevelopmentExpense",),
    },
    "operating_income": {
        "label": "Operating income",
        "tags": ("OperatingIncomeLoss",),
        "ifrs_tags": ("ProfitLossFromOperatingActivities",),
    },
    "net_income": {
        "label": "Net income",
        "tags": ("NetIncomeLoss",),
        "ifrs_tags": ("ProfitLossAttributableToOwnersOfParent", "ProfitLoss"),
    },
    "total_assets": {
        "label": "Total assets",
        "tags": ("Assets",),
        "ifrs_tags": ("Assets",),
    },
    "operating_cash_flow": {
        "label": "Operating cash flow",
        "tags": ("NetCashProvidedByUsedInOperatingActivities",),
        "ifrs_tags": ("CashFlowsFromUsedInOperatingActivities",),
    },
    "total_liabilities": {
        "label": "Total liabilities",
        "tags": ("Liabilities",),
        "ifrs_tags": ("Liabilities",),
    },
    "stockholders_equity": {
        "label": "Shareholders' equity",
        "tags": (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
        "ifrs_tags": ("EquityAttributableToOwnersOfParent", "Equity"),
    },
    "cash": {
        "label": "Cash and cash equivalents",
        "tags": (
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            "Cash",
        ),
        "ifrs_tags": ("CashAndCashEquivalents",),
    },
    "long_term_debt": {
        "label": "Long-term debt",
        "tags": ("LongTermDebt", "LongTermDebtNoncurrent"),
        "ifrs_tags": ("NoncurrentPortionOfNoncurrentBorrowings", "LongtermBorrowings"),
    },
    "eps_diluted": {
        "label": "Diluted earnings per share",
        "tags": ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
        "ifrs_tags": ("DilutedEarningsLossPerShare", "BasicAndDilutedEarningsLossPerShare"),
        "unit": "USD/shares",
    },
    "dividends_per_share": {
        "label": "Dividends per share",
        "tags": ("CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"),
        "ifrs_tags": ("DividendsRecognisedAsDistributionsToOwnersPerShare",),
        "unit": "USD/shares",
    },
    "dividends_paid": {
        "label": "Dividends paid",
        "tags": ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends"),
        "ifrs_tags": ("DividendsPaidClassifiedAsFinancingActivities", "DividendsPaid"),
    },
}
# Metrics shown in USD totals (charts in billions); per-share figures are excluded.
TOTAL_METRICS = [key for key, spec in METRICS.items() if spec.get("unit", "USD") == "USD"]

RATIOS: dict[str, tuple[str, ...]] = {
    "revenue_growth": ("revenue", "revenue"),
    "operating_margin": ("operating_income", "revenue"),
    "net_margin": ("net_income", "revenue"),
    "research_intensity": ("research_and_development", "revenue"),
    "payout_ratio": ("dividends_paid", "net_income"),
}


class SECClientError(RuntimeError):
    """Raised when SEC data cannot be retrieved or interpreted."""


def validate_sec_user_agent(user_agent: str) -> str:
    value = user_agent.strip()
    if "@" not in value or "." not in value.rsplit("@", 1)[-1]:
        raise ValueError("Enter a contact email in the SEC user-agent field.")
    return value


def _get_json(url: str, user_agent: str) -> dict[str, Any]:
    headers = {
        "User-Agent": validate_sec_user_agent(user_agent),
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov" if "data.sec.gov" in url else "www.sec.gov",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise SECClientError("The SEC returned an unexpected response.")
    return payload


_CACHE_SECONDS = 900
_CACHE_MAX_ENTRIES = 12
_json_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_cache_lock = threading.Lock()


def _get_json_cached(url: str, user_agent: str) -> dict[str, Any]:
    """_get_json with a 15-minute in-memory cache, so fetching several metrics for one company
    downloads its (large) company-facts file once instead of once per metric."""
    now = time.monotonic()
    with _cache_lock:
        hit = _json_cache.get(url)
        if hit and now - hit[0] < _CACHE_SECONDS:
            return hit[1]
    payload = _get_json(url, user_agent)
    with _cache_lock:
        _json_cache[url] = (now, payload)
        while len(_json_cache) > _CACHE_MAX_ENTRIES:
            _json_cache.pop(min(_json_cache, key=lambda key: _json_cache[key][0]))
    return payload


def clear_sec_cache() -> None:
    with _cache_lock:
        _json_cache.clear()


def _get_text(url: str, user_agent: str) -> str:
    headers = {
        "User-Agent": validate_sec_user_agent(user_agent),
        "Accept-Encoding": "gzip, deflate",
    }
    response = requests.get(url, headers=headers, timeout=45)
    response.raise_for_status()
    return response.text


# Tickers whose SEC directory entry moved to a new holding company that has no filing history yet.
# They point at the registrant that holds the 10-Ks. Review once the new entity files its first 10-K.
CIK_OVERRIDES: dict[str, tuple[int, str]] = {
    # July 2026: XOM moved to ExxonMobil Holdings Corp (CIK 2115436); the 10-Ks are under Exxon Mobil Corp.
    "XOM": (34088, "Exxon Mobil Corp"),
}


def find_company(ticker: str, user_agent: str) -> dict[str, Any]:
    normalized = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.-]{1,10}", normalized):
        raise ValueError("Enter a valid public-company ticker symbol.")

    if normalized in CIK_OVERRIDES:
        cik, name = CIK_OVERRIDES[normalized]
        return {"ticker": normalized, "cik": cik, "name": name}
    companies = _get_json_cached("https://www.sec.gov/files/company_tickers.json", user_agent)
    for company in companies.values():
        if str(company.get("ticker", "")).upper() == normalized:
            return {
                "ticker": normalized,
                "cik": int(company["cik_str"]),
                "name": str(company["title"]),
            }
    raise SECClientError(f"The SEC company directory does not contain ticker {normalized}.")


def get_company_facts(ticker: str, user_agent: str) -> tuple[dict[str, Any], dict[str, Any]]:
    company = find_company(ticker, user_agent)
    cik = f"{company['cik']:010d}"
    facts = _get_json_cached(f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik}.json", user_agent)
    return company, facts


# US companies file a 10-K each year; foreign companies listed in the US file a 20-F (or a 40-F from Canada).
ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}

# US companies tag figures with US GAAP concepts; companies reporting under IFRS (most of Europe) use ifrs-full.
TAXONOMIES = ("us-gaap", "ifrs-full")

# Approximate yearly average exchange rates (US dollars per unit of currency, ECB/Fed reference averages),
# so European figures can be compared with US companies. Years after the last entry use the last known rate.
USD_PER_UNIT: dict[str, dict[int, float]] = {
    "EUR": {2015: 1.110, 2016: 1.107, 2017: 1.130, 2018: 1.181, 2019: 1.120, 2020: 1.142,
            2021: 1.183, 2022: 1.053, 2023: 1.081, 2024: 1.082, 2025: 1.130},
    "GBP": {2015: 1.528, 2016: 1.355, 2017: 1.289, 2018: 1.335, 2019: 1.277, 2020: 1.284,
            2021: 1.375, 2022: 1.237, 2023: 1.243, 2024: 1.278, 2025: 1.320},
    "CHF": {2015: 1.040, 2016: 1.015, 2017: 1.016, 2018: 1.023, 2019: 1.006, 2020: 1.066,
            2021: 1.094, 2022: 1.048, 2023: 1.113, 2024: 1.136, 2025: 1.210},
}
USD_PER_UNIT["DKK"] = {year: rate / 7.45 for year, rate in USD_PER_UNIT["EUR"].items()}  # the krone is pegged to the euro

# Ordinary shares represented by one US-listed share (ADR), so per-share figures match the price traded in the US.
SHARES_PER_US_LISTING: dict[str, float] = {"AZN": 0.5, "FMS": 0.5, "EVO": 0.5, "SHEL": 2.0}


def usd_rate(currency: str, year: int) -> float | None:
    """US dollars per unit of `currency` in `year` (1.0 for USD); None for a currency without rates."""
    if currency == "USD":
        return 1.0
    rates = USD_PER_UNIT.get(currency)
    if not rates:
        return None
    if year in rates:
        return rates[year]
    return rates[max(rates)] if year > max(rates) else rates[min(rates)]


def shares_per_us_listing(ticker: str) -> float:
    return SHARES_PER_US_LISTING.get(ticker.upper(), 1.0)


def _pick_unit(units: dict[str, list[dict[str, Any]]], per_share: bool) -> str | None:
    """The unit a concept is mostly reported in (e.g. EUR for SAP), preferring USD on a tie."""
    candidates = []
    for unit, entries in units.items():
        currency, _, rest = unit.partition("/")
        if per_share != (rest == "shares") or len(currency) != 3 or not currency.isupper():
            continue
        if currency != "USD" and currency not in USD_PER_UNIT:
            continue
        annual = sum(1 for entry in entries if entry.get("form") in ANNUAL_FORMS)
        candidates.append((annual, currency == "USD", unit))
    return max(candidates)[2] if candidates and max(candidates)[0] else None


def _period_days(item: dict[str, Any]) -> int | None:
    from datetime import date

    if not item.get("start") or not item.get("end"):
        return None
    return (date.fromisoformat(item["end"]) - date.fromisoformat(item["start"])).days


def _period_year(end: str) -> int:
    from datetime import date, timedelta

    # Ten days back keeps 52/53-week years that end in the first days of January in the right year.
    return (date.fromisoformat(end) - timedelta(days=10)).year


def _annual_values(entries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """One value per fiscal year from a concept's 10-K entries.

    A 10-K repeats earlier years for comparison, and the SEC tags every entry with the *filing's*
    fiscal year ("fy"), so "fy" can't label a value on its own. Instead each value is placed by the
    period it covers (its "end" date). The newest period in each filing is the year that filing is
    about, which tells us how the company numbers its fiscal years (e.g. a year ending January 2025
    may be "fiscal 2024" or "fiscal 2025"). When a period appears in several filings, the latest
    filing wins, so restated figures replace originals. Quarter-length periods are skipped.
    """
    from collections import Counter

    usable = []
    for item in entries:
        if item.get("form") not in ANNUAL_FORMS or not isinstance(item.get("fy"), int):
            continue
        days = _period_days(item)
        if days is not None and not 300 <= days <= 380:
            continue  # a quarter or other partial period
        usable.append(item)

    current_period: dict[str, tuple[str, int]] = {}
    for item in usable:
        if item.get("end"):
            filing = str(item.get("accn") or item.get("filed"))
            if item["end"] > current_period.get(filing, ("", 0))[0]:
                current_period[filing] = (item["end"], item["fy"])
    offsets = Counter(fy - _period_year(end) for end, fy in current_period.values())
    offset = offsets.most_common(1)[0][0] if offsets else 0

    result: dict[int, dict[str, Any]] = {}
    for item in usable:
        if item.get("end"):
            year = _period_year(item["end"]) + offset
        elif item.get("fp") == "FY":  # incomplete data: fall back to the filing's own year
            year = item["fy"]
        else:
            continue
        current = result.get(year)
        if current is None or (str(item.get("filed", "")), str(item.get("end", ""))) > (
            str(current.get("filed", "")),
            str(current.get("end", "")),
        ):
            result[year] = item
    return result


def extract_metric_records(
    company: dict[str, Any],
    facts: dict[str, Any],
    metric: str,
    years: int,
) -> list[dict[str, Any]]:
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    if not 1 <= years <= 10:
        raise ValueError("Years must be between 1 and 10.")

    all_facts = facts.get("facts", {})
    unit = METRICS[metric].get("unit", "USD")
    per_share = unit.endswith("/shares")

    def latest_year(taxonomy: str) -> int:
        return max(
            (
                int(entry["fy"])
                for concept in all_facts.get(taxonomy, {}).values()
                for unit_entries in concept.get("units", {}).values()
                for entry in unit_entries
                if entry.get("form") in ANNUAL_FORMS and entry.get("fp") == "FY" and isinstance(entry.get("fy"), int)
            ),
            default=0,
        )

    newest = {taxonomy: latest_year(taxonomy) for taxonomy in TAXONOMIES}
    cutoff_year = max(newest.values()) - years + 1
    # The taxonomy with the newest annual data goes first (e.g. Fresenius moved from US GAAP to IFRS).
    tag_sources = [
        (taxonomy, tag)
        for taxonomy in sorted(TAXONOMIES, key=lambda name: -newest[name])
        for tag in METRICS[metric]["tags" if taxonomy == "us-gaap" else "ifrs_tags"]
    ]
    available_by_year: dict[int, tuple[dict[str, Any], str]] = {}

    # Tags are tried in order; a later tag only fills years the earlier ones don't cover
    # (companies sometimes switch revenue tags between years).
    for taxonomy, tag in tag_sources:
        concept = all_facts.get(taxonomy, {}).get(tag)
        if not concept:
            continue
        units = concept.get("units", {})
        reported_unit = _pick_unit(units, per_share)  # SAP, say, adds a few USD convenience figures to its EUR ones
        if not reported_unit:
            continue
        for year, item in _annual_values(units.get(reported_unit, [])).items():
            if year >= cutoff_year and year not in available_by_year:
                available_by_year[year] = (item, reported_unit.split("/")[0])

    records: list[dict[str, Any]] = []
    for year, (item, currency) in sorted(available_by_year.items()):
        rate = usd_rate(currency, _period_year(item["end"]) if item.get("end") else year)
        if rate is None:
            continue
        reported_value = float(item["val"])
        if metric == "dividends_paid":
            reported_value = abs(reported_value)  # IFRS cash-flow statements often show payments as negatives
        value = reported_value * rate * (shares_per_us_listing(company["ticker"]) if per_share else 1.0)
        accession = str(item.get("accn", ""))
        accession_path = accession.replace("-", "")
        source_url = ""
        if accession_path:
            source_url = f"{SEC_ARCHIVES}/{company['cik']}/{accession_path}/{quote(accession)}-index.html"
        records.append(
            {
                "ticker": company["ticker"],
                "company": company["name"],
                "metric": metric,
                "metric_label": METRICS[metric]["label"],
                "fiscal_year": year,
                "value_usd": value,
                "unit": unit,
                "reported_currency": currency,
                "reported_value": reported_value,
                "usd_rate": rate,
                "period_end": str(item.get("end", "")),
                "filed": str(item.get("filed", "")),
                "accession": accession,
                "source_url": source_url,
            }
        )
    return records


def get_metric_records(ticker: str, metric: str, years: int, user_agent: str) -> list[dict[str, Any]]:
    company, facts = get_company_facts(ticker, user_agent)
    return extract_metric_records(company, facts, metric, years)


def get_recent_10k_filings(
    ticker: str,
    years: int,
    user_agent: str,
) -> list[dict[str, Any]]:
    if not 1 <= years <= 10:
        raise ValueError("Years must be between 1 and 10.")
    company = find_company(ticker, user_agent)
    cik = f"{company['cik']:010d}"
    submissions = _get_json(f"{SEC_DATA}/submissions/CIK{cik}.json", user_agent)
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filings: list[dict[str, Any]] = []
    for index, form in enumerate(forms):
        if form not in ANNUAL_FORMS:
            continue
        accession = recent["accessionNumber"][index]
        document = recent["primaryDocument"][index]
        filed = recent["filingDate"][index]
        report_date = recent.get("reportDate", [""] * len(forms))[index]
        report_year = int(report_date[:4]) if report_date else int(filed[:4])
        accession_path = accession.replace("-", "")
        filings.append(
            {
                "ticker": company["ticker"],
                "company": company["name"],
                "report_year": report_year,
                "form": form,
                "filed": filed,
                "accession": accession,
                "document_url": f"{SEC_ARCHIVES}/{company['cik']}/{accession_path}/{document}",
                "filing_url": (
                    f"{SEC_ARCHIVES}/{company['cik']}/{accession_path}/"
                    f"{quote(accession)}-index.html"
                ),
            }
        )

    filings.sort(key=lambda filing: (filing["report_year"], filing["filed"]), reverse=True)
    latest_by_year: dict[int, dict[str, Any]] = {}
    for filing in filings:
        latest_by_year.setdefault(filing["report_year"], filing)
    return list(latest_by_year.values())[:years]


def download_filing_text(document_url: str, user_agent: str) -> str:
    html = _get_text(document_url, user_agent)
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "ix:header"]):
        element.decompose()
    text = soup.get_text("\n", strip=True)
    if len(text) < 500:
        raise SECClientError("The filing document did not contain enough readable text.")
    return text
