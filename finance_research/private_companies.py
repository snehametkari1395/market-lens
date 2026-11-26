"""Funding data for a hand-picked set of US private companies, from SEC Form D filings.

Private companies don't publish financial statements, but when they raise money under
Regulation D they must file a Form D notice with the SEC. It says how much the round was
for, how much was sold, the date of the first sale and how many investors took part.
It does NOT contain revenue, profit or valuation.
"""
from __future__ import annotations

import threading
import xml.etree.ElementTree as ET
from typing import Any

from requests import RequestException

from finance_research.sec_client import SEC_ARCHIVES, SEC_DATA, SECClientError, _get_json_cached, _get_text

# key -> (display name, SEC CIK, what the company does). Legal names differ for some (Rippling = People Center, Inc.).
PRIVATE_COMPANIES: dict[str, dict[str, Any]] = {
    "stripe": {"name": "Stripe", "cik": 1691342, "about": "Online payments"},
    "databricks": {"name": "Databricks", "cik": 1587468, "about": "Data and AI platform"},
    "ramp": {"name": "Ramp", "cik": 1803782, "about": "Corporate cards and spend management"},
    "rippling": {"name": "Rippling", "cik": 1772138, "about": "HR and payroll software"},
    "figure-ai": {"name": "Figure AI", "cik": 2014185, "about": "Humanoid robots"},
    "gusto": {"name": "Gusto", "cik": 1639415, "about": "Small-business payroll"},
    "deel": {"name": "Deel", "cik": 1813695, "about": "Global hiring and payroll"},
    "notion": {"name": "Notion", "cik": 1781814, "about": "Workspace and notes software"},
    "discord": {"name": "Discord", "cik": 1763840, "about": "Chat and community platform"},
    "perplexity": {"name": "Perplexity", "cik": 1972171, "about": "AI search engine"},
}

FORM_D_TYPES = {"D", "D/A"}
FORM_D_NOTE = (
    "Form D amounts are what the company reported selling in each private offering. They are not revenue, "
    "profit or valuation, and a funding round can be split across several filings or left unreported."
)

_xml_cache: dict[str, dict[str, Any]] = {}  # filings never change, so cache them for the process lifetime
_xml_lock = threading.Lock()


def resolve_private_company(name: str) -> tuple[str, dict[str, Any]]:
    """Match a key or display name (case-insensitive, e.g. 'Stripe', 'figure ai', 'People Center')."""
    wanted = "".join(ch for ch in name.lower() if ch.isalnum())
    for key, company in PRIVATE_COMPANIES.items():
        if wanted in {key.replace("-", ""), "".join(ch for ch in company["name"].lower() if ch.isalnum())}:
            return key, company
    if wanted == "peoplecenter":
        return "rippling", PRIVATE_COMPANIES["rippling"]
    options = ", ".join(company["name"] for company in PRIVATE_COMPANIES.values())
    raise ValueError(f"Private-company data is available for: {options}.")


def _text(node: ET.Element | None, path: str) -> str:
    found = node.find(path) if node is not None else None
    return (found.text or "").strip() if found is not None else ""


def _number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None  # e.g. "Indefinite"


def parse_form_d(xml_text: str) -> dict[str, Any]:
    root = ET.fromstring(xml_text)
    offering = root.find("offeringData")
    amounts = offering.find("offeringSalesAmounts") if offering is not None else None
    securities = offering.find("typesOfSecuritiesOffered") if offering is not None else None
    security_types = [
        child.tag.removeprefix("is").removesuffix("Type")
        for child in (securities if securities is not None else [])
        if child.tag.startswith("is") and (child.text or "").strip().lower() == "true"
    ]
    return {
        "issuer": _text(root, "primaryIssuer/entityName"),
        "state": _text(root, "primaryIssuer/issuerAddress/stateOrCountryDescription"),
        "industry": _text(offering, "industryGroup/industryGroupType"),
        "revenue_range": _text(offering, "issuerSize/revenueRange") or "Not given",
        "is_amendment": _text(offering, "typeOfFiling/newOrAmendment/isAmendment").lower() == "true",
        "previous_accession": _text(offering, "typeOfFiling/newOrAmendment/previousAccessionNumber"),
        "first_sale": _text(offering, "typeOfFiling/dateOfFirstSale/value") or None,
        "securities": security_types,
        "offering_amount": _number(_text(amounts, "totalOfferingAmount")),
        "amount_sold": _number(_text(amounts, "totalAmountSold")),
        "investors": int(_number(_text(offering, "investors/totalNumberAlreadyInvested")) or 0),
    }


def _form_d(cik: int, accession: str, document: str, user_agent: str) -> dict[str, Any]:
    url = f"{SEC_ARCHIVES}/{cik}/{accession.replace('-', '')}/{document.rsplit('/', 1)[-1]}"
    with _xml_lock:
        if url in _xml_cache:
            return _xml_cache[url]
    parsed = parse_form_d(_get_text(url, user_agent))
    with _xml_lock:
        _xml_cache[url] = parsed
    return parsed


def get_private_funding(name: str, user_agent: str, limit: int = 25) -> dict[str, Any]:
    """Funding offerings (newest first) for one private company, keeping only the latest version of
    each offering when it was amended."""
    key, company = resolve_private_company(name)
    submissions = _get_json_cached(f"{SEC_DATA}/submissions/CIK{company['cik']:010d}.json", user_agent)
    if submissions.get("tickers"):
        raise SECClientError(f"{company['name']} now has a stock ticker, so it is no longer private.")
    recent = submissions.get("filings", {}).get("recent", {})
    rows = zip(recent.get("form", []), recent.get("filingDate", []), recent.get("accessionNumber", []), recent.get("primaryDocument", []))
    filings = [(form, filed, accession, doc) for form, filed, accession, doc in rows if form in FORM_D_TYPES][:limit]

    offerings: dict[str, dict[str, Any]] = {}  # original accession -> latest filing of that offering
    unreadable = 0
    for form, filed, accession, document in filings:  # newest first
        try:
            data = _form_d(company["cik"], accession, document, user_agent)
        except (RequestException, ET.ParseError):
            unreadable += 1  # one slow or malformed filing shouldn't hide the rest
            continue
        original = data["previous_accession"] or accession
        if original in offerings:
            continue  # a newer amendment of this offering is already kept
        offerings[original] = {
            "filed": filed,
            "form": form,
            **{k: v for k, v in data.items() if k not in {"issuer", "state", "previous_accession", "is_amendment"}},
            "filing_url": f"{SEC_ARCHIVES}/{company['cik']}/{accession.replace('-', '')}/",
        }
    if filings and not offerings:
        raise SECClientError(f"The SEC did not return {company['name']}'s Form D filings. Try again shortly.")
    rounds = sorted(offerings.values(), key=lambda row: row["first_sale"] or row["filed"], reverse=True)
    result = {
        "company": company["name"],
        "key": key,
        "legal_name": submissions.get("name", ""),
        "about": company["about"],
        "cik": company["cik"],
        "state_of_incorporation": submissions.get("stateOfIncorporation", ""),
        "offerings": rounds,
        "total_reported_sold": sum(row["amount_sold"] or 0 for row in rounds),
        "source": "SEC Form D filings",
        "note": FORM_D_NOTE,
    }
    if unreadable:
        result["skipped_filings"] = unreadable
    return result
