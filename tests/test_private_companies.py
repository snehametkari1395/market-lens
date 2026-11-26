import pytest

from finance_research import private_companies as pc
from finance_research.conversation import extract_citations
from finance_research.sec_client import SECClientError
from finance_research.tools import build_tools

FORM_D = """<?xml version="1.0"?>
<edgarSubmission>
  <primaryIssuer><cik>0001587468</cik><entityName>Databricks, Inc.</entityName>
    <issuerAddress><stateOrCountryDescription>CALIFORNIA</stateOrCountryDescription></issuerAddress></primaryIssuer>
  <offeringData>
    <industryGroup><industryGroupType>Other Technology</industryGroupType></industryGroup>
    <issuerSize><revenueRange>Decline to Disclose</revenueRange></issuerSize>
    <typeOfFiling><newOrAmendment><isAmendment>{amend}</isAmendment>{prev}</newOrAmendment>
      <dateOfFirstSale><value>{first}</value></dateOfFirstSale></typeOfFiling>
    <typesOfSecuritiesOffered><isEquityType>true</isEquityType><isDebtType>false</isDebtType></typesOfSecuritiesOffered>
    <offeringSalesAmounts><totalOfferingAmount>{offered}</totalOfferingAmount><totalAmountSold>{sold}</totalAmountSold></offeringSalesAmounts>
    <investors><totalNumberAlreadyInvested>{investors}</totalNumberAlreadyInvested></investors>
  </offeringData>
</edgarSubmission>"""


def form_d(sold, first="2026-08-12", offered="5000000000", investors=136, prev=""):
    return FORM_D.format(
        amend="true" if prev else "false",
        prev=f"<previousAccessionNumber>{prev}</previousAccessionNumber>" if prev else "",
        first=first, offered=offered, sold=sold, investors=investors,
    )


def test_parse_form_d_reads_amounts_and_types() -> None:
    data = pc.parse_form_d(form_d("4999997255", offered="Indefinite"))
    assert data["amount_sold"] == 4999997255 and data["offering_amount"] is None
    assert data["securities"] == ["Equity"] and data["investors"] == 136
    assert data["first_sale"] == "2026-08-12" and data["revenue_range"] == "Decline to Disclose"


def test_resolve_accepts_names_and_rejects_others() -> None:
    assert pc.resolve_private_company("Figure AI")[0] == "figure-ai"
    assert pc.resolve_private_company("people center")[0] == "rippling"
    with pytest.raises(ValueError, match="available for"):
        pc.resolve_private_company("SpaceX")


def _fake_sec(monkeypatch, submissions, documents):
    monkeypatch.setattr(pc, "_get_json_cached", lambda url, ua: submissions)
    monkeypatch.setattr(pc, "_get_text", lambda url, ua: documents[url.split("/")[-2]])
    pc._xml_cache.clear()


def test_amendments_replace_their_original_offering(monkeypatch) -> None:
    submissions = {
        "name": "Databricks, Inc.", "tickers": [], "stateOfIncorporation": "DE",
        "filings": {"recent": {
            "form": ["D/A", "8-K", "D", "D"],
            "filingDate": ["2026-09-01", "2026-08-30", "2026-08-27", "2025-01-10"],
            "accessionNumber": ["0001-26-3", "0001-26-9", "0001-26-2", "0001-25-1"],
            "primaryDocument": ["xslFormDX01/primary_doc.xml"] * 4,
        }},
    }
    documents = {
        "0001263": form_d("5000000000", prev="0001-26-2"),
        "0001262": form_d("4000000000"),
        "0001251": form_d("10000000", first="2024-12-20", investors=3),
    }
    _fake_sec(monkeypatch, submissions, documents)
    result = pc.get_private_funding("databricks", "Test test@example.com")
    assert [row["amount_sold"] for row in result["offerings"]] == [5000000000, 10000000]
    assert result["offerings"][0]["form"] == "D/A"
    assert result["total_reported_sold"] == 5010000000
    assert result["offerings"][0]["filing_url"].endswith("/1587468/0001263/")

    citations = extract_citations([{"name": "get_private_company_funding", "content": result}])
    assert citations[0]["kind"] == "SEC Form D filing" and "Databricks" in citations[0]["label"]


def test_company_that_went_public_is_refused(monkeypatch) -> None:
    _fake_sec(monkeypatch, {"tickers": ["XYZ"], "filings": {"recent": {}}}, {})
    with pytest.raises(SECClientError, match="no longer private"):
        pc.get_private_funding("stripe", "Test test@example.com")


def test_tool_is_registered() -> None:
    names = [item.name for item in build_tools("Test test@example.com", None)]
    assert "get_private_company_funding" in names
