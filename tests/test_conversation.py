import json

from finance_research.conversation import (
    answer_help_question,
    conversation_to_json,
    conversation_to_markdown,
    extract_citations,
)
from finance_research.rag import _relevance_from_distance

SEC_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/"

TOOL_RESULTS = [
    {
        "name": "fetch_sec_financials",
        "content": {
            "records": [
                {"company": "Apple Inc.", "ticker": "AAPL", "fiscal_year": 2024, "metric_label": "Revenue", "source_url": SEC_URL},
                {"company": "Apple Inc.", "ticker": "AAPL", "fiscal_year": 2024, "metric_label": "Revenue", "source_url": SEC_URL},
            ]
        },
    },
    {
        "name": "calculate_financial_ratio",
        "content": {"ticker": "AAPL", "fiscal_year": 2024, "ratio": "net_margin", "sources": [SEC_URL]},
    },
    {
        "name": "search_filing_knowledge",
        "content": {
            "results": [
                {"company": "Microsoft", "report_year": 2024, "section": "Item 1A — Risk Factors", "source_url": FILING_URL}
            ],
            "retrieval": {"query": "risks", "ticker": "MSFT"},
        },
    },
    {"name": "fetch_stock_quote", "content": {"error": "Market data is not configured."}},
]


def test_extract_citations_dedupes_and_covers_each_tool() -> None:
    citations = extract_citations(TOOL_RESULTS)
    kinds = [item["kind"] for item in citations]
    assert kinds == ["SEC financial data", "SEC financial data", "SEC filing excerpt"]
    assert "Item 1A" in citations[2]["label"]
    assert all(item["url"].startswith("https://") for item in citations)


def test_extract_citations_ignores_non_dict_content() -> None:
    assert extract_citations([{"name": "x", "content": "plain text"}]) == []


def test_markdown_export_includes_sources_and_tools() -> None:
    messages = [
        {"role": "user", "content": "Apple margin?", "timestamp": "2026-11-21 10:00"},
        {
            "role": "assistant",
            "content": "Net margin was 24%.",
            "tools": TOOL_RESULTS,
            "citations": extract_citations(TOOL_RESULTS),
        },
    ]
    markdown = conversation_to_markdown(messages, "openai/gpt-4o-mini")
    assert "## You · 2026-11-21 10:00" in markdown
    assert "**Sources**" in markdown and SEC_URL in markdown
    assert "_Tools used: fetch_sec_financials, calculate_financial_ratio" in markdown


def test_json_export_round_trips() -> None:
    payload = json.loads(conversation_to_json([{"role": "user", "content": "hi"}], "m"))
    assert payload["model"] == "m"
    assert payload["messages"][0]["content"] == "hi"


def test_help_guide_matches_topics_and_falls_back() -> None:
    assert answer_help_question("Why do I need an OpenRouter key?")["title"] == "OpenRouter API key"
    assert answer_help_question("How do I index filings?")["title"] == "Indexing filings"
    assert answer_help_question("How do I export my chat?")["title"] == "Saving or exporting a chat"
    assert answer_help_question("How do I get started?")["title"] == "Getting started"
    assert answer_help_question("zzz qqq")["title"] == "Topics I can help with"


def test_relevance_from_distance_is_bounded_cosine() -> None:
    assert _relevance_from_distance(0.0) == 1.0
    assert _relevance_from_distance(0.5) == 0.75
    assert _relevance_from_distance(3.0) == 0.0
