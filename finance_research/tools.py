from __future__ import annotations

import json
from typing import Any

import requests
from langchain_core.tools import tool

from finance_research.events import get_company_events
from finance_research.private_companies import PRIVATE_COMPANIES, get_private_funding
from finance_research.rag import DEFAULT_STRATEGY, SEARCH_STRATEGIES, search_filings
from finance_research.security import sanitize_untrusted
from finance_research.sec_client import (
    METRICS,
    RATIOS,
    get_metric_records,
)


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False)


def _ratio_records(
    ticker: str,
    ratio: str,
    fiscal_year: int,
    user_agent: str,
) -> dict[str, object]:
    if ratio not in RATIOS:
        raise ValueError(f"Unsupported ratio: {ratio}. Choose from {', '.join(RATIOS)}.")
    numerator_metric, denominator_metric = RATIOS[ratio]
    numerator = get_metric_records(ticker, numerator_metric, 10, user_agent)
    denominator = get_metric_records(ticker, denominator_metric, 10, user_agent)
    numerator_by_year = {row["fiscal_year"]: row for row in numerator}
    denominator_by_year = {row["fiscal_year"]: row for row in denominator}

    if ratio == "revenue_growth":
        prior_year = fiscal_year - 1
        if fiscal_year not in numerator_by_year or prior_year not in numerator_by_year:
            raise ValueError(f"Revenue data for both {prior_year} and {fiscal_year} is required.")
        prior = numerator_by_year[prior_year]["value_usd"]
        current = numerator_by_year[fiscal_year]["value_usd"]
        if prior == 0:
            raise ValueError("Revenue growth is undefined because prior-year revenue is zero.")
        value = (current / prior - 1) * 100
        inputs = {str(prior_year): prior, str(fiscal_year): current}
        formula = "(current-year revenue / prior-year revenue - 1) × 100"
    else:
        if fiscal_year not in numerator_by_year or fiscal_year not in denominator_by_year:
            raise ValueError(f"Required filing data for fiscal year {fiscal_year} is unavailable.")
        numerator_value = numerator_by_year[fiscal_year]["value_usd"]
        denominator_value = denominator_by_year[fiscal_year]["value_usd"]
        if denominator_value == 0:
            raise ValueError(f"{ratio} is undefined because revenue is zero.")
        value = numerator_value / denominator_value * 100
        inputs = {
            numerator_metric: numerator_value,
            denominator_metric: denominator_value,
        }
        formula = f"({numerator_metric} / {denominator_metric}) × 100"

    source_rows = [
        numerator_by_year.get(fiscal_year),
        denominator_by_year.get(fiscal_year),
    ]
    if ratio == "revenue_growth":
        source_rows.append(numerator_by_year.get(fiscal_year - 1))

    return {
        "ticker": ticker.upper(),
        "ratio": ratio,
        "fiscal_year": fiscal_year,
        "value_percent": round(value, 2),
        "formula": formula,
        "inputs_usd": inputs,
        "sources": sorted(
            {
                row["source_url"]
                for row in source_rows
                if row and row.get("source_url")
            }
        ),
    }


TOOL_DESCRIPTIONS = {
    "fetch_sec_financials": "Fetch annual financial figures from SEC company facts. Metric must be one of: "
    + ", ".join(METRICS) + ".",
    "calculate_financial_ratio": "Calculate revenue_growth, operating_margin, net_margin, or research_intensity "
    "for one company and fiscal year using SEC figures.",
    "compare_companies": "Compare a supported SEC financial metric across two to five public-company tickers.",
    "search_filing_knowledge": "Search ingested SEC 10-K filing passages and return excerpts with section/year/source "
    "citations. Write the question in English: the filings are in English. Leave report_year out to search the "
    "latest indexed filings; only set it when the user asks about a specific fiscal year.",
    "fetch_stock_quote": "Fetch a latest stock quote from Alpha Vantage; requires a user-provided API key.",
    "get_company_events": "List a company's recent major events from SEC Form 8-K filings (earnings releases, "
    "executive changes, acquisitions, financing, cybersecurity incidents) with filing dates and links. "
    "days is how far back to look (1-730, default 180).",
    "get_private_company_funding": "Get a US private company's funding history from SEC Form D filings: amount "
    "raised per offering, first-sale date, number of investors and security type. No revenue, profit or valuation. "
    "Works only for: " + ", ".join(company["name"] for company in PRIVATE_COMPANIES.values()) + ".",
}


class ToolKit:
    """The research tools as plain methods returning dicts.

    The in-app agent (LangChain tools) and the Market Lens MCP server both wrap
    these same methods, so behaviour is identical whichever way they're called.
    """

    def __init__(
        self,
        user_agent: str,
        vector_store: object | None,
        market_data_key: str = "",
        search_strategy: str = DEFAULT_STRATEGY,
    ) -> None:
        self.user_agent = user_agent
        self.vector_store = vector_store
        self.market_data_key = market_data_key
        self.search_strategy = search_strategy

    def fetch_sec_financials(self, ticker: str, metric: str, years: int = 5) -> dict[str, Any]:
        if metric not in METRICS:
            raise ValueError(f"Choose one of these metric keys: {', '.join(METRICS)}.")
        records = get_metric_records(ticker, metric, years, self.user_agent)
        if not records:
            return {"message": "No SEC annual figures were found for this metric.", "ticker": ticker}
        return {"records": records, "source": "SEC Company Facts"}

    def calculate_financial_ratio(self, ticker: str, ratio: str, fiscal_year: int) -> dict[str, Any]:
        return _ratio_records(ticker, ratio, fiscal_year, self.user_agent)

    def compare_companies(self, tickers: list[str], metric: str, years: int = 5) -> dict[str, Any]:
        if metric not in METRICS:
            raise ValueError(f"Choose one of these metric keys: {', '.join(METRICS)}.")
        normalized = list(dict.fromkeys(ticker.strip().upper() for ticker in tickers))
        if not 2 <= len(normalized) <= 5:
            raise ValueError("Provide between two and five distinct ticker symbols.")
        records = [row for ticker in normalized for row in get_metric_records(ticker, metric, years, self.user_agent)]
        return {
            "metric": metric,
            "metric_label": METRICS[metric]["label"],
            "records": records,
            "note": "Fiscal years may not align exactly across companies.",
        }

    def search_filing_knowledge(
        self, ticker: str, question: str, report_year: int | None = None, strategy: str | None = None
    ) -> dict[str, Any]:
        if self.vector_store is None:
            return {"message": "Filing search is unavailable until an OpenRouter key is configured."}
        strategy = strategy if strategy in SEARCH_STRATEGIES else self.search_strategy
        results = search_filings(self.vector_store, ticker, question, report_year=report_year, strategy=strategy)
        year_note = None
        if not results and report_year is not None:
            # Models often guess a year from their training data; fall back to whatever years are indexed.
            results = search_filings(self.vector_store, ticker, question, strategy=strategy)
            if results:
                years = sorted({item.get("report_year") for item in results if item.get("report_year")}, reverse=True)
                year_note = (
                    f"No passages are indexed for FY{report_year}. These come from the indexed "
                    f"filing(s) for FY{', FY'.join(str(year) for year in years)}; say which year you quote."
                )
        flagged_chunks = []
        for item in results:
            item["text"], reasons = sanitize_untrusted(item["text"])
            if reasons:
                item["safety_flags"] = reasons
                flagged_chunks.append({"rank": item["rank"], "reasons": reasons})
        retrieval = {
            "query": question,
            "ticker": ticker.strip().upper(),
            "report_year": report_year,
            "top_k": 5,
            "strategy": strategy,
            "embedding_model": "openai/text-embedding-3-small",
        }
        if not results:
            return {
                "message": f"No indexed filing passages found for {ticker.upper()}. Ingest its 10-K filings first.",
                "results": [],
                "retrieval": retrieval,
            }
        payload: dict[str, Any] = {"results": results, "retrieval": retrieval}
        if year_note:
            payload["note"] = year_note
        if flagged_chunks:
            payload["safety"] = {
                "note": "Instruction-like text was removed from these filing excerpts before the model saw them.",
                "flagged_chunks": flagged_chunks,
            }
        return payload

    def get_company_events(self, ticker: str, days: int = 180) -> dict[str, Any]:
        events = get_company_events(ticker, self.user_agent, days=max(1, min(int(days), 730)), limit=15)
        if not events:
            return {"message": f"No 8-K filings in the last {days} days.", "ticker": ticker.upper(), "events": []}
        return {
            "ticker": events[0]["ticker"],
            "company": events[0]["company"],
            "days": days,
            "source": "SEC Form 8-K filings",
            "events": [
                {
                    "filed": event["filed"],
                    "form": event["form"],
                    "category": event["category"],
                    "headline": event["headline"],
                    "items": [f"{item['code']} {item['label']}" for item in event["items"]],
                    "filing_url": event["filing_url"],
                }
                for event in events
            ],
            "note": "Item codes say what kind of event was reported; open the filing for the details.",
        }

    def get_private_company_funding(self, company: str) -> dict[str, Any]:
        return get_private_funding(company, self.user_agent)

    def fetch_stock_quote(self, ticker: str) -> dict[str, Any]:
        if not self.market_data_key.strip():
            return {"error": "Market data is not configured. Add an Alpha Vantage API key in the sidebar.", "ticker": ticker.upper()}
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": ticker.strip().upper(), "apikey": self.market_data_key.strip()},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        quote_data = payload.get("Global Quote")
        if not quote_data or "05. price" not in quote_data:
            message = payload.get("Note") or payload.get("Error Message") or "No quote data was returned."
            return {"error": message, "ticker": ticker.upper()}
        return {
            "ticker": quote_data.get("01. symbol", ticker.upper()),
            "price": quote_data.get("05. price"),
            "change_percent": quote_data.get("10. change percent"),
            "latest_trading_day": quote_data.get("07. latest trading day"),
            "source": "Alpha Vantage",
        }


def build_tools(
    user_agent: str,
    vector_store: object | None,
    market_data_key: str = "",
    search_strategy: str = DEFAULT_STRATEGY,
) -> list[object]:
    kit = ToolKit(user_agent, vector_store, market_data_key, search_strategy)

    @tool(description=TOOL_DESCRIPTIONS["fetch_sec_financials"])
    def fetch_sec_financials(ticker: str, metric: str, years: int = 5) -> str:
        return _json(kit.fetch_sec_financials(ticker, metric, years))

    @tool(description=TOOL_DESCRIPTIONS["calculate_financial_ratio"])
    def calculate_financial_ratio(ticker: str, ratio: str, fiscal_year: int) -> str:
        return _json(kit.calculate_financial_ratio(ticker, ratio, fiscal_year))

    @tool(description=TOOL_DESCRIPTIONS["compare_companies"])
    def compare_companies(tickers: list[str], metric: str, years: int = 5) -> str:
        return _json(kit.compare_companies(tickers, metric, years))

    @tool(description=TOOL_DESCRIPTIONS["search_filing_knowledge"])
    def search_filing_knowledge(ticker: str, question: str, report_year: int | None = None) -> str:
        return _json(kit.search_filing_knowledge(ticker, question, report_year))

    @tool(description=TOOL_DESCRIPTIONS["fetch_stock_quote"])
    def fetch_stock_quote(ticker: str) -> str:
        return _json(kit.fetch_stock_quote(ticker))

    @tool("get_company_events", description=TOOL_DESCRIPTIONS["get_company_events"])
    def get_company_events_tool(ticker: str, days: int = 180) -> str:
        return _json(kit.get_company_events(ticker, days))

    @tool(description=TOOL_DESCRIPTIONS["get_private_company_funding"])
    def get_private_company_funding(company: str) -> str:
        return _json(kit.get_private_company_funding(company))

    return [
        fetch_sec_financials,
        calculate_financial_ratio,
        compare_companies,
        search_filing_knowledge,
        fetch_stock_quote,
        get_company_events_tool,
        get_private_company_funding,
    ]
