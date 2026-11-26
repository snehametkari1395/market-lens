from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from openai import APIError
from requests import RequestException

import altair as alt

from finance_research.agent import AgentRunError, complete, run_research_agent
from finance_research.analytics import (
    FACTORS,
    KPI_LABELS,
    add_ratios,
    build_panel,
    factor_scorecard,
    indexed_series,
    kpi_table,
    peer_ranks,
)
from finance_research.events import CATEGORIES as EVENT_CATEGORIES
from finance_research.events import event_text, get_company_events
from finance_research.private_companies import FORM_D_NOTE, PRIVATE_COMPANIES, get_private_funding
from finance_research.events import summary_prompt as event_summary_prompt
from finance_research.factcheck import verify_answer
from finance_research.prices import PriceDataError, fetch_price_history, indexed_prices, price_growth
from finance_research.evaluation import evaluate_generation, evaluate_retrieval, generate_test_set, list_runs, save_run
from finance_research.experiments import Experiment, assign_variant, summarize_experiment, verdict
from finance_research.i18n import LANGUAGES, translate
from finance_research.mcp_server import LocalServer
from finance_research.scheduler import KnowledgeBaseScheduler, SchedulerConfig, task_scheduler_command
from finance_research.conversation import (
    HELP_SUGGESTIONS,
    answer_help_question,
    conversation_to_csv,
    conversation_to_json,
    conversation_to_markdown,
    conversation_to_pdf,
    extract_citations,
    format_citations_markdown,
)
from finance_research.auth import AuthError, UserStore
from finance_research.mcp_tools import DEFAULT_MCP_URL, MCPError, build_mcp_tools, list_mcp_tools
from finance_research.monitoring import Monitor, summarize_events
from finance_research.providers import PROVIDERS, price_for
from finance_research.rag import (
    CANDIDATES_PER_METHOD,
    DEFAULT_STRATEGY,
    SEARCH_STRATEGIES,
    create_vector_store,
    index_summary,
    ingest_recent_filings,
    load_refresh_status,
    refresh_knowledge_base,
    search_filings,
)
from finance_research.ratelimit import RateLimiter, RateLimitExceeded, RateLimits, mask_key, openrouter_balance, verify_api_key
from finance_research.sec_client import METRICS, TOTAL_METRICS, SECClientError, clear_sec_cache, get_metric_records, validate_sec_user_agent
from finance_research.screening import build_research_shortlist
from finance_research.security import PromptInjectionError, check_user_question, strip_untrusted_markers
from finance_research.tools import build_tools
from finance_research.usage import add_usage, estimate_cost, fetch_openrouter_pricing, format_cost, summarize_usage

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
VECTOR_DIR = DATA_DIR / "chroma"
REFRESH_STATUS = DATA_DIR / "kb_refresh_status.json"
DEFAULT_MODEL = "openai/gpt-4o-mini"
SECTORS = [
    {
        "key": "information-technology",
        "title": "Information technology",
        "companies": {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google (Alphabet)", "META": "Meta", "ORCL": "Oracle"},
        "description": "Software, cloud, devices, and digital advertising platforms.",
    },
    {
        "key": "semiconductors",
        "title": "Semiconductors",
        "companies": {"NVDA": "NVIDIA", "AMD": "AMD", "INTC": "Intel", "AVGO": "Broadcom", "QCOM": "Qualcomm"},
        "description": "Chip design and manufacturing, including AI infrastructure demand.",
    },
    {
        "key": "retail",
        "title": "E-commerce & retail",
        "companies": {"AMZN": "Amazon", "WMT": "Walmart", "COST": "Costco", "TGT": "Target", "HD": "Home Depot"},
        "description": "Online commerce compared with large-scale physical retail.",
    },
    {
        "key": "automotive",
        "title": "Automotive & EV",
        "companies": {"TSLA": "Tesla", "GM": "General Motors", "F": "Ford", "RIVN": "Rivian", "LCID": "Lucid"},
        "description": "Electric and traditional automakers, margins, and R&D spend.",
    },
    {
        "key": "financials",
        "title": "Banking & financials",
        "companies": {"JPM": "JPMorgan Chase", "BAC": "Bank of America", "GS": "Goldman Sachs", "V": "Visa", "MA": "Mastercard"},
        "description": "Banks and payment networks. Revenue tags can differ for banks.",
    },
    {
        "key": "healthcare",
        "title": "Healthcare & pharma",
        "companies": {"JNJ": "Johnson & Johnson", "PFE": "Pfizer", "LLY": "Eli Lilly", "UNH": "UnitedHealth", "MRK": "Merck"},
        "description": "Drug makers and health insurers, with heavy R&D in pharma.",
    },
    {
        "key": "consumer-staples",
        "title": "Consumer staples",
        "companies": {"KO": "Coca-Cola", "PEP": "PepsiCo", "PG": "Procter & Gamble", "MDLZ": "Mondelez", "CL": "Colgate-Palmolive"},
        "description": "Everyday household brands with steady, mature revenue.",
    },
    {
        "key": "energy",
        "title": "Energy",
        "companies": {"XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips", "SLB": "SLB (Schlumberger)", "OXY": "Occidental"},
        "description": "Oil and gas producers and services, whose results track commodity prices.",
    },
    {
        "key": "media-telecom",
        "title": "Media & telecom",
        "companies": {"NFLX": "Netflix", "DIS": "Disney", "CMCSA": "Comcast", "VZ": "Verizon", "T": "AT&T"},
        "description": "Streaming, entertainment, and the networks that deliver them.",
    },
    {
        "key": "industrials",
        "title": "Industrials & logistics",
        "companies": {"CAT": "Caterpillar", "DE": "Deere", "HON": "Honeywell", "UPS": "UPS", "FDX": "FedEx"},
        "description": "Heavy machinery, conglomerates, and parcel delivery tied to the economic cycle.",
    },
    {
        "key": "aerospace-defense",
        "title": "Aerospace & defense",
        "companies": {"BA": "Boeing", "LMT": "Lockheed Martin", "RTX": "RTX", "NOC": "Northrop Grumman", "GD": "General Dynamics"},
        "description": "Aircraft makers and defense contractors with long, backlog-driven contracts.",
    },
    {
        "key": "travel",
        "title": "Travel & hospitality",
        "companies": {"BKNG": "Booking Holdings", "MAR": "Marriott", "HLT": "Hilton", "DAL": "Delta Air Lines", "UAL": "United Airlines"},
        "description": "Airlines, hotels, and online booking, sensitive to demand and fuel costs.",
    },
]
# Banks' deposits are liabilities by design, so balance-sheet "health" ratios don't compare fairly with other companies.
BANK_TICKERS = {"JPM", "BAC", "GS"}
COMPANY_NAMES = {ticker: name for sector in SECTORS for ticker, name in sector["companies"].items()}
TICKERS = list(COMPANY_NAMES)


def _company_label(ticker: str) -> str:
    name = COMPANY_NAMES.get(ticker)
    return f"{name} ({ticker})" if name else ticker



def _initial_secret(name: str) -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, ""))
    except st.errors.StreamlitSecretNotFoundError:
        return ""


@st.cache_resource(show_spinner=False)
def _cached_vector_store(api_key: str, persist_directory: str):
    return create_vector_store(api_key, persist_directory)


@st.cache_data(ttl=120, show_spinner=False)
def _cached_openrouter_balance(api_key: str) -> dict[str, float | None] | None:
    return openrouter_balance(api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_metrics(ticker: str, metric: str, years: int, user_agent: str) -> list[dict[str, Any]]:
    return get_metric_records(ticker, metric, years, user_agent)


@st.cache_data(ttl=3600, show_spinner=False)
def _analytics_records(tickers: tuple[str, ...], years: int, user_agent: str) -> list[dict[str, Any]]:
    from finance_research.analytics import ANALYTICS_METRICS

    return [
        row
        for ticker in tickers
        for metric in ANALYTICS_METRICS
        for row in _cached_metrics(ticker, metric, years, user_agent)
    ]


@st.cache_resource(show_spinner=False)
def _monitor() -> Monitor:
    return Monitor(DATA_DIR / "logs")


@st.cache_resource(show_spinner=False)
def _rate_limiter() -> RateLimiter:
    """One limiter shared by every browser session, so opening new tabs doesn't reset the limits."""
    return RateLimiter()


@st.cache_resource(show_spinner=False)
def _user_store() -> UserStore:
    return UserStore(DATA_DIR / "users.json")


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_mcp_tool_specs(url: str) -> list[dict[str, Any]]:
    return list_mcp_tools(url)


@st.cache_resource(show_spinner=False)
def _kb_scheduler() -> KnowledgeBaseScheduler:
    """One scheduler per server process; it keeps running when browser tabs close."""
    return KnowledgeBaseScheduler(on_event=lambda kind, **fields: _monitor().event(kind, **fields))


@st.cache_resource(show_spinner=False)
def _server_registry() -> dict[str, LocalServer]:
    """Survives reruns (the script's own globals don't), so an old server can be stopped before a new one starts."""
    return {}


@st.cache_resource(show_spinner=False, max_entries=1)
def _market_lens_mcp_server(sec_email: str, openrouter_key: str, alpha_key: str) -> tuple[str, list[dict[str, Any]]]:
    """Start (or restart, when the keys change) the app's own MCP server on localhost.

    Returns its URL and tool list. Keys go to the child process through its
    environment only; they are not written anywhere.
    """
    _local_servers = _server_registry()
    for server in _local_servers.values():
        server.stop()
    _local_servers.clear()
    server = LocalServer(
        {
            "SEC_USER_AGENT_EMAIL": sec_email,
            "OPENROUTER_API_KEY": openrouter_key,
            "ALPHA_VANTAGE_API_KEY": alpha_key,
            "MARKET_LENS_VECTOR_DIR": str(VECTOR_DIR),
        }
    )
    _local_servers[server.url] = server
    specs = server.wait_until_ready()
    _monitor().event("mcp_server_started", url=server.url, tools=[spec["name"] for spec in specs])
    return server.url, specs


# Categorical palette (fixed order, validated for colour-vision deficiency on adjacent pairs).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _color_scale(entities: list[str]) -> alt.Scale:
    """Colour follows the entity: the same company keeps the same colour as filters change."""
    return alt.Scale(domain=entities, range=SERIES_COLORS[: len(entities)])


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_pricing() -> dict[str, dict[str, float]]:
    try:
        return fetch_openrouter_pricing()
    except (RequestException, ValueError):
        return {}


def _tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    return [message for message in messages if isinstance(message, ToolMessage)]


def _tool_results_from_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    args_by_id = {
        call["id"]: call.get("args", {})
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    results = []
    for tool_message in _tool_messages(messages):
        try:
            content = json.loads(str(tool_message.content))
        except json.JSONDecodeError:
            content = {"result": str(tool_message.content)}
        results.append(
            {
                "name": tool_message.name,
                "args": args_by_id.get(tool_message.tool_call_id, {}),
                "content": content,
            }
        )
    return results


TOOL_LABELS = {
    "fetch_sec_financials": "SEC financials",
    "calculate_financial_ratio": "Ratio calculation",
    "compare_companies": "Company comparison",
    "search_filing_knowledge": "Filing search",
    "fetch_stock_quote": "Stock quote",
    "get_company_events": "Company events (8-K)",
    "get_private_company_funding": "Private-company funding (Form D)",
}
RATIO_LABELS = {
    "revenue_growth": "Revenue growth",
    "operating_margin": "Operating margin",
    "net_margin": "Net margin",
    "research_intensity": "R&D intensity",
}


def _tool_outcome(content: Any) -> tuple[str, str]:
    """Short human-readable result of a tool call, and whether it succeeded."""
    if not isinstance(content, dict):
        return "Returned text", "ok"
    if content.get("error"):
        return _shorten(content["error"], 80), "error"
    if "records" in content:
        tickers = sorted({row.get("ticker", "") for row in content["records"]})
        return f"{len(content['records'])} figure(s) for {', '.join(tickers)}", "ok"
    if "value_percent" in content:
        return f"{content['value_percent']:.2f}%", "ok"
    if "results" in content:
        found = len(content["results"])
        return (f"{found} passage(s)", "ok") if found else ("No indexed passages", "empty")
    if "price" in content:
        return f"${float(content['price']):,.2f}", "ok"
    if content.get("message"):
        return _shorten(content["message"], 80), "empty"
    return "Done", "ok"


def _render_tool_visuals(tools: list[dict[str, Any]], key: str) -> None:
    """Visualise what each tool call returned: a call timeline, then charts per result type."""
    status_icons = {"ok": "✅", "empty": "⚪", "error": "⚠️"}
    timeline = []
    for step, tool_result in enumerate(tools, start=1):
        outcome, state = _tool_outcome(tool_result.get("content"))
        args = tool_result.get("args") or {}
        timeline.append(
            {
                "Step": step,
                "Tool": TOOL_LABELS.get(tool_result["name"], tool_result["name"]),
                "Asked for": ", ".join(f"{name}={value}" for name, value in args.items()) or "—",
                "Result": f"{status_icons[state]} {outcome}",
            }
        )
    st.markdown("**Tool call timeline**")
    st.dataframe(pd.DataFrame(timeline), hide_index=True, width="stretch", key=f"{key}-timeline")

    # Financial records from fetch_sec_financials / compare_companies, merged per metric.
    records = [
        row
        for tool_result in tools
        if isinstance(tool_result.get("content"), dict)
        for row in tool_result["content"].get("records", []) or []
        if isinstance(row, dict) and row.get("value_usd") is not None
    ]
    if records:
        frame = pd.DataFrame(records).drop_duplicates(subset=["ticker", "fiscal_year", "metric_label"])
        for metric_label, metric_rows in frame.groupby("metric_label", sort=False):
            per_share = str(metric_rows.get("unit", pd.Series(["USD"])).iloc[0]) == "USD/shares"
            unit_label = "USD per share" if per_share else "USD billions"
            st.markdown(f"**{metric_label}** ({unit_label}, as reported to the SEC)")
            pivot = metric_rows.pivot(index="fiscal_year", columns="ticker", values="value_usd").sort_index()
            if not per_share:
                pivot = pivot / 1e9
            pivot.index = pivot.index.astype(str)
            if len(pivot.index) > 1:
                st.line_chart(pivot, y_label=unit_label, height=260)
            else:
                st.bar_chart(pivot.T, y_label=unit_label, height=220)

    # Ratio calculations: the result plus the two inputs behind it.
    ratios = [
        tool_result["content"]
        for tool_result in tools
        if tool_result["name"] == "calculate_financial_ratio"
        and isinstance(tool_result.get("content"), dict)
        and "value_percent" in tool_result["content"]
    ]
    if ratios:
        st.markdown("**Calculated ratios**")
        with st.container(horizontal=True):
            for ratio in ratios:
                st.metric(
                    f"{ratio['ticker']} · {RATIO_LABELS.get(ratio['ratio'], ratio['ratio'])} · FY{ratio['fiscal_year']}",
                    f"{ratio['value_percent']:.2f}%",
                    border=True,
                )
        for index, ratio in enumerate(ratios):
            inputs = ratio.get("inputs_usd") or {}
            if inputs:
                st.caption(f"{ratio['ticker']} · {ratio.get('formula', '')}")
                inputs_frame = pd.DataFrame(
                    {"Input": [str(name).replace("_", " ") for name in inputs], "USD billions": [v / 1e9 for v in inputs.values()]}
                )
                st.bar_chart(inputs_frame, x="Input", y="USD billions", horizontal=True, height=60 + 45 * len(inputs))

    quotes = [
        tool_result["content"]
        for tool_result in tools
        if tool_result["name"] == "fetch_stock_quote"
        and isinstance(tool_result.get("content"), dict)
        and tool_result["content"].get("price")
    ]
    if quotes:
        st.markdown("**Stock quotes** (Alpha Vantage, may be delayed)")
        with st.container(horizontal=True):
            for quote in quotes:
                st.metric(
                    f"{quote['ticker']} · {quote.get('latest_trading_day', '')}",
                    f"${float(quote['price']):,.2f}",
                    delta=quote.get("change_percent"),
                    border=True,
                )


def _shorten(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _render_retrieval(retrieval: dict[str, Any], results: list[dict[str, Any]], key: str) -> None:
    """Show the RAG pipeline for one filing search: steps, relevance chart, and retrieved chunks."""
    year = retrieval.get("report_year")
    strategy = retrieval.get("strategy", "vector")
    scope = f"Only {retrieval.get('ticker', '?')} chunks" + (f" from FY{year}" if year else ", all years")
    if strategy == "hybrid":
        steps = [
            ("1 · Question", f"“{_shorten(retrieval.get('query', ''), 70)}”"),
            ("2 · Filter", scope),
            ("3a · Vector", f"Top {CANDIDATES_PER_METHOD} by meaning ({retrieval.get('embedding_model', 'embeddings')})"),
            ("3b · Keyword", f"Top {CANDIDATES_PER_METHOD} by BM25 word match"),
            ("4 · Fuse", f"Reciprocal Rank Fusion keeps the best {len(results)}"),
            ("5 · Answer", "Chunks passed to the model as cited evidence"),
        ]
    else:
        method = (
            f"Converted to a vector with {retrieval.get('embedding_model', 'the embedding model')}"
            if strategy == "vector"
            else "Scored by BM25 keyword matching"
        )
        steps = [
            ("1 · Question", f"“{_shorten(retrieval.get('query', ''), 70)}”"),
            ("2 · " + ("Embed" if strategy == "vector" else "Tokenise"), method),
            ("3 · Filter", scope),
            ("4 · Retrieve", f"{len(results)} best chunk(s) of top {retrieval.get('top_k', len(results))}"),
            ("5 · Answer", "Chunks passed to the model as cited evidence"),
        ]
    st.caption(f"Search method: **{SEARCH_STRATEGIES.get(strategy, strategy)}**")
    for column, (title, detail) in zip(st.columns(len(steps)), steps):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(detail)

    if not results:
        st.info("No indexed chunks matched this search. Index the company's filings in the Filing library first.")
        return

    chart = pd.DataFrame(
        {
            "Chunk": [
                f"#{item['rank']} · FY{item.get('report_year')} · {_shorten(item.get('section') or 'Narrative', 32)}"
                for item in results
            ],
            "Relevance": [item.get("relevance", 0.0) for item in results],
        }
    )
    explanation = {
        "vector": "cosine similarity to your question, 0–1",
        "keyword": "BM25 score relative to the best match, 0–1",
        "hybrid": "fused score relative to a chunk ranked first by both methods, 0–1",
    }[strategy]
    st.markdown(f"**Relevance of each retrieved chunk** ({explanation})")
    st.bar_chart(chart, x="Chunk", y="Relevance", horizontal=True, sort=False, height=60 + 38 * len(results))
    if strategy == "hybrid":
        st.markdown("**Where each chunk ranked in the two searches** (— means it wasn't in that list)")
        st.dataframe(
            pd.DataFrame(
                {
                    "Final rank": [item["rank"] for item in results],
                    "Vector rank": [item.get("vector_rank") or "—" for item in results],
                    "Keyword rank": [item.get("keyword_rank") or "—" for item in results],
                    "Cosine": [item.get("cosine") for item in results],
                    "BM25": [item.get("bm25") for item in results],
                    "RRF score": [item.get("rrf_score") for item in results],
                }
            ),
            hide_index=True,
            key=f"{key}-fusion",
        )

    for item in results:
        with st.container(border=True, key=f"{key}-chunk-{item['rank']}"):
            st.markdown(
                f"**#{item['rank']} · {item.get('company') or item.get('ticker')} · FY{item.get('report_year')} · "
                f"{item.get('section') or 'Filing narrative'}**"
            )
            st.progress(float(item.get("relevance", 0.0)), text=f"Relevance {item.get('relevance', 0.0):.2f}")
            if item.get("safety_flags"):
                st.warning(
                    "Instruction-like text was removed from this excerpt before the model saw it: "
                    + ", ".join(item["safety_flags"]),
                    icon=":material/shield:",
                )
            st.caption(_shorten(strip_untrusted_markers(item.get("text", "")), 420))
            if item.get("source_url"):
                st.markdown(f"[Open SEC filing]({item['source_url']})")


def _render_usage_caption(usage: dict[str, Any]) -> None:
    if not usage:
        return
    model_note = f" · {usage['model']}" if usage.get("model") else ""
    st.caption(
        f":material/token: {usage.get('input_tokens', 0):,} input + {usage.get('output_tokens', 0):,} output tokens "
        f"over {usage.get('model_calls', 0)} model call(s) · {format_cost(usage.get('cost_usd'))}{model_note}"
    )


FIGURE_STATUS = {
    "matched": "✓ Found in the source data",
    "calculated": "✓ Calculated from the source data",
    "unverified": "⚠ Not found: check this figure",
}


def _render_figure_check(message: dict[str, Any], key: str) -> None:
    """Show whether each number in the answer appears in (or follows from) the data the tools returned."""
    check = message.get("verification")
    if check is None and message.get("tools"):
        check = verify_answer(str(message.get("content", "")), message["tools"])
    if not check or not check.get("checked"):
        return
    supported = check["matched"] + check["calculated"]
    if not check.get("has_tool_data"):
        st.warning(
            f"This answer contains {check['checked']} figure(s), but no data was fetched to check them against. "
            "Treat them as unverified.",
            icon=":material/fact_check:",
        )
    elif check["unverified"]:
        st.warning(
            f"**Figure check:** {supported} of {check['checked']} figures match the source data. "
            f"{check['unverified']} couldn't be matched; check before relying on {'it' if check['unverified'] == 1 else 'them'}.",
            icon=":material/fact_check:",
        )
    else:
        st.caption(f":material/verified: **Figure check:** all {check['checked']} figures match the source data.")
    with st.expander("Figure check details", icon=":material/fact_check:"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Figure": row["text"], "Result": FIGURE_STATUS[row["status"]], "Matched to": row["source"] or "—"}
                    for row in check["figures"]
                ]
            ),
            hide_index=True,
            key=f"{key}-figures",
        )
        st.caption(
            "Each number in the answer is looked up in the tool results, allowing for rounding. "
            "This confirms a figure exists in the sources, not that it's attached to the right company or year."
        )


def _render_assistant_extras(message: dict[str, Any], key: str) -> None:
    _render_usage_caption(message.get("usage") or {})
    _render_figure_check(message, key)
    citations = message.get("citations") or []
    if citations:
        st.markdown("**Sources**\n\n" + format_citations_markdown(citations))

    tools = message.get("tools") or []
    flagged = [
        tool["content"]["safety"]
        for tool in tools
        if isinstance(tool.get("content"), dict) and tool["content"].get("safety")
    ]
    if flagged:
        st.info(
            f"Prompt-injection guard: instruction-like text was removed from "
            f"{sum(len(item['flagged_chunks']) for item in flagged)} filing excerpt(s) before the model read them.",
            icon=":material/shield:",
        )
    if tools:
        with st.expander("Tool results, visualised", icon=":material/insights:", expanded=False):
            _render_tool_visuals(tools, key=f"{key}-visuals")
    searches = [
        tool["content"]
        for tool in tools
        if tool.get("name") == "search_filing_knowledge"
        and isinstance(tool.get("content"), dict)
        and tool["content"].get("retrieval")
    ]
    for index, search in enumerate(searches):
        with st.expander(f"How filing search found this · {search['retrieval'].get('ticker', '')}", icon=":material/account_tree:"):
            _render_retrieval(search["retrieval"], search.get("results", []), key=f"{key}-search-{index}")

    if tools:
        with st.expander(f"Tool activity · {len(tools)} result(s)", icon=":material/build:"):
            for tool_result in tools:
                st.markdown(f"**{tool_result['name']}**")
                st.json(tool_result["content"], expanded=False)


def _chat_history() -> list[BaseMessage]:
    history = st.session_state.get("agent_history", [])
    return [message for message in history if isinstance(message, (HumanMessage, AIMessage))]


st.set_page_config(
    page_title="Market Lens",
    page_icon=":material/monitoring:",
    layout="wide",
)

st.html(
    """
    <style>
    [class*="st-key-sector-card-"] {
        background: #0c2148;
        border: 1px solid #203a66;
        border-top: 4px solid #19f39a;
        border-radius: 14px;
        padding: 18px 20px 12px;
        min-height: 215px;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.25);
        color: #f2f6ff;
    }
    [class*="st-key-sector-card-"] h3,
    [class*="st-key-sector-card-"] p {
        color: #f2f6ff;
    }
    [class*="st-key-sector-card-"] {
        container-type: inline-size;  /* lets the title size itself to the card width */
    }
    [class*="st-key-sector-card-"] h3 {
        font-size: min(1.75rem, 8.6cqi);  /* shrink long titles so every title stays on one line */
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    [class*="st-key-sector-card-"] [data-testid="stCaptionContainer"] {
        min-height: 4.6em;  /* room for three lines of tickers, so descriptions line up across a row */
    }
    [class*="st-key-sector-card-"] button {
        background: #19f39a;
        color: #04112d;
        border: 0;
        border-radius: 10px;
        min-height: 46px;
        min-width: 68%;
        font-weight: 650;
    }
    [class*="st-key-sector-card-"] button p {
        color: #04112d;
        font-weight: 650;
    }
    [class*="st-key-sector-card-"] button {
        cursor: pointer;
        transition: background 120ms ease, transform 120ms ease;
    }
    [class*="st-key-sector-card-"] button:hover {
        background: #5ff7b8;
        color: #04112d;
        border: 0;
        transform: translateY(-1px);
    }
    [class*="st-key-sector-card-active-"] {
        border: 1px solid #19f39a;
        border-top: 4px solid #19f39a;
        box-shadow: 0 0 0 3px rgba(25, 243, 154, 0.25), 0 12px 28px rgba(0, 0, 0, 0.25);
    }
    [class*="st-key-sector-card-active-"] button {
        background: transparent;
        border: 1px solid #19f39a;
    }
    [class*="st-key-sector-card-active-"] button p {
        color: #19f39a;
    }
    </style>
    """
)

monitor = _monitor()
limiter = _rate_limiter()
user_store = _user_store()


# ---------------------------------------------------------------------------
# Sign-in and personalisation
# ---------------------------------------------------------------------------


def _apply_preferences(preferences: dict[str, Any]) -> None:
    """Load a user's saved defaults into the widgets' session state."""
    st.session_state["company_selection"] = [t for t in preferences.get("watchlist", []) if t in TICKERS] or ["AAPL", "MSFT"]
    if preferences.get("default_metric") in METRICS:
        st.session_state["dashboard_metric"] = preferences["default_metric"]
    st.session_state["years"] = int(preferences.get("years") or 5)
    provider_key = preferences.get("provider") if preferences.get("provider") in PROVIDERS else "openrouter"
    st.session_state["provider"] = provider_key
    st.session_state["model_name"] = preferences.get("model") or PROVIDERS[provider_key].default_model
    st.session_state["mcp_enabled"] = bool(preferences.get("mcp_enabled"))
    if preferences.get("language") in LANGUAGES:
        st.session_state["language"] = preferences["language"]
    if preferences.get("search_strategy") in SEARCH_STRATEGIES:
        st.session_state["search_strategy"] = preferences["search_strategy"]
    if preferences.get("tool_source") in TOOL_SOURCES:
        st.session_state["tool_source"] = preferences["tool_source"]


TOOL_SOURCES = {"builtin": "Built-in (in the app)", "mcp_server": "Market Lens MCP server"}
EXPERIMENT_FILE = DATA_DIR / "experiment.json"


def _load_experiment_settings() -> dict[str, Any]:
    defaults = {"enabled": False, "name": "rag-strategy-1", "variant_a": "vector", "variant_b": "hybrid", "share_b": 0.5}
    try:
        return {**defaults, **json.loads(EXPERIMENT_FILE.read_text(encoding="utf-8"))}
    except (FileNotFoundError, json.JSONDecodeError):
        return defaults


def _save_experiment_settings(settings: dict[str, Any]) -> None:
    EXPERIMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def _render_sign_in() -> None:
    st.title("Market Lens")
    st.caption("Evidence-led public-company research")
    _, centre, _ = st.columns([1, 2, 1])
    with centre, st.container(border=True):
        sign_in, sign_up, guest = st.tabs(["Sign in", "Create account", "Continue as guest"])
        with sign_in, st.form("sign_in_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary", width="stretch"):
                try:
                    limiter.check(f"login:{username.lower()}", RateLimits(per_minute=5, per_hour=20, daily_budget_usd=1e9))
                    profile = user_store.authenticate(username, password)
                except (AuthError, RateLimitExceeded) as error:
                    # Only log names of real accounts: people sometimes type a password into the username box.
                    logged_name = username.strip() if user_store.exists(username) else "[unknown username]"
                    monitor.event("login_failed", level="warning", username=logged_name, reason=str(error))
                    st.error(str(error), icon=":material/lock:")
                else:
                    st.session_state["user"] = profile
                    _apply_preferences(profile["preferences"])
                    monitor.event("login", username=profile["username"])
                    st.rerun()
        with sign_up, st.form("sign_up_form"):
            new_username = st.text_input("Choose a username", help="3–32 letters, numbers, dot, dash or underscore.")
            display_name = st.text_input("Display name (optional)")
            new_password = st.text_input("Password", type="password", help="At least 8 characters.")
            confirm = st.text_input("Confirm password", type="password")
            if st.form_submit_button("Create account", type="primary", width="stretch"):
                try:
                    if new_password != confirm:
                        raise AuthError("The passwords don't match.")
                    profile = user_store.create_user(new_username, new_password, display_name)
                except AuthError as error:
                    st.error(str(error), icon=":material/error:")
                else:
                    st.session_state["user"] = profile
                    _apply_preferences(profile["preferences"])
                    monitor.event("signup", username=profile["username"])
                    st.rerun()
        with guest:
            st.write("Use the app without an account. Your settings won't be saved between visits.")
            if st.button("Continue as guest", width="stretch"):
                st.session_state["user"] = {"username": "guest", "guest": True, "preferences": {}}
                monitor.event("login", username="guest")
                st.rerun()
        st.caption(
            "Accounts are stored on this computer in data/users.json with salted, hashed passwords. "
            "API keys are never saved."
        )


if "user" not in st.session_state:
    _render_sign_in()
    st.stop()

current_user: dict[str, Any] = st.session_state["user"]
is_guest = bool(current_user.get("guest"))
if is_guest:
    st.session_state.setdefault("guest_id", os.urandom(6).hex())
user_id = f"guest:{st.session_state['guest_id']}" if is_guest else current_user["username"]
is_admin = not is_guest and user_store.is_admin(current_user["username"])

# Keys saved on the server (environment variables / secrets.toml) are the owner's money,
# so only admins get them pre-filled. Everyone else pastes their own key in the sidebar.
for widget_key, secret_name in (
    ("openrouter_key", "OPENROUTER_API_KEY"),
    ("openai_key", "OPENAI_API_KEY"),
    ("anthropic_key", "ANTHROPIC_API_KEY"),
    ("alpha_key", "ALPHA_VANTAGE_API_KEY"),
):
    st.session_state.setdefault(widget_key, _initial_secret(secret_name) if is_admin else "")
st.session_state.setdefault("company_selection", ["AAPL", "MSFT"])
st.session_state.setdefault("years", 5)
st.session_state.setdefault("provider", "openrouter")
st.session_state.setdefault("model_name", PROVIDERS[st.session_state["provider"]].default_model)
st.session_state.setdefault("mcp_enabled", False)
st.session_state.setdefault("mcp_url", DEFAULT_MCP_URL)
DEFAULT_LIMITS = RateLimits()  # 5/min, 40/hour, $1/day: the most a non-admin can set
ADMIN_LIMIT_CEILING = RateLimits(per_minute=60, per_hour=500, daily_budget_usd=100.0)


def _limit_ceiling() -> RateLimits:
    return ADMIN_LIMIT_CEILING if is_admin else DEFAULT_LIMITS


def _secret_number(name: str, default: float) -> float:
    try:
        return float(_initial_secret(name) or default)
    except ValueError:
        return default


SHARED_KEY = _initial_secret("OPENROUTER_API_KEY").strip()
SHARED_MODEL = _initial_secret("SHARED_MODEL").strip() or "openai/gpt-4o-mini"
SHARED_LIMITS = RateLimits(per_minute=10, per_hour=120, daily_budget_usd=_secret_number("SHARED_DAILY_BUDGET_USD", 1.0))
SHARED_BUCKET = "__shared_key__"  # one allowance for everyone using the shared key


def _check_shared_allowance(active: bool) -> None:
    """Raise RateLimitExceeded when the app-wide allowance on the shared key is used up."""
    if not active:
        return
    try:
        limiter.check(SHARED_BUCKET, SHARED_LIMITS)
    except RateLimitExceeded as error:
        if "spending" in str(error):
            raise RateLimitExceeded(
                "Today's free AI allowance for this app is used up. Try again tomorrow, "
                "or paste your own OpenRouter key in the sidebar.",
                error.retry_after,
            ) from None
        raise RateLimitExceeded(f"The app's shared key is busy. {error}", error.retry_after) from None


def _record_shared_cost(active: bool, cost: float | None) -> None:
    if active:
        limiter.record_cost(SHARED_BUCKET, cost)


st.session_state.setdefault("limit_per_minute", DEFAULT_LIMITS.per_minute)
st.session_state.setdefault("limit_per_hour", DEFAULT_LIMITS.per_hour)
st.session_state.setdefault("limit_daily_budget", DEFAULT_LIMITS.daily_budget_usd)
st.session_state.setdefault("language", "en")
st.session_state.setdefault("search_strategy", DEFAULT_STRATEGY)
st.session_state.setdefault("tool_source", "builtin")


def T(key: str) -> str:
    """Translate an interface string into the chosen language."""
    return translate(key, st.session_state.get("language", "en"))


experiment_settings = _load_experiment_settings()
experiment = Experiment(
    name=experiment_settings["name"],
    variant_a=experiment_settings["variant_a"],
    variant_b=experiment_settings["variant_b"],
    share_b=float(experiment_settings["share_b"]),
)
if experiment_settings["enabled"]:
    ab_variant, ab_strategy = assign_variant(user_id, experiment)
else:
    ab_variant, ab_strategy = None, None

MAIN_TABS_KEY = f"main_tabs_{st.session_state['language']}"  # labels change with the language


def _sync_company_pickers() -> None:
    """Make the Analytics and Company events pickers follow the main company selection.

    Each tab has its own picker (Analytics allows at most five companies), so without this they would keep
    showing whichever companies they started with. Results for other companies are cleared, not left on screen.
    """
    selection = [ticker for ticker in st.session_state.get("company_selection", []) if ticker in TICKERS][:5]
    if not selection:
        return
    for picker in ("analytics_tickers", "event_tickers"):
        st.session_state[picker] = list(selection)
    if st.session_state.get("analytics_order") != selection:
        for stale in ("analytics_records", "analytics_prices", "analytics_order"):
            st.session_state.pop(stale, None)
    if (st.session_state.get("company_events") or {}).get("order") != selection:
        st.session_state.pop("company_events", None)


def _explore_sector(group: dict[str, Any]) -> None:
    """Runs before the page redraws: select the sector's companies, open the overview tab and load its data."""
    st.session_state["company_selection"] = list(group["companies"])
    _sync_company_pickers()
    st.session_state["active_sector"] = group["key"]
    st.session_state["auto_load_overview"] = True
    st.session_state[MAIN_TABS_KEY] = T("tab_overview")
    st.session_state["sector_toast"] = f"Comparing {group['title']}: {', '.join(group['companies'])}."
    monitor.event("sector_explored", username=user_id, sector=group["key"])


if "sector_toast" in st.session_state:
    st.toast(st.session_state.pop("sector_toast"), icon=":material/check_circle:")

KEY_WIDGETS = {"openrouter": "openrouter_key", "openai": "openai_key", "anthropic": "anthropic_key"}


def _reset_model() -> None:
    st.session_state["model_name"] = PROVIDERS[st.session_state["provider"]].default_model


def _forget_keys() -> None:
    for widget_key in ("openrouter_key", "openai_key", "anthropic_key", "alpha_key"):
        st.session_state[widget_key] = ""
    st.session_state.pop("key_checks", None)
    monitor.event("keys_cleared", username=current_user["username"])


def _sign_out() -> None:
    monitor.event("logout", username=current_user["username"])
    for state_key in list(st.session_state.keys()):
        del st.session_state[state_key]


with st.sidebar:
    st.title("Market Lens")
    st.caption("Evidence-led public-company research")
    with st.container(horizontal=True, vertical_alignment="center"):
        name = current_user.get("preferences", {}).get("display_name") or current_user["username"]
        st.markdown(f":material/account_circle: **{'Guest' if is_guest else name}**")
        st.space("stretch")
        st.button(T("sign_out"), on_click=_sign_out, type="tertiary", icon=":material/logout:")

    language = st.selectbox(
        f":material/translate: {T('language')}",
        options=list(LANGUAGES),
        format_func=lambda code: LANGUAGES[code],
        key="language",
        help="Changes the interface and the language the assistant answers in. SEC filings themselves are in English.",
    )

    st.markdown(f"**{T('model_heading')}**")
    provider_key = st.selectbox(
        T("provider"),
        options=list(PROVIDERS),
        format_func=lambda key: PROVIDERS[key].label,
        key="provider",
        on_change=_reset_model,
        help="Choose who runs the research assistant. Filing search embeddings always use OpenRouter.",
    )
    provider = PROVIDERS[provider_key]
    model_options = list(provider.models)
    if st.session_state["model_name"] not in model_options:
        model_options.append(st.session_state["model_name"])
    model_name = st.selectbox(
        "Model",
        options=model_options,
        key="model_name",
        accept_new_options=provider.free_text_model,
        help=provider.help + (" You can type any model id the provider offers." if provider.free_text_model else ""),
    )
    if provider_key != "openrouter":
        st.text_input(provider.key_label, type="password", key=KEY_WIDGETS[provider_key])
    openrouter_key = st.text_input(
        "OpenRouter API key",
        type="password",
        key="openrouter_key",
        help="Used for chat when OpenRouter is the provider, and always for filing-search embeddings.",
    )
    chat_key = st.session_state[KEY_WIDGETS[provider_key]]
    sec_email = st.text_input(
        T("sec_email"),
        value=_initial_secret("SEC_USER_AGENT_EMAIL"),
        help="SEC requests require an identifying User-Agent that includes a contact email.",
    )
    market_data_key = st.text_input(
        "Alpha Vantage key (optional)",
        type="password",
        key="alpha_key",
        help="Enables the stock quote tool. Market data is separate from SEC filing data.",
    )

    # Shared key: the owner's OpenRouter key from the server's secrets lets visitors use the AI features
    # without a key of their own. It is only used here on the server: it is never put in a widget, so it
    # never reaches the browser. Visitors on it get a fixed low-cost model and an app-wide allowance.
    using_shared_key = bool(SHARED_KEY) and not chat_key.strip()
    shared_embeddings = bool(SHARED_KEY) and not openrouter_key.strip()
    if using_shared_key:
        provider_key, provider = "openrouter", PROVIDERS["openrouter"]
        chat_key, model_name = SHARED_KEY, SHARED_MODEL
    if shared_embeddings:
        openrouter_key = SHARED_KEY
    if using_shared_key or shared_embeddings:
        st.caption(
            f":material/key: No key needed: the AI features use the app's shared key with **{SHARED_MODEL}** "
            f"(up to ${SHARED_LIMITS.daily_budget_usd:.2f} a day for everyone). Paste your own key to choose any model."
        )

    with st.expander("API keys & limits", icon=":material/key:"):
        key_rows = [
            ("openrouter", "OpenRouter", st.session_state["openrouter_key"]),
            ("openai", "OpenAI", st.session_state["openai_key"]),
            ("anthropic", "Anthropic", st.session_state["anthropic_key"]),
            ("alphavantage", "Alpha Vantage", st.session_state["alpha_key"]),
        ]
        checks = st.session_state.setdefault("key_checks", {})
        for check_key, label, value in key_rows:
            result = checks.get(check_key)
            mark = "" if result is None else (" ✅" if result[0] else " ❌")
            st.markdown(f"**{label}:** `{mask_key(value)}`{mark}")
            if result:
                st.caption(result[1])
        with st.container(horizontal=True):
            if st.button("Test keys", icon=":material/verified:"):
                for check_key, label, value in key_rows:
                    if value.strip():
                        checks[check_key] = verify_api_key(check_key, value)
                        monitor.event("key_checked", provider=check_key, ok=checks[check_key][0])
                st.rerun()
            st.button("Forget keys", icon=":material/key_off:", on_click=_forget_keys)
        st.caption("Keys live only in this browser session. They are never written to disk or logs.")
        st.markdown("**Rate limits** (per user)")
        ceiling = _limit_ceiling()
        st.number_input("Questions per minute", 1, ceiling.per_minute, key="limit_per_minute")
        st.number_input("Questions per hour", 1, ceiling.per_hour, key="limit_per_hour")
        st.number_input(
            "Daily spending limit (USD)", 0.01, ceiling.daily_budget_usd, step=0.25, key="limit_daily_budget", format="%.2f"
        )
        if not is_admin:
            st.caption("You can lower these limits. Only an admin can raise them above the defaults.")

    with st.expander("Remote MCP tools", icon=":material/hub:"):
        st.toggle(
            "Let the assistant use MCP tools",
            key="mcp_enabled",
            help="Adds tools from a public remote MCP server (streamable HTTP) to the research assistant.",
        )
        st.text_input("MCP server URL", key="mcp_url", help=f"Default: {DEFAULT_MCP_URL} (DeepWiki, no sign-in needed).")
        if st.session_state["mcp_enabled"]:
            try:
                specs = _cached_mcp_tool_specs(st.session_state["mcp_url"])
                st.caption(f"Connected · {len(specs)} tool(s): " + ", ".join(spec["name"] for spec in specs))
            except (MCPError, ValueError) as error:
                st.error(str(error), icon=":material/cloud_off:")

    with st.expander("Retrieval & tools", icon=":material/manage_search:"):
        st.selectbox(
            T("search_method"),
            options=list(SEARCH_STRATEGIES),
            format_func=lambda key: SEARCH_STRATEGIES[key],
            key="search_strategy",
            disabled=ab_variant is not None,
            help="Hybrid combines keyword (BM25) and vector search with Reciprocal Rank Fusion.",
        )
        if ab_variant is not None:
            st.caption(
                f":material/science: An A/B test is running. You're in variant **{ab_variant}** "
                f"({SEARCH_STRATEGIES[ab_strategy]})."
            )
        st.radio(
            "Run the research tools",
            options=list(TOOL_SOURCES),
            format_func=lambda key: TOOL_SOURCES[key],
            key="tool_source",
            help="The MCP option starts the app's own MCP server on this computer and calls the tools through it.",
        )
    search_strategy = ab_strategy or st.session_state["search_strategy"]

    years = st.slider(T("history"), min_value=1, max_value=10, key="years")
    selected_tickers = st.multiselect(
        T("companies"),
        options=TICKERS,
        format_func=_company_label,
        key="company_selection",
        on_change=_sync_company_pickers,
        help="Pick companies to compare, or use Explore companies on a sector card.",
    )
    if not is_guest and st.button(T("save_defaults"), icon=":material/bookmark:", width="stretch"):
        st.session_state["user"] = user_store.update_preferences(
            current_user["username"],
            {
                "watchlist": selected_tickers,
                "default_metric": st.session_state.get("dashboard_metric", "revenue"),
                "years": years,
                "provider": provider_key,
                "model": model_name,
                "mcp_enabled": st.session_state["mcp_enabled"],
                "language": language,
                "search_strategy": st.session_state["search_strategy"],
                "tool_source": st.session_state["tool_source"],
            },
        )
        st.toast("Saved. These settings load whenever you sign in.", icon=":material/bookmark_added:")
    st.divider()
    usage_slot = st.container()  # filled at the end of the run, after any new answer is counted
    st.caption(T("research_only"))

_ceiling = _limit_ceiling()  # enforced here too, not only by the widgets' max values
user_limits = RateLimits(
    per_minute=min(int(st.session_state["limit_per_minute"]), _ceiling.per_minute),
    per_hour=min(int(st.session_state["limit_per_hour"]), _ceiling.per_hour),
    daily_budget_usd=min(float(st.session_state["limit_daily_budget"]), _ceiling.daily_budget_usd),
)

st.title(T("page_title"))
st.markdown(T("page_intro"))

st.subheader(T("explore_heading"))
st.caption(T("explore_caption"))
for row_start in range(0, len(SECTORS), 3):
    cards = st.columns(3)
    for column, group in zip(cards, SECTORS[row_start : row_start + 3]):
        with column:
            is_active = st.session_state.get("active_sector") == group["key"] and st.session_state.get(
                "company_selection"
            ) == list(group["companies"])
            with st.container(key=f"sector-card-{'active-' if is_active else ''}{group['key']}", height="stretch"):
                st.subheader(group["title"], anchor=False)  # no anchor icon, so long titles don't wrap early
                st.caption(" · ".join(_company_label(ticker) for ticker in group["companies"]))
                st.write(group["description"])
                st.space("stretch")  # pushes the button to the bottom so buttons line up across a row
                st.button(
                    ("✓ " if is_active else "") + T("explore_button"),
                    key=f"sector-action-{group['key']}",
                    on_click=_explore_sector,
                    args=(group,),
                    icon=":material/arrow_downward:" if not is_active else None,
                    width="stretch",
                )

overview, analytics_tab, events_tab, private_tab, shortlist, assistant, filings, evaluation_tab, monitoring, guide = st.tabs(
    [
        T("tab_overview"),
        T("tab_analytics"),
        T("tab_events"),
        T("tab_private"),
        T("tab_shortlist"),
        T("tab_assistant"),
        T("tab_filings"),
        T("tab_evaluation"),
        T("tab_monitoring"),
        T("tab_guide"),
    ],
    key=MAIN_TABS_KEY,
    on_change="rerun",  # lets the Explore buttons switch to the overview tab
)

with overview:
    st.subheader(T("compare_heading"))
    metric_key = st.selectbox(
        T("metric"),
        options=TOTAL_METRICS,
        format_func=lambda key: METRICS[key]["label"],
        key="dashboard_metric",
    )
    with st.container(horizontal=True):
        load_clicked = st.button(T("load_sec"), type="primary", icon=":material/download:")
        refresh_clicked = st.button(
            T("refresh_sec"),
            icon=":material/refresh:",
            help="SEC figures are cached for an hour. This clears the cache and fetches the latest filings data.",
        )
    if refresh_clicked:
        _cached_metrics.clear()
        _analytics_records.clear()
        clear_sec_cache()
        monitor.event("sec_cache_cleared", username=user_id)
        st.toast("Cleared cached SEC data; loading the latest figures.", icon=":material/refresh:")
    auto_load = st.session_state.pop("auto_load_overview", False)
    if auto_load and not sec_email.strip():
        st.warning(
            "Companies selected. Add your **SEC contact email** in the sidebar, then press **Load SEC data** to see them.",
            icon=":material/mail:",
        )
        auto_load = False
    if load_clicked or refresh_clicked or auto_load:
        try:
            user_agent = validate_sec_user_agent(sec_email)
            if not selected_tickers:
                st.warning("Select at least one company.")
            else:
                with st.status("Fetching annual SEC data…", expanded=True) as status:
                    rows = [
                        row
                        for ticker in selected_tickers
                        for row in _cached_metrics(ticker, metric_key, years, user_agent)
                    ]
                    status.update(label="SEC data loaded", state="complete")
                st.session_state["overview_rows"] = rows
                st.session_state["overview_metric"] = metric_key
                st.session_state["overview_loaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        except (ValueError, RequestException, SECClientError) as error:
            st.error(str(error), icon=":material/error:")

    overview_rows = st.session_state.get("overview_rows", [])
    if overview_rows:
        frame = pd.DataFrame(overview_rows)
        latest = frame.sort_values("fiscal_year").groupby("ticker", as_index=False).tail(1)
        with st.container(horizontal=True):
            for row in latest.to_dict("records"):
                st.metric(
                    f"{row['company']} · FY{row['fiscal_year']}",
                    f"${row['value_usd'] / 1_000_000_000:,.1f}B",
                    border=True,
                )
        chart_data = frame.pivot(index="fiscal_year", columns="ticker", values="value_usd").sort_index()
        st.subheader(f"{METRICS[st.session_state.get('overview_metric', metric_key)]['label']} by fiscal year")
        st.line_chart(
            chart_data / 1_000_000_000,
            y_label="USD billions",
            alt="Annual SEC-reported financial metric by company, in billions of US dollars",
        )
        st.dataframe(
            frame[["company", "ticker", "fiscal_year", "metric_label", "value_usd", "filed", "source_url"]],
            hide_index=True,
            column_config={
                "value_usd": st.column_config.NumberColumn("Value (USD)", format="$%.0f"),
                "source_url": st.column_config.LinkColumn("SEC source"),
            },
            alt="SEC-reported annual financial data and source links",
        )
    else:
        st.info("Choose companies and a metric, then load SEC data to see a sourced comparison.")

PRICE_YEARS = 10


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_events(ticker: str, days: int, user_agent: str) -> list[dict[str, Any]]:
    return get_company_events(ticker, user_agent, days=days, limit=25)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_private_funding(company_key: str, user_agent: str) -> dict[str, Any]:
    return get_private_funding(company_key, user_agent)


def _usd_short(value: float | None) -> str:
    if value is None:
        return "—"
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= size:
            return f"${value / size:,.1f}{suffix}"
    return f"${value:,.0f}"


def _starter_questions(tickers: list[str]) -> list[tuple[str, str, str]]:
    """(icon, button label, question) built from the companies picked in the sidebar."""
    first = tickers[0] if tickers else "AAPL"
    second = next((ticker for ticker in tickers[1:] if ticker != first), "MSFT" if first != "MSFT" else "AAPL")
    name, other = COMPANY_NAMES.get(first, first), COMPANY_NAMES.get(second, second)
    return [
        (":material/compare_arrows:", f"Compare {name} and {other} revenue",
         f"Compare {name} ({first}) and {other} ({second}) revenue over the last 5 years. Which grew faster?"),
        (":material/percent:", f"{name}'s net margin, explained",
         f"What was {name} ({first})'s net margin in its latest fiscal year, and how is it calculated?"),
        (":material/event_note:", f"What has {name} announced lately?",
         f"What major events has {name} ({first}) reported in SEC 8-K filings in the last 90 days?"),
        (":material/warning:", f"Risks in {name}'s 10-K",
         f"What are the main risks {name} ({first}) describes in its latest 10-K?"),
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def _price_history(ticker: str, years: int, use_alpha: bool, _alpha_key: str) -> tuple[pd.DataFrame, str]:
    """Monthly adjusted prices, cached for an hour. The key itself is excluded from the cache hash."""
    return fetch_price_history(ticker, years, _alpha_key if use_alpha else "")


def _load_price_histories(tickers: list[str], alpha_key: str) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, str]]:
    histories: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    errors: dict[str, str] = {}
    for ticker in tickers:
        try:
            histories[ticker], sources[ticker] = _price_history(ticker, PRICE_YEARS, bool(alpha_key.strip()), alpha_key.strip())
        except (PriceDataError, RequestException, ValueError, KeyError) as error:
            errors[ticker] = str(error)
    return histories, sources, errors


def _percent(value: Any) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.1%}"


with analytics_tab:
    st.subheader("Advanced analytics")
    st.caption(
        "Growth, profitability and efficiency computed from SEC-reported annual figures. "
        "Ratios are derived values; check the underlying filings before relying on them."
    )
    # Starting value via session state: _sync_company_pickers sets it later, and a widget default would clash.
    st.session_state.setdefault("analytics_tickers", [t for t in selected_tickers if t in TICKERS][:5] or ["AAPL", "MSFT"])
    analytics_tickers = st.multiselect(
        T("companies"),
        options=TICKERS,
        format_func=_company_label,
        max_selections=5,
        key="analytics_tickers",
        help="Up to five companies, so every series keeps a distinct, colour-blind-safe colour.",
    )
    analytics_years = st.slider("Years of history", 3, 10, max(years, 5), key="analytics_years")
    if st.button("Run analysis", type="primary", icon=":material/analytics:", key="run_analytics"):
        try:
            user_agent = validate_sec_user_agent(sec_email)
            if not analytics_tickers:
                raise ValueError("Choose at least one company.")
            with st.status("Fetching six SEC metrics per company…") as status:
                records = _analytics_records(tuple(analytics_tickers), analytics_years, user_agent)
                status.update(label=f"Fetching {PRICE_YEARS} years of share prices…")
                histories, sources, price_errors = _load_price_histories(list(analytics_tickers), market_data_key)
                status.update(label="Analysis ready", state="complete")
            st.session_state["analytics_prices"] = (histories, sources, price_errors)
            st.session_state["analytics_records"] = records
            st.session_state["analytics_order"] = list(analytics_tickers)
            monitor.event("analytics_run", username=user_id, companies=len(analytics_tickers), years=analytics_years)
        except (ValueError, RequestException, SECClientError) as error:
            st.error(str(error), icon=":material/error:")

    analytics_records = st.session_state.get("analytics_records")
    if analytics_records:
        order = st.session_state["analytics_order"]
        ratios = add_ratios(build_panel(analytics_records))
        kpis = kpi_table(ratios).set_index("ticker").reindex(order).reset_index().dropna(subset=["company"])
        colors = _color_scale(order)

        st.markdown("#### Headline KPIs · latest fiscal year")
        for row in kpis.to_dict("records"):
            with st.container(border=True):
                st.markdown(f"**{row['company']} ({row['ticker']}) · FY{row['fiscal_year']}**")
                with st.container(horizontal=True):
                    st.metric("Revenue", "—" if row["revenue_b"] is None else f"${row['revenue_b']:,.1f}B")
                    st.metric(
                        f"Revenue CAGR ({row['cagr_years']}y)",
                        _percent(row["revenue_cagr"]),
                    )
                    st.metric(
                        "Revenue growth",
                        _percent(row["revenue_growth"]),
                        delta=None if row["revenue_growth"] is None or pd.isna(row["revenue_growth"]) else f"{row['revenue_growth']:+.1%} YoY",
                    )
                    st.metric("Operating margin", _percent(row["operating_margin"]))
                    st.metric("Net margin", _percent(row["net_margin"]))
                    st.metric("R&D intensity", _percent(row["rd_intensity"]))
                    st.metric("Return on assets", _percent(row["return_on_assets"]))
                    st.metric(
                        "Debt-to-equity",
                        "—" if row["debt_to_equity"] is None or pd.isna(row["debt_to_equity"]) else f"{row['debt_to_equity']:.2f}×",
                        help="Long-term debt ÷ shareholders' equity. Not shown when equity is zero or negative, or debt isn't reported.",
                    )
                    st.metric(
                        "Net cash",
                        "—" if row["net_cash_b"] is None or pd.isna(row["net_cash_b"]) else f"{'−' if row['net_cash_b'] < 0 else ''}${abs(row['net_cash_b']):,.1f}B",
                        help="Cash and cash equivalents minus long-term debt. Negative means more debt than cash.",
                    )

        growth_col, margin_col = st.columns(2)
        with growth_col:
            st.markdown("#### Revenue growth, indexed")
            st.caption("Each company's revenue set to 100 in the first year all of them report.")
            indexed = indexed_series(ratios)
            if not indexed.empty:
                hover = alt.selection_point(fields=["fiscal_year"], nearest=True, on="pointerover", empty=False)
                base = alt.Chart(indexed).encode(
                    x=alt.X("fiscal_year:O", title="Fiscal year"),
                    y=alt.Y("index:Q", title="Index (first common year = 100)", scale=alt.Scale(zero=False)),
                    color=alt.Color("ticker:N", scale=colors, title="Company"),
                )
                lines = base.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True))
                rule = (
                    alt.Chart(indexed)
                    .mark_rule(color="#898781")
                    .encode(
                        x="fiscal_year:O",
                        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
                        tooltip=[alt.Tooltip("fiscal_year:O", title="Year")]
                        + [alt.Tooltip(f"{ticker}:Q", title=ticker, format=".0f") for ticker in order if ticker in set(indexed["ticker"])],
                    )
                    .transform_pivot("ticker", value="index", groupby=["fiscal_year"])
                    .add_params(hover)
                )
                labels = (
                    base.transform_window(rank="rank()", sort=[alt.SortField("fiscal_year", order="descending")], groupby=["ticker"])
                    .transform_filter(alt.datum.rank == 1)
                    .mark_text(align="left", dx=6, fontSize=11)
                    .encode(text="ticker:N", color=alt.value("#52514e"))
                )
                st.altair_chart((lines + rule + labels).properties(height=300), width="stretch")
        with margin_col:
            margin_choice = st.segmented_control(
                "Margin",
                ["operating_margin", "net_margin", "rd_intensity", "return_on_assets", "liabilities_to_assets"],
                format_func=lambda key: KPI_LABELS[key],
                default="net_margin",
                key="analytics_margin",
            )
            margin_key = margin_choice or "net_margin"
            st.markdown(f"#### {KPI_LABELS[margin_key]} over time")
            margin_data = ratios.dropna(subset=[margin_key])[["ticker", "fiscal_year", margin_key]]
            if not margin_data.empty:
                margin_chart = (
                    alt.Chart(margin_data)
                    .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True))
                    .encode(
                        x=alt.X("fiscal_year:O", title="Fiscal year"),
                        y=alt.Y(f"{margin_key}:Q", title=KPI_LABELS[margin_key], axis=alt.Axis(format="%")),
                        color=alt.Color("ticker:N", scale=colors, title="Company"),
                        tooltip=[
                            alt.Tooltip("ticker:N", title="Company"),
                            alt.Tooltip("fiscal_year:O", title="Year"),
                            alt.Tooltip(f"{margin_key}:Q", title=KPI_LABELS[margin_key], format=".1%"),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(margin_chart, width="stretch")
            else:
                st.caption("Not enough data for this ratio.")

        heat_col, scatter_col = st.columns(2)
        with heat_col:
            st.markdown("#### Year-over-year revenue growth")
            st.caption("Blue = growth, red = decline, grey = flat.")
            heat = ratios.dropna(subset=["revenue_growth"])[["ticker", "fiscal_year", "revenue_growth"]]
            if not heat.empty:
                limit = max(0.05, float(heat["revenue_growth"].abs().quantile(0.95)))
                heatmap = alt.Chart(heat).encode(
                    x=alt.X("fiscal_year:O", title="Fiscal year"),
                    y=alt.Y("ticker:N", title=None, sort=order),
                )
                cells = heatmap.mark_rect(cornerRadius=3, stroke="#fcfcfb", strokeWidth=2).encode(
                    color=alt.Color(
                        "revenue_growth:Q",
                        title="YoY growth",
                        scale=alt.Scale(domain=[-limit, 0, limit], range=["#e34948", "#e6e6e6", "#2a78d6"], interpolate="lab", clamp=True),
                        legend=alt.Legend(format="%"),
                    ),
                    tooltip=[
                        alt.Tooltip("ticker:N", title="Company"),
                        alt.Tooltip("fiscal_year:O", title="Year"),
                        alt.Tooltip("revenue_growth:Q", title="Growth", format="+.1%"),
                    ],
                )
                text = heatmap.mark_text(fontSize=11).encode(
                    text=alt.Text("revenue_growth:Q", format="+.0%"), color=alt.value("#0b0b0b")
                )
                st.altair_chart((cells + text).properties(height=60 + 40 * len(order)), width="stretch")
        with scatter_col:
            st.markdown("#### Growth vs profitability · latest year")
            st.caption("Top-right companies grew faster and kept more of each dollar of revenue.")
            scatter = kpis.dropna(subset=["revenue_growth", "net_margin"])
            if not scatter.empty:
                points = alt.Chart(scatter).encode(
                    x=alt.X("revenue_growth:Q", title="Revenue growth (YoY)", axis=alt.Axis(format="%")),
                    y=alt.Y("net_margin:Q", title="Net margin", axis=alt.Axis(format="%")),
                )
                dots = points.mark_circle(size=160, opacity=1, stroke="#fcfcfb", strokeWidth=2).encode(
                    color=alt.Color("ticker:N", scale=colors, title="Company"),
                    tooltip=[
                        alt.Tooltip("company:N"),
                        alt.Tooltip("revenue_growth:Q", title="Growth", format="+.1%"),
                        alt.Tooltip("net_margin:Q", title="Net margin", format=".1%"),
                    ],
                )
                names = points.mark_text(dy=-14, fontSize=11).encode(text="ticker:N", color=alt.value("#52514e"))
                st.altair_chart((dots + names).properties(height=300), width="stretch")

        # ---------------- Share price history ----------------
        histories, price_sources, price_errors = st.session_state.get("analytics_prices", ({}, {}, {}))
        price_order = [ticker for ticker in order if ticker in histories]
        stats = {ticker: price_growth(histories[ticker]) for ticker in price_order}
        st.markdown(f"#### Share price · last {PRICE_YEARS} years")
        if price_order:
            view = st.segmented_control(
                "Show",
                ["growth", "price"],
                format_func={"growth": "Growth of $1,000", "price": "Share price (USD)"}.get,
                default="growth",
                key="price_view",
            ) or "growth"
            log_scale = st.toggle(
                "Log scale",
                value=False,
                key="price_log",
                help="On a log scale equal distances mean equal percentage changes, which makes fast and slow growers easier to compare.",
            )
            if view == "growth":
                series = indexed_prices({ticker: histories[ticker] for ticker in price_order})
                series["value"] = series["index"] * 10  # index 100 -> $1,000
                y_title, value_format = "Value of $1,000 invested (USD)", "$,.0f"
                st.caption(
                    "What $1,000 invested in the first month all selected companies traded would be worth, "
                    "with dividends reinvested and adjusted for stock splits."
                )
            else:
                series = pd.concat(
                    [histories[ticker].assign(ticker=ticker, value=histories[ticker]["close"]) for ticker in price_order],
                    ignore_index=True,
                )
                y_title, value_format = "Share price (USD)", "$,.2f"
                st.caption("Monthly closing price as traded, not adjusted for dividends. Splits are reflected by the data source.")
            series = series[["ticker", "date", "value"]]
            hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
            y_scale = alt.Scale(type="log") if log_scale else alt.Scale(zero=view == "growth")
            base = alt.Chart(series).encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")),
                y=alt.Y("value:Q", title=y_title, scale=y_scale, axis=alt.Axis(format="$,.0f")),
                color=alt.Color("ticker:N", scale=colors, title="Company"),
            )
            lines = base.mark_line(strokeWidth=2)
            points = base.mark_point(size=64, filled=True).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
            rule = (
                alt.Chart(series)
                .mark_rule(color="#898781")
                .encode(
                    x="date:T",
                    opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
                    tooltip=[alt.Tooltip("date:T", title="Month", format="%b %Y")]
                    + [alt.Tooltip(f"{ticker}:Q", title=ticker, format=value_format) for ticker in price_order],
                )
                .transform_pivot("ticker", value="value", groupby=["date"])
                .add_params(hover)
            )
            end_labels = (
                base.transform_window(rank="rank()", sort=[alt.SortField("date", order="descending")], groupby=["ticker"])
                .transform_filter(alt.datum.rank == 1)
                .mark_text(align="left", dx=6, fontSize=11)
                .encode(text="ticker:N", color=alt.value("#52514e"))
            )
            st.altair_chart((lines + points + rule + end_labels).properties(height=340), width="stretch")

            eps = {
                row["ticker"]: (None if row.get("eps_diluted") is None or pd.isna(row["eps_diluted"]) else float(row["eps_diluted"]))
                for row in kpis.to_dict("records")
            }
            summary = pd.DataFrame(
                [
                    {
                        "Company": _company_label(ticker),
                        "Period": f"{stats[ticker]['start']:%b %Y} – {stats[ticker]['end']:%b %Y}" if stats[ticker].get("start") is not None else "—",
                        "Start price": stats[ticker].get("start_price"),
                        "Latest price": stats[ticker].get("end_price"),
                        "Total return": stats[ticker]["total_return"],
                        "Per year": stats[ticker]["annual_return"],
                        "Worst fall": stats[ticker]["max_drawdown"],
                        "$1,000 became": stats[ticker]["value_of_1000"],
                        "Latest EPS": eps.get(ticker),
                        "P/E": (stats[ticker]["end_price"] / eps[ticker])
                        if eps.get(ticker) and eps[ticker] > 0 and stats[ticker].get("end_price")
                        else None,
                    }
                    for ticker in price_order
                ]
            )
            st.dataframe(
                summary,
                hide_index=True,
                column_config={
                    "Start price": st.column_config.NumberColumn(format="$%.2f"),
                    "Latest price": st.column_config.NumberColumn(format="$%.2f"),
                    "Total return": st.column_config.NumberColumn(format="percent"),
                    "Per year": st.column_config.NumberColumn(format="percent", help="Annualised return including reinvested dividends (CAGR)."),
                    "Worst fall": st.column_config.NumberColumn(format="percent", help="Largest drop from a previous high (maximum drawdown), using month-end prices."),
                    "$1,000 became": st.column_config.NumberColumn(format="dollar"),
                    "Latest EPS": st.column_config.NumberColumn(format="$%.2f", help="Diluted earnings per share from the latest 10-K."),
                    "P/E": st.column_config.NumberColumn(
                        format="%.1f×",
                        help="Price-to-earnings: latest monthly price ÷ latest annual diluted EPS. "
                        "Blank when earnings are negative. A high P/E means investors pay more for each dollar of past profit.",
                    ),
                },
            )
            used_sources = sorted(set(price_sources.values()))
            st.caption(
                f"Source: {', '.join(used_sources)} (monthly, split- and dividend-adjusted). "
                + ("Yahoo Finance's chart service is unofficial; add an Alpha Vantage key in the sidebar to use an official API. " if "Yahoo Finance" in used_sources else "")
                + "Past share-price performance does not predict future returns."
            )
        for ticker, message in price_errors.items():
            st.warning(f"{_company_label(ticker)}: {message}", icon=":material/show_chart:")
        if not price_order and not price_errors:
            st.caption("Press **Run analysis** again to load share prices.")

        # ---------------- Factor scorecard ----------------
        st.markdown("#### Factor scorecard")
        st.caption(
            "Each factor turns one measure into a 0–100 score on a fixed scale, so a score means the same for every company. "
            "This summarises reported fundamentals; it is not a buy or sell recommendation."
        )
        with st.popover("Adjust factor weights", icon=":material/tune:"):
            st.caption("0 leaves a factor out. Higher numbers count more in the overall score.")
            weights = {
                key: st.slider(spec["label"], 0, 3, 1, key=f"factor_weight_{key}", help=f"{spec['measure']}: {spec['formula']}")
                for key, spec in FACTORS.items()
            }
        card = factor_scorecard(
            ratios,
            kpis,
            annual_returns={ticker: stats[ticker]["annual_return"] for ticker in price_order},
            not_applicable={"financial_health": BANK_TICKERS},
            weights=weights,
        )
        active = [key for key in FACTORS if weights.get(key)]
        if not card.empty and active:
            cells = pd.DataFrame(
                [
                    {
                        "ticker": row["ticker"],
                        "factor": FACTORS[key]["label"],
                        "score": row[f"{key}_score"],
                        "raw": "—"
                        if row[f"{key}_value"] is None or pd.isna(row[f"{key}_value"])
                        else (f"{row[f'{key}_value']:.2f}×" if FACTORS[key]["format"] == "ratio" else f"{row[f'{key}_value']:.1%}"),
                        "measure": FACTORS[key]["measure"],
                    }
                    for row in card.to_dict("records")
                    for key in active
                ]
            )
            score_col, overall_col = st.columns([3, 2])
            with score_col:
                factor_order = [FACTORS[key]["label"] for key in active]
                company_order = list(card["ticker"])
                grid = alt.Chart(cells).encode(
                    x=alt.X("factor:N", sort=factor_order, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=110)),
                    y=alt.Y("ticker:N", sort=company_order, title=None),
                )
                tiles = grid.mark_rect(cornerRadius=3, stroke="#fcfcfb", strokeWidth=2).encode(
                    color=alt.Color(
                        "score:Q",
                        title="Score",
                        scale=alt.Scale(domain=[0, 100], range=["#e8f0fb", "#1d5fb4"]),
                    ),
                    tooltip=[
                        alt.Tooltip("ticker:N", title="Company"),
                        alt.Tooltip("factor:N", title="Factor"),
                        alt.Tooltip("measure:N", title="Measure"),
                        alt.Tooltip("raw:N", title="Value"),
                        alt.Tooltip("score:Q", title="Score", format=".0f"),
                    ],
                )
                labels = grid.mark_text(fontSize=11).encode(
                    text=alt.condition("isValid(datum.score)", alt.Text("score:Q", format=".0f"), alt.value("n/a")),
                    color=alt.condition("datum.score > 55", alt.value("#ffffff"), alt.value("#0b0b0b")),
                )
                st.altair_chart((tiles + labels).properties(height=80 + 44 * len(company_order)), width="stretch")
            with overall_col:
                overall = card.assign(company=card["ticker"].map(lambda ticker: COMPANY_NAMES.get(ticker, ticker)))[
                    ["ticker", "company", "overall_score", "factors_used"]
                ].rename(
                    columns={"ticker": "Ticker", "company": "Company", "overall_score": "Overall score", "factors_used": "Factors used"}
                )
                st.dataframe(
                    overall,
                    hide_index=True,
                    column_config={
                        "Overall score": st.column_config.ProgressColumn("Overall score", min_value=0, max_value=100, format="%.0f"),
                        "Factors used": st.column_config.NumberColumn(help="Factors with data and a weight above 0. Missing data is left out, not scored as zero."),
                    },
                )
        elif not active:
            st.info("All factor weights are 0. Raise at least one in **Adjust factor weights**.")
        with st.expander("How the scores are calculated", icon=":material/calculate:"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Factor": spec["label"],
                            "Measure": spec["measure"],
                            "Formula": spec["formula"],
                            "Scores 0 at": f"{spec['floor']:.2f}×" if spec["format"] == "ratio" else f"{spec['floor']:.0%}",
                            "Scores 100 at": f"{spec['ceiling']:.2f}×" if spec["format"] == "ratio" else f"{spec['ceiling']:.0%}",
                        }
                        for spec in FACTORS.values()
                    ]
                ),
                hide_index=True,
            )
            st.markdown(
                "Scores rise in a straight line between the two thresholds and are capped at 0 and 100. "
                "The overall score is the weighted average of the factors a company has data for. "
                "Not covered: debt, valuation (price compared with earnings), and future outlook. "
                "Banks and energy companies usually report no R&D, so Innovation is skipped for them rather than scored as zero. "
                "Financial health is skipped for banks, because customer deposits count as liabilities and every bank would score near zero."
            )

        st.markdown("#### Peer ranking")
        st.caption("1 = best among the selected companies on each measure. Missing data ranks last.")
        ranks = peer_ranks(kpis)
        st.dataframe(
            ranks.rename(columns={**KPI_LABELS, "company": "Company", "ticker": "Ticker", "average_rank": "Average rank"}),
            hide_index=True,
            column_config={"Average rank": st.column_config.ProgressColumn("Average rank", min_value=1, max_value=max(len(ranks), 1), format="%.2f")},
        )
        with st.expander("Data table", icon=":material/table:"):
            table = ratios.drop(columns=["liabilities_derived"], errors="ignore").copy()
            for column in ("revenue", "operating_income", "net_income", "research_and_development", "total_assets", "operating_cash_flow", "total_liabilities", "stockholders_equity", "cash", "long_term_debt", "net_cash"):
                table[column] = table[column] / 1e9
            st.dataframe(
                table,
                hide_index=True,
                column_config={
                    **{c: st.column_config.NumberColumn(f"{c.replace('_', ' ').title()} (USD B)", format="%.2f") for c in ("revenue", "operating_income", "net_income", "research_and_development", "total_assets", "operating_cash_flow", "total_liabilities", "stockholders_equity", "cash", "long_term_debt", "net_cash")},
                    **{c: st.column_config.NumberColumn(KPI_LABELS.get(c, c), format="percent") for c in ("operating_margin", "net_margin", "rd_intensity", "return_on_assets", "revenue_growth")},
                    "cash_conversion": st.column_config.NumberColumn("Cash conversion (OCF ÷ net income)", format="%.2f×"),
                    "liabilities_to_assets": st.column_config.NumberColumn("Liabilities ÷ assets", format="percent"),
                    "debt_to_equity": st.column_config.NumberColumn("Debt-to-equity", format="%.2f×"),
                    "eps_diluted": st.column_config.NumberColumn("Diluted EPS (USD)", format="$%.2f"),
                },
            )
    else:
        st.info("Pick up to five companies and press **Run analysis**.")

with events_tab:
    st.subheader("Company events")
    st.caption(
        "Major news straight from the source: companies must file a Form 8-K with the SEC within four business days "
        "of events such as earnings, executive changes, deals or cybersecurity incidents."
    )
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.session_state.setdefault("event_tickers", [t for t in selected_tickers if t in TICKERS][:5] or ["AAPL", "MSFT"])
        event_tickers = st.multiselect(
            T("companies"),
            options=TICKERS,
            format_func=_company_label,
            max_selections=5,
            key="event_tickers",
        )
        event_days = st.segmented_control(
            "Look back",
            [30, 90, 180, 365],
            format_func=lambda days: {30: "1 month", 90: "3 months", 180: "6 months", 365: "1 year"}[days],
            default=180,
            key="event_days",
        ) or 180
    if st.button("Load events", type="primary", icon=":material/event_note:", key="load_events"):
        try:
            user_agent = validate_sec_user_agent(sec_email)
            if not event_tickers:
                raise ValueError("Choose at least one company.")
            loaded, failed = [], {}
            with st.spinner("Reading 8-K filings from SEC EDGAR…"):
                for ticker in event_tickers:
                    try:
                        loaded += _cached_events(ticker, int(event_days), user_agent)
                    except (ValueError, RequestException, SECClientError) as error:
                        failed[ticker] = str(error)
            st.session_state["company_events"] = {"events": loaded, "failed": failed, "order": list(event_tickers), "days": int(event_days)}
            monitor.event("events_loaded", username=user_id, companies=len(event_tickers), days=int(event_days), events=len(loaded))
        except ValueError as error:
            st.error(str(error), icon=":material/error:")

    event_state = st.session_state.get("company_events")
    if not event_state:
        st.info("Pick companies and press **Load events** to see their recent SEC 8-K filings.")
    else:
        for ticker, message in event_state["failed"].items():
            st.warning(f"{_company_label(ticker)}: {message}", icon=":material/warning:")
        events = sorted(event_state["events"], key=lambda event: (event["filed"], event["accession"]), reverse=True)
        if not events:
            st.info(f"No 8-K filings in the last {event_state['days']} days for these companies.")
        else:
            present = [category for category in EVENT_CATEGORIES if any(event["category"] == category for event in events)]
            chosen = st.pills(
                "Show",
                present,
                selection_mode="multi",
                default=present,
                key=f"event_filter_{'-'.join(present)}",
            ) or []
            shown = [event for event in events if event["category"] in chosen]

            counts = st.columns(min(4, len(present)) or 1)
            for column, category in zip(counts, present[:4]):
                column.metric(category, sum(event["category"] == category for event in events))

            if shown:
                timeline = pd.DataFrame(
                    [
                        {
                            "date": pd.Timestamp(event["filed"]),
                            "ticker": event["ticker"],
                            "category": event["category"],
                            "headline": event["headline"],
                        }
                        for event in shown
                    ]
                )
                category_scale = alt.Scale(domain=EVENT_CATEGORIES, range=SERIES_COLORS[: len(EVENT_CATEGORIES)])
                chart = (
                    alt.Chart(timeline)
                    .mark_circle(size=150, opacity=1, stroke="#fcfcfb", strokeWidth=2)
                    .encode(
                        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y")),
                        y=alt.Y("ticker:N", title=None, sort=event_state["order"]),
                        color=alt.Color("category:N", scale=category_scale, title="Event type"),
                        tooltip=[
                            alt.Tooltip("ticker:N", title="Company"),
                            alt.Tooltip("date:T", title="Filed", format="%d %b %Y"),
                            alt.Tooltip("category:N", title="Type"),
                            alt.Tooltip("headline:N", title="Event"),
                        ],
                    )
                    .properties(height=70 + 44 * len(event_state["order"]))
                )
                st.altair_chart(chart, width="stretch")

            summaries = st.session_state.setdefault("event_summaries", {})
            for event in shown[:40]:
                with st.container(border=True):
                    with st.container(horizontal=True, vertical_alignment="center"):
                        st.markdown(f"**{event['headline']}**")
                        st.space("stretch")
                        st.badge(event["category"], color="gray")
                    filed = datetime.strptime(event["filed"], "%Y-%m-%d").strftime("%d %b %Y")
                    st.caption(
                        f"{_company_label(event['ticker'])} · {event['form']} filed {filed} · "
                        + ", ".join(f"Item {item['code']}" for item in event["items"])
                    )
                    summary = summaries.get(event["accession"])
                    if summary:
                        st.markdown(summary["text"])
                        st.caption(
                            f":material/smart_toy: AI summary of the filing · {summary['model']} · {format_cost(summary['cost'])}"
                            + (" · instruction-like text was removed from the filing first" if summary["flags"] else "")
                        )
                    with st.container(horizontal=True):
                        st.link_button("Open filing", event["filing_url"], icon=":material/open_in_new:", type="tertiary")
                        if not summary and st.button(
                            "Summarise with AI",
                            key=f"summarise-{event['accession']}",
                            icon=":material/auto_awesome:",
                            type="tertiary",
                        ):
                            try:
                                if not chat_key.strip():
                                    raise ValueError(f"Add your {provider.key_label} in the sidebar to summarise filings.")
                                limiter.check(user_id, user_limits)
                                _check_shared_allowance(using_shared_key)
                                with st.spinner("Reading the filing and summarising…"):
                                    text = event_text(event, validate_sec_user_agent(sec_email))
                                    prompt, flags = event_summary_prompt(event, text)
                                    reply, usage = complete(provider_key, chat_key, model_name.strip(), prompt, max_tokens=700)
                                cost = estimate_cost(usage, price_for(provider_key, model_name.strip(), _cached_pricing()))
                                limiter.record_cost(user_id, cost)
                                _record_shared_cost(using_shared_key, cost)
                                st.session_state["usage_total"] = add_usage(st.session_state.get("usage_total", {}), usage, cost)
                                summaries[event["accession"]] = {"text": reply.strip(), "model": model_name.strip(), "cost": cost, "flags": flags}
                                monitor.event("event_summarised", username=user_id, ticker=event["ticker"], cost_usd=cost)
                                st.rerun()
                            except RateLimitExceeded as error:
                                st.warning(str(error), icon=":material/timer:")
                            except Exception as error:  # SEC, network or provider errors: show, don't crash the tab
                                monitor.event("event_summary_error", level="error", username=user_id, error=str(error)[:300])
                                st.error(f"Couldn't summarise this filing: {error}", icon=":material/error:")
            if len(shown) > 40:
                st.caption(f"Showing the 40 most recent of {len(shown)} events.")
            st.caption(
                "Event types come from the item codes companies tick on Form 8-K. AI summaries can contain mistakes: "
                "open the filing to check the details."
            )

with private_tab:
    st.subheader("Private companies")
    st.caption(
        "Private companies don't publish financial statements, but when they raise money they file a **Form D** with the SEC. "
        "That shows how much they raised and when: not revenue, profit or valuation."
    )
    private_key = st.pills(
        "Company",
        list(PRIVATE_COMPANIES),
        format_func=lambda key: PRIVATE_COMPANIES[key]["name"],
        default="stripe",
        key="private_company",
    ) or "stripe"
    try:
        with st.spinner("Reading Form D filings from SEC EDGAR…"):
            funding = _cached_private_funding(private_key, validate_sec_user_agent(sec_email))
    except ValueError as error:
        st.info(str(error), icon=":material/info:")
    except (RequestException, SECClientError) as error:
        st.error(f"Couldn't load the SEC filings: {error}", icon=":material/error:")
    else:
        offerings = funding["offerings"]
        st.markdown(f"**{funding['company']}** · {funding['about']} · legal name *{funding['legal_name']}*")
        if not offerings:
            st.info("No Form D filings found for this company.")
        else:
            latest = offerings[0]
            with st.container(horizontal=True):
                st.metric("Total reported raised", _usd_short(funding["total_reported_sold"]), border=True)
                st.metric("Offerings", len(offerings), border=True)
                st.metric("Latest offering", _usd_short(latest["amount_sold"]), latest["first_sale"] or latest["filed"], delta_color="off", border=True)
                st.metric("Investors (latest)", latest["investors"] or "—", border=True)
            frame = pd.DataFrame(
                [
                    {
                        "First sale": pd.Timestamp(row["first_sale"] or row["filed"]),
                        "Raised (USD M)": (row["amount_sold"] or 0) / 1e6,
                        "Offered (USD M)": (row["offering_amount"] / 1e6) if row["offering_amount"] is not None else None,
                        "Investors": row["investors"],
                        "Securities": ", ".join(row["securities"]),
                        "Filed": row["filed"],
                        "Filing": row["filing_url"],
                    }
                    for row in offerings
                ]
            )
            chart = (
                alt.Chart(frame)
                .mark_bar(color=SERIES_COLORS[0], size=14, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("First sale:T", title=None, axis=alt.Axis(format="%Y")),
                    y=alt.Y("Raised (USD M):Q", title="Raised (USD millions)"),
                    tooltip=[
                        alt.Tooltip("First sale:T", format="%d %b %Y"),
                        alt.Tooltip("Raised (USD M):Q", format=",.1f"),
                        alt.Tooltip("Investors:Q"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, width="stretch")
            st.dataframe(
                frame,
                hide_index=True,
                width="stretch",
                column_config={
                    "First sale": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Raised (USD M)": st.column_config.NumberColumn(format="%.1f"),
                    "Offered (USD M)": st.column_config.NumberColumn(format="%.1f"),
                    "Filing": st.column_config.LinkColumn(display_text="Open"),
                },
            )
        if funding.get("skipped_filings"):
            st.caption(f"{funding['skipped_filings']} older filing(s) couldn't be read from the SEC and are left out.")
        st.caption(FORM_D_NOTE + " Ask the assistant about these companies too, e.g. *How much has Databricks raised?*")

with shortlist:
    st.subheader("Build a research shortlist")
    st.write(
        "Set the financial conditions you want to screen for. Companies that meet them are "
        "research candidates—not personalized buy recommendations or predictions."
    )
    with st.form("research_screen_form"):
        screen_tickers = st.multiselect(
            "Companies to screen",
            options=TICKERS,
            default=["AAPL", "MSFT"],
            format_func=_company_label,
            max_selections=5,
            help="Screen up to five companies at once so the results stay reviewable.",
        )
        growth_col, margin_col = st.columns(2)
        minimum_growth = growth_col.slider(
            "Minimum latest-year revenue growth (%)",
            min_value=-50,
            max_value=50,
            value=0,
            help="Growth is calculated from consecutive reported fiscal years.",
        )
        minimum_margin = margin_col.slider(
            "Minimum latest-year net margin (%)",
            min_value=-50,
            max_value=50,
            value=5,
            help="Net margin is net income divided by revenue.",
        )
        run_screen = st.form_submit_button(
            "Run transparent screen",
            type="primary",
            icon=":material/filter_alt:",
        )

    if run_screen:
        try:
            user_agent = validate_sec_user_agent(sec_email)
            if not screen_tickers:
                st.warning("Select at least one company to screen.")
            else:
                records_by_ticker: dict[str, dict[str, list[dict[str, Any]]]] = {}
                with st.status("Retrieving reported figures and applying your criteria…", expanded=True) as status:
                    for ticker in screen_tickers:
                        records_by_ticker[ticker] = {
                            metric: _cached_metrics(ticker, metric, years, user_agent)
                            for metric in ("revenue", "net_income")
                        }
                    results = build_research_shortlist(
                        records_by_ticker,
                        float(minimum_growth),
                        float(minimum_margin),
                    )
                    status.update(label="Screen complete", state="complete")
                st.session_state["screen_results"] = results
                st.session_state["screen_criteria"] = {
                    "minimum_growth": minimum_growth,
                    "minimum_margin": minimum_margin,
                }
        except (ValueError, RequestException, SECClientError) as error:
            st.error(str(error), icon=":material/error:")

    results = st.session_state.get("screen_results", [])
    if results:
        criteria = st.session_state.get("screen_criteria", {})
        matches = [
            item for item in results if item["growth_match"] and item["margin_match"]
        ]
        st.caption(
            f"{len(matches)} of {len(results)} companies meet both selected criteria: "
            f"revenue growth ≥ {criteria.get('minimum_growth', 0)}% and "
            f"net margin ≥ {criteria.get('minimum_margin', 0)}%."
        )
        for item in results:
            passed = int(item["growth_match"]) + int(item["margin_match"])
            heading = (
                f"{item['company']} ({item['ticker']})"
                if item["fiscal_year"] is None
                else f"{item['company']} ({item['ticker']}) · FY{item['fiscal_year']}"
            )
            with st.container(border=True):
                st.subheader(heading)
                st.badge(
                    f"Meets {passed} of 2 criteria",
                    icon=":material/check_circle:" if passed == 2 else ":material/info:",
                    color="green" if passed == 2 else "orange",
                )
                metric_col, margin_col = st.columns(2)
                growth_value = item["revenue_growth_percent"]
                margin_value = item["net_margin_percent"]
                metric_col.metric(
                    "Latest revenue growth",
                    f"{growth_value:.1f}%" if growth_value is not None else "Unavailable",
                    border=True,
                )
                margin_col.metric(
                    "Latest net margin",
                    f"{margin_value:.1f}%" if margin_value is not None else "Unavailable",
                    border=True,
                )
                st.write(item["explanation"])
                if item["sources"]:
                    st.markdown(
                        "Sources: "
                        + " · ".join(
                            f"[FY{source['fiscal_year']} SEC filing]({source['url']})"
                            for source in item["sources"]
                        )
                    )
                else:
                    st.caption("No SEC source links were available for the required figures.")
        st.warning(
            "This screen uses only two historical fundamentals. It does not assess valuation, "
            "future performance, risk tolerance, diversification, or your personal circumstances."
        )
    else:
        st.info("Choose companies and thresholds, then run the screen to see sourced explanations.")

def _record_feedback(answer_id: str) -> None:
    score = st.session_state.get(f"feedback-{answer_id}")
    message = next((m for m in st.session_state.get("chat_messages", []) if m.get("answer_id") == answer_id), None)
    if message is None or score is None:
        return
    message["feedback"] = score
    monitor.event(
        "answer_feedback",
        username=user_id,
        answer_id=answer_id,
        score=int(score),
        experiment=message.get("experiment"),
        variant=message.get("variant"),
        strategy=message.get("strategy"),
        provider=(message.get("usage") or {}).get("provider"),
        model=(message.get("usage") or {}).get("model"),
    )


def _render_feedback(message: dict[str, Any]) -> None:
    if not message.get("answer_id"):
        return
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(T("was_helpful"))
        st.feedback(
            "thumbs",
            key=f"feedback-{message['answer_id']}",
            on_change=_record_feedback,
            args=(message["answer_id"],),
        )


def _top_relevance(tool_results: list[dict[str, Any]]) -> float | None:
    scores = [
        result.get("relevance")
        for item in tool_results
        if item["name"] == "search_filing_knowledge" and isinstance(item["content"], dict)
        for result in item["content"].get("results", []) or []
        if result.get("relevance") is not None
    ]
    return max(scores) if scores else None


with assistant:
    st.subheader(T("assistant_heading"))
    st.caption(T("assistant_hint"))
    if ab_variant is not None:
        st.caption(
            f":material/science: Filing search uses **{SEARCH_STRATEGIES[search_strategy]}** for you as part of an A/B test. "
            "Rate answers with 👍/👎 to help decide the winner."
        )
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
        st.session_state["agent_history"] = []

    chat_messages = st.session_state["chat_messages"]
    if chat_messages:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(f"{len(chat_messages)} message(s) in this session's history")
            st.space("stretch")
            file_stamp = datetime.now().strftime("%Y%m%d-%H%M")
            with st.popover(T("export"), icon=":material/download:"):
                st.caption("Download this conversation with its sources.")
                try:
                    pdf_bytes = conversation_to_pdf(chat_messages, model_name.strip())
                    st.download_button(
                        "PDF · printable",
                        data=pdf_bytes,
                        file_name=f"market-lens-chat-{file_stamp}.pdf",
                        mime="application/pdf",
                        icon=":material/picture_as_pdf:",
                        width="stretch",
                    )
                except Exception as error:  # a PDF layout problem should not break the other exports
                    st.warning(f"PDF export failed: {error}", icon=":material/warning:")
                st.download_button(
                    "CSV · spreadsheet",
                    data=conversation_to_csv(chat_messages),
                    file_name=f"market-lens-chat-{file_stamp}.csv",
                    mime="text/csv",
                    icon=":material/table_view:",
                    width="stretch",
                )
                st.download_button(
                    "Markdown · readable text",
                    data=conversation_to_markdown(chat_messages, model_name.strip()),
                    file_name=f"market-lens-chat-{file_stamp}.md",
                    mime="text/markdown",
                    icon=":material/description:",
                    width="stretch",
                )
                st.download_button(
                    "JSON · full record",
                    data=conversation_to_json(chat_messages, model_name.strip()),
                    file_name=f"market-lens-chat-{file_stamp}.json",
                    mime="application/json",
                    icon=":material/data_object:",
                    width="stretch",
                )
            if st.button(T("clear_chat"), icon=":material/delete_sweep:"):
                st.session_state["chat_messages"] = []
                st.session_state["agent_history"] = []
                st.rerun()

    for index, item in enumerate(chat_messages):
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("timestamp"):
                st.caption(item["timestamp"])
            if item["role"] == "assistant":
                _render_assistant_extras(item, key=f"history-{index}")
                _render_feedback(item)

    starter = None
    starters = _starter_questions(selected_tickers)
    if not chat_messages:
        st.markdown(f"**{T('starter_heading')}**")
        rows = [st.columns(2), st.columns(2)]
        for index, (icon, label, text) in enumerate(starters):
            if rows[index // 2][index % 2].button(label, key=f"starter-{index}", icon=icon, width="stretch", help=text):
                starter = text
        st.caption("Questions use the companies selected in the sidebar. The 10-K question needs that company's filings indexed in the Filing library.")
    else:
        with st.popover("Example questions", icon=":material/lightbulb:"):
            for index, (icon, label, text) in enumerate(starters):
                if st.button(label, key=f"starter-more-{index}", icon=icon, type="tertiary", help=text):
                    starter = text

    question = st.chat_input(
        T("chat_placeholder"),
        max_chars=4000,
        submit_mode="disable",
    ) or starter
    if question:
        asked_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state["chat_messages"].append({"role": "user", "content": question, "timestamp": asked_at})
        with st.chat_message("user"):
            st.markdown(question)
            st.caption(asked_at)
        with st.chat_message("assistant"):
            started = time.perf_counter()
            log_context = {
                "username": user_id,
                "provider": provider_key,
                "model": model_name,
                "question_chars": len(question),
                "question_preview": question[:120],
                "language": language,
                "strategy": search_strategy,
                "tool_source": st.session_state["tool_source"],
                "experiment": experiment.name if ab_variant else None,
                "variant": ab_variant,
            }
            try:
                check_user_question(question)  # cheapest check first, before any setup or network call
                user_agent = validate_sec_user_agent(sec_email)
                if not chat_key.strip():
                    raise ValueError(f"Add your {provider.key_label} in the sidebar to use the research assistant.")
                limiter.check(user_id, user_limits)
                _check_shared_allowance(using_shared_key or shared_embeddings)
                if st.session_state["tool_source"] == "mcp_server":
                    with st.spinner("Connecting to the Market Lens MCP server…"):
                        server_url, server_specs = _market_lens_mcp_server(sec_email.strip(), openrouter_key.strip(), market_data_key.strip())
                    # The search strategy is set by the app (A/B test or your choice), not by the model.
                    tools = build_mcp_tools(
                        server_url, server_specs, trusted=True,
                        fixed_arguments={"search_filing_knowledge": {"strategy": search_strategy}},
                    )
                else:
                    vector_store = _cached_vector_store(openrouter_key, str(VECTOR_DIR)) if openrouter_key.strip() else None
                    tools = build_tools(user_agent, vector_store, market_data_key, search_strategy)
                if st.session_state["mcp_enabled"]:
                    mcp_url = st.session_state["mcp_url"]
                    tools += build_mcp_tools(mcp_url, _cached_mcp_tool_specs(mcp_url))
                with st.status(f"Researching with {model_name}…", expanded=False) as status:
                    answer, agent_messages = run_research_agent(
                        question,
                        chat_key,
                        model_name.strip(),
                        tools,
                        history=_chat_history(),
                        provider=provider_key,
                        language=language,
                    )
                    status.update(label="Research complete", state="complete")
                tool_results = _tool_results_from_messages(agent_messages)
                usage = summarize_usage(agent_messages)
                cost = estimate_cost(usage, price_for(provider_key, model_name.strip(), _cached_pricing()))
                usage["cost_usd"] = cost
                usage["provider"] = provider_key
                usage["model"] = model_name
                st.session_state["usage_total"] = add_usage(st.session_state.get("usage_total", {}), usage, cost)
                limiter.record_cost(user_id, cost)
                _record_shared_cost(using_shared_key, cost)
                answer_id = os.urandom(8).hex()
                top_relevance = _top_relevance(tool_results)
                verification = verify_answer(answer, tool_results)
                monitor.event(
                    "chat_answer",
                    **log_context,
                    answer_id=answer_id,
                    top_relevance=top_relevance,
                    latency_s=round(time.perf_counter() - started, 2),
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    total_tokens=usage["total_tokens"],
                    model_calls=usage["model_calls"],
                    cost_usd=cost,
                    tools=[item["name"] for item in tool_results],
                    tool_errors=sum(1 for item in tool_results if isinstance(item["content"], dict) and item["content"].get("error")),
                    figures_checked=verification["checked"],
                    figures_unverified=verification["unverified"],
                )
                reply = {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "tools": tool_results,
                    "citations": extract_citations(tool_results),
                    "verification": verification,
                    "usage": usage,
                    "answer_id": answer_id,
                    "strategy": search_strategy,
                    "experiment": log_context["experiment"],
                    "variant": ab_variant,
                }
                st.markdown(answer)
                st.caption(reply["timestamp"])
                _render_assistant_extras(reply, key=f"new-{len(st.session_state['chat_messages'])}")
                st.session_state["chat_messages"].append(reply)
                _render_feedback(reply)
                st.session_state["agent_history"] = [
                    *st.session_state["agent_history"],
                    HumanMessage(content=question),
                    *[
                        message
                        for message in agent_messages
                        if isinstance(message, AIMessage) and not message.tool_calls
                    ][-1:],
                ]
            except PromptInjectionError as error:
                monitor.event("injection_blocked", level="warning", **log_context, reason=str(error))
                st.warning(str(error), icon=":material/shield:")
                st.session_state["chat_messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Blocked by the prompt-injection guard: {error}",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                )
            except RateLimitExceeded as error:
                monitor.event("rate_limited", level="warning", **log_context, reason=str(error))
                st.warning(str(error), icon=":material/timer:")
                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": str(error), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
                )
            except Exception as error:  # provider, network, SEC and tool errors: log, then show a readable message
                monitor.event(
                    "chat_error",
                    level="error",
                    **log_context,
                    latency_s=round(time.perf_counter() - started, 2),
                    error_type=type(error).__name__,
                    error=str(error)[:500],
                )
                if not isinstance(error, (ValueError, RequestException, SECClientError, APIError, AgentRunError, MCPError)):
                    error = RuntimeError(f"{type(error).__name__}: {error}")
                st.error(str(error), icon=":material/error:")
                st.session_state["chat_messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Unable to complete that request: {error}",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                )

with filings:
    st.subheader("Build the SEC filing knowledge base")
    st.write(
        "Fetch recent 10-K filings from SEC EDGAR, split them into overlapping text chunks, "
        "and index the chunks with ticker, fiscal year, section, and source metadata."
    )
    ingest_col, status_col = st.columns([1, 2], vertical_alignment="bottom")
    with ingest_col:
        ingest_ticker = st.selectbox("Company to index", options=TICKERS, format_func=_company_label, key="ingest_ticker")
        ingest_years = st.slider("10-K filings to load", 1, 10, 3, key="ingest_years")
    with status_col:
        if st.button("Fetch and index filings", type="primary", icon=":material/library_add:"):
            try:
                user_agent = validate_sec_user_agent(sec_email)
                if not openrouter_key.strip():
                    raise ValueError("Add your OpenRouter API key to create document embeddings.")
                _check_shared_allowance(shared_embeddings)
                with st.status("Downloading, chunking, and indexing SEC filings…", expanded=True) as status:
                    vector_store = _cached_vector_store(openrouter_key, str(VECTOR_DIR))
                    count = ingest_recent_filings(ingest_ticker, ingest_years, user_agent, vector_store)
                    status.update(label="Filing index updated", state="complete")
                # ~1,400 characters (~350 tokens) per chunk at text-embedding-3-small's $0.02 per million tokens.
                _record_shared_cost(shared_embeddings, count * 350 * 0.02 / 1_000_000)
                st.success(f"Indexed {count:,} text chunks for {ingest_ticker}.")
                monitor.event("filings_indexed", username=user_id, ticker=ingest_ticker, chunks=count)
            except (ValueError, RequestException, SECClientError, APIError, RateLimitExceeded) as error:
                monitor.event("filings_index_error", level="error", username=user_id, ticker=ingest_ticker, error=str(error))
                st.error(str(error), icon=":material/error:")

    st.subheader("Keep the knowledge base up to date")
    st.write(
        "Companies file a new 10-K each year. This checks SEC EDGAR for filings newer than what's indexed "
        "and adds only the new ones, so the assistant always searches the latest reports. "
        "Run a check now, or turn on automated updates."
    )

    def _run_refresh(source: str) -> list[dict[str, Any]]:
        indexed = sorted({row["ticker"] for row in index_summary(VECTOR_DIR)}) if VECTOR_DIR.exists() else []
        if not indexed:
            return []
        report = refresh_knowledge_base(
            indexed,
            validate_sec_user_agent(sec_email),
            _cached_vector_store(openrouter_key, str(VECTOR_DIR)),
            REFRESH_STATUS,
        )
        monitor.event(
            "kb_refresh",
            username=user_id,
            source=source,
            companies=len(report),
            new_filings=sum(len(item["new_filings"]) for item in report),
            chunks_added=sum(item["chunks_added"] for item in report),
            errors=sum(1 for item in report if item["error"]),
        )
        return report

    # Background refreshes can embed many filings, so they need the user's own key, not the shared one.
    refresh_ready = bool(openrouter_key.strip()) and not shared_embeddings and bool(sec_email.strip())
    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button("Check for new filings now", icon=":material/sync:", disabled=not refresh_ready):
            try:
                with st.status("Checking SEC EDGAR for new 10-K filings…", expanded=False) as status:
                    report = _run_refresh("manual")
                    status.update(label="Check complete", state="complete")
                added = sum(len(item["new_filings"]) for item in report)
                if not report:
                    st.info("Index at least one company first.")
                else:
                    st.success(f"Checked {len(report)} company(ies): {added} new filing(s) added.")
            except (ValueError, RequestException, SECClientError, APIError) as error:
                st.error(str(error), icon=":material/error:")
    if not refresh_ready:
        st.caption("Needs your SEC contact email and OpenRouter key (for embeddings).")

    scheduler = _kb_scheduler()
    with st.container(border=True):
        st.markdown("**:material/schedule: Automated updates**")
        st.caption(
            "Runs inside the app's server on a fixed schedule, even with no browser open. "
            "It stops when the app stops; for updates that survive restarts, use Windows Task Scheduler below."
        )
        sched_status = scheduler.status()
        interval_choice = st.select_slider(
            "Check every", options=[1, 6, 12, 24, 72, 168], value=int(sched_status["interval_hours"] or 24),
            format_func=lambda hours: f"{hours} h" if hours < 24 else f"{hours // 24} day(s)", key="kb_interval",
        )
        with st.container(horizontal=True):
            if st.button(
                "Start automated updates" if not sched_status["enabled"] else "Apply new schedule",
                icon=":material/play_arrow:", type="primary", disabled=not refresh_ready, key="kb_start",
            ):
                scheduler.start(
                    SchedulerConfig(interval_hours=interval_choice, sec_email=sec_email.strip(), openrouter_key=openrouter_key.strip()),
                    run_now=not sched_status["enabled"],
                )
                st.toast("Automated updates are on. The first check runs now.", icon=":material/schedule:")
                st.rerun()
            if sched_status["enabled"] and st.button("Stop", icon=":material/stop:", key="kb_stop"):
                scheduler.stop()
                st.rerun()

        @st.fragment(run_every=30 if sched_status["enabled"] else None)
        def _scheduler_status() -> None:
            current = scheduler.status()
            if not current["enabled"]:
                st.caption(":material/pause_circle: Off.")
            else:
                state = "checking now…" if current["running"] else f"next check {current['next_run']} UTC"
                st.caption(f":material/check_circle: On · every {current['interval_hours']:g} h · {state}")
            last = current["last_run"]
            if last:
                if last.get("ok"):
                    st.caption(
                        f"Last run {last['finished_at']} UTC: {last['companies']} company(ies) checked, "
                        f"{last['new_filings']} new filing(s), {last['chunks_added']} chunks added, {last['errors']} error(s)."
                    )
                else:
                    st.warning(f"Last run failed: {last.get('error')}", icon=":material/error:")

        _scheduler_status()

        with st.expander("Run it from Windows Task Scheduler instead", icon=":material/terminal:"):
            st.write(
                "This runs one update per day even when the app isn't running. Set the two environment variables "
                "for your Windows user first (once), then create the task:"
            )
            st.code(
                'setx SEC_USER_AGENT_EMAIL "you@example.com"\nsetx OPENROUTER_API_KEY "your-key"\n'
                + task_scheduler_command(str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT)),
                language="powershell",
            )
            st.caption("Run one update by hand: `.venv\\Scripts\\python.exe -m finance_research.scheduler`")

    @st.fragment(run_every=60 if scheduler.enabled else None)
    def _knowledge_base_status() -> None:
        status_rows = load_refresh_status(REFRESH_STATUS)
        if status_rows:
            frame = pd.DataFrame(status_rows.values())
            frame["new_filings"] = frame["new_filings"].apply(lambda items: ", ".join(items) or "none")
            st.dataframe(
                frame[["ticker", "latest_filing", "checked_at", "new_filings", "chunks_added", "error"]],
                hide_index=True,
                column_config={
                    "ticker": "Company",
                    "latest_filing": "Latest 10-K on SEC",
                    "checked_at": "Last checked (UTC)",
                    "new_filings": "Added in last check",
                    "chunks_added": "Chunks added",
                    "error": "Problem",
                },
            )
        else:
            st.caption("No update checks yet.")

    _knowledge_base_status()

    st.subheader("What's in the index")
    try:
        summary = index_summary(VECTOR_DIR)
    except Exception as error:  # a corrupt or locked local index should not break the page
        summary = []
        st.warning(f"Couldn't read the local filing index: {error}", icon=":material/warning:")
    if summary:
        summary_frame = pd.DataFrame(summary)
        summary_frame["Fiscal year"] = "FY" + summary_frame["report_year"].astype(str)
        st.bar_chart(
            summary_frame,
            x="ticker",
            y="chunks",
            color="Fiscal year",
            x_label="Company",
            y_label="Indexed text chunks",
            height=280,
        )
        st.caption(
            f"{int(summary_frame['chunks'].sum()):,} chunks across {summary_frame['ticker'].nunique()} "
            "company(ies), stored locally under data/chroma and excluded from Git."
        )

        st.subheader("Try a retrieval")
        st.write(
            "See exactly what the assistant would retrieve for a question, without calling the chat model. "
            "This uses one embedding request."
        )
        indexed_tickers = sorted(summary_frame["ticker"].unique())
        with st.form("retrieval_preview_form"):
            query_col, ticker_col, year_col = st.columns([3, 1, 1])
            preview_query = query_col.text_input("Question", placeholder="What are the main supply chain risks?")
            preview_ticker = ticker_col.selectbox("Company", indexed_tickers, format_func=_company_label)
            year_options = ["All years"] + [
                int(year) for year in sorted(summary_frame["report_year"].unique(), reverse=True)
            ]
            preview_year = year_col.selectbox("Fiscal year", year_options)
            preview_strategy = st.segmented_control(
                "Search method",
                [*SEARCH_STRATEGIES, "compare"],
                format_func=lambda key: "Compare all three" if key == "compare" else SEARCH_STRATEGIES[key],
                default=DEFAULT_STRATEGY,
            )
            run_preview = st.form_submit_button("Show retrieval", icon=":material/account_tree:")
        if run_preview:
            try:
                if not openrouter_key.strip():
                    raise ValueError("Add your OpenRouter API key to embed the question.")
                _check_shared_allowance(shared_embeddings)
                report_year = None if preview_year == "All years" else int(preview_year)
                store = _cached_vector_store(openrouter_key, str(VECTOR_DIR))
                strategies = list(SEARCH_STRATEGIES) if preview_strategy == "compare" else [preview_strategy or DEFAULT_STRATEGY]
                columns = st.columns(len(strategies)) if len(strategies) > 1 else [st.container()]
                for column, strategy in zip(columns, strategies):
                    with column:
                        if len(strategies) > 1:
                            st.markdown(f"##### {SEARCH_STRATEGIES[strategy]}")
                        results = search_filings(store, preview_ticker, preview_query, report_year=report_year, strategy=strategy)
                        if len(strategies) > 1:
                            for item in results:
                                with st.container(border=True):
                                    st.markdown(
                                        f"**#{item['rank']}** · FY{item.get('report_year')} · {_shorten(item.get('section') or 'Narrative', 40)}"
                                    )
                                    st.progress(float(item.get("relevance") or 0), text=f"Relevance {item.get('relevance', 0):.2f}")
                                    st.caption(_shorten(item.get("text", ""), 220))
                        else:
                            _render_retrieval(
                                {
                                    "query": preview_query,
                                    "ticker": preview_ticker,
                                    "report_year": report_year,
                                    "top_k": 5,
                                    "strategy": strategy,
                                    "embedding_model": "openai/text-embedding-3-small",
                                },
                                results,
                                key="preview",
                            )
            except (ValueError, APIError, RateLimitExceeded) as error:
                st.error(str(error), icon=":material/error:")
    else:
        st.info("No filings are indexed yet. Fetch a company's 10-K filings above to build the knowledge base.")

EVAL_DIR = DATA_DIR / "eval"
TESTSET_FILE = EVAL_DIR / "testset.json"
STRATEGY_COLORS = alt.Scale(domain=list(SEARCH_STRATEGIES), range=SERIES_COLORS[:3])
METRIC_LABELS = {
    "hit_rate": "Hit rate@k",
    "recall_at_1": "Recall@1",
    "mrr": "MRR",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "answer_correctness": "Answer correctness",
}


def _metric_bars(metrics: dict[str, dict[str, Any]], names: list[str], key: str) -> None:
    rows = [
        {"strategy": strategy, "metric": METRIC_LABELS[name], "score": values.get(name)}
        for strategy, values in metrics.items()
        for name in names
        if values.get(name) is not None
    ]
    if not rows:
        return
    chart = (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("strategy:N", title=None, sort=list(SEARCH_STRATEGIES), axis=None),
            x=alt.X("score:Q", title="Score (0–1)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("strategy:N", scale=STRATEGY_COLORS, title="Search method",
                            legend=alt.Legend(labelExpr="datum.value == 'hybrid' ? 'Hybrid' : datum.value == 'vector' ? 'Vector' : 'Keyword'")),
            row=alt.Row("metric:N", title=None, sort=[METRIC_LABELS[name] for name in names], header=alt.Header(labelAngle=0, labelAlign="left")),
            tooltip=[alt.Tooltip("strategy:N", title="Method"), alt.Tooltip("metric:N"), alt.Tooltip("score:Q", format=".3f")],
        )
        .properties(height=22 * len(metrics), width="container")
    )
    st.altair_chart(chart, width="stretch", key=key)


def _show_run(run: dict[str, Any], key: str) -> None:
    retrieval = run.get("retrieval") or {}
    generation = run.get("generation") or {}
    if retrieval:
        st.markdown(f"**Retrieval** · does the ground-truth chunk come back in the top {retrieval['k']}?")
        frame = pd.DataFrame(retrieval["metrics"]).T.rename(index=SEARCH_STRATEGIES)
        st.dataframe(
            frame,
            column_config={
                "hit_rate": st.column_config.NumberColumn("Hit rate@k", format="percent"),
                "recall_at_1": st.column_config.NumberColumn("Recall@1", format="percent"),
                "mrr": st.column_config.NumberColumn("MRR", format="%.3f"),
                "avg_latency_ms": st.column_config.NumberColumn("Avg search time (ms)", format="%.0f"),
            },
        )
        _metric_bars(retrieval["metrics"], ["hit_rate", "recall_at_1", "mrr"], key=f"{key}-retrieval")
    if generation:
        st.markdown("**Answer quality** · scored 0–1 by an LLM judge, using the RAGAs metric definitions")
        frame = pd.DataFrame(generation["metrics"]).T.rename(index=SEARCH_STRATEGIES)
        st.dataframe(frame, column_config={name: st.column_config.NumberColumn(METRIC_LABELS[name], format="%.2f") for name in ("faithfulness", "answer_relevancy", "context_precision", "answer_correctness")})
        _metric_bars(generation["metrics"], ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness"], key=f"{key}-generation")
        with st.expander("Per-question judgements", icon=":material/fact_check:"):
            st.dataframe(pd.DataFrame(generation["rows"]), hide_index=True)
    if retrieval:
        with st.expander("Per-question retrieval ranks", icon=":material/format_list_numbered:"):
            details = pd.DataFrame(retrieval["details"])
            st.dataframe(details.pivot_table(index=["id", "question"], columns="strategy", values="rank", aggfunc="first").reset_index(), hide_index=True)
    config = run.get("config") or {}
    if config:
        st.caption(
            f"{config.get('questions', '?')} question(s) · companies {', '.join(config.get('tickers', []))} · "
            f"generator/judge {config.get('model', '?')}"
            + (f" · {run.get('generation', {}).get('tokens', {}).get('input_tokens', 0):,} + "
               f"{run.get('generation', {}).get('tokens', {}).get('output_tokens', 0):,} judge tokens" if generation else "")
        )


with evaluation_tab:
    st.subheader("Evaluate the RAG system")
    st.write(
        "Measures how well filing search finds the right passage and how good the resulting answers are, for each "
        "search method. Use it to choose a default, and alongside the live A/B test in Monitoring."
    )
    with st.expander("How the evaluation works", icon=":material/help:"):
        st.markdown(
            "1. **Test set.** Random passages are sampled from the indexed 10-Ks, and a model writes one question "
            "each passage answers, plus a reference answer. That passage is the ground truth.\n"
            "2. **Retrieval metrics.** Each question is searched with every method: *hit rate@k* (the ground-truth "
            "passage is in the top k), *recall@1* (it's first), and *MRR* (mean of 1 ÷ its rank).\n"
            "3. **Answer metrics (optional, uses the LLM).** For each question, an answer is written from the "
            "retrieved passages, then a judge model scores it using the RAGAs definitions: *faithfulness* (claims "
            "backed by the passages), *answer relevancy*, *context precision* (relevant passages ranked high), and "
            "*answer correctness* (matches the reference)."
        )
    try:
        eval_summary = index_summary(VECTOR_DIR)
    except Exception:
        eval_summary = []
    eval_tickers = sorted({row["ticker"] for row in eval_summary})
    # Evaluation runs many model calls, so it needs the user's own keys, not the app's shared key.
    eval_ready = bool(openrouter_key.strip()) and bool(chat_key.strip()) and bool(eval_tickers) and not (using_shared_key or shared_embeddings)
    if not eval_tickers:
        st.info("Index at least one company's 10-K filings in the Filing library first.")
    elif using_shared_key or shared_embeddings:
        st.info("Evaluation makes many model calls, so it needs your own OpenRouter key in the sidebar.")
    elif not eval_ready:
        st.info(f"Needs an OpenRouter key (embeddings) and your {provider.key_label} (question writing and judging).")

    def _completer(max_tokens: int = 800):
        return lambda prompt: complete(provider_key, chat_key, model_name.strip(), prompt, max_tokens=max_tokens)

    st.markdown("#### 1 · Test set")
    if "eval_testset" not in st.session_state and TESTSET_FILE.exists():
        st.session_state["eval_testset"] = json.loads(TESTSET_FILE.read_text(encoding="utf-8"))
    with st.form("eval_testset_form"):
        set_col, count_col = st.columns([3, 1])
        testset_tickers = set_col.multiselect("Companies", eval_tickers, default=eval_tickers[:2], format_func=_company_label)
        per_company = count_col.number_input("Questions per company", 1, 15, 5)
        build_set = st.form_submit_button(
            "Generate test set", icon=":material/quiz:", disabled=not eval_ready,
            help=f"Uses {model_name} once per question.",
        )
    if build_set:
        try:
            with st.status("Writing questions from sampled filing passages…") as status:
                test_set = generate_test_set(
                    _cached_vector_store(openrouter_key, str(VECTOR_DIR)), testset_tickers, int(per_company), _completer(600)
                )
                status.update(label=f"{len(test_set)} questions ready", state="complete")
            EVAL_DIR.mkdir(parents=True, exist_ok=True)
            TESTSET_FILE.write_text(json.dumps(test_set, indent=2, ensure_ascii=False), encoding="utf-8")
            st.session_state["eval_testset"] = test_set
            monitor.event("eval_testset_built", username=user_id, questions=len(test_set), model=model_name)
        except Exception as error:
            st.error(f"{type(error).__name__}: {error}", icon=":material/error:")
    test_set = st.session_state.get("eval_testset") or []
    if test_set:
        st.caption(f"{len(test_set)} question(s), saved to data/eval/testset.json. Edit or remove rows before running if you like.")
        edited = st.data_editor(
            pd.DataFrame(test_set)[["id", "ticker", "report_year", "question", "reference_answer", "section", "ground_truth_chunk"]],
            hide_index=True,
            num_rows="dynamic",
            disabled=["id", "ticker", "report_year", "section", "ground_truth_chunk"],
            column_config={"ground_truth_chunk": None},
            key="eval_testset_editor",
        )
        test_set = edited.dropna(subset=["question"]).to_dict("records")

    st.markdown("#### 2 · Run")
    with st.form("eval_run_form"):
        strategies_col, k_col = st.columns([3, 1])
        eval_strategies = strategies_col.multiselect(
            "Search methods", list(SEARCH_STRATEGIES), default=list(SEARCH_STRATEGIES), format_func=lambda key: SEARCH_STRATEGIES[key]
        )
        eval_k = k_col.number_input("k (results per search)", 1, 10, 5)
        judge = st.toggle(
            "Also score answers with an LLM judge (RAGAs metrics)",
            help="Two model calls per question per method. Retrieval metrics alone only use embeddings.",
        )
        if test_set:
            calls = 2 * len(test_set) * max(1, len(eval_strategies))
            st.caption(f"Answer scoring would make about {calls} calls to {model_name}.")
        run_eval = st.form_submit_button("Run evaluation", type="primary", icon=":material/science:", disabled=not (eval_ready and test_set))
    if run_eval:
        try:
            store = _cached_vector_store(openrouter_key, str(VECTOR_DIR))
            started = time.perf_counter()
            with st.status("Evaluating retrieval…", expanded=True) as status:
                run: dict[str, Any] = {
                    "config": {
                        "questions": len(test_set),
                        "tickers": sorted({item["ticker"] for item in test_set}),
                        "strategies": eval_strategies,
                        "model": f"{provider_key}:{model_name}",
                    },
                    "retrieval": evaluate_retrieval(store, test_set, eval_strategies, k=int(eval_k)),
                }
                if judge:
                    status.update(label="Generating and judging answers…")
                    progress = st.progress(0.0)
                    run["generation"] = evaluate_generation(
                        store, test_set, eval_strategies, _completer(600), _completer(500), k=min(int(eval_k), 4),
                        on_progress=lambda done, total: progress.progress(done / total, text=f"{done}/{total}"),
                    )
                status.update(label="Evaluation complete", state="complete")
            path = save_run(EVAL_DIR, run)
            monitor.event(
                "eval_run", username=user_id, questions=len(test_set), strategies=eval_strategies, judged=judge,
                duration_s=round(time.perf_counter() - started, 1),
                **{f"hit_rate_{s}": m["hit_rate"] for s, m in run["retrieval"]["metrics"].items()},
            )
            st.success(f"Saved to data/eval/{path.name}.")
        except Exception as error:
            monitor.event("eval_error", level="error", username=user_id, error=str(error))
            st.error(f"{type(error).__name__}: {error}", icon=":material/error:")

    st.markdown("#### 3 · Results")
    runs = list_runs(EVAL_DIR)
    if runs:
        chosen = st.selectbox(
            "Run", range(len(runs)),
            format_func=lambda i: f"{runs[i]['created_at']} · {runs[i].get('config', {}).get('questions', '?')} questions"
            + (" · with answer scoring" if runs[i].get("generation") else ""),
            key="eval_run_choice",
        )
        _show_run(runs[chosen], key=f"eval-{chosen}")
    else:
        st.caption("No evaluation runs yet.")

with monitoring:
    st.subheader("Monitoring")
    if not is_admin:
        st.info(
            "Monitoring shows every user's activity, so only the app's admin can see it. "
            "Your own usage is below.",
            icon=":material/admin_panel_settings:",
        )
        own_limits = limiter.status(user_id, user_limits)
        with st.container(horizontal=True):
            st.metric("This minute", f"{own_limits['last_minute']:.0f} / {own_limits['per_minute']}", border=True)
            st.metric("This hour", f"{own_limits['last_hour']:.0f} / {own_limits['per_hour']}", border=True)
            st.metric("Spent today", f"${own_limits['spent_today']:.4f} / ${own_limits['daily_budget_usd']:.2f}", border=True)
    else:
        st.caption(
            "Every question, answer, error, blocked request and background job is logged to data/logs "
            "(events.jsonl for this dashboard, market_lens.log as plain text). Keys and emails are redacted."
        )
        window_label = st.segmented_control(
            "Time window", ["Last hour", "Last 24 hours", "Last 7 days", "All"], default="Last 24 hours", key="monitor_window"
        )
        window_hours = {"Last hour": 1, "Last 24 hours": 24, "Last 7 days": 168, "All": None}[window_label or "Last 24 hours"]
        events = monitor.read_events(since_hours=window_hours)
        stats = summarize_events(events)
        with st.container(horizontal=True):
            st.metric("Answers", stats["answers"], border=True)
            st.metric("Error rate", f"{stats['error_rate']:.0%}", border=True)
            st.metric("Blocked", stats["blocked"], border=True, help="Prompt-injection blocks and rate-limit hits")
            st.metric("Median latency", f"{stats['median_latency_s']:.1f} s", border=True)
            st.metric("p90 latency", f"{stats['p90_latency_s']:.1f} s", border=True)
            st.metric("Tokens", f"{stats['tokens']:,}", border=True)
            st.metric("Cost", f"${stats['cost_usd']:.4f}", border=True)
            st.metric(
                "Figures verified",
                "—" if stats["figure_check_rate"] is None else f"{stats['figure_check_rate']:.0%}",
                border=True,
                help=f"Share of numbers in answers found in, or calculated from, the tool data. {stats['figures_checked']} checked.",
            )

        if events:
            frame = pd.DataFrame(events)
            frame["time"] = pd.to_datetime(frame["ts"])
            answers = frame[frame["kind"] == "chat_answer"]
            chart_col, tools_col = st.columns(2)
            with chart_col:
                st.markdown("**Response time per answer** (seconds)")
                if not answers.empty:
                    st.line_chart(answers, x="time", y="latency_s", height=240)
                else:
                    st.caption("No answers in this window.")
            with tools_col:
                st.markdown("**Tool calls**")
                if stats["tool_counts"]:
                    tools_frame = pd.DataFrame(
                        {"Tool": [TOOL_LABELS.get(name, name) for name in stats["tool_counts"]], "Calls": list(stats["tool_counts"].values())}
                    )
                    st.bar_chart(tools_frame, x="Tool", y="Calls", horizontal=True, height=240)
                else:
                    st.caption("No tool calls in this window.")
            if not answers.empty and "model" in answers:
                st.markdown("**Tokens and cost by model**")
                by_model = (
                    answers.assign(cost_usd=answers["cost_usd"].fillna(0))
                    .groupby(["provider", "model"], as_index=False)
                    .agg(answers=("kind", "size"), tokens=("total_tokens", "sum"), cost_usd=("cost_usd", "sum"), avg_latency_s=("latency_s", "mean"))
                )
                st.dataframe(
                    by_model,
                    hide_index=True,
                    column_config={
                        "cost_usd": st.column_config.NumberColumn("Cost (USD)", format="$%.5f"),
                        "avg_latency_s": st.column_config.NumberColumn("Avg latency (s)", format="%.1f"),
                    },
                )
            problems = frame[frame["level"].isin(["warning", "error"])]
            st.markdown("**Recent warnings and errors**")
            if problems.empty:
                st.caption("None in this window.")
            else:
                detail_cols = [col for col in ("ts", "kind", "username", "error_type", "error", "reason") if col in problems]
                st.dataframe(problems.sort_values("ts", ascending=False)[detail_cols].head(50), hide_index=True)
            with st.expander("All events", icon=":material/list:"):
                st.dataframe(frame.sort_values("ts", ascending=False).drop(columns=["time"]), hide_index=True)
            st.download_button(
                "Download event log (JSONL)",
                data=monitor.events_path.read_bytes(),
                file_name="market-lens-events.jsonl",
                mime="application/jsonl",
                icon=":material/download:",
            )
        else:
            st.info("No events in this window yet. Ask the research assistant a question to see data here.")

        st.markdown("**Your rate limits**")
        limit_status = limiter.status(user_id, user_limits)
        with st.container(horizontal=True):
            st.metric("This minute", f"{limit_status['last_minute']:.0f} / {limit_status['per_minute']}", border=True)
            st.metric("This hour", f"{limit_status['last_hour']:.0f} / {limit_status['per_hour']}", border=True)
            st.metric("Spent today", f"${limit_status['spent_today']:.4f} / ${limit_status['daily_budget_usd']:.2f}", border=True)

        st.divider()
        st.subheader(":material/science: A/B test: retrieval strategies")
        st.caption(
            "Splits users between two filing-search methods (each user always gets the same one), then compares "
            "thumbs-up rates, retrieval relevance, speed and cost. Applies to every user of this app."
        )
        with st.form("experiment_form"):
            exp_cols = st.columns([1, 1, 1, 1])
            exp_enabled = exp_cols[0].toggle("Running", value=experiment_settings["enabled"])
            exp_a = exp_cols[1].selectbox("Variant A", list(SEARCH_STRATEGIES), index=list(SEARCH_STRATEGIES).index(experiment_settings["variant_a"]), format_func=SEARCH_STRATEGIES.get)
            exp_b = exp_cols[2].selectbox("Variant B", list(SEARCH_STRATEGIES), index=list(SEARCH_STRATEGIES).index(experiment_settings["variant_b"]), format_func=SEARCH_STRATEGIES.get)
            exp_share = exp_cols[3].slider("Users in B", 0.0, 1.0, float(experiment_settings["share_b"]), 0.05, format="%.2f")
            exp_name = st.text_input("Experiment name", experiment_settings["name"], help="Change the name to start a fresh experiment; results are grouped by name.")
            if st.form_submit_button("Save experiment", icon=":material/save:", disabled=is_guest, help=None if not is_guest else "Sign in to change the experiment."):
                try:
                    Experiment(name=exp_name, variant_a=exp_a, variant_b=exp_b, share_b=exp_share)
                    _save_experiment_settings({"enabled": exp_enabled, "name": exp_name.strip(), "variant_a": exp_a, "variant_b": exp_b, "share_b": exp_share})
                    monitor.event("experiment_updated", username=user_id, enabled=exp_enabled, name=exp_name, variant_a=exp_a, variant_b=exp_b, share_b=exp_share)
                    st.rerun()
                except ValueError as error:
                    st.error(str(error), icon=":material/error:")

        all_events = monitor.read_events(since_hours=None, limit=50000)
        summary = summarize_experiment(all_events, experiment.name)
        st.markdown(
            f"**{experiment.name}** · A = {SEARCH_STRATEGIES[experiment.variant_a]}, B = {SEARCH_STRATEGIES[experiment.variant_b]} · "
            + ("running" if experiment_settings["enabled"] else "paused")
            + (f" · you are in **{ab_variant}**" if ab_variant else "")
        )
        variant_cols = st.columns(2)
        for column, label in zip(variant_cols, ("A", "B")):
            stats = summary["variants"][label]
            with column.container(border=True):
                st.markdown(f"**Variant {label}** · {SEARCH_STRATEGIES.get(getattr(experiment, f'variant_{label.lower()}'))}")
                with st.container(horizontal=True):
                    st.metric("Answers", stats["answers"])
                    st.metric("Users", stats["users"])
                    st.metric("👍 rate", "—" if stats["satisfaction"] is None else f"{stats['satisfaction']:.0%}", help=f"{stats['thumbs_up']} of {stats['votes']} votes")
                    st.metric("Avg top relevance", "—" if stats["avg_top_relevance"] is None else f"{stats['avg_top_relevance']:.2f}", help="Only answers that searched filings")
                    st.metric("Median latency", "—" if stats["median_latency_s"] is None else f"{stats['median_latency_s']:.1f} s")
        st.info(verdict(summary), icon=":material/analytics:")
        if summary["test"]:
            st.caption(
                f"Two-proportion z-test on thumbs-up rate: z = {summary['test']['z']:.2f}, p = {summary['test']['p_value']:.3f}. "
                "Relevance scores aren't comparable across methods (each uses its own scale), so judge by feedback and the offline RAG evaluation."
            )


with guide:
    st.subheader("Setup checklist")
    try:
        validate_sec_user_agent(sec_email)
        email_ready = True
    except ValueError:
        email_ready = False
    try:
        index_ready = bool(index_summary(VECTOR_DIR))
    except Exception:
        index_ready = False
    checklist = [
        (email_ready, "SEC contact email", "Needed for all SEC data. Add it in the sidebar."),
        (bool(chat_key.strip()), f"{provider.key_label} ({provider.label})", "Needed for the research assistant with the chosen provider."),
        (bool(openrouter_key.strip()), "OpenRouter API key", "Needed for filing search (embeddings), whichever chat provider you use."),
        (index_ready, "Filings indexed", "Index at least one company in the Filing library to ask about 10-K text."),
        (bool(market_data_key.strip()), "Alpha Vantage key (optional)", "Only needed for stock quotes."),
    ]
    for done, label, detail in checklist:
        icon = ":material/check_circle:" if done else ":material/radio_button_unchecked:"
        st.markdown(f"{icon} **{label}** — {'ready' if done else detail}")

    st.subheader("Ask the guide")
    st.caption("A built-in helper for using Market Lens. It works without any API key.")
    if "help_messages" not in st.session_state:
        st.session_state["help_messages"] = [
            {
                "role": "assistant",
                "content": "Hi! Ask me how to use Market Lens, or pick a question below.",
            }
        ]

    with st.container(horizontal=True):
        suggestion = None
        for index, text in enumerate(HELP_SUGGESTIONS):
            if st.button(text, key=f"help-suggestion-{index}", type="tertiary"):
                suggestion = text

    help_question = st.chat_input("Ask the guide, e.g. “How do I index filings?”", key="help_input") or suggestion
    if help_question:
        reply = answer_help_question(help_question)
        st.session_state["help_messages"] += [
            {"role": "user", "content": help_question},
            {"role": "assistant", "content": f"**{reply['title']}**\n\n{reply['answer']}"},
        ]

    for item in st.session_state["help_messages"]:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

with usage_slot:
    totals = st.session_state.get("usage_total")
    st.markdown("**Session usage**")
    if totals:
        tokens_col, cost_col = st.columns(2)
        tokens_col.metric("Tokens", f"{totals['total_tokens']:,}")
        cost_col.metric("Cost", f"${totals['cost_usd']:.4f}")
        st.caption(
            f"{totals['answers']} answer(s) · {totals['input_tokens']:,} in / {totals['output_tokens']:,} out"
            + (f" · {totals['unpriced_answers']} without a known price" if totals.get("unpriced_answers") else "")
            + ". Prices from OpenRouter's public model list."
        )
    else:
        st.caption("Tokens and cost appear here after your first research question.")

    openrouter_key_value = st.session_state.get("openrouter_key", "").strip()
    if openrouter_key_value:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("**OpenRouter balance**")
            st.space("stretch")
            if st.button("Refresh", icon=":material/refresh:", type="tertiary", help="Check again now", key="refresh_balance"):
                _cached_openrouter_balance.clear()
        balance = _cached_openrouter_balance(openrouter_key_value)
        if balance is None:
            st.caption("Couldn't read the balance for this key.")
        else:
            if balance["account_remaining"] is not None:
                st.metric("Account credit left", f"${balance['account_remaining']:.2f}")
            if balance["key_remaining"] is not None and balance["key_limit"] is not None:
                st.metric("Left on this key", f"${balance['key_remaining']:.2f}")
                st.progress(
                    min(max(balance["key_remaining"] / balance["key_limit"], 0.0), 1.0) if balance["key_limit"] else 0.0,
                    text=f"${balance['key_usage'] or 0:.2f} used of the ${balance['key_limit']:.2f} key limit",
                )
            elif balance["key_usage"] is not None:
                st.caption(f"${balance['key_usage']:.2f} used on this key (no spending limit set on it).")
            if balance["account_remaining"] is None:
                st.caption(
                    "Your total account balance is at [openrouter.ai/settings/credits](https://openrouter.ai/settings/credits)."
                )

st.divider()
st.caption(T("disclaimer"))
