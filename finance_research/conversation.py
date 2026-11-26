"""Chat helpers that do not depend on Streamlit: citations, export, and the help guide."""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Source citations
# ---------------------------------------------------------------------------


def extract_citations(tool_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Collect the sources behind an answer from the tool results the agent used.

    Citations come from the tools' own outputs, not from the model's text, so a
    listed source is always one the app actually retrieved.
    """
    citations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(label: str, url: str, kind: str) -> None:
        key = (label, url)
        if key in seen:
            return
        seen.add(key)
        citations.append({"label": label, "url": url, "kind": kind})

    for result in tool_results:
        content = result.get("content")
        if not isinstance(content, dict):
            continue
        name = result.get("name", "")

        for row in content.get("records", []) or []:
            if isinstance(row, dict) and row.get("source_url"):
                add(
                    f"{row.get('company') or row.get('ticker')} · FY{row.get('fiscal_year')} · "
                    f"{row.get('metric_label') or row.get('metric')} (SEC Company Facts)",
                    str(row["source_url"]),
                    "SEC financial data",
                )

        for url in content.get("sources", []) or []:
            if isinstance(url, str) and url:
                add(
                    f"{content.get('ticker', '')} · FY{content.get('fiscal_year', '')} · "
                    f"inputs for {str(content.get('ratio', 'ratio')).replace('_', ' ')}",
                    url,
                    "SEC financial data",
                )

        for passage in content.get("results", []) or []:
            if isinstance(passage, dict) and passage.get("source_url"):
                add(
                    f"{passage.get('company') or passage.get('ticker')} · FY{passage.get('report_year')} 10-K · "
                    f"{passage.get('section') or 'Filing narrative'}",
                    str(passage["source_url"]),
                    "SEC filing excerpt",
                )

        for event in content.get("events", []) or []:
            if isinstance(event, dict) and event.get("filing_url"):
                add(
                    f"{content.get('company') or content.get('ticker', '')} · 8-K filed {event.get('filed')} · "
                    f"{event.get('headline', 'company event')}",
                    str(event["filing_url"]),
                    "SEC 8-K filing",
                )

        for offering in content.get("offerings", []) or []:
            if isinstance(offering, dict) and offering.get("filing_url"):
                add(
                    f"{content.get('company', '')} · Form {offering.get('form', 'D')} filed {offering.get('filed')}",
                    str(offering["filing_url"]),
                    "SEC Form D filing",
                )

        if name == "fetch_stock_quote" and content.get("source") == "Alpha Vantage":
            add(
                f"{content.get('ticker', '')} quote · {content.get('latest_trading_day', 'latest')}",
                "https://www.alphavantage.co/",
                "Market data",
            )
    return citations


def format_citations_markdown(citations: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{index}. [{item['label']}]({item['url']}) — {item['kind']}"
        for index, item in enumerate(citations, start=1)
    )


# ---------------------------------------------------------------------------
# Conversation export
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def conversation_to_markdown(messages: list[dict[str, Any]], model_name: str = "") -> str:
    lines = [
        "# Market Lens research conversation",
        "",
        f"Exported: {now_iso()}" + (f" · Model: {model_name}" if model_name else ""),
        "",
        "> Educational research based on SEC filings. Not personalized investment advice.",
        "",
    ]
    for message in messages:
        speaker = "You" if message.get("role") == "user" else "Assistant"
        stamp = f" · {message['timestamp']}" if message.get("timestamp") else ""
        lines += [f"## {speaker}{stamp}", "", str(message.get("content", "")).strip(), ""]
        citations = message.get("citations") or []
        if citations:
            lines += ["**Sources**", "", format_citations_markdown(citations), ""]
        tools_used = [tool["name"] for tool in message.get("tools") or [] if tool.get("name")]
        if tools_used:
            lines += [f"_Tools used: {', '.join(dict.fromkeys(tools_used))}_", ""]
    return "\n".join(lines).rstrip() + "\n"


def conversation_to_json(messages: list[dict[str, Any]], model_name: str = "") -> str:
    return json.dumps(
        {
            "app": "Market Lens",
            "exported_at": now_iso(),
            "model": model_name,
            "messages": messages,
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )


CSV_COLUMNS = [
    "timestamp",
    "role",
    "content",
    "tools_used",
    "sources",
    "input_tokens",
    "output_tokens",
    "cost_usd",
]


def conversation_to_csv(messages: list[dict[str, Any]]) -> str:
    """One row per message, suitable for a spreadsheet. UTF-8 with BOM so Excel shows symbols correctly."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for message in messages:
        usage = message.get("usage") or {}
        cost = usage.get("cost_usd")
        writer.writerow(
            {
                "timestamp": message.get("timestamp", ""),
                "role": message.get("role", ""),
                "content": str(message.get("content", "")).strip(),
                "tools_used": "; ".join(
                    dict.fromkeys(tool["name"] for tool in message.get("tools") or [] if tool.get("name"))
                ),
                "sources": "; ".join(item["url"] for item in message.get("citations") or []),
                "input_tokens": usage.get("input_tokens", ""),
                "output_tokens": usage.get("output_tokens", ""),
                "cost_usd": "" if cost is None else f"{cost:.6f}",
            }
        )
    return "﻿" + buffer.getvalue()


# Core PDF fonts only cover Latin-1, so map common typographic characters first.
_PDF_REPLACEMENTS = {
    "·": "-", "—": "-", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...",
    "×": "x", "≥": ">=", "≤": "<=", "→": "->", "•": "-", " ": " ",
}


def _pdf_text(text: str) -> str:
    for source, target in _PDF_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", text)  # [label](url) -> label (url)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)  # headings become plain lines
    text = text.replace("`", "")
    return text.encode("latin-1", "replace").decode("latin-1")


def conversation_to_pdf(messages: list[dict[str, Any]], model_name: str = "") -> bytes:
    from fpdf import FPDF  # imported lazily so the rest of the app works without fpdf2

    for message in messages:
        text = str(message.get("content", ""))
        for source, target in _PDF_REPLACEMENTS.items():
            text = text.replace(source, target)
        if any(ord(char) > 255 for char in text):
            raise ValueError(
                "PDF export supports Latin-script languages only. Use Markdown or JSON for this conversation."
            )

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(width, 9, "Market Lens research conversation", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(90, 90, 90)
    meta = f"Exported: {now_iso()}" + (f" - Model: {model_name}" if model_name else "")
    pdf.multi_cell(width, 5, _pdf_text(meta), new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(
        width,
        5,
        "Educational research based on SEC filings. Not personalized investment advice.",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(4)

    for message in messages:
        is_user = message.get("role") == "user"
        speaker = "You" if is_user else "Assistant"
        stamp = f" - {message['timestamp']}" if message.get("timestamp") else ""
        pdf.set_text_color(25, 70, 160) if is_user else pdf.set_text_color(10, 120, 80)
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(width, 6, _pdf_text(speaker + stamp), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(20, 20, 20)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(
            width, 5, _pdf_text(str(message.get("content", "")).strip()), markdown=True, new_x="LMARGIN", new_y="NEXT"
        )

        citations = message.get("citations") or []
        if citations:
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 9)
            pdf.multi_cell(width, 5, "Sources", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)
            for index, item in enumerate(citations, start=1):
                pdf.multi_cell(
                    width,
                    4,
                    _pdf_text(f"{index}. {item['label']} - {item['kind']}\n   {item['url']}"),
                    new_x="LMARGIN",
                    new_y="NEXT",
                )

        usage = message.get("usage") or {}
        tools_used = list(dict.fromkeys(tool["name"] for tool in message.get("tools") or [] if tool.get("name")))
        footer = []
        if tools_used:
            footer.append("Tools: " + ", ".join(tools_used))
        if usage:
            cost = usage.get("cost_usd")
            footer.append(
                f"Tokens: {usage.get('input_tokens', 0):,} in / {usage.get('output_tokens', 0):,} out"
                + ("" if cost is None else f" - Cost: ${cost:.5f}")
            )
        if footer:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(90, 90, 90)
            pdf.multi_cell(width, 4, _pdf_text(" | ".join(footer)), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Help guide (rule-based, works without any API key)
# ---------------------------------------------------------------------------

HELP_TOPICS: list[dict[str, Any]] = [
    {
        "title": "Getting started",
        "keywords": ["start", "started", "begin", "first", "how do i use", "setup", "set up", "what can", "help"],
        "answer": (
            "1. Enter your **SEC contact email** in the sidebar. The SEC requires it for data requests.\n"
            "2. On **Company overview**, pick companies (or click *Explore companies* on a sector card) "
            "and press **Load SEC data**.\n"
            "3. Add an **OpenRouter API key** to use the research assistant and filing search.\n"
            "4. On **Filing library**, index a company's 10-K filings, then ask the assistant about them."
        ),
    },
    {
        "title": "OpenRouter API key",
        "keywords": ["openrouter", "api key", "key", "chat not working", "embedding", "llm", "model"],
        "answer": (
            "The OpenRouter key powers **filing search** (turning 10-K text into searchable embeddings) and, "
            "when OpenRouter is the chosen provider, the **research assistant** chat. You can chat through "
            "OpenAI or Anthropic directly instead. Charts, the shortlist and sector buttons work without any "
            "key. Create a key at openrouter.ai, add a little credit, and paste it into the sidebar."
        ),
    },
    {
        "title": "SEC contact email",
        "keywords": ["email", "sec", "user agent", "user-agent", "contact", "403", "rate"],
        "answer": (
            "The SEC asks every automated client to identify itself with a contact email. The app sends "
            "it in the request header only. Without it, SEC data and filings won't load."
        ),
    },
    {
        "title": "Indexing filings",
        "keywords": ["index", "ingest", "10-k", "filing", "filings", "library", "no indexed", "rag"],
        "answer": (
            "Open **Filing library**, choose a company and how many years of 10-Ks to load, then press "
            "**Fetch and index filings**. The app downloads each filing, splits it into overlapping chunks, "
            "tags them with ticker, year and section, and stores them locally in `data/chroma`. "
            "Only indexed companies can be searched by the assistant."
        ),
    },
    {
        "title": "How retrieval works",
        "keywords": ["retrieval", "visual", "chunk", "similarity", "relevance", "vector", "how does search"],
        "answer": (
            "When you ask about filing text, your question is turned into an embedding, matched against "
            "the indexed chunks for that company (and year, if given), and the closest passages are passed "
            "to the model. Open **How filing search found this** under an answer, or use **Try a retrieval** "
            "in the Filing library, to see each step and the relevance score of every chunk."
        ),
    },
    {
        "title": "Sources and citations",
        "keywords": ["source", "sources", "citation", "cite", "trust", "verify", "where"],
        "answer": (
            "Every assistant answer lists the sources its tools actually retrieved: SEC Company Facts "
            "records, 10-K excerpts (with year and section), and market data. Sources are taken from the "
            "tool results, not from the model's wording, so check the numbers against those links."
        ),
    },
    {
        "title": "Saving or exporting a chat",
        "keywords": ["export", "download", "save", "history", "clear", "markdown", "json", "conversation", "pdf", "csv"],
        "answer": (
            "Your conversation stays for this browser session. Use the **Export** menu above the chat: "
            "**PDF** for a printable copy with sources, **CSV** for a spreadsheet (one row per message, with "
            "tokens and cost), **Markdown** for readable text, or **JSON** for the full record including tool "
            "output. **Clear conversation** starts fresh."
        ),
    },
    {
        "title": "Choosing a model provider",
        "keywords": ["provider", "anthropic", "claude", "openai", "gpt", "gemini", "switch model", "multi-model", "which model"],
        "answer": (
            "Pick a **Provider** and **Model** at the top of the sidebar. **OpenRouter** gives one key for many "
            "models (you can type any model id). **OpenAI (direct)** uses an OpenAI key, and **Anthropic Claude "
            "(direct)** uses an Anthropic key through the official Anthropic SDK. Filing search still needs an "
            "OpenRouter key for embeddings, whichever chat provider you choose. Each answer shows which model wrote it."
        ),
    },
    {
        "title": "Accounts and saved preferences",
        "keywords": ["account", "login", "log in", "sign in", "sign up", "password", "profile", "preferences", "my preferences", "my settings", "guest", "watchlist", "defaults"],
        "answer": (
            "Create an account on the sign-in screen, or continue as a guest. When signed in, press **Save these as "
            "my defaults** in the sidebar to remember your companies, metric, history length, provider, model and "
            "MCP setting; they load every time you sign in. Passwords are stored as salted hashes in data/users.json; "
            "API keys are never saved."
        ),
    },
    {
        "title": "Remote MCP tools",
        "keywords": ["mcp", "remote tools", "deepwiki", "server", "model context protocol"],
        "answer": (
            "Open **Remote MCP tools** in the sidebar and turn on *Let the assistant use MCP tools*. The default "
            "server is DeepWiki (public, no sign-in), which answers questions about GitHub repositories; you can "
            "enter any public https MCP server that speaks streamable HTTP. MCP results are treated as untrusted "
            "text and pass through the prompt-injection filter."
        ),
    },
    {
        "title": "Rate limits and API keys",
        "keywords": ["limit", "limits", "rate", "budget", "too many", "slow down", "test key", "forget", "keys"],
        "answer": (
            "Open **API keys & limits** in the sidebar. **Test keys** checks each key with a free request and shows "
            "only a masked version. **Forget keys** clears them from this session. You can also set how many "
            "questions per minute and hour are allowed and a daily spending limit; when one is reached the "
            "assistant pauses and tells you when you can continue."
        ),
    },
    {
        "title": "Monitoring and logs",
        "keywords": ["monitor", "monitoring", "log", "logs", "the logs", "log file", "logging", "latency", "errors", "dashboard", "events"],
        "answer": (
            "The **Monitoring** tab shows answers, error rate, response times, tokens, cost, tool usage and recent "
            "errors for the last hour, day, week or all time, plus your rate-limit status. Logs are written to "
            "data/logs (events.jsonl and market_lens.log) with keys and emails redacted."
        ),
    },
    {
        "title": "Keeping filings up to date",
        "keywords": ["update", "updates", "new filings", "latest", "real-time", "realtime", "refresh", "fresh", "stale"],
        "answer": (
            "In **Filing library**, press **Check for new filings now** to look for 10-Ks newer than what's "
            "indexed and add only those, or turn on the 30-minute auto-check while the page is open. On "
            "**Company overview**, **Refresh from SEC now** skips the one-hour cache and loads the latest figures."
        ),
    },
    {
        "title": "Hybrid search",
        "keywords": ["hybrid", "bm25", "keyword search", "vector search", "search method", "fusion", "rrf", "semantic"],
        "answer": (
            "Filing search can use **Vector** (meaning), **Keyword** (BM25 exact-word matching) or **Hybrid**, the "
            "default, which runs both and merges them with Reciprocal Rank Fusion. Hybrid catches exact terms like "
            "product names and numbers that vector search can miss. Choose under **Retrieval & tools** in the "
            "sidebar, or compare all three side by side in **Filing library → Try a retrieval**."
        ),
    },
    {
        "title": "A/B testing",
        "keywords": ["a/b", "ab test", "experiment", "variant", "thumbs", "feedback", "split test"],
        "answer": (
            "In **Monitoring**, start an A/B test between two search methods. Each user is assigned to one variant "
            "for good. Rate answers with 👍/👎; the panel compares thumbs-up rates with a significance test, plus "
            "relevance, speed and cost, and says when there's enough data to pick a winner."
        ),
    },
    {
        "title": "Automated updates",
        "keywords": ["automated", "automatic", "schedule", "scheduler", "task scheduler", "daily", "background"],
        "answer": (
            "In **Filing library → Automated updates**, pick an interval and press **Start automated updates**. "
            "The app's server then checks SEC EDGAR for new 10-Ks on that schedule, even with no browser open. "
            "To keep updating after the app stops, use the Windows Task Scheduler command shown there."
        ),
    },
    {
        "title": "Languages",
        "keywords": ["language", "languages", "spanish", "french", "german", "portuguese", "hindi", "translate", "idioma"],
        "answer": (
            "Pick a **Language** at the top of the sidebar: English, Español, Français, Deutsch, Português or हिन्दी. "
            "The main interface changes and the assistant answers in that language. Filings stay in English, so the "
            "assistant still searches them in English. PDF export supports Latin-script languages; use Markdown for Hindi."
        ),
    },
    {
        "title": "Analytics dashboard",
        "keywords": ["analytics", "cagr", "margins", "heatmap", "ranking", "peer", "kpi", "indexed"],
        "answer": (
            "The **Analytics** tab compares up to five companies: revenue, CAGR, growth, margins, R&D intensity and "
            "return on assets; indexed revenue growth; margin trends; a year-over-year growth heatmap; growth vs "
            "profitability; a 10-year share-price chart; a factor scorecard; and a peer ranking."
        ),
    },
    {
        "title": "Share price history",
        "keywords": ["share price", "stock chart", "10 years", "price history", "stock grown", "growth of $1,000", "drawdown"],
        "answer": (
            "In **Analytics**, press **Run analysis** to see each company's last 10 years of monthly share prices. "
            "**Growth of $1,000** shows what $1,000 invested would be worth with dividends reinvested; **Share price** "
            "shows the traded price. The table lists total return, return per year, and the worst fall from a high. "
            "Prices come from Alpha Vantage when you add a key, otherwise from Yahoo Finance's public chart service. "
            "Past performance does not predict future returns."
        ),
    },
    {
        "title": "Company events (8-K)",
        "keywords": ["events", "8-k", "news", "announced", "announcement", "earnings release", "ceo change", "latest events"],
        "answer": (
            "Open **Company events**, pick up to five companies and a time range, and press **Load events**. You'll "
            "see each company's SEC Form 8-K filings: earnings, executive changes, deals, financing and more, on a "
            "timeline and as a list. **Summarise with AI** reads a filing (and its press release) and writes a short "
            "summary; it needs your AI key. The chat assistant can look up events too: ask \"What has Apple announced lately?\""
        ),
    },
    {
        "title": "Example questions",
        "keywords": ["example question", "what can i ask", "where to start", "starter", "ideas", "sample question"],
        "answer": (
            "In **Research assistant**, an empty chat shows four example questions built from the companies selected in "
            "the sidebar. Click one to ask it. Once you've started chatting, they're under **Example questions**."
        ),
    },
    {
        "title": "Figure check",
        "keywords": ["figure check", "verified", "unverified", "fact check", "wrong number", "hallucination", "check numbers"],
        "answer": (
            "Under each answer, the **figure check** looks up every number in the data the tools returned, allowing "
            "for rounding. ✓ means it was found or can be calculated from it (like a growth rate); ⚠ means it couldn't "
            "be matched, so check it in the sources. It can't tell whether a correct number is attached to the right "
            "company or year."
        ),
    },
    {
        "title": "Financial health and P/E",
        "keywords": ["debt", "p/e", "pe ratio", "valuation", "net cash", "leverage", "financial health", "eps", "liabilities"],
        "answer": (
            "**Analytics** shows each company's debt-to-equity, net cash (cash minus long-term debt) and liabilities ÷ "
            "assets, and the share-price table adds diluted EPS and the P/E ratio (latest price ÷ latest annual EPS). "
            "The scorecard's **Financial health** factor uses liabilities ÷ assets; it's skipped for banks, whose "
            "deposits count as liabilities."
        ),
    },
    {
        "title": "Factor scorecard",
        "keywords": ["scorecard", "factor", "score", "weights", "recommend", "best company", "rating"],
        "answer": (
            "The **Factor scorecard** in Analytics scores companies 0–100 on growth, profitability, efficiency, cash "
            "quality, financial health, consistency, innovation and shareholder return, each on a fixed published scale. Use **Adjust "
            "factor weights** to change what counts most. Missing data is skipped, not scored as zero. It summarises "
            "reported figures and is not a buy or sell recommendation."
        ),
    },
    {
        "title": "Market Lens MCP server",
        "keywords": ["mcp server", "own mcp", "claude desktop", "expose tools", "tool source", "stdio"],
        "answer": (
            "All six research tools are also an MCP server. In the app, choose **Market Lens MCP server** under "
            "**Retrieval & tools** to run the assistant's tools through it. Other MCP clients can use it with "
            "`python -m finance_research.mcp_server` (stdio) or `--http --port 8765`."
        ),
    },
    {
        "title": "RAG evaluation",
        "keywords": ["evaluation", "evaluate", "ragas", "faithfulness", "hit rate", "mrr", "test set", "benchmark"],
        "answer": (
            "The **RAG evaluation** tab builds a test set from your indexed filings, measures hit rate, recall@1 and "
            "MRR for each search method, and can score answers with an LLM judge on faithfulness, answer relevancy, "
            "context precision and answer correctness (the RAGAs metrics). Runs are saved in data/eval."
        ),
    },
    {
        "title": "Token usage and cost",
        "keywords": ["token", "tokens", "cost", "costs", "price", "spend", "spent", "usage", "how much"],
        "answer": (
            "Under every assistant answer you'll see how many input and output tokens it used and what it "
            "cost. The sidebar's **Session usage** adds up all answers in this session. Costs use the "
            "per-token prices in OpenRouter's public model list for the model you chose."
        ),
    },
    {
        "title": "Tool result charts",
        "keywords": ["chart", "charts", "graph", "visualise", "visualize", "visualisation", "tool", "tools", "timeline"],
        "answer": (
            "Open **Tool results, visualised** under an answer. It shows a timeline of every tool the "
            "assistant called (what it asked for and what came back), charts of the SEC figures it fetched, "
            "the inputs behind each calculated ratio, and any stock quotes."
        ),
    },
    {
        "title": "Prompt-injection protection",
        "keywords": ["injection", "safety", "secure", "security", "blocked", "jailbreak", "shield", "guard"],
        "answer": (
            "Market Lens guards against attempts to hijack the assistant in three ways: questions that try "
            "to override its rules are blocked; instruction-like text inside retrieved 10-K excerpts is "
            "removed and the rest is marked as untrusted data; and an answer that repeats the assistant's "
            "internal instructions is withheld. These are pattern checks, so a cleverly worded attack can still get "
            "through; the bigger safeguard is that all tools are read-only, so the model can't change anything."
        ),
    },
    {
        "title": "Research shortlist",
        "keywords": ["shortlist", "screen", "threshold", "growth", "margin", "filter"],
        "answer": (
            "The **Research shortlist** checks up to five companies against two historical conditions: "
            "latest revenue growth and net margin. Each result explains the calculation and links to the "
            "SEC filings used. It is a research aid, not a recommendation."
        ),
    },
    {
        "title": "Explore companies buttons",
        "keywords": ["explore", "sector", "card", "button", "group", "compare companies"],
        "answer": (
            "Click **Explore companies** on a sector card: the app selects that sector's companies, opens the "
            "**Company overview** tab and loads their SEC data (add your SEC contact email in the sidebar first). "
            "The chosen card is highlighted with a ✓."
        ),
    },
    {
        "title": "Stock quotes",
        "keywords": ["price", "quote", "stock price", "alpha vantage", "market data"],
        "answer": (
            "Stock quotes come from Alpha Vantage, not the SEC. Add a free Alpha Vantage key in the "
            "sidebar, then ask the assistant for a quote. Free keys are rate-limited."
        ),
    },
    {
        "title": "Investment advice",
        "keywords": ["buy", "sell", "invest", "advice", "should i", "recommend", "portfolio"],
        "answer": (
            "Market Lens explains reported financials and filing evidence. It does not give personalized "
            "buy/sell recommendations or predict prices."
        ),
    },
]

HELP_SUGGESTIONS = [
    "How do I get started?",
    "What is a 10-K?",
    "What does net margin mean?",
    "Why do I need an OpenRouter key?",
    "How do I index filings?",
    "How does retrieval work?",
    "How do I export my chat?",
    "How much does each answer cost?",
    "How is the app protected from prompt injection?",
    "How do I use Claude or OpenAI?",
    "How do I save my preferences?",
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9-]+", text.lower()))


def answer_help_question(question: str) -> dict[str, str]:
    """Return the best-matching help topic, or a menu of topics when nothing matches."""
    text = question.lower().strip()
    words = _tokens(text)
    best_topic: dict[str, Any] | None = None
    best_score = 0
    for topic in HELP_TOPICS:
        score = 0
        for keyword in topic["keywords"]:
            if " " in keyword:
                score += 3 if keyword in text else 0
            elif keyword in words:
                score += 2
        if score > best_score:
            best_topic, best_score = topic, score

    from finance_research.glossary import is_definition_question, lookup_term

    term = lookup_term(question)
    if term and (best_topic is None or is_definition_question(question)):
        return {"title": term["term"], "answer": f"{term['meaning']}\n\n**Why it matters:** {term['why']}"}

    if best_topic is None:
        menu = "\n".join(f"- {topic['title']}" for topic in HELP_TOPICS)
        return {
            "title": "Topics I can help with",
            "answer": "I'm not sure about that one. Ask what a term means (e.g. “What is a 10-K?”) or about one of these:\n\n"
            + menu,
        }
    return {"title": best_topic["title"], "answer": best_topic["answer"]}
