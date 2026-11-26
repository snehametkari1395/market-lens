# Market Lens

Market Lens is a Streamlit research assistant for comparing public companies using SEC filings and reported financial data. It explains evidence and calculations; it does not make personalized investment recommendations.

## Features in the starter build

- SEC company-facts retrieval for annual revenue, R&D expense, operating income, net income, assets, and operating cash flow
- Side-by-side company selection and a financial trend chart
- A transparent research shortlist using user-selected revenue-growth and net-margin thresholds, with per-company explanations and SEC filing links
- SEC 10-K download, chunking, metadata tagging, and Chroma vector search
- OpenRouter chat with LangChain tool calling
- Tools for fetching SEC data, calculating ratios, comparing companies, searching filings, listing 8-K company events, and requesting a stock quote
- Source URLs, fiscal periods, units, validation, and explicit errors for unavailable information
- Numbered source citations under every assistant answer, built from the data the tools actually retrieved
- Figure check: every number in an answer is matched against the tool data (allowing for rounding) or recognised as a growth rate or ratio calculated from it; unmatched figures are flagged, and the Monitoring tab tracks the share verified
- Conversation history with timestamps, Markdown/JSON export, and a clear-conversation button
- RAG visualization: step-by-step retrieval view with per-chunk relevance scores, an index overview chart, and a "Try a retrieval" panel in the Filing library
- A Help & guide tab with a live setup checklist and a built-in guide chatbot that works without an API key
- Token usage and cost under every answer, plus session totals in the sidebar (priced from OpenRouter's public model list)
- "Tool results, visualised": a timeline of each tool call plus charts of fetched SEC figures, ratio inputs, and quotes
- Conversation export as PDF, CSV, Markdown, or JSON
- Prompt-injection protection: blocks rule-override questions, redacts instruction-like text in retrieved filing excerpts and MCP results and marks them as untrusted data, and withholds answers that leak the system prompt
- Multi-model support: OpenRouter (any model), OpenAI direct, or Anthropic Claude direct via the official Anthropic SDK (with server-side refusal fallbacks enabled)
- Real-time knowledge-base updates: check SEC EDGAR for new 10-Ks and index only the new ones, manually or every 30 minutes; "Refresh from SEC now" bypasses the one-hour data cache
- User accounts (salted PBKDF2 password hashes in `data/users.json`) with saved preferences: watchlist, metric, history length, provider, model, and MCP setting; guest mode available
- Remote MCP tools: connect the assistant to any public streamable-HTTP MCP server (default: DeepWiki)
- Rate limiting (questions per minute/hour and a daily spending limit, per user; non-admins can only lower the defaults) and API key management (masked display, key testing, forget keys)
- Logging and monitoring: redacted JSON-lines event log and rotating text log in `data/logs/`, plus a Monitoring tab with answers, error rate, latency, tokens, cost, tool usage, and recent errors
- Hybrid search: BM25 keyword search and vector search merged with Reciprocal Rank Fusion (default), with vector-only and keyword-only also available and a side-by-side comparison in the Filing library
- A/B testing of retrieval strategies: deterministic per-user assignment, thumbs-up/down feedback on answers, and a per-variant comparison with a two-proportion z-test (Monitoring tab)
- Automated knowledge-base updates: a background scheduler in the app server, plus `python -m finance_research.scheduler` for Windows Task Scheduler or cron
- Multi-language support: English, Spanish, French, German, Portuguese and Hindi for the main interface, with answers in the chosen language (filings are still searched in English)
- Advanced analytics dashboard: KPIs, revenue CAGR, indexed growth, margin trends, a YoY growth heatmap, growth vs profitability, and peer ranking for up to five companies
- Financial health and valuation: liabilities ÷ assets, debt-to-equity, net cash, diluted EPS and price-to-earnings (P/E)
- 10-year share-price history: growth of $1,000 (dividend-adjusted) or traded price, with total and annual return and worst drawdown; Alpha Vantage when a key is set, otherwise Yahoo Finance's public chart service
- Company events: recent SEC Form 8-K filings (earnings, executive changes, deals, financing, cybersecurity incidents) on a timeline and list, with optional AI summaries of each filing and its press release; also available to the chat assistant as a tool
- One-click example questions in the chat, built from the companies selected in the sidebar
- Factor scorecard: 0–100 scores on eight factors (growth, profitability, efficiency, cash quality, financial health, consistency, innovation, shareholder return) using fixed, published scales and adjustable weights; a transparent summary, not a recommendation
- The research tools as an MCP server (`finance_research/mcp_server.py`), usable by any MCP client and selectable as the assistant's tool source in the app
- RAG evaluation: synthetic test sets from indexed filings, retrieval metrics (hit rate@k, recall@1, MRR) per strategy, and RAGAs-style LLM-judged faithfulness, answer relevancy, context precision and answer correctness

## End-to-end smoke test (real API calls)

The automated tests use stand-in AI responses. To check the real thing before a demo, run:

```powershell
$env:OPENROUTER_API_KEY = "your-key"            # this terminal window only
$env:SEC_USER_AGENT_EMAIL = "you@example.com"
.\.venv\Scripts\python.exe scripts\smoke_test.py      # options: --ticker MSFT --model openai/gpt-4o-mini --skip-index --scratch
```

It checks the key, indexes the company's latest 10-K (filling the knowledge base), tries all three search methods, asks the assistant three questions (SEC figures, a 10-K question, recent 8-K events) with the figure check, and summarises an 8-K. It prints PASS/WARN/FAIL per step and the total cost (a few cents with a small model), and saves a report without the key to `data/smoke/last_report.json`.

## Run the tools as an MCP server

```powershell
$env:SEC_USER_AGENT_EMAIL = "you@example.com"
$env:OPENROUTER_API_KEY = "your-key"            # for filing search
.\.venv\Scripts\python.exe -m finance_research.mcp_server              # stdio, e.g. for Claude Desktop
.\.venv\Scripts\python.exe -m finance_research.mcp_server --http       # http://127.0.0.1:8765/mcp
```

## Automated knowledge-base updates without the app

```powershell
.\.venv\Scripts\python.exe -m finance_research.scheduler                # one update pass, then exit
```

The Filing library shows a ready-made `schtasks` command that runs this daily.

Companies are grouped by sector in the `SECTORS` list in `streamlit_app.py`; add a sector or company there with any ticker in the SEC company directory. SEC coverage and metric availability vary by issuer. Stock quotes require an Alpha Vantage API key and are not supplied by the SEC.

## Setup on Windows

From the project directory in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

In VS Code, choose `.venv\Scripts\python.exe` with **Python: Select Interpreter**. If Windows reports that `python3` cannot be found, avoid the Microsoft Store `python3` alias; use the activated environment's `python` command or the explicit path:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Credentials and data access

**Admin.** The first account created on this computer is the admin. To choose admins yourself, set `MARKET_LENS_ADMINS` to a comma-separated list of usernames before starting the app. Only admins:

- get the API keys saved on the server (environment variables or `.streamlit/secrets.toml`) pre-filled. Everyone else, including guests, enters their own key, so visitors can't spend the owner's credit;
- can raise rate limits above the defaults (5 questions/minute, 40/hour, $1/day);
- can see the Monitoring tab, which shows every user's activity.


- Enter an OpenRouter API key in the sidebar for chat and embeddings.
- Enter a contact email in the SEC contact field. The app uses it in the SEC-required identifying User-Agent.
- An Alpha Vantage key is optional. It enables the stock quote tool and is used first for share-price history.
- Keys entered in the sidebar are kept only in the current Streamlit session. For local development, they can instead be set as environment variables before launching Streamlit:

```powershell
$env:OPENROUTER_API_KEY = "your-key"
$env:SEC_USER_AGENT_EMAIL = "you@example.com"
$env:ALPHA_VANTAGE_API_KEY = "your-key"
$env:OPENAI_API_KEY = "your-key"        # optional, for OpenAI direct
$env:ANTHROPIC_API_KEY = "your-key"     # optional, for Anthropic Claude direct
```

Filing search always uses OpenRouter embeddings, so an OpenRouter key is still needed for the Filing library and filing questions even when chatting through OpenAI or Anthropic.

Vector data is stored locally in `data/chroma/` and is excluded from Git. SEC endpoints may rate-limit requests; use the application with an identifying User-Agent and avoid unnecessary repeated downloads.

## Run and test

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
.\.venv\Scripts\python.exe -m pytest
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository. `.gitignore` already keeps out `.venv/`, `data/chroma/`, `data/users.json`, logs and `.streamlit/secrets.toml`.
2. On [share.streamlit.io](https://share.streamlit.io), choose **Create app → Deploy a public app from GitHub**, pick the repository and branch, set **Main file path** to `streamlit_app.py`, and under **Advanced settings** choose Python 3.13.
3. Paste these secrets in **Advanced settings → Secrets** (TOML):

   ```toml
   SEC_USER_AGENT_EMAIL = "contact@example.com"   # shown in the sidebar to every visitor
   MARKET_LENS_ADMINS = "your-admin-username"      # required: otherwise the first account created becomes admin
   OPENROUTER_API_KEY = "sk-or-v1-..."             # the shared key: visitors can use the AI without their own
   SHARED_DAILY_BUDGET_USD = "1.00"                # optional: app-wide daily allowance on the shared key (default $1)
   SHARED_MODEL = "openai/gpt-4o-mini"             # optional: the model visitors get on the shared key
   ```

4. After the app starts, open it and create the admin account straight away with the username from `MARKET_LENS_ADMINS`.

Notes for the hosted version:

- **Shared key:** visitors without a key of their own use `OPENROUTER_API_KEY` for the assistant, 8-K summaries and filing search. The key is used only on the server, never put in a sidebar field, so it never reaches visitors' browsers (admins see it pre-filled). Shared use is fixed to `SHARED_MODEL` and stops for everyone once `SHARED_DAILY_BUDGET_USD` is spent that day; evaluation runs and automated knowledge-base updates need the visitor's own key. Visitors can still paste their own key to choose any model.
- **Hard spending cap:** the in-app allowance resets when the app restarts. For a limit that can't be exceeded, create a separate OpenRouter key for this app and set a credit limit on it (OpenRouter → Keys → Edit → Credit limit), e.g. $5.
- **Knowledge base ships with the app:** `data/chroma/` is committed with the latest 10-K of AAPL, MSFT, NVDA, JPM, AMZN, TSLA, JNJ and XOM (built with `scripts/build_knowledge_base.py`), so the hosted app starts with filings to search.
- **Storage is temporary:** accounts, logs and any filings indexed on the hosted app are reset when it restarts or goes to sleep. To keep more companies, index them locally with `scripts/build_knowledge_base.py`, then commit and push `data/chroma/`.
- **Price history** comes from Yahoo Finance when no Alpha Vantage key is set; Yahoo sometimes blocks cloud servers, so paste an Alpha Vantage key if the chart doesn't load.

## How it works

### Retrieval-augmented generation (RAG)

1. **Ingest:** the Filing library downloads a company's 10-K annual reports from SEC EDGAR (`sec_client.py`).
2. **Chunk:** each report is split into overlapping passages, tagged with ticker, year, section and source URL (`rag.py`).
3. **Embed and store:** passages are turned into vectors with OpenRouter's `text-embedding-3-small` and stored in a local Chroma database.
4. **Retrieve:** a question is matched two ways, by meaning (vector similarity) and by exact words (BM25 keyword search), and the two rankings are merged with Reciprocal Rank Fusion.
5. **Generate:** the best passages go to the model, which answers using only them and cites each one. The passages are wrapped as untrusted text, so instructions hidden inside a filing are ignored.

### Tool calling

The model never makes up figures: it is given six tools (`tools.py`) and decides which to call.

1. The app sends the question plus a description of each tool: its name, purpose and input fields.
2. Instead of answering, the model replies "call `fetch_sec_financials` with ticker=AAPL, metric=revenue".
3. The app runs that Python function (fetching real SEC data, for example) and sends the result back.
4. The model reads the result and either calls another tool or writes the final answer. This repeats for up to five rounds (`agent.py`).
5. The app builds numbered citations from the tool results, so every figure links to its source.

The tools: SEC financials, ratio calculation, company comparison, filing search, 8-K company events, and stock quote. The same tools are also served by an MCP server (`mcp_server.py`).

### Project structure

| Path | Responsibility |
|---|---|
| `streamlit_app.py` | The user interface only |
| `finance_research/agent.py` | The tool-calling loop for OpenRouter, OpenAI and Anthropic |
| `finance_research/tools.py` | The research tools |
| `finance_research/sec_client.py` | SEC data: company facts, fiscal-year matching, filings |
| `finance_research/rag.py` | Chunking, embeddings, hybrid search |
| `finance_research/events.py` | 8-K company events |
| `finance_research/analytics.py`, `prices.py`, `screening.py` | Ratios, scorecard, share prices, shortlist |
| `finance_research/security.py`, `auth.py`, `ratelimit.py`, `monitoring.py` | Prompt-injection guards, accounts, limits, logging |
| `tests/` | Automated tests (run with `pytest`) |

## Current limitations

**Data**
- Only US companies that file 10-Ks with the SEC, using US-GAAP labels. Companies choose different labels for the same item, so some figures are missing or not exactly comparable (banks report revenue differently, for example).
- Fiscal years end in different months (Apple in September, Microsoft in June, Walmart in January), so "the same year" covers different periods.
- Annual figures only; quarterly (10-Q) data is not used yet.
- Share prices are monthly. Without an Alpha Vantage key they come from Yahoo Finance's unofficial chart service, which may change or limit requests.
- P/E uses the latest annual EPS, which can be up to a year old.
- No news articles, analyst forecasts or live prices.

**AI**
- Answers can still contain mistakes. The figure check flags numbers that aren't in the source data, but it can't tell whether a correct number is attached to the right company or year, and it doesn't check text-only claims.
- Retrieval quality depends on chunking and search; the evaluation tab measures it, but scores have not been run regularly.
- Answers are not streamed, so long answers take several seconds to appear.
- The AI features need the user's own API key and cost a small amount per question.

**Scale**
- Accounts, logs and the vector database are local files on one computer, which suits a single user or a demo, not many users at once.
- Filings must be indexed before they can be searched.

**By design**
- No buy/sell recommendations or portfolio advice. The shortlist and scorecard summarise historical, reported data only.

## Possible improvements

**Short term**
- Stream answers word by word so the chat feels faster.
- Add quarterly (10-Q) figures and earnings-call transcripts to the data and the knowledge base.
- Add a reranking model after hybrid search, and run the RAG evaluation after every change.
- More valuation measures (price-to-sales, price-to-book) using shares outstanding.

**Medium term**
- A profile page per company combining key numbers, price chart, scorecard, top 10-K risks and latest events.
- Alerts when a watched company files a new 8-K or 10-K, using the existing scheduler.
- Load any SEC-listed company on demand instead of a fixed list.
- A plain-English mode with a glossary for beginners.

**Long term**
- Deploy with a proper database (e.g. PostgreSQL), a hosted vector store and OAuth sign-in for many users.
- Track answer quality over time from the 👍/👎 feedback.
- Support annual reports from other countries' regulators.
