"""Plain-English help for newcomers: what the app is, what each tab does, and what the terms mean (no Streamlit)."""

from __future__ import annotations

import re

APP_IN_ONE_LINE = (
    "Market Lens helps you understand US public companies using the official reports they file with the "
    "US regulator (the SEC), and lets you ask an AI assistant questions about them in plain English."
)

QUICK_START = [
    "**Pick companies.** Click *Explore companies* on a sector card, or choose companies in the sidebar.",
    "**Look at the numbers.** *Company overview* and *Analytics* show revenue, profit and growth as charts.",
    "**Ask a question.** In *Research assistant*, type something like “What risks does Apple mention?”. "
    "The answer quotes the company's own annual report.",
    "**Stuck on a word?** The glossary below explains every term the app uses.",
]

# (tab, what it shows, when to use it)
TAB_TOUR = [
    ("Company overview", "Key figures (revenue, profit, cash…) for the selected companies, side by side.", "Compare companies at a glance."),
    ("Valuation", "Share-price charts plus market cap, P/E and other valuation ratios for the selected companies.", "See how expensive a stock is compared with its earnings and peers."),
    ("Analytics", "Growth, margins and other ratios over several years, plus share-price growth.", "See trends and which company is improving."),
    ("Company events", "Recent major news each company officially reported (Form 8-K): results, new CEO, deals…", "Find out what happened recently."),
    ("Calendar", "Upcoming earnings announcements and new stock listings (IPOs).", "See what's coming up."),
    ("Watchlist", "Your own list of companies, a simple portfolio, price alerts and dividends.", "Keep track of companies you care about."),
    ("Private companies", "How much money well-known private companies raised (Form D filings).", "Learn about companies that aren't on the stock market."),
    ("Research shortlist", "Checks companies against conditions you set, e.g. growth above 10%.", "Narrow a list down to candidates worth reading about."),
    ("Research assistant", "An AI chatbot that answers questions using SEC data and the annual reports.", "Ask anything in plain English."),
    ("Filing library", "The annual reports (10-Ks) the assistant can search, and tools to add more.", "See or extend what the assistant has read."),
    ("RAG evaluation", "Tests how well the assistant finds the right passages and answers correctly.", "Technical: measure answer quality."),
    ("Monitoring", "Usage, costs and errors across the app (admin only).", "Technical: run the app."),
    ("Help & guide", "This page: quick start, tab tour, glossary and a help bot.", "Whenever something is unclear."),
]

# category -> list of (term, plain meaning, why it matters, extra words people may type)
GLOSSARY: dict[str, list[tuple[str, str, str, list[str]]]] = {
    "Reports & the SEC": [
        ("SEC", "The US Securities and Exchange Commission: the government body that regulates the stock market. Public companies must send it regular reports.", "All the company data in Market Lens comes from these official reports, not from opinions.", ["securities and exchange commission"]),
        ("SEC EDGAR", "The SEC's free online database where every company report is published.", "Market Lens downloads its figures and documents from EDGAR.", ["edgar"]),
        ("10-K (annual report)", "The detailed yearly report a public company must file: its business, risks, and audited financial statements.", "It's the most reliable source of a company's yearly numbers; the assistant quotes from it.", ["10-k", "10k", "annual report"]),
        ("10-Q (quarterly report)", "A shorter, unaudited report filed for each of the first three quarters of the year.", "Gives more recent numbers between annual reports.", ["10-q", "10q", "quarterly report"]),
        ("8-K (major event report)", "A report a company must file within four business days of important news: results, a new CEO, a deal, a cyber attack…", "Shown in the Company events tab: news straight from the company.", ["8-k", "8k"]),
        ("Form D", "A short notice a private company files when it raises money from investors.", "The only official data most private companies publish; used in the Private companies tab.", ["form d"]),
        ("20-F (foreign annual report)", "The annual report a non-US company listed in the US files with the SEC instead of a 10-K.", "It's where the app gets figures and passages for SAP, ASML and the other European companies.", ["20-f", "20f"]),
        ("IFRS", "International Financial Reporting Standards: the accounting rules used in Europe and most of the world. US companies use US GAAP instead.", "The app reads both, so European and US companies can be compared, but small definition differences remain.", ["ifrs", "us gaap", "gaap"]),
        ("ADR (US-listed share)", "American depositary receipt: a certificate traded in the US that stands for shares of a foreign company, sometimes more or fewer than one.", "Per-share figures for European companies are adjusted to match one US-listed share, so P/E uses the right price.", ["adr", "ads", "depositary"]),
        ("Currency conversion", "European companies report in euros, Danish kroner and so on. The app converts their figures to US dollars at each year's average exchange rate.", "Makes companies comparable, but part of a change over time can come from the exchange rate rather than the business.", ["exchange rate", "euro", "eur", "currency", "fx"]),
        ("Ticker", "The short code a company trades under on the stock market, e.g. AAPL for Apple.", "You pick companies by ticker.", ["symbol", "ticker symbol"]),
        ("Fiscal year (FY)", "A company's financial year. It doesn't always match the calendar: Apple's ends in late September.", "Charts are labelled by fiscal year, so 2024 means the company's 2024 financial year.", ["fy", "financial year"]),
        ("Filing", "Any official report a company submits to the SEC.", "“Filings” is just the general word for 10-Ks, 10-Qs, 8-Ks and so on.", ["filings"]),
    ],
    "Money & profit": [
        ("Revenue", "All the money a company brings in from selling its products or services, before any costs. Also called sales.", "Shows how big the business is and whether it's growing.", ["sales", "turnover", "top line"]),
        ("Net income", "What's left of revenue after every cost, interest and tax: the company's profit.", "Shows whether the company actually makes money.", ["profit", "earnings", "bottom line", "net profit"]),
        ("Operating income", "Profit from the main business, before interest and taxes.", "Shows how profitable day-to-day operations are.", ["operating profit"]),
        ("Operating margin", "Operating income ÷ revenue. A 30% margin means 30 cents of operating profit per dollar of sales.", "Lets you compare profitability of companies of different sizes.", []),
        ("Net margin", "Net income ÷ revenue: the share of each dollar of sales kept as final profit.", "Higher usually means a more efficient or more powerful business.", ["profit margin"]),
        ("Revenue growth (YoY)", "How much revenue changed compared with the year before (year over year), in %.", "Shows whether the business is expanding (+) or shrinking (−). Blue/red in the growth heatmap.", ["yoy", "year over year", "growth"]),
        ("R&D intensity", "Research and development spending ÷ revenue.", "Shows how much a company invests in future products.", ["r&d", "research and development"]),
        ("Cash", "Money the company holds, including very safe short-term investments (cash equivalents).", "A cushion for hard times and for investing.", ["cash and cash equivalents"]),
        ("Total assets", "Everything the company owns that has value: cash, buildings, equipment, investments…", "The size of the company's resources.", ["assets"]),
        ("Liabilities", "Everything the company owes: loans, bills, other obligations.", "Too many liabilities compared with assets can be risky.", []),
        ("Shareholders' equity", "Assets minus liabilities: what would belong to the owners if everything were sold and paid off.", "Used in ratios like debt-to-equity.", ["equity", "book value"]),
        ("Return on assets (ROA)", "Net income ÷ total assets.", "How well the company turns what it owns into profit.", ["roa"]),
        ("Liabilities to assets", "Total liabilities ÷ total assets.", "How much of the company is funded by what it owes. Lower is generally safer.", []),
        ("Debt-to-equity", "Long-term debt ÷ shareholders' equity.", "How much the company relies on borrowing. High values mean more risk if business slows.", ["debt to equity", "leverage"]),
        ("Net cash", "Cash minus long-term debt. Negative means more debt than cash.", "A quick view of financial strength.", []),
    ],
    "Shares & returns": [
        ("Share price", "The price of one share of the company on the stock market.", "Shown in charts and the watchlist; it moves every trading day.", ["stock price", "price", "quote"]),
        ("EPS (earnings per share)", "Net income ÷ number of shares. “Diluted” EPS also counts shares that could be created (e.g. from employee options).", "Turns company profit into profit per share.", ["eps", "diluted eps"]),
        ("P/E ratio", "Share price ÷ EPS: how many dollars investors pay for one dollar of yearly profit.", "A rough sense of how “expensive” a share is. High can mean high expectations.", ["p/e", "pe", "price to earnings"]),
        ("Dividend", "Cash a company pays its shareholders out of its profits, usually every quarter.", "A way investors get money back without selling shares.", ["dividends"]),
        ("Dividend yield", "Dividends paid in the last 12 months ÷ share price.", "Like an interest rate on the share: 3% means $3 a year per $100 invested.", ["yield"]),
        ("Payout ratio", "Dividends paid ÷ net income.", "Shows how much of its profit a company hands out. Above 100% may not last.", []),
        ("CAGR (per-year return)", "Compound annual growth rate: the steady yearly growth that would give the same overall result.", "Makes returns over different periods comparable.", ["cagr", "annualised return", "annualized return"]),
        ("Worst fall (max drawdown)", "The biggest drop from a previous high, in %.", "Shows how bumpy the ride has been for investors.", ["drawdown", "max drawdown"]),
        ("Log scale", "A chart scale where equal distances mean equal % changes.", "Makes fast- and slow-growing companies easier to compare on one chart.", ["logarithmic"]),
        ("IPO", "Initial public offering: when a company first sells shares on the stock market.", "Shown in the Calendar tab.", ["initial public offering"]),
        ("Earnings report", "The company's announcement of its latest quarterly results.", "Share prices often move a lot around these dates.", ["earnings date", "earnings call"]),
        ("Sector", "A group of companies in the same kind of business, e.g. semiconductors or banks.", "The sector cards let you explore similar companies together.", ["industry"]),
    ],
    "AI & search": [
        ("Research assistant", "The AI chatbot in Market Lens. It looks things up with tools before it answers.", "Answers are based on real data and quotes, not the AI's memory.", ["chatbot", "assistant", "ai"]),
        ("Knowledge base", "The collection of annual-report passages the assistant can search.", "The assistant can only quote reports that are in it; see the Filing library.", ["vector store", "index", "indexed"]),
        ("RAG (retrieval-augmented generation)", "Before answering, the AI first finds (retrieves) the most relevant passages from the reports and then writes its answer from them.", "Keeps answers grounded in the documents, with sources you can check.", ["rag", "retrieval"]),
        ("Embedding", "A list of numbers that captures the meaning of a piece of text, so texts about the same idea end up close together.", "Lets the app find passages by meaning, not just exact words.", ["embeddings", "vector"]),
        ("Hybrid search", "Combines keyword search (exact words) with meaning-based search, then merges the results.", "Finds the right passage more often than either method alone.", ["bm25", "keyword search"]),
        ("Tool calling", "The AI decides to use a tool, like “get revenue from SEC” or “search the reports”, then answers using the result.", "Lets the assistant fetch exact, up-to-date numbers instead of guessing.", ["tools", "tool"]),
        ("Tokens", "The small pieces of text an AI model reads and writes; usage and cost are counted in tokens.", "Shown under each answer with its cost.", ["token"]),
        ("Model", "The AI engine that writes the answers, e.g. GPT-4o mini or Claude.", "Different models differ in quality, speed and price.", ["llm"]),
        ("API key", "A password-like code that lets the app use an online service such as an AI model.", "Visitors don't need one: the app uses its own shared key with a daily limit.", ["key", "openrouter"]),
    ],
}

_DEFINITION_WORDS = re.compile(r"\b(what|what's|whats|mean|means|meaning|define|definition|explain|stand for|stands for)\b")


def all_terms() -> list[dict[str, str]]:
    """Every glossary entry as a flat list of dicts (category, term, meaning, why)."""
    return [
        {"category": category, "term": term, "meaning": meaning, "why": why}
        for category, entries in GLOSSARY.items()
        for term, meaning, why, _aliases in entries
    ]


def search_glossary(query: str) -> list[dict[str, str]]:
    """Entries whose term, aliases or explanation contain the query (term matches first)."""
    text = query.lower().strip()
    if not text:
        return all_terms()
    named, described = [], []
    for category, entries in GLOSSARY.items():
        for term, meaning, why, aliases in entries:
            entry = {"category": category, "term": term, "meaning": meaning, "why": why}
            if text in term.lower() or any(text in alias for alias in aliases):
                named.append(entry)
            elif text in meaning.lower() or text in why.lower():
                described.append(entry)
    return named + described


def _names(term: str, aliases: list[str]) -> list[str]:
    main = term.lower()
    names = [main, re.sub(r"\s*\(.*\)", "", main)]
    inside = re.search(r"\((.*)\)", main)
    if inside:
        names.append(inside.group(1))
    return [name for name in names + aliases if len(name) >= 2]


def lookup_term(question: str) -> dict[str, str] | None:
    """The glossary entry a question names (longest matching name wins), or None."""
    text = " " + re.sub(r"[^a-z0-9&/'\- ]", " ", question.lower()) + " "
    best, best_length = None, 0
    for category, entries in GLOSSARY.items():
        for term, meaning, why, aliases in entries:
            for name in _names(term, aliases):
                if f" {name} " in text and len(name) > best_length:
                    best = {"category": category, "term": term, "meaning": meaning, "why": why}
                    best_length = len(name)
    return best


def is_definition_question(question: str) -> bool:
    return bool(_DEFINITION_WORDS.search(question.lower()))
