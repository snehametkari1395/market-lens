"""Index the latest annual report (10-K, or 20-F for European companies) of a set of companies into data/chroma, so the app starts with a knowledge base.

Usage (PowerShell, from the project folder):

    $env:OPENROUTER_API_KEY = "your-key"
    $env:SEC_USER_AGENT_EMAIL = "you@example.com"
    .\\.venv\\Scripts\\python.exe scripts\\build_knowledge_base.py                 # the default list
    .\\.venv\\Scripts\\python.exe scripts\\build_knowledge_base.py AAPL MSFT --years 2

Filings that are already indexed are skipped, so it's safe to run again. Embeddings cost about
$0.002 per 10-K with openai/text-embedding-3-small.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# One well-known company from each of the main sectors.
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "AMZN", "TSLA", "JNJ", "XOM", "SAP", "ASML"]
EMBEDDING_PRICE_PER_CHUNK = 350 * 0.02 / 1_000_000  # ~1,400 characters (~350 tokens) per chunk


def main() -> int:
    parser = argparse.ArgumentParser(description="Index recent 10-K filings into the app's knowledge base.")
    parser.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--years", type=int, default=1, help="10-K filings per company (default: the latest one)")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    from finance_research.rag import create_vector_store, index_summary, ingest_recent_filings
    from finance_research.sec_client import validate_sec_user_agent

    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    email = os.getenv("SEC_USER_AGENT_EMAIL", "").strip()
    if not key or not email:
        print("Set OPENROUTER_API_KEY and SEC_USER_AGENT_EMAIL in this terminal first (see the top of this file).")
        return 2
    user_agent = validate_sec_user_agent(email)
    directory = ROOT / "data" / "chroma"
    store = create_vector_store(key, directory)

    failed = []
    total = 0
    for ticker in [t.upper() for t in args.tickers]:
        started = time.perf_counter()
        try:
            chunks = ingest_recent_filings(ticker, args.years, user_agent, store, skip_indexed=True)
        except Exception as error:  # report and carry on with the next company
            failed.append(ticker)
            print(f"FAIL  {ticker}: {type(error).__name__}: {error}")
            continue
        total += chunks
        note = f"{chunks:,} passages" if chunks else "already indexed"
        print(f"OK    {ticker}: {note} ({time.perf_counter() - started:.0f}s)")

    print("-" * 60)
    for row in index_summary(directory):
        print(f"{row['ticker']:<6} FY{row['report_year']}  {row['chunks']:>5,} passages")
    print(f"Added {total:,} passages · embedding cost ≈ ${total * EMBEDDING_PRICE_PER_CHUNK:.4f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
