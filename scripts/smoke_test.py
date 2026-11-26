"""End-to-end smoke test with real API calls.

Runs the parts of Market Lens that the automated tests can only fake: the API key, 10-K indexing
and embeddings, filing search, the tool-calling assistant, the figure check and an 8-K summary.
It costs a few cents with a small model, and it fills the app's knowledge base as a side effect.

Usage (PowerShell, from the project folder):

    $env:OPENROUTER_API_KEY = "your-key"            # this terminal window only
    $env:SEC_USER_AGENT_EMAIL = "you@example.com"
    .\\.venv\\Scripts\\python.exe scripts\\smoke_test.py

Options: --ticker AAPL  --model openai/gpt-4o-mini  --skip-index  --scratch (index into a
temporary folder instead of the app's knowledge base). The key is read from the environment and
is never printed or saved. A report (without the key) is written to data/smoke/last_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EMBEDDING_PRICE_PER_TOKEN = 0.02 / 1_000_000  # openai/text-embedding-3-small on OpenRouter


class Step:
    def __init__(self, name: str) -> None:
        self.name = name
        self.status = "PASS"
        self.notes: list[str] = []
        self.seconds = 0.0
        self.cost = 0.0

    def warn(self, note: str) -> None:
        if self.status == "PASS":
            self.status = "WARN"
        self.notes.append(note)

    def fail(self, note: str) -> None:
        self.status = "FAIL"
        self.notes.append(note)


def run_step(steps: list[Step], name: str, body: Callable[[Step], None]) -> Step:
    step = Step(name)
    print(f"\n▶ {name}…", flush=True)
    started = time.perf_counter()
    try:
        body(step)
    except Exception as error:  # report every failure and carry on with the next step
        step.fail(f"{type(error).__name__}: {error}")
        step.notes.append(traceback.format_exc(limit=3).strip().splitlines()[-1])
    step.seconds = time.perf_counter() - started
    steps.append(step)
    for note in step.notes:
        print(f"   {note}")
    print(f"   {step.status} in {step.seconds:.1f}s" + (f" · ${step.cost:.4f}" if step.cost else ""))
    return step


def tool_results(messages: list[Any]) -> list[dict[str, Any]]:
    from langchain_core.messages import ToolMessage

    results = []
    for message in messages:
        if isinstance(message, ToolMessage):
            try:
                content = json.loads(str(message.content))
            except json.JSONDecodeError:
                content = {"result": str(message.content)}
            results.append({"name": message.name, "content": content})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--skip-index", action="store_true", help="don't download or index a 10-K")
    parser.add_argument("--scratch", action="store_true", help="index into a temporary folder, not the app's knowledge base")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    from finance_research.agent import complete, run_research_agent
    from finance_research.events import event_text, get_company_events, summary_prompt
    from finance_research.factcheck import verify_answer
    from finance_research.providers import price_for
    from finance_research.rag import SEARCH_STRATEGIES, create_vector_store, ingest_recent_filings, search_filings
    from finance_research.ratelimit import verify_api_key
    from finance_research.sec_client import get_recent_10k_filings, validate_sec_user_agent
    from finance_research.tools import build_tools
    from finance_research.usage import estimate_cost, fetch_openrouter_pricing, summarize_usage

    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    email = os.getenv("SEC_USER_AGENT_EMAIL", "").strip()
    if not key or not email:
        print("Set OPENROUTER_API_KEY and SEC_USER_AGENT_EMAIL in this terminal first (see the top of this file).")
        return 2
    user_agent = validate_sec_user_agent(email)
    ticker = args.ticker.upper()
    vector_dir = Path(tempfile.mkdtemp(prefix="market-lens-smoke-")) if args.scratch else ROOT / "data" / "chroma"
    try:
        prices = fetch_openrouter_pricing()
    except Exception:
        prices = {}
    price = price_for("openrouter", args.model, prices)

    print(f"Market Lens smoke test · {ticker} · model {args.model} · knowledge base: {vector_dir}")
    steps: list[Step] = []
    report_answers: list[dict[str, Any]] = []
    state: dict[str, Any] = {}

    def check_key(step: Step) -> None:
        ok, message = verify_api_key("openrouter", key)
        step.notes.append(message)
        if not ok:
            raise RuntimeError("The OpenRouter key was rejected, so the remaining AI steps will fail.")
        if price is None:
            step.warn(f"No price found for {args.model}; costs will show as $0.")

    def index_filing(step: Step) -> None:
        state["store"] = create_vector_store(key, vector_dir)
        if args.skip_index:
            step.warn("Skipped (--skip-index).")
            return
        filing = get_recent_10k_filings(ticker, 1, user_agent)[0]
        step.notes.append(f"Latest 10-K: FY{filing['report_year']}, filed {filing['filed']}")
        chunks = ingest_recent_filings(ticker, 1, user_agent, state["store"], skip_indexed=True)
        if chunks:
            step.cost = chunks * 1400 / 4 * EMBEDDING_PRICE_PER_TOKEN  # ~4 characters per token
            step.notes.append(f"Indexed {chunks} passages (embedding cost estimated).")
        else:
            step.notes.append("Already indexed; nothing new to embed.")

    def search(step: Step) -> None:
        for strategy in SEARCH_STRATEGIES:
            results = search_filings(state["store"], ticker, "supply chain risks and manufacturing", strategy=strategy)
            if not results:
                step.fail(f"{strategy}: no passages returned")
                continue
            top = results[0]
            step.notes.append(f"{strategy}: {len(results)} passages · top section '{top.get('section')}' · relevance {top.get('relevance', 0):.2f}")

    def ask(question: str, expected_tools: tuple[str, ...]) -> Callable[[Step], None]:
        def body(step: Step) -> None:
            tools = build_tools(user_agent, state.get("store"))
            answer, messages = run_research_agent(question, key, args.model, tools, provider="openrouter")
            results = tool_results(messages)
            usage = summarize_usage(messages)
            step.cost = estimate_cost(usage, price) or 0.0
            names = [result["name"] for result in results]
            check = verify_answer(answer, results)
            step.notes.append(f"Tools used: {', '.join(names) or 'none'} · {usage['total_tokens']:,} tokens")
            step.notes.append(
                f"Figure check: {check['matched'] + check['calculated']} of {check['checked']} figures supported"
                + (f", {check['unverified']} unverified" if check["unverified"] else "")
            )
            preview = " ".join(answer.split())[:220]
            step.notes.append(f"Answer: {preview}{'…' if len(answer) > 220 else ''}")
            if not set(expected_tools) & set(names):
                step.fail(f"Expected the assistant to call {' or '.join(expected_tools)}.")
            searches = [r for r in results if r["name"] == "search_filing_knowledge" and isinstance(r["content"], dict)]
            if "search_filing_knowledge" in expected_tools and searches and not any(r["content"].get("results") for r in searches):
                step.fail("Filing search returned no passages, so the answer isn't based on the 10-K.")
            errors = [result for result in results if isinstance(result["content"], dict) and result["content"].get("error")]
            if errors:
                step.warn(f"{len(errors)} tool call(s) returned an error: {errors[0]['content']['error'][:150]}")
            if check["unverified"]:
                step.warn("Some figures couldn't be matched to the tool data; read the answer.")
            report_answers.append({"question": question, "answer": answer, "tools": names, "figure_check": check})

        return body

    def summarise_event(step: Step) -> None:
        events = get_company_events(ticker, user_agent, days=365, limit=5)
        if not events:
            step.warn("No 8-K filings in the last year to summarise.")
            return
        event = next((item for item in events if item["category"] == "Earnings"), events[0])
        prompt, flags = summary_prompt(event, event_text(event, user_agent))
        summary, usage = complete("openrouter", key, args.model, prompt, max_tokens=700)
        step.cost = estimate_cost(usage, price) or 0.0
        step.notes.append(f"{event['filed']} · {event['headline']}")
        step.notes.append("Summary: " + " ".join(summary.split())[:260])
        if flags:
            step.warn(f"Instruction-like text was removed from the filing: {flags}")
        if len(summary.strip()) < 40:
            step.fail("The summary is empty or too short.")

    run_step(steps, "1. API key", check_key)
    run_step(steps, f"2. Index {ticker}'s latest 10-K", index_filing)
    if "store" in state:
        run_step(steps, "3. Filing search (hybrid, vector, keyword)", search)
    run_step(steps, "4. Assistant: SEC figures", ask(
        f"What were {ticker}'s revenue and net income in its two most recent fiscal years, and how much did revenue grow?",
        ("fetch_sec_financials", "compare_companies", "calculate_financial_ratio"),
    ))
    run_step(steps, "5. Assistant: 10-K filing question", ask(
        f"What does {ticker} say about supply chain risks in its latest 10-K? Quote the filing.", ("search_filing_knowledge",)
    ))
    run_step(steps, "6. Assistant: recent 8-K events", ask(
        f"What major events has {ticker} reported in SEC 8-K filings in the last 90 days?", ("get_company_events",)
    ))
    run_step(steps, "7. 8-K summary", summarise_event)

    total_cost = sum(step.cost for step in steps)
    print("\n" + "=" * 72)
    for step in steps:
        print(f"{step.status:<5} {step.name}")
    failed = [step for step in steps if step.status == "FAIL"]
    print("-" * 72)
    print(f"{len(steps) - len(failed)} of {len(steps)} steps passed · total {sum(s.seconds for s in steps):.0f}s · cost ≈ ${total_cost:.4f}")

    report_dir = ROOT / "data" / "smoke"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "model": args.model,
        "knowledge_base": str(vector_dir),
        "total_cost_usd": round(total_cost, 5),
        "steps": [{"name": s.name, "status": s.status, "seconds": round(s.seconds, 1), "cost_usd": round(s.cost, 5), "notes": s.notes} for s in steps],
        "answers": report_answers,
    }
    (report_dir / "last_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report: {report_dir / 'last_report.json'}")
    if not args.scratch and not args.skip_index:
        print("Restart the app so it picks up the newly indexed filing.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
