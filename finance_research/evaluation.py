"""RAG evaluation: retrieval metrics plus RAGAs-style LLM-judged answer metrics.

1. Build a test set: sample indexed 10-K chunks and have a model write one question
   (and a reference answer) that only that chunk answers. The chunk is the ground truth.
2. Retrieval metrics per search strategy: hit rate@k, recall@1 and MRR (did the
   ground-truth chunk come back, and how high).
3. Generation metrics (same definitions as RAGAs): for each question, answer from the
   retrieved chunks, then an LLM judge scores faithfulness, answer relevancy, context
   precision and answer correctness from 0 to 1.
"""

from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from finance_research.rag import SEARCH_STRATEGIES, search_filings
from finance_research.security import scan_text

Completer = Callable[[str], tuple[str, dict[str, int]]]

QUESTION_PROMPT = """You are building a test set for a search system over SEC 10-K filings.
Read the passage from {company}'s FY{year} 10-K ({section}) and write ONE specific factual question
that this passage answers. Rules:
- The question must make sense on its own (name the company; don't say "this passage").
- Prefer concrete facts: amounts, names, counts, dates, specific risks or policies.
- Then write a short reference answer (one or two sentences) using only the passage.
Reply with JSON only: {{"question": "...", "answer": "..."}}

Passage:
\"\"\"{text}\"\"\""""

ANSWER_PROMPT = """Answer the question using only the context passages from SEC 10-K filings.
If the context doesn't contain the answer, say so. Be concise (at most four sentences).

Question: {question}

Context:
{contexts}"""

JUDGE_PROMPT = """You are grading a retrieval-augmented answer. Score each metric from 0 to 1.

- faithfulness: share of the answer's claims that are supported by the retrieved contexts
  (1 = every claim supported; an answer that only says the context lacks the answer scores 1).
- answer_relevancy: how directly the answer addresses the question (1 = fully on point).
- context_precision: for each retrieved context, in order, is it relevant to the question? Give a list of 0/1.
- answer_correctness: how well the answer matches the reference answer's facts (1 = same facts).

Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Retrieved contexts:
{contexts}

Reply with JSON only:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": [0, 1], "answer_correctness": 0.0, "reason": "one sentence"}}"""


def parse_json_reply(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (tolerates code fences and extra text)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object in reply: {text[:120]}")
    return json.loads(match.group(0))


def chunk_id(item: dict[str, Any]) -> str:
    return f"{item.get('ticker')}|{item.get('report_year')}|{item.get('source_url')}|{item.get('chunk_index')}"


def sample_chunks(vector_store, ticker: str, count: int, seed: int = 7, min_chars: int = 500) -> list[dict[str, Any]]:
    """Random narrative chunks for one company, skipping short or table-like ones."""
    stored = vector_store.get(where={"ticker": ticker.upper()}, include=["documents", "metadatas"])
    candidates = []
    for text, metadata in zip(stored.get("documents") or [], stored.get("metadatas") or []):
        letters = sum(ch.isalpha() for ch in text)
        if len(text) >= min_chars and letters / max(len(text), 1) > 0.6 and not scan_text(text).flagged:
            candidates.append({**metadata, "text": text})
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def generate_test_set(
    vector_store, tickers: list[str], per_company: int, complete: Completer, seed: int = 7
) -> list[dict[str, Any]]:
    test_set = []
    for ticker in tickers:
        for chunk in sample_chunks(vector_store, ticker, per_company, seed=seed):
            try:
                reply, _ = complete(
                    QUESTION_PROMPT.format(
                        company=chunk.get("company") or ticker,
                        year=chunk.get("report_year"),
                        section=chunk.get("section") or "narrative",
                        text=chunk["text"][:3000],
                    )
                )
                parsed = parse_json_reply(reply)
            except (ValueError, json.JSONDecodeError):
                continue
            if not parsed.get("question"):
                continue
            test_set.append(
                {
                    "id": f"q{len(test_set) + 1}",
                    "ticker": ticker.upper(),
                    "report_year": chunk.get("report_year"),
                    "question": str(parsed["question"]).strip(),
                    "reference_answer": str(parsed.get("answer", "")).strip(),
                    "ground_truth_chunk": chunk_id(chunk),
                    "section": chunk.get("section"),
                }
            )
    return test_set


def evaluate_retrieval(
    vector_store, test_set: list[dict[str, Any]], strategies: list[str], k: int = 5
) -> dict[str, Any]:
    per_strategy: dict[str, Any] = {}
    details = []
    for strategy in strategies:
        if strategy not in SEARCH_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        hits = reciprocal_ranks = top1 = 0.0
        latencies = []
        for item in test_set:
            started = time.perf_counter()
            results = search_filings(vector_store, item["ticker"], item["question"], count=k, strategy=strategy)
            latencies.append(time.perf_counter() - started)
            ids = [chunk_id(result) for result in results]
            rank = ids.index(item["ground_truth_chunk"]) + 1 if item["ground_truth_chunk"] in ids else None
            hits += 1 if rank else 0
            top1 += 1 if rank == 1 else 0
            reciprocal_ranks += (1 / rank) if rank else 0
            details.append({"id": item["id"], "strategy": strategy, "rank": rank, "question": item["question"]})
        total = len(test_set) or 1
        per_strategy[strategy] = {
            "hit_rate": hits / total,
            "recall_at_1": top1 / total,
            "mrr": reciprocal_ranks / total,
            "avg_latency_ms": 1000 * sum(latencies) / len(latencies) if latencies else 0.0,
        }
    return {"k": k, "metrics": per_strategy, "details": details}


def _contexts_block(results: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{index}] ({result.get('section') or 'Filing'}, FY{result.get('report_year')}) {result['text'][:1500]}"
        for index, result in enumerate(results, start=1)
    )


def _clip(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def average_precision(relevance: list[int]) -> float | None:
    """RAGAs context precision: mean of precision@i over the positions i that hold a relevant context."""
    if not relevance:
        return None
    hits, total = 0, 0.0
    for position, relevant in enumerate(relevance, start=1):
        if relevant:
            hits += 1
            total += hits / position
    return total / hits if hits else 0.0


def evaluate_generation(
    vector_store,
    test_set: list[dict[str, Any]],
    strategies: list[str],
    answer_with: Completer,
    judge_with: Completer,
    k: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    rows = []
    total_steps = len(test_set) * len(strategies)
    tokens = {"input_tokens": 0, "output_tokens": 0}

    def add(usage: dict[str, int]) -> None:
        for key in tokens:
            tokens[key] += int(usage.get(key, 0))

    step = 0
    for strategy in strategies:
        for item in test_set:
            step += 1
            results = search_filings(vector_store, item["ticker"], item["question"], count=k, strategy=strategy)
            contexts = _contexts_block(results)
            row: dict[str, Any] = {"id": item["id"], "strategy": strategy, "question": item["question"]}
            try:
                answer, usage = answer_with(ANSWER_PROMPT.format(question=item["question"], contexts=contexts))
                add(usage)
                verdict_text, usage = judge_with(
                    JUDGE_PROMPT.format(
                        question=item["question"], reference=item["reference_answer"], answer=answer, contexts=contexts
                    )
                )
                add(usage)
                verdict = parse_json_reply(verdict_text)
                relevance = [1 if int(value) else 0 for value in verdict.get("context_precision", [])][: len(results)]
                row.update(
                    {
                        "answer": answer,
                        "faithfulness": _clip(verdict.get("faithfulness")),
                        "answer_relevancy": _clip(verdict.get("answer_relevancy")),
                        "context_precision": average_precision(relevance),
                        "answer_correctness": _clip(verdict.get("answer_correctness")),
                        "judge_reason": str(verdict.get("reason", ""))[:300],
                    }
                )
            except Exception as error:  # one failed item shouldn't sink the run
                row["error"] = f"{type(error).__name__}: {error}"[:300]
            rows.append(row)
            if on_progress:
                on_progress(step, total_steps)

    metrics = {}
    for strategy in strategies:
        scored = [row for row in rows if row["strategy"] == strategy and "error" not in row]
        metrics[strategy] = {
            name: (sum(row[name] for row in scored if row[name] is not None) / max(1, sum(row[name] is not None for row in scored)))
            if scored
            else None
            for name in ("faithfulness", "answer_relevancy", "context_precision", "answer_correctness")
        }
        metrics[strategy]["scored"] = len(scored)
        metrics[strategy]["failed"] = sum(1 for row in rows if row["strategy"] == strategy and "error" in row)
    return {"k": k, "metrics": metrics, "rows": rows, "tokens": tokens}


def save_run(eval_dir: str | Path, run: dict[str, Any]) -> Path:
    path = Path(eval_dir)
    path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **run}
    target = path / f"run-{stamp}.json"
    target.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def list_runs(eval_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(eval_dir)
    if not path.exists():
        return []
    runs = []
    for file in sorted(path.glob("run-*.json"), reverse=True):
        try:
            runs.append({"file": file.name, **json.loads(file.read_text(encoding="utf-8"))})
        except json.JSONDecodeError:
            continue
    return runs
