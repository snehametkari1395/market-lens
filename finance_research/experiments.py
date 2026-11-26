"""A/B testing of RAG retrieval strategies.

Each user is assigned to a variant by hashing their id, so the same person always
sees the same strategy. Answers log their variant; thumbs-up/down feedback and
retrieval quality are then compared per variant with a two-proportion z-test.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from finance_research.rag import SEARCH_STRATEGIES


@dataclass(frozen=True)
class Experiment:
    name: str = "rag-strategy-1"
    variant_a: str = "vector"
    variant_b: str = "hybrid"
    share_b: float = 0.5  # fraction of users who get variant B

    def __post_init__(self) -> None:
        if self.variant_a not in SEARCH_STRATEGIES or self.variant_b not in SEARCH_STRATEGIES:
            raise ValueError("Variants must be search strategies.")
        if self.variant_a == self.variant_b:
            raise ValueError("Choose two different strategies to compare.")
        if not 0.0 <= self.share_b <= 1.0:
            raise ValueError("The share of users in variant B must be between 0 and 1.")


def assign_variant(user_id: str, experiment: Experiment) -> tuple[str, str]:
    """Deterministic assignment: returns (variant label, strategy)."""
    digest = hashlib.sha256(f"{experiment.name}:{user_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return ("B", experiment.variant_b) if bucket < experiment.share_b else ("A", experiment.variant_a)


def two_proportion_z_test(successes_a: int, total_a: int, successes_b: int, total_b: int) -> dict[str, float] | None:
    """Two-sided z-test for a difference in proportions. None when either group is empty."""
    if total_a == 0 or total_b == 0:
        return None
    rate_a, rate_b = successes_a / total_a, successes_b / total_b
    pooled = (successes_a + successes_b) / (total_a + total_b)
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / total_a + 1 / total_b))
    if standard_error == 0:
        return {"rate_a": rate_a, "rate_b": rate_b, "difference": rate_b - rate_a, "z": 0.0, "p_value": 1.0}
    z = (rate_b - rate_a) / standard_error
    return {
        "rate_a": rate_a,
        "rate_b": rate_b,
        "difference": rate_b - rate_a,
        "z": z,
        "p_value": math.erfc(abs(z) / math.sqrt(2)),
    }


def summarize_experiment(events: list[dict[str, Any]], experiment_name: str) -> dict[str, Any]:
    """Per-variant stats from the event log: answers, feedback, latency, retrieval relevance, cost."""
    answers = [e for e in events if e.get("kind") == "chat_answer" and e.get("experiment") == experiment_name]
    feedback = [e for e in events if e.get("kind") == "answer_feedback" and e.get("experiment") == experiment_name]
    # Keep only the latest vote per answer, so changing your mind doesn't double count.
    latest_vote: dict[str, dict[str, Any]] = {}
    for event in feedback:
        latest_vote[event.get("answer_id", "")] = event

    variants: dict[str, dict[str, Any]] = {}
    for label in ("A", "B"):
        rows = [e for e in answers if e.get("variant") == label]
        votes = [e for e in latest_vote.values() if e.get("variant") == label]
        relevance = [float(e["top_relevance"]) for e in rows if e.get("top_relevance") is not None]
        latencies = sorted(float(e.get("latency_s") or 0) for e in rows)
        variants[label] = {
            "strategy": rows[0].get("strategy") if rows else (votes[0].get("strategy") if votes else ""),
            "answers": len(rows),
            "users": len({e.get("username") for e in rows}),
            "votes": len(votes),
            "thumbs_up": sum(1 for e in votes if e.get("score") == 1),
            "satisfaction": (sum(1 for e in votes if e.get("score") == 1) / len(votes)) if votes else None,
            "median_latency_s": latencies[len(latencies) // 2] if latencies else None,
            "avg_top_relevance": (sum(relevance) / len(relevance)) if relevance else None,
            "filing_searches": sum(1 for e in rows if e.get("top_relevance") is not None),
            "avg_cost_usd": (sum(float(e.get("cost_usd") or 0) for e in rows) / len(rows)) if rows else None,
        }
    test = two_proportion_z_test(
        variants["A"]["thumbs_up"], variants["A"]["votes"], variants["B"]["thumbs_up"], variants["B"]["votes"]
    )
    return {"variants": variants, "test": test}


def verdict(summary: dict[str, Any], alpha: float = 0.05, min_votes: int = 20) -> str:
    variants, test = summary["variants"], summary["test"]
    votes_a, votes_b = variants["A"]["votes"], variants["B"]["votes"]
    if test is None or min(votes_a, votes_b) < min_votes:
        return (
            f"Not enough feedback yet: {votes_a} vote(s) for A and {votes_b} for B. "
            f"Collect at least {min_votes} per variant before drawing conclusions."
        )
    better = "B" if test["difference"] > 0 else "A"
    if test["p_value"] < alpha:
        return (
            f"Variant {better} ({variants[better]['strategy']}) has a significantly higher thumbs-up rate "
            f"({test['rate_b'] if better == 'B' else test['rate_a']:.0%} vs "
            f"{test['rate_a'] if better == 'B' else test['rate_b']:.0%}, p = {test['p_value']:.3f})."
        )
    return f"No significant difference yet (p = {test['p_value']:.2f}). Keep collecting feedback."
