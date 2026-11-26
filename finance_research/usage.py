"""Token usage and cost calculation for OpenRouter model calls."""

from __future__ import annotations

from typing import Any

import requests
from langchain_core.messages import AIMessage, BaseMessage

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def summarize_usage(messages: list[BaseMessage]) -> dict[str, int]:
    """Add up the tokens reported for every model call in one agent run."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "model_calls": 0}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        usage = message.usage_metadata or {}
        if not usage:
            token_usage = (message.response_metadata or {}).get("token_usage") or {}
            usage = {
                "input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
            }
        totals["input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["output_tokens"] += int(usage.get("output_tokens") or 0)
        totals["total_tokens"] += int(
            usage.get("total_tokens") or (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
        )
        totals["model_calls"] += 1
    return totals


def fetch_openrouter_pricing(timeout: int = 15) -> dict[str, dict[str, float]]:
    """Return USD-per-token prices for every OpenRouter model. No API key is needed."""
    response = requests.get(OPENROUTER_MODELS_URL, timeout=timeout)
    response.raise_for_status()
    prices: dict[str, dict[str, float]] = {}
    for model in response.json().get("data", []):
        pricing = model.get("pricing") or {}
        try:
            prices[model["id"]] = {
                "input": float(pricing.get("prompt") or 0),
                "output": float(pricing.get("completion") or 0),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return prices


def estimate_cost(usage: dict[str, int], price: dict[str, float] | None) -> float | None:
    """Cost in USD, or None when the model's price is unknown."""
    if not price:
        return None
    return usage.get("input_tokens", 0) * price["input"] + usage.get("output_tokens", 0) * price["output"]


def format_cost(cost: float | None) -> str:
    if cost is None:
        return "price unknown"
    if cost == 0:
        return "$0 (free model)"
    if cost < 0.01:
        return f"${cost:.5f}"
    return f"${cost:.4f}"


def add_usage(total: dict[str, Any], usage: dict[str, Any], cost: float | None) -> dict[str, Any]:
    """Return a new running total with one more answer's usage added."""
    combined = {
        key: int(total.get(key, 0)) + int(usage.get(key, 0))
        for key in ("input_tokens", "output_tokens", "total_tokens", "model_calls")
    }
    combined["answers"] = int(total.get("answers", 0)) + 1
    combined["cost_usd"] = float(total.get("cost_usd", 0.0)) + (cost or 0.0)
    combined["unpriced_answers"] = int(total.get("unpriced_answers", 0)) + (1 if cost is None else 0)
    return combined
