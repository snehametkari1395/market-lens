"""Model providers the research assistant can use."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    key_env: str
    key_label: str
    models: tuple[str, ...]
    default_model: str
    help: str
    free_text_model: bool = False  # True when any model id the provider offers may be typed


PROVIDERS: dict[str, Provider] = {
    "openrouter": Provider(
        key="openrouter",
        label="OpenRouter (many models)",
        key_env="OPENROUTER_API_KEY",
        key_label="OpenRouter API key",
        models=(
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-sonnet-5-5",
            "google/gemini-2.5-flash",
            "meta-llama/llama-3.3-70b-instruct",
        ),
        default_model="openai/gpt-4o-mini",
        help="One key for hundreds of models from OpenAI, Anthropic, Google, Meta and others.",
        free_text_model=True,
    ),
    "openai": Provider(
        key="openai",
        label="OpenAI (direct)",
        key_env="OPENAI_API_KEY",
        key_label="OpenAI API key",
        models=("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"),
        default_model="gpt-4o-mini",
        help="Calls OpenAI's API directly with your OpenAI key.",
        free_text_model=True,
    ),
    "anthropic": Provider(
        key="anthropic",
        label="Anthropic Claude (direct)",
        key_env="ANTHROPIC_API_KEY",
        key_label="Anthropic API key",
        models=("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"),
        default_model="claude-opus-5-5",
        help="Calls Claude through the official Anthropic SDK with your Anthropic key.",
    ),
}

# USD per token for direct Anthropic calls (Anthropic first-party list prices).
ANTHROPIC_PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00 / 1e6, "output": 20.00 / 1e6},
    "claude-sonnet-5-5": {"input": 2.00 / 1e6, "output": 10.00 / 1e6},
    "claude-haiku-5-5": {"input": 0.10 / 1e6, "output": 0.50 / 1e6},
}


def price_for(provider: str, model: str, openrouter_prices: dict[str, dict[str, float]]) -> dict[str, float] | None:
    """Per-token price for a provider/model pair, or None when unknown."""
    if provider == "anthropic":
        return ANTHROPIC_PRICES.get(model)
    if provider == "openai":
        return openrouter_prices.get(f"openai/{model}")
    return openrouter_prices.get(model)
