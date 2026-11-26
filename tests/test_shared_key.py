"""The app's shared OpenRouter key: visitors can use the AI without a key, and the key never reaches the browser."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SECRET = "sk-or-v1-shared-owner-key-DO-NOT-LEAK"


def _all_text(at: AppTest) -> str:
    """Everything the browser would receive for display: element values, labels and text."""
    parts = []
    for element in [*at.main, *at.sidebar]:  # every rendered element, nested ones included
        for attribute in ("value", "label", "body", "placeholder", "help"):
            value = getattr(element, attribute, None)
            if isinstance(value, str):
                parts.append(value)
    parts += [str(at.session_state[key]) for key in ("openrouter_key", "openai_key", "anthropic_key", "alpha_key")]
    return "\n".join(parts)


def _start_as_guest(monkeypatch, budget: str | None = None):
    import streamlit as st

    st.cache_resource.clear()  # a fresh rate limiter, so the shared allowance starts at $0
    calls: list[dict] = []

    def fake_agent(question, api_key, model, tools, **kwargs):
        calls.append({"key": api_key, "model": model, "provider": kwargs.get("provider")})
        message = AIMessage(content="Apple's revenue was strong.", usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100})
        return message.content, [message]

    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    monkeypatch.setenv("SEC_USER_AGENT_EMAIL", "tester@example.com")
    monkeypatch.setenv("MARKET_LENS_ADMINS", "owner")
    if budget:
        monkeypatch.setenv("SHARED_DAILY_BUDGET_USD", budget)
    monkeypatch.setattr("finance_research.agent.run_research_agent", fake_agent)
    monkeypatch.setattr(
        "finance_research.usage.fetch_openrouter_pricing", lambda: {"openai/gpt-4o-mini": {"input": 1.5e-7, "output": 6e-7}}
    )
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=90)
    at.run()
    next(button for button in at.button if button.label == "Continue as guest").click().run()
    assert not at.exception, at.exception
    return at, calls


def test_guest_uses_shared_key_without_seeing_it(monkeypatch):
    at, calls = _start_as_guest(monkeypatch)
    assert at.session_state["openrouter_key"] == ""  # the key widget stays empty for visitors
    page_text = _all_text(at)
    assert "No key needed" in page_text and "OpenRouter API key" in page_text  # the check really reads the page
    assert SECRET not in page_text

    at.chat_input[0].set_value("What was Apple's revenue last year?").run()
    assert not at.exception, at.exception
    assert calls == [{"key": SECRET, "model": "openai/gpt-4o-mini", "provider": "openrouter"}]
    assert SECRET not in _all_text(at)


def test_shared_allowance_stops_the_assistant(monkeypatch):
    # Each fake answer costs about $0.0002, so a $0.0001 allowance is used up by the first one.
    at, calls = _start_as_guest(monkeypatch, budget="0.0001")
    at.chat_input[0].set_value("What was Apple's revenue last year?").run()
    at.chat_input[0].set_value("And Microsoft's?").run()
    assert not at.exception, at.exception
    assert len(calls) == 1
    assert any("free AI allowance" in warning.value for warning in at.warning)


def test_refresh_from_sec_button_runs(monkeypatch):
    # Regression: the button cleared a cache function that was only defined further down the script.
    at, _ = _start_as_guest(monkeypatch)
    next(button for button in at.button if button.label == "Refresh from SEC now").click().run()
    assert not at.exception, at.exception
