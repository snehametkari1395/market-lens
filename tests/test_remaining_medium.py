import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from finance_research import agent as agent_module
from finance_research import mcp_tools
from finance_research.agent import anthropic_tool_definitions, run_research_agent
from finance_research.auth import AuthError, UserStore, hash_password, verify_password
from finance_research.conversation import answer_help_question
from finance_research.monitoring import Monitor, redact, summarize_events
from finance_research.providers import ANTHROPIC_PRICES, PROVIDERS, price_for
from finance_research.rag import load_refresh_status, refresh_knowledge_base
from finance_research.ratelimit import RateLimiter, RateLimitExceeded, RateLimits, mask_key
from finance_research.usage import summarize_usage


@tool
def fetch_sec_financials(ticker: str, metric: str, years: int = 5) -> str:
    """Fetch annual financial figures from SEC company facts."""
    return json.dumps({"records": [{"ticker": ticker, "metric": metric, "value_usd": 1.0}]})


# --- Multi-model support --------------------------------------------------------


def test_providers_and_prices() -> None:
    assert set(PROVIDERS) == {"openrouter", "openai", "anthropic"}
    assert PROVIDERS["anthropic"].default_model == "claude-opus-5-5"
    openrouter_prices = {"openai/gpt-4o-mini": {"input": 1e-7, "output": 4e-7}}
    assert price_for("openai", "gpt-4o-mini", openrouter_prices) == openrouter_prices["openai/gpt-4o-mini"]
    assert price_for("openrouter", "openai/gpt-4o-mini", openrouter_prices) == openrouter_prices["openai/gpt-4o-mini"]
    assert price_for("anthropic", "claude-opus-5-5", {}) == ANTHROPIC_PRICES["claude-opus-5-5"]
    assert price_for("anthropic", "unknown", {}) is None


def test_anthropic_tool_definitions_use_input_schema() -> None:
    [definition] = anthropic_tool_definitions([fetch_sec_financials])
    assert definition["name"] == "fetch_sec_financials"
    assert definition["input_schema"]["type"] == "object"
    assert set(definition["input_schema"]["properties"]) == {"ticker", "metric", "years"}


class _FakeAnthropic:
    """Stands in for anthropic.Anthropic: first asks for a tool, then answers."""

    calls: list[dict] = []

    def __init__(self, api_key: str) -> None:
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        _FakeAnthropic.calls.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot, as sent
        usage = SimpleNamespace(input_tokens=100, output_tokens=20)
        if len(_FakeAnthropic.calls) == 1:
            block = SimpleNamespace(type="tool_use", id="toolu_1", name="fetch_sec_financials",
                                    input={"ticker": "AAPL", "metric": "revenue"})
            return SimpleNamespace(stop_reason="tool_use", content=[block], usage=usage)
        text = SimpleNamespace(type="text", text="Apple revenue was reported (SEC).")
        return SimpleNamespace(stop_reason="end_turn", content=[text], usage=usage)


def test_anthropic_loop_runs_tools_and_reports_usage(monkeypatch) -> None:
    import anthropic

    _FakeAnthropic.calls = []
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    answer, messages = run_research_agent(
        "Apple revenue?", "sk-ant-test", "claude-opus-5-5", [fetch_sec_financials], provider="anthropic"
    )
    assert answer == "Apple revenue was reported (SEC)."
    first, second = _FakeAnthropic.calls
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default"
    assert "temperature" not in first
    tool_result_message = second["messages"][-1]
    assert tool_result_message["role"] == "user"
    assert tool_result_message["content"][0]["type"] == "tool_result"
    assert tool_result_message["content"][0]["tool_use_id"] == "toolu_1"
    assert [m.name for m in messages if isinstance(m, ToolMessage)] == ["fetch_sec_financials"]
    assert summarize_usage(messages)["total_tokens"] == 240


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_research_agent("Apple revenue?", "key", "model", [], provider="nope")


def test_tool_errors_are_returned_to_the_model() -> None:
    @tool
    def broken(ticker: str) -> str:
        """Always fails."""
        raise ValueError("bad ticker")

    result, is_error = agent_module._run_tool({"broken": broken}, "broken", {"ticker": "X"})
    assert is_error and "bad ticker" in json.loads(result)["error"]


# --- Real-time knowledge base updates ---------------------------------------------


class _FakeStore:
    def __init__(self, indexed_ids: set[str]) -> None:
        self.ids = indexed_ids

    def get(self, ids, include):
        return {"ids": [i for i in ids if i in self.ids]}


def test_refresh_only_adds_new_filings(monkeypatch, tmp_path) -> None:
    filings = [
        {"ticker": "AAPL", "report_year": 2025, "accession": "0002", "filed": "2025-11-01"},
        {"ticker": "AAPL", "report_year": 2024, "accession": "0001", "filed": "2024-11-01"},
    ]
    import finance_research.rag as rag

    monkeypatch.setattr(rag, "get_recent_10k_filings", lambda ticker, years, ua: filings[:years])
    ingested = []
    monkeypatch.setattr(rag, "ingest_recent_filings", lambda *a, **k: ingested.append(a) or 42)
    store = _FakeStore({"AAPL-2024-0001-0"})
    status_path = tmp_path / "status.json"

    report = refresh_knowledge_base(["AAPL"], "ua", store, status_path, years=2)
    assert report[0]["new_filings"] == ["FY2025 (filed 2025-11-01)"]
    assert report[0]["chunks_added"] == 42
    assert load_refresh_status(status_path)["AAPL"]["latest_filing"] == "FY2025 (filed 2025-11-01)"

    store.ids.add("AAPL-2025-0002-0")
    report = refresh_knowledge_base(["AAPL"], "ua", store, status_path, years=2)
    assert report[0]["new_filings"] == [] and report[0]["chunks_added"] == 0
    assert len(ingested) == 1  # nothing re-downloaded on the second check


# --- Authentication and personalisation -----------------------------------------


def test_password_hashing() -> None:
    stored = hash_password("correct horse")
    assert stored.startswith("pbkdf2_sha256$") and "correct horse" not in stored
    assert verify_password("correct horse", stored)
    assert not verify_password("wrong", stored)


def test_user_store_signup_login_and_preferences(tmp_path) -> None:
    store = UserStore(tmp_path / "users.json")
    profile = store.create_user("Alex_1", "password123", "Alex")
    assert profile["preferences"]["display_name"] == "Alex"
    with pytest.raises(AuthError):
        store.create_user("alex_1", "password123")  # usernames are case-insensitive
    with pytest.raises(AuthError):
        store.create_user("ab", "password123")
    with pytest.raises(AuthError):
        store.create_user("newuser", "short")
    with pytest.raises(AuthError):
        store.authenticate("Alex_1", "wrong-password")
    with pytest.raises(AuthError):
        store.authenticate("nobody", "password123")
    assert store.authenticate("ALEX_1", "password123")["username"] == "Alex_1"

    updated = store.update_preferences("Alex_1", {"watchlist": ["NVDA"], "provider": "anthropic", "is_admin": True})
    assert updated["preferences"]["watchlist"] == ["NVDA"]
    assert "is_admin" not in updated["preferences"]
    assert "password123" not in (tmp_path / "users.json").read_text()


# --- MCP ---------------------------------------------------------------------------


def test_mcp_url_validation() -> None:
    assert mcp_tools.validate_mcp_url(" https://mcp.deepwiki.com/mcp ") == "https://mcp.deepwiki.com/mcp"
    for bad in ("http://example.com/mcp", "https://localhost/mcp", "ftp://x", "not a url"):
        with pytest.raises(ValueError):
            mcp_tools.validate_mcp_url(bad)


def test_mcp_tools_are_wrapped_and_sanitised(monkeypatch) -> None:
    calls = []

    def fake_call(url, name, arguments, trusted=False):
        calls.append((name, arguments))
        return {"tool": name, "result": "ok"}

    monkeypatch.setattr(mcp_tools, "call_mcp_tool", fake_call)
    specs = [{"name": "ask_wiki_question", "description": "Ask about a repo",
              "input_schema": {"type": "object", "properties": {"repoName": {"type": "string"}, "question": {"type": "string"}},
                               "required": ["repoName", "question"]}}]
    [wrapped] = mcp_tools.build_mcp_tools("https://mcp.deepwiki.com/mcp", specs)
    assert wrapped.name == "mcp_deepwiki_ask_wiki_question"
    assert "untrusted" in wrapped.description
    out = json.loads(wrapped.invoke({"repoName": "a/b", "question": "what?"}))
    assert out["result"] == "ok"
    assert calls == [("ask_wiki_question", {"repoName": "a/b", "question": "what?"})]
    [definition] = anthropic_tool_definitions([wrapped])
    assert set(definition["input_schema"]["properties"]) == {"repoName", "question"}


# --- Rate limiting and key management ---------------------------------------------


def test_rate_limiter_per_minute_hour_and_budget() -> None:
    now = [1000.0]
    limiter = RateLimiter(RateLimits(per_minute=2, per_hour=3, daily_budget_usd=0.05), clock=lambda: now[0])
    limiter.check("u")
    limiter.check("u")
    with pytest.raises(RateLimitExceeded) as minute:
        limiter.check("u")
    assert 0 < minute.value.retry_after <= 60
    limiter.check("other-user")  # limits are per user

    now[0] += 61
    limiter.check("u")
    with pytest.raises(RateLimitExceeded, match="per hour"):
        limiter.check("u")

    now[0] += 3600
    limiter.record_cost("u", 0.06)
    with pytest.raises(RateLimitExceeded, match="Daily spending limit"):
        limiter.check("u")
    status = limiter.status("u")
    assert status["spent_today"] == pytest.approx(0.06)


def test_mask_key_never_shows_whole_key() -> None:
    assert mask_key("") == "not set"
    assert mask_key("sk-or-v1-abcdef1234567890") == "sk-or…7890"
    assert mask_key("short") == "•••••"


# --- Logging and monitoring -------------------------------------------------------


def test_redaction_removes_keys_and_emails() -> None:
    text = redact("key sk-ant-api03-SECRETSECRET user a.b@example.com apikey=abc123")
    assert "SECRET" not in text and "example.com" not in text and "abc123" not in text


def test_monitor_writes_and_summarises_events(tmp_path) -> None:
    monitor = Monitor(tmp_path / "logs")
    monitor.event("chat_answer", latency_s=2.0, total_tokens=100, cost_usd=0.001, tools=["compare_companies"],
                  question_preview="email me at x@y.com")
    monitor.event("chat_answer", latency_s=4.0, total_tokens=50, cost_usd=None, tools=[])
    monitor.event("chat_error", level="error", error="boom")
    monitor.event("rate_limited", level="warning")
    events = monitor.read_events()
    assert len(events) == 4
    assert "x@y.com" not in (tmp_path / "logs" / "events.jsonl").read_text()
    assert (tmp_path / "logs" / "market_lens.log").exists()
    stats = summarize_events(events)
    assert stats["answers"] == 2 and stats["errors"] == 1 and stats["blocked"] == 1
    assert stats["error_rate"] == pytest.approx(1 / 3)
    assert stats["tokens"] == 150 and stats["cost_usd"] == pytest.approx(0.001)
    assert stats["tool_counts"] == {"compare_companies": 1}


def test_help_guide_covers_remaining_features() -> None:
    assert answer_help_question("How do I use Claude or OpenAI?")["title"] == "Choosing a model provider"
    assert answer_help_question("How do I save my preferences?")["title"] == "Accounts and saved preferences"
    assert answer_help_question("What is MCP?")["title"] == "Remote MCP tools"
    assert answer_help_question("Where are the logs?")["title"] == "Monitoring and logs"
    assert answer_help_question("How do I get new filings?")["title"] == "Keeping filings up to date"
