"""Runs scripts/smoke_test.py end to end with fakes, so the real run only fails on real problems."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("smoke_test", ROOT / "scripts" / "smoke_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _agent(question, api_key, model, tools, provider="openrouter"):
    if "8-K" in question:
        name, content = "get_company_events", {"events": [{"filed": "2026-11-02", "headline": "Quarterly or annual results"}]}
        answer = "Apple reported quarterly results on 2026-11-02."
    elif "10-K" in question:
        name, content = "search_filing_knowledge", {"results": [{"ticker": "AAPL", "report_year": 2026, "text": "Supply disruptions..."}]}
        answer = "Apple warns that supply disruptions could hurt results."
    else:
        name = "fetch_sec_financials"
        content = {"records": [{"ticker": "AAPL", "metric": "revenue", "metric_label": "Revenue", "fiscal_year": 2026, "value_usd": 478.83e9}]}
        answer = "Revenue was $478.8 billion."
    call = AIMessage(content="", tool_calls=[{"id": "c1", "name": name, "args": {}}], usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})
    final = AIMessage(content=answer, usage_metadata={"input_tokens": 200, "output_tokens": 50, "total_tokens": 250})
    return answer, [call, ToolMessage(content=json.dumps(content), tool_call_id="c1", name=name), final]


def test_smoke_script_runs_every_step_with_fakes(monkeypatch, tmp_path, capsys):
    script = _load_script()
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("SEC_USER_AGENT_EMAIL", "tester@example.com")
    monkeypatch.setattr("sys.argv", ["smoke_test.py", "--scratch"])

    import finance_research.agent as agent
    import finance_research.events as events
    import finance_research.rag as rag
    import finance_research.ratelimit as ratelimit
    import finance_research.sec_client as sec_client
    import finance_research.usage as usage

    monkeypatch.setattr(ratelimit, "verify_api_key", lambda provider, key: (True, "Key accepted."))
    monkeypatch.setattr(usage, "fetch_openrouter_pricing", lambda: {"openai/gpt-4o-mini": {"input": 1e-7, "output": 4e-7}})
    monkeypatch.setattr(rag, "create_vector_store", lambda key, directory: object())
    monkeypatch.setattr(sec_client, "get_recent_10k_filings", lambda t, y, ua: [{"report_year": 2026, "filed": "2026-10-31"}])
    monkeypatch.setattr(rag, "ingest_recent_filings", lambda t, y, ua, store, skip_indexed: 320)
    monkeypatch.setattr(rag, "search_filings", lambda store, t, q, strategy: [{"section": "Risk Factors", "relevance": 0.8}])
    monkeypatch.setattr(agent, "run_research_agent", _agent)
    monkeypatch.setattr(agent, "complete", lambda *args, **kwargs: ("- Revenue rose 8% to $102.5 billion in the quarter.", {"input_tokens": 900, "output_tokens": 60}))
    monkeypatch.setattr(events, "get_company_events", lambda t, ua, days, limit: [{"filed": "2026-11-02", "headline": "Quarterly or annual results", "category": "Earnings", "items": [{"code": "2.02", "label": "Results"}], "company": "Apple", "ticker": "AAPL", "form": "8-K"}])
    monkeypatch.setattr(events, "event_text", lambda event, ua: "Apple announced results. " * 20)

    exit_code = script.main()
    output = capsys.readouterr().out

    assert exit_code == 0, output
    assert "7 of 7 steps passed" in output
    assert "sk-test-not-real" not in output
    report = json.loads((tmp_path / "data" / "smoke" / "last_report.json").read_text(encoding="utf-8"))
    assert [step["status"] for step in report["steps"]] == ["PASS"] * 7
    assert "sk-test-not-real" not in json.dumps(report)
    assert report["answers"][0]["figure_check"]["matched"] == 1
    assert report["total_cost_usd"] > 0


def test_smoke_script_reports_a_failed_step(monkeypatch, tmp_path, capsys):
    script = _load_script()
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("SEC_USER_AGENT_EMAIL", "tester@example.com")
    monkeypatch.setattr("sys.argv", ["smoke_test.py", "--scratch", "--skip-index"])

    import finance_research.agent as agent
    import finance_research.rag as rag
    import finance_research.ratelimit as ratelimit
    import finance_research.usage as usage

    monkeypatch.setattr(ratelimit, "verify_api_key", lambda provider, key: (False, "Invalid key."))
    monkeypatch.setattr(usage, "fetch_openrouter_pricing", lambda: {})
    monkeypatch.setattr(rag, "create_vector_store", lambda key, directory: object())
    monkeypatch.setattr(rag, "search_filings", lambda store, t, q, strategy: [])

    def broken_agent(*args, **kwargs):
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(agent, "run_research_agent", broken_agent)
    monkeypatch.setattr("finance_research.events.get_company_events", lambda *a, **k: [])

    assert script.main() == 1
    output = capsys.readouterr().out
    assert "FAIL  1. API key" in output and "401 Unauthorized" in output
