import asyncio
import json

import pandas as pd
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from finance_research.analytics import add_ratios, build_panel, cagr, indexed_series, kpi_table, peer_ranks
from finance_research.evaluation import (
    average_precision,
    chunk_id,
    evaluate_generation,
    evaluate_retrieval,
    generate_test_set,
    list_runs,
    parse_json_reply,
    save_run,
)
from finance_research.experiments import Experiment, assign_variant, summarize_experiment, two_proportion_z_test, verdict
from finance_research.i18n import LANGUAGES, STRINGS, language_instruction, translate
from finance_research.rag import BM25, reciprocal_rank_fusion, search_filings, tokenize
from finance_research.scheduler import KnowledgeBaseScheduler, SchedulerConfig, task_scheduler_command
from finance_research.tools import ToolKit

PASSAGES = [
    ("Item 1A — Risk Factors", "Our supply chain depends on contract manufacturers in Asia, and disruptions could delay products. " * 6),
    ("Item 7 — Management's Discussion", "Services revenue grew because of App Store, advertising and cloud subscriptions. " * 6),
    ("Item 1A — Risk Factors", "We face intense competition in smartphones and personal computers from many rivals. " * 6),
    ("Item 8 — Financial Statements", "The company repurchased shares and paid dividends totalling ninety billion dollars. " * 6),
    ("Item 1 — Business", "Employees numbered approximately one hundred sixty thousand full-time equivalents worldwide. " * 6),
]


@pytest.fixture()
def store(tmp_path):
    chroma = Chroma(
        collection_name="sec_filings",
        embedding_function=DeterministicFakeEmbedding(size=64),
        persist_directory=str(tmp_path / "chroma"),
    )
    documents = [
        Document(
            page_content=text,
            metadata={"ticker": "AAPL", "company": "Apple Inc.", "report_year": 2024, "section": section,
                      "chunk_index": index, "source_url": "https://www.sec.gov/aapl-2024", "accession": "0001"},
        )
        for index, (section, text) in enumerate(PASSAGES)
    ]
    chroma.add_documents(documents, ids=[f"AAPL-2024-0001-{i}" for i in range(len(documents))])
    return chroma


# --- Hybrid search -----------------------------------------------------------------


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("What are the supply-chain risks in Asia?") == ["supply-chain", "risks", "asia"]


def test_bm25_ranks_matching_document_first() -> None:
    index = BM25(["apple supply chain asia", "services revenue growth", "supply of dividends"])
    scores = index.scores("supply chain")
    assert scores[0] > scores[2] > scores[1] == 0


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    a = {"text": "a", "ticker": "X", "report_year": 1, "source_url": "u", "chunk_index": 0}
    b = {"text": "b", "ticker": "X", "report_year": 1, "source_url": "u", "chunk_index": 1}
    c = {"text": "c", "ticker": "X", "report_year": 1, "source_url": "u", "chunk_index": 2}
    fused = reciprocal_rank_fusion({"vector": [a, b], "keyword": [b, c]})
    assert [item["text"] for item in fused] == ["b", "a", "c"]
    assert fused[0]["vector_rank"] == 2 and fused[0]["keyword_rank"] == 1


def test_search_strategies_on_real_index(store) -> None:
    keyword = search_filings(store, "aapl", "employees full-time equivalents", strategy="keyword")
    assert keyword[0]["chunk_index"] == 4 and keyword[0]["relevance"] == 1.0
    hybrid = search_filings(store, "AAPL", "employees full-time equivalents", strategy="hybrid", count=3)
    assert 4 in [item["chunk_index"] for item in hybrid]
    assert all(0 <= item["relevance"] <= 1 for item in hybrid)
    assert all(item["strategy"] == "hybrid" and "rrf_score" in item for item in hybrid)
    vector = search_filings(store, "AAPL", "employees", strategy="vector", count=2)
    assert len(vector) == 2 and "cosine" in vector[0]
    with pytest.raises(ValueError):
        search_filings(store, "AAPL", "x", strategy="magic")


def test_toolkit_search_sanitises_and_reports_strategy(store) -> None:
    kit = ToolKit("ua", store, search_strategy="keyword")
    payload = kit.search_filing_knowledge("AAPL", "dividends repurchased")
    assert payload["retrieval"]["strategy"] == "keyword"
    assert payload["results"][0]["text"].startswith("<<untrusted_external_text>>")
    assert kit.search_filing_knowledge("AAPL", "dividends", strategy="vector")["retrieval"]["strategy"] == "vector"


def test_toolkit_search_falls_back_when_the_model_guesses_a_missing_year(store) -> None:
    kit = ToolKit("ua", store, search_strategy="keyword")
    payload = kit.search_filing_knowledge("AAPL", "dividends repurchased", report_year=1999)
    assert payload["results"], "should fall back to the indexed years"
    assert "FY1999" in payload["note"]
    assert kit.search_filing_knowledge("MSFT", "dividends", report_year=1999)["results"] == []  # nothing indexed for MSFT


# --- A/B testing ---------------------------------------------------------------


def test_variant_assignment_is_stable_and_respects_share() -> None:
    experiment = Experiment(share_b=0.5)
    assert assign_variant("alice", experiment) == assign_variant("alice", experiment)
    labels = [assign_variant(f"user{i}", experiment)[0] for i in range(2000)]
    assert 0.45 < labels.count("B") / len(labels) < 0.55
    assert all(assign_variant(f"u{i}", Experiment(share_b=0.0))[0] == "A" for i in range(50))
    with pytest.raises(ValueError):
        Experiment(variant_a="hybrid", variant_b="hybrid")


def test_z_test_and_verdict() -> None:
    result = two_proportion_z_test(40, 100, 60, 100)
    assert result["p_value"] < 0.01 and result["difference"] == pytest.approx(0.2)
    assert two_proportion_z_test(0, 0, 1, 1) is None

    events = [{"kind": "chat_answer", "experiment": "e", "variant": v, "username": f"u{i}", "latency_s": 1, "top_relevance": 0.5,
               "strategy": "vector" if v == "A" else "hybrid", "answer_id": f"{v}{i}"} for i in range(30) for v in "AB"]
    events += [{"kind": "answer_feedback", "experiment": "e", "variant": "A", "answer_id": f"A{i}", "score": int(i < 12)} for i in range(30)]
    events += [{"kind": "answer_feedback", "experiment": "e", "variant": "B", "answer_id": f"B{i}", "score": int(i < 25)} for i in range(30)]
    events.append({"kind": "answer_feedback", "experiment": "e", "variant": "B", "answer_id": "B0", "score": 0})  # changed vote
    summary = summarize_experiment(events, "e")
    assert summary["variants"]["A"]["votes"] == 30 and summary["variants"]["B"]["thumbs_up"] == 24
    assert summary["variants"]["B"]["strategy"] == "hybrid"
    assert "Variant B" in verdict(summary) and "significantly" in verdict(summary)
    assert "Not enough feedback" in verdict(summarize_experiment(events[:10], "e"))


# --- Automated knowledge-base updates --------------------------------------------


def test_scheduler_runs_updates_and_records_status() -> None:
    calls, events = [], []
    scheduler = KnowledgeBaseScheduler(
        on_event=lambda kind, **fields: events.append(kind),
        update=lambda config: calls.append(config) or [{"ticker": "AAPL", "new_filings": ["FY2025"], "chunks_added": 9, "error": ""}],
    )
    with pytest.raises(ValueError):
        scheduler.start(SchedulerConfig(interval_hours=24, sec_email="", openrouter_key="k"))
    scheduler.start(SchedulerConfig(interval_hours=24, sec_email="a@b.co", openrouter_key="k"))
    report = scheduler.run_once()
    assert report[0]["chunks_added"] == 9 and len(calls) == 1
    status = scheduler.status()
    assert status["enabled"] and status["interval_hours"] == 24 and status["next_run"]
    assert status["last_run"]["new_filings"] == 1 and status["last_run"]["ok"]
    scheduler.stop()
    assert not scheduler.status()["enabled"]
    assert events == ["kb_scheduler_started", "kb_refresh", "kb_scheduler_stopped"]


def test_scheduler_survives_failures() -> None:
    def boom(config):
        raise RuntimeError("SEC is down")

    scheduler = KnowledgeBaseScheduler(update=boom)
    scheduler.start(SchedulerConfig(interval_hours=1, sec_email="a@b.co", openrouter_key="k"))
    assert scheduler.run_once() == []
    assert scheduler.status()["last_run"]["ok"] is False and "SEC is down" in scheduler.status()["last_run"]["error"]
    scheduler.stop()


def test_task_scheduler_command() -> None:
    command = task_scheduler_command(r"C:\app\.venv\Scripts\python.exe", r"C:\app", hour=6)
    assert command.startswith("schtasks /Create /SC DAILY /ST 06:00") and "finance_research.scheduler" in command


# --- Multi-language ------------------------------------------------------------------


def test_every_string_is_translated_into_every_language() -> None:
    for key, entry in STRINGS.items():
        assert set(entry) == set(LANGUAGES), f"{key} is missing a language"
    assert translate("tab_assistant", "es") == "Asistente"
    assert translate("tab_assistant", "xx") == "Research assistant"
    assert translate("no_such_key", "fr") == "no_such_key"


def test_language_instruction() -> None:
    assert language_instruction("en") == ""
    instruction = language_instruction("hi")
    assert "Hindi" in instruction and "English" in instruction  # answers in Hindi, searches in English


# --- Analytics ------------------------------------------------------------------------


def _records():
    rows = []
    data = {
        "AAA": {"revenue": [100, 120, 150], "net_income": [10, 15, 30], "operating_income": [12, 18, 36],
                "research_and_development": [5, 6, 7], "total_assets": [200, 220, 260], "operating_cash_flow": [14, 18, 33]},
        "BBB": {"revenue": [80, 76, 84], "net_income": [8, -2, 4], "operating_income": [9, 1, 6],
                "research_and_development": [1, 1, 1], "total_assets": [100, 100, 100], "operating_cash_flow": [9, 3, 7]},
    }
    for ticker, metrics in data.items():
        for metric, values in metrics.items():
            for year, value in zip((2022, 2023, 2024), values):
                rows.append({"ticker": ticker, "company": ticker + " Inc", "metric": metric, "fiscal_year": year, "value_usd": value * 1e9})
    return rows


def test_analytics_kpis_and_ranks() -> None:
    ratios = add_ratios(build_panel(_records()))
    aaa_2024 = ratios[(ratios.ticker == "AAA") & (ratios.fiscal_year == 2024)].iloc[0]
    assert aaa_2024["net_margin"] == pytest.approx(0.2)
    assert aaa_2024["revenue_growth"] == pytest.approx(0.25)
    bbb_2023 = ratios[(ratios.ticker == "BBB") & (ratios.fiscal_year == 2023)].iloc[0]
    assert pd.isna(bbb_2023["cash_conversion"])  # undefined for a loss year

    kpis = kpi_table(ratios).set_index("ticker")
    assert kpis.loc["AAA", "revenue_cagr"] == pytest.approx(cagr(100, 150, 2))
    assert kpis.loc["AAA", "cagr_years"] == 2
    ranks = peer_ranks(kpi_table(ratios))
    assert ranks.iloc[0]["ticker"] == "AAA" and ranks.iloc[0]["net_margin"] == 1

    indexed = indexed_series(ratios)
    assert indexed[(indexed.ticker == "AAA") & (indexed.fiscal_year == 2024)]["index"].iloc[0] == pytest.approx(150)
    assert indexed[(indexed.ticker == "BBB") & (indexed.fiscal_year == 2022)]["index"].iloc[0] == pytest.approx(100)
    assert cagr(0, 10, 2) is None and cagr(10, -5, 2) is None


# --- Tools as an MCP server ---------------------------------------------------------


def test_mcp_server_exposes_the_research_tools(store) -> None:
    from finance_research.mcp_server import create_server

    server = create_server(ToolKit("ua", store, search_strategy="keyword"))

    async def scenario():
        tools = await server.list_tools()
        names = sorted(tool.name for tool in tools)
        result = await server.call_tool("search_filing_knowledge", {"ticker": "AAPL", "question": "dividends repurchased"})
        return names, result

    names, result = asyncio.run(scenario())
    assert names == ["calculate_financial_ratio", "compare_companies", "fetch_sec_financials", "fetch_stock_quote", "get_company_events", "get_private_company_funding", "get_upcoming_earnings", "get_valuation", "search_filing_knowledge"]
    text = json.dumps(result, default=str)
    assert "dividends" in text and "keyword" in text


# --- RAG evaluation ----------------------------------------------------------------


def test_parse_json_reply_tolerates_wrapping() -> None:
    assert parse_json_reply('Sure!\n```json\n{"question": "Q?", "answer": "A"}\n```') == {"question": "Q?", "answer": "A"}
    with pytest.raises(ValueError):
        parse_json_reply("no json here")


def test_average_precision_matches_ragas_definition() -> None:
    assert average_precision([1, 0, 1]) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert average_precision([0, 0]) == 0.0
    assert average_precision([]) is None


def test_end_to_end_evaluation(store, tmp_path) -> None:
    def question_writer(prompt):
        # Echo distinctive words from the passage so keyword search can find it.
        passage = prompt.split('"""')[1]
        words = " ".join(passage.split()[:8])
        return json.dumps({"question": f"What does Apple say: {words}?", "answer": words}), {"input_tokens": 10, "output_tokens": 5}

    test_set = generate_test_set(store, ["AAPL"], 3, question_writer)
    assert len(test_set) == 3 and all(item["ground_truth_chunk"].startswith("AAPL|2024|") for item in test_set)

    retrieval = evaluate_retrieval(store, test_set, ["keyword", "hybrid", "vector"], k=3)
    assert retrieval["metrics"]["keyword"]["hit_rate"] == 1.0
    assert retrieval["metrics"]["keyword"]["mrr"] == 1.0
    assert 0 <= retrieval["metrics"]["vector"]["hit_rate"] <= 1
    assert len(retrieval["details"]) == 9

    answerer = lambda prompt: ("An answer from the context.", {"input_tokens": 50, "output_tokens": 10})
    judge = lambda prompt: (json.dumps({"faithfulness": 1, "answer_relevancy": 0.8, "context_precision": [1, 0, 1], "answer_correctness": 0.5}),
                            {"input_tokens": 80, "output_tokens": 20})
    generation = evaluate_generation(store, test_set, ["vector"], answerer, judge, k=3)  # vector always returns k results
    metrics = generation["metrics"]["vector"]
    assert metrics["faithfulness"] == 1.0 and metrics["answer_relevancy"] == pytest.approx(0.8)
    assert metrics["context_precision"] == pytest.approx((1 + 2 / 3) / 2)
    assert metrics["scored"] == 3 and metrics["failed"] == 0
    assert generation["tokens"] == {"input_tokens": 390, "output_tokens": 90}

    path = save_run(tmp_path / "eval", {"retrieval": retrieval, "generation": generation})
    runs = list_runs(tmp_path / "eval")
    assert runs[0]["file"] == path.name and runs[0]["retrieval"]["k"] == 3
    assert chunk_id({"ticker": "A", "report_year": 1, "source_url": "u", "chunk_index": 2}) == "A|1|u|2"
