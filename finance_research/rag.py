from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finance_research.sec_client import download_filing_text, get_recent_10k_filings

SECTION_PATTERN = re.compile(
    r"(?i)\bITEM\s+(1A|1B|7A|1|7|8)\.?\s+([A-Z][A-Z ,&'’()/-]{2,})"
)


def create_vector_store(api_key: str, persist_directory: str | Path) -> Chroma:
    if not api_key.strip():
        raise ValueError("An OpenRouter API key is required to create embeddings.")
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    return Chroma(
        collection_name="sec_filings",
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def _section_for_text(text: str) -> str:
    match = SECTION_PATTERN.search(text)
    if not match:
        return "Filing narrative"
    item = match.group(1).upper()
    label = " ".join(match.group(2).split()).title()
    return f"Item {item} — {label[:90]}"


def _first_chunk_id(filing: dict[str, Any]) -> str:
    return f"{filing['ticker']}-{filing['report_year']}-{filing['accession']}-0"


def is_filing_indexed(vector_store: Chroma, filing: dict[str, Any]) -> bool:
    return bool(vector_store.get(ids=[_first_chunk_id(filing)], include=[]).get("ids"))


def ingest_recent_filings(
    ticker: str,
    years: int,
    user_agent: str,
    vector_store: Chroma,
    skip_indexed: bool = False,
) -> int:
    filings = get_recent_10k_filings(ticker, years, user_agent)
    if not filings:
        raise ValueError(f"No recent annual reports (10-K or 20-F) were found for {ticker.upper()}.")
    if skip_indexed:
        filings = [filing for filing in filings if not is_filing_indexed(vector_store, filing)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
        add_start_index=True,
    )
    documents: list[Document] = []
    ids: list[str] = []
    for filing in filings:
        text = download_filing_text(filing["document_url"], user_agent)
        base = Document(
            page_content=text,
            metadata={
                "ticker": filing["ticker"],
                "company": filing["company"],
                "report_year": filing["report_year"],
                "filed": filing["filed"],
                "accession": filing["accession"],
                "source_url": filing["filing_url"],
            },
        )
        chunks = splitter.split_documents([base])
        for index, chunk in enumerate(chunks):
            chunk.metadata["section"] = _section_for_text(chunk.page_content)
            chunk.metadata["chunk_index"] = index
            chunk_id = (
                f"{filing['ticker']}-{filing['report_year']}-"
                f"{filing['accession']}-{index}"
            )
            documents.append(chunk)
            ids.append(chunk_id)

    if documents:
        vector_store.add_documents(documents, ids=ids)
    return len(documents)


def refresh_knowledge_base(
    tickers: list[str],
    user_agent: str,
    vector_store: Chroma,
    status_path: str | Path,
    years: int = 1,
) -> list[dict[str, Any]]:
    """Check the SEC for 10-K filings newer than what's indexed and add only the new ones.

    Records each check in a small JSON status file so the app can show when the
    knowledge base was last brought up to date.
    """
    report = []
    status = load_refresh_status(status_path)
    for ticker in tickers:
        checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            latest = get_recent_10k_filings(ticker, years, user_agent)
            new = [filing for filing in latest if not is_filing_indexed(vector_store, filing)]
            chunks = ingest_recent_filings(ticker, years, user_agent, vector_store, skip_indexed=True) if new else 0
            entry = {
                "ticker": ticker,
                "checked_at": checked_at,
                "new_filings": [f"FY{filing['report_year']} (filed {filing['filed']})" for filing in new],
                "chunks_added": chunks,
                "latest_filing": f"FY{latest[0]['report_year']} (filed {latest[0]['filed']})" if latest else "none",
                "error": "",
            }
        except Exception as error:  # one failing company shouldn't stop the others
            entry = {"ticker": ticker, "checked_at": checked_at, "new_filings": [], "chunks_added": 0,
                     "latest_filing": status.get(ticker, {}).get("latest_filing", "unknown"), "error": str(error)[:200]}
        status[ticker] = entry
        report.append(entry)
    Path(status_path).parent.mkdir(parents=True, exist_ok=True)
    Path(status_path).write_text(json.dumps(status, indent=2), encoding="utf-8")
    return report


def load_refresh_status(status_path: str | Path) -> dict[str, Any]:
    path = Path(status_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Retrieval: vector, keyword (BM25) and hybrid (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

SEARCH_STRATEGIES = {
    "hybrid": "Hybrid (keyword + vector)",
    "vector": "Vector (semantic)",
    "keyword": "Keyword (BM25)",
}
DEFAULT_STRATEGY = "hybrid"
RRF_K = 60  # standard Reciprocal Rank Fusion constant
CANDIDATES_PER_METHOD = 20

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the their this to was were what which "
    "with how does do did our we us any about into than then them these those can will would could should".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS and len(token) > 1]


class BM25:
    """Okapi BM25 keyword scoring over a fixed set of chunks."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(text) for text in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_length = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0
        self.term_freqs: list[dict[str, int]] = []
        document_freq: dict[str, int] = {}
        for tokens in self.doc_tokens:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.term_freqs.append(counts)
            for token in counts:
                document_freq[token] = document_freq.get(token, 0) + 1
        total = len(self.doc_tokens)
        self.idf = {
            token: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for token, freq in document_freq.items()
        }

    def scores(self, query: str) -> list[float]:
        query_tokens = tokenize(query)
        results = []
        for counts, length in zip(self.term_freqs, self.doc_lengths):
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token)
                if not frequency:
                    continue
                norm = 1 - self.b + self.b * (length / self.avg_length if self.avg_length else 1)
                score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / (frequency + self.k1 * norm)
            results.append(score)
        return results


_bm25_cache: dict[tuple, tuple[BM25, list[str], list[dict[str, Any]]]] = {}


def _filters(ticker: str, report_year: int | None) -> dict[str, Any]:
    if report_year is None:
        return {"ticker": ticker}
    return {"$and": [{"ticker": ticker}, {"report_year": report_year}]}


def _chunk_key(metadata: dict[str, Any], text: str) -> tuple:
    return (
        metadata.get("ticker"),
        metadata.get("report_year"),
        metadata.get("accession") or metadata.get("source_url"),
        metadata.get("chunk_index"),
        text[:40],
    )


def _keyword_index(vector_store: Chroma, ticker: str, report_year: int | None):
    """BM25 index over one company's (and optionally one year's) chunks, cached until the chunk count changes."""
    stored = vector_store.get(where=_filters(ticker, report_year), include=["documents", "metadatas"])
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    cache_key = (id(vector_store), ticker, report_year, len(documents))
    if cache_key not in _bm25_cache:
        _bm25_cache[cache_key] = (BM25(documents), documents, metadatas)
    return _bm25_cache[cache_key]


def _result(document_text: str, metadata: dict[str, Any], **scores: Any) -> dict[str, Any]:
    return {
        "text": document_text,
        "ticker": metadata.get("ticker"),
        "company": metadata.get("company"),
        "report_year": metadata.get("report_year"),
        "section": metadata.get("section"),
        "chunk_index": metadata.get("chunk_index"),
        "source_url": metadata.get("source_url"),
        **scores,
    }


def vector_candidates(vector_store: Chroma, query: str, ticker: str, report_year: int | None, count: int):
    scored = vector_store.similarity_search_with_score(query, k=count, filter=_filters(ticker, report_year))
    return [
        _result(document.page_content, document.metadata, cosine=_relevance_from_distance(distance))
        for document, distance in scored
    ]


def keyword_candidates(vector_store: Chroma, query: str, ticker: str, report_year: int | None, count: int):
    index, documents, metadatas = _keyword_index(vector_store, ticker, report_year)
    scores = index.scores(query)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [_result(documents[i], metadatas[i], bm25=round(scores[i], 3)) for i in ranked[:count] if scores[i] > 0]


def reciprocal_rank_fusion(ranked_lists: dict[str, list[dict[str, Any]]], k: int = RRF_K) -> list[dict[str, Any]]:
    """Merge ranked lists: each item scores sum(1 / (k + rank)) over the lists it appears in."""
    merged: dict[tuple, dict[str, Any]] = {}
    for method, items in ranked_lists.items():
        for rank, item in enumerate(items, start=1):
            key = _chunk_key(item, item["text"])
            entry = merged.setdefault(key, {**item, "rrf_score": 0.0})
            entry.update({name: value for name, value in item.items() if name in ("cosine", "bm25")})
            entry[f"{method}_rank"] = rank
            entry["rrf_score"] += 1 / (k + rank)
    return sorted(merged.values(), key=lambda item: item["rrf_score"], reverse=True)


def search_filings(
    vector_store: Chroma,
    ticker: str,
    query: str,
    report_year: int | None = None,
    count: int = 5,
    strategy: str = DEFAULT_STRATEGY,
) -> list[dict[str, Any]]:
    normalized = ticker.strip().upper()
    if not normalized or len(normalized) > 10:
        raise ValueError("Enter a valid ticker symbol.")
    if not query.strip() or len(query) > 2000:
        raise ValueError("Enter a filing question between 1 and 2,000 characters.")
    if not 1 <= count <= 10:
        raise ValueError("Result count must be between 1 and 10.")
    if report_year is not None and not 1990 <= report_year <= 2100:
        raise ValueError("Enter a valid fiscal year.")
    if strategy not in SEARCH_STRATEGIES:
        raise ValueError(f"Unknown search strategy: {strategy}. Choose from {', '.join(SEARCH_STRATEGIES)}.")

    if strategy == "vector":
        results = vector_candidates(vector_store, query, normalized, report_year, count)
        for item in results:
            item["relevance"] = item["cosine"]
    elif strategy == "keyword":
        results = keyword_candidates(vector_store, query, normalized, report_year, count)
        top = max((item["bm25"] for item in results), default=0) or 1
        for item in results:
            item["relevance"] = round(item["bm25"] / top, 3)
    else:
        fused = reciprocal_rank_fusion(
            {
                "vector": vector_candidates(vector_store, query, normalized, report_year, CANDIDATES_PER_METHOD),
                "keyword": keyword_candidates(vector_store, query, normalized, report_year, CANDIDATES_PER_METHOD),
            }
        )
        results = fused[:count]
        best_possible = 2 / (RRF_K + 1)  # ranked first by both methods
        for item in results:
            item["relevance"] = round(item["rrf_score"] / best_possible, 3)
            item["rrf_score"] = round(item["rrf_score"], 5)

    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
        item["strategy"] = strategy
    return results


def _relevance_from_distance(distance: float) -> float:
    """Convert Chroma's default squared-L2 distance into a 0–1 relevance score.

    OpenAI embeddings are unit-length, so squared L2 distance d equals 2 − 2·cos;
    1 − d/2 recovers cosine similarity.
    """
    return round(max(0.0, min(1.0, 1 - float(distance) / 2)), 3)


def index_summary(persist_directory: str | Path) -> list[dict[str, Any]]:
    """Count indexed chunks per company and fiscal year, without needing an API key."""
    if not Path(persist_directory).exists():
        return []
    store = Chroma(collection_name="sec_filings", persist_directory=str(persist_directory))
    metadatas = store.get(include=["metadatas"]).get("metadatas") or []
    counts: dict[tuple[str, int], int] = {}
    for metadata in metadatas:
        key = (str(metadata.get("ticker", "?")), int(metadata.get("report_year") or 0))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"ticker": ticker, "report_year": year, "chunks": chunks}
        for (ticker, year), chunks in sorted(counts.items())
    ]
