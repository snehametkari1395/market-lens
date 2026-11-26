"""Check the figures in an assistant answer against the data its tools returned.

Citations show what the tools fetched, not whether the answer quotes it correctly. This module pulls
the numbers out of the answer text, puts them in common units, and looks for each one in the tool
results, allowing for rounding. Each figure ends up as:

- "matched":    found in the tool data (e.g. "$391.0 billion" ~ 391,035,000,000 from SEC Company Facts)
- "calculated": not in the data, but equal to a growth rate or ratio computed from it
                (same company, same metric across consecutive years, or two metrics in the same year)
- "unverified": not found; it may be wrong, or worked out in a way the checker can't reproduce

It checks that a number exists in the sources, not that the answer attaches it to the right company
or year, and it does not check text-only claims.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

SCALES = {
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}
APPROXIMATE_WORDS = ("about", "around", "approximately", "roughly", "nearly", "almost", "over", "under", "more than", "less than", "~", "≈")
APPROXIMATE_TOLERANCE = 0.03  # 3% for "about $390 billion"

_FIGURE = re.compile(
    r"""
    (?P<currency>(?:US)?\$)?\s?
    (?P<sign>[-−–])?
    (?P<currency2>(?:US)?\$)?
    (?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?:\s?(?P<scale>trillion|billion|million|thousand|tn|bn|mn|[TBMK])\b)?
    \s?(?P<suffix>%|\s?percent\b|\s?per\s?cent\b|×|x\b|\s?times\b)?
    """,
    re.IGNORECASE | re.VERBOSE,
)
_URL = re.compile(r"\(?https?://\S+\)?")
_CITATION = re.compile(r"\[\d{1,2}\]")


@dataclass
class Figure:
    text: str
    number: float         # the digits as written, e.g. 391.0 in "$391.0 billion"
    value: float          # in base units: dollars, a fraction for percentages, a plain ratio for multiples
    kind: str             # money | percent | multiple | number
    decimals: int
    scale: float
    approximate: bool
    status: str = "unverified"
    source: str = ""
    candidates: list[float] = field(default_factory=list, repr=False)


def _skip(text: str, match: re.Match[str], value: float, has_unit: bool, raw: str) -> bool:
    before = text[max(0, match.start() - 8) : match.start()]
    after = text[match.end() : match.end() + 3]
    if re.search(r"(Item\s*|FY\s?|Q|H|Form\s*|No\.\s*|#)$", before, re.IGNORECASE):
        return True  # Item 2.02, FY2024, Q3, Form 8, #2
    if re.match(r"\s?-\s?[KQ]\b", after):
        return True  # 10-K, 10-Q, 8-K
    if has_unit:
        return False
    line_start = text.rfind("\n", 0, match.start()) + 1
    if re.fullmatch(r"\s*", text[line_start : match.start()]) and re.match(r"[.)]\s", text[match.end() : match.end() + 2]):
        return True  # numbered list item "1. "
    if "." not in raw and "," not in raw:
        if 1900 <= value <= 2100:
            return True  # a year
        if value < 100:
            return True  # small counts ("3 companies", "5 years") aren't financial figures
    return False


def extract_figures(text: str) -> list[Figure]:
    """Numbers worth checking in an answer, with their units."""
    clean = _CITATION.sub(" ", _URL.sub(" ", text or ""))
    figures: list[Figure] = []
    for match in _FIGURE.finditer(clean):
        raw = match.group("number")
        number = float(raw.replace(",", ""))
        currency = bool(match.group("currency") or match.group("currency2"))
        scale_word = (match.group("scale") or "").lower()
        suffix = (match.group("suffix") or "").strip().lower()
        # A bare "M"/"B" after a number only counts as a scale when it looks like money.
        if scale_word in {"t", "b", "m", "k"} and not currency:
            scale_word = ""
        scale = SCALES.get(scale_word, 1.0)
        has_unit = currency or bool(scale_word) or bool(suffix)
        if _skip(clean, match, number, has_unit, raw):
            continue
        sign = -1.0 if match.group("sign") else 1.0
        if suffix in {"%", "percent", "per cent"} or "percent" in suffix:
            kind, value = "percent", sign * number / 100
        elif suffix in {"×", "x", "times"}:
            kind, value = "multiple", sign * number
        elif currency:
            kind, value = "money", sign * number * scale
        else:
            kind, value = "number", sign * number * scale
        context = clean[max(0, match.start() - 25) : match.start()].lower()
        figures.append(
            Figure(
                text=match.group(0).strip(),
                number=number,
                value=value,
                kind=kind,
                decimals=len(raw.split(".")[1]) if "." in raw else 0,
                scale=scale,
                approximate=any(word in context[-14:] for word in APPROXIMATE_WORDS),
            )
        )
    return figures


# ---------------------------------------------------------------------------
# Known numbers from tool results
# ---------------------------------------------------------------------------


@dataclass
class Known:
    value: float
    label: str


def _record_label(row: dict[str, Any]) -> str:
    year = f"FY{row['fiscal_year']}" if row.get("fiscal_year") else ""
    return " · ".join(str(part) for part in (row.get("ticker"), year, row.get("metric_label") or row.get("metric")) if part)


def _walk(node: Any, label: str, out: list[Known]) -> None:
    if isinstance(node, bool) or node is None:
        return
    if isinstance(node, (int, float)):
        out.append(Known(float(node), label))
    elif isinstance(node, str):
        for figure in _numbers_in_text(node):
            out.append(Known(figure, label))
    elif isinstance(node, dict):
        if "value_usd" in node and ("metric" in node or "metric_label" in node):
            label = _record_label(node) + " (SEC)"
        for key, value in node.items():
            if key in {"source_url", "document_url", "filing_url", "url", "accession", "filed", "fiscal_year", "report_year", "rank", "cik"}:
                continue
            _walk(value, label, out)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _walk(value, label, out)


def _numbers_in_text(text: str) -> list[float]:
    """Every number in a piece of tool text (filing excerpts, quote strings), in base units."""
    values = []
    for match in _FIGURE.finditer(text):
        number = float(match.group("number").replace(",", ""))
        scale = SCALES.get((match.group("scale") or "").lower(), 1.0)
        values.append(number * scale)
        if (match.group("suffix") or "").strip() in {"%", "percent"}:
            values.append(number / 100)
    return values


def known_numbers(tool_results: Iterable[dict[str, Any]]) -> list[Known]:
    out: list[Known] = []
    for result in tool_results:
        content = result.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except ValueError:
                pass
        name = str(result.get("name", "tool"))
        if name == "search_filing_knowledge" and isinstance(content, dict):
            for passage in content.get("results", []) or []:
                tag = f"{passage.get('ticker', '')} FY{passage.get('report_year', '')} 10-K excerpt".strip()
                _walk(passage.get("text", ""), tag, out)
            continue
        _walk(content, name.replace("_", " "), out)
    return out


def _metric_label(metric: str) -> str:
    from finance_research.sec_client import METRICS

    return str(METRICS.get(metric, {}).get("label", metric.replace("_", " "))).lower()


def _derived_numbers(tool_results: Iterable[dict[str, Any]]) -> list[Known]:
    """Growth rates and ratios the answer may have calculated from SEC records (meaningful pairs only)."""
    rows = []
    for result in tool_results:
        content = result.get("content")
        if isinstance(content, dict):
            rows += [row for row in content.get("records", []) or [] if isinstance(row, dict) and row.get("value_usd") is not None]
    derived: list[Known] = []
    by_series: dict[tuple[str, str], dict[int, float]] = {}
    by_year: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        ticker, metric, year = str(row.get("ticker")), str(row.get("metric")), row.get("fiscal_year")
        if not isinstance(year, int):
            continue
        by_series.setdefault((ticker, metric), {})[year] = float(row["value_usd"])
        by_year.setdefault((ticker, year), {})[metric] = float(row["value_usd"])
    for (ticker, metric), series in by_series.items():
        years = sorted(series)
        for start_index, start in enumerate(years):
            for end in years[start_index + 1 :]:
                first, last = series[start], series[end]
                if first:
                    derived.append(Known(last / first - 1, f"{ticker} {_metric_label(metric)} growth, FY{start}–FY{end}"))
                    derived.append(Known(last - first, f"{ticker} {_metric_label(metric)} change, FY{start}–FY{end}"))
                if first > 0 and last > 0 and end > start:
                    derived.append(Known((last / first) ** (1 / (end - start)) - 1, f"{ticker} {_metric_label(metric)} CAGR, FY{start}–FY{end}"))
    for (ticker, year), metrics in by_year.items():
        for top, top_value in metrics.items():
            for bottom, bottom_value in metrics.items():
                if top != bottom and bottom_value:
                    derived.append(Known(top_value / bottom_value, f"{ticker} FY{year} {_metric_label(top)} ÷ {_metric_label(bottom)}"))
    # Differences between companies for the same metric and year ("$12B more than").
    for (ticker, year), metrics in by_year.items():
        for (other, other_year), other_metrics in by_year.items():
            if other_year == year and other > ticker:
                for metric in set(metrics) & set(other_metrics):
                    derived.append(Known(metrics[metric] - other_metrics[metric], f"{ticker} minus {other} {_metric_label(metric)}, FY{year}"))
                    if other_metrics[metric]:
                        derived.append(Known(metrics[metric] / other_metrics[metric], f"{ticker} ÷ {other} {_metric_label(metric)}, FY{year}"))
    return derived


def _candidates(figure: Figure) -> list[float]:
    """What the figure could mean in base units. Unscaled numbers may be in thousands, millions or billions."""
    values = [figure.value]
    if figure.kind == "percent":
        values.append(figure.value * 100)  # tools sometimes return "value_percent": 24.31
    if figure.kind in {"money", "number"} and figure.scale == 1.0:
        values += [figure.value * factor for factor in (1e3, 1e6, 1e9)]
    if figure.kind == "multiple":
        values.append(figure.value / 100)
    return values


def _tolerance(figure: Figure, candidate: float) -> float:
    """Half of the last digit shown, in the candidate's units: "$391.0 billion" allows ±$0.05 billion."""
    factor = abs(candidate / figure.number) if figure.number else 1.0
    tolerance = 0.5 * 10 ** (-figure.decimals) * factor
    if figure.approximate:
        tolerance = max(tolerance, abs(candidate) * APPROXIMATE_TOLERANCE)
    return tolerance * 1.0001 + 1e-12


def _find(figure: Figure, pool: list[Known]) -> Known | None:
    # Compare sizes, not signs: "a loss of $2.1 billion" should match -2.1e9, "fell 5%" a growth of -0.05.
    for candidate in _candidates(figure):
        tolerance = _tolerance(figure, candidate)
        for known in pool:
            if abs(abs(known.value) - abs(candidate)) <= tolerance:
                return known
    return None


def verify_answer(answer: str, tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Check every figure in the answer. Returns counts plus one row per figure."""
    figures = extract_figures(answer)
    known = known_numbers(tool_results)
    derived = _derived_numbers(tool_results)
    for figure in figures:
        figure.candidates = _candidates(figure)
        match = _find(figure, known)
        if match:
            figure.status, figure.source = "matched", match.label
            continue
        match = _find(figure, derived)
        if match:
            figure.status, figure.source = "calculated", match.label
    counts = {status: sum(figure.status == status for figure in figures) for status in ("matched", "calculated", "unverified")}
    return {
        "checked": len(figures),
        **counts,
        "has_tool_data": bool(known),
        "figures": [
            {key: value for key, value in asdict(figure).items() if key != "candidates"} for figure in figures
        ],
    }
