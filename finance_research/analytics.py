"""Financial analytics computed from SEC company-facts records (pure pandas, no Streamlit)."""

from __future__ import annotations

from typing import Any

import pandas as pd

ANALYTICS_METRICS = (
    "revenue",
    "operating_income",
    "net_income",
    "research_and_development",
    "total_assets",
    "operating_cash_flow",
    "total_liabilities",
    "stockholders_equity",
    "cash",
    "long_term_debt",
    "eps_diluted",
)

KPI_LABELS = {
    "revenue_b": "Revenue (USD B)",
    "revenue_cagr": "Revenue CAGR",
    "revenue_growth": "Latest revenue growth",
    "operating_margin": "Operating margin",
    "net_margin": "Net margin",
    "rd_intensity": "R&D intensity",
    "return_on_assets": "Return on assets",
    "cash_conversion": "Cash conversion",
    "liabilities_to_assets": "Liabilities ÷ assets",
    "debt_to_equity": "Debt-to-equity",
}
# For ranking: True when a higher value is better.
HIGHER_IS_BETTER = {
    "revenue_b": True,
    "revenue_cagr": True,
    "revenue_growth": True,
    "operating_margin": True,
    "net_margin": True,
    "rd_intensity": True,
    "return_on_assets": True,
    "cash_conversion": True,
    "liabilities_to_assets": False,
    "debt_to_equity": False,
}


def build_panel(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (ticker, fiscal_year), one column per metric, values in USD."""
    if not records:
        return pd.DataFrame(columns=["ticker", "company", "fiscal_year", *ANALYTICS_METRICS])
    frame = pd.DataFrame(records)
    metric_column = "metric" if "metric" in frame else "metric_key"
    panel = (
        frame.pivot_table(index=["ticker", "fiscal_year"], columns=metric_column, values="value_usd", aggfunc="last")
        .reset_index()
        .rename_axis(columns=None)
    )
    names = frame.groupby("ticker")["company"].last()
    panel.insert(1, "company", panel["ticker"].map(names))
    for metric in ANALYTICS_METRICS:
        if metric not in panel:
            panel[metric] = float("nan")
    return panel.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator != 0)
    return numerator / denominator


def add_ratios(panel: pd.DataFrame) -> pd.DataFrame:
    """Margins and efficiency ratios per company-year, plus year-over-year revenue growth."""
    out = panel.copy()
    out["operating_margin"] = _safe_ratio(out["operating_income"], out["revenue"])
    out["net_margin"] = _safe_ratio(out["net_income"], out["revenue"])
    out["rd_intensity"] = _safe_ratio(out["research_and_development"], out["revenue"])
    out["return_on_assets"] = _safe_ratio(out["net_income"], out["total_assets"])
    out["cash_conversion"] = _safe_ratio(out["operating_cash_flow"], out["net_income"].where(out["net_income"] > 0))
    # Some companies (e.g. Delta) don't report a total-liabilities line: assets minus equity is the same thing.
    out["liabilities_derived"] = out["total_liabilities"].isna() & out["total_assets"].notna() & out["stockholders_equity"].notna()
    out["total_liabilities"] = out["total_liabilities"].fillna(out["total_assets"] - out["stockholders_equity"])
    out["liabilities_to_assets"] = _safe_ratio(out["total_liabilities"], out["total_assets"])
    # Debt-to-equity is not meaningful when equity is zero or negative (common after large buybacks).
    out["debt_to_equity"] = _safe_ratio(out["long_term_debt"], out["stockholders_equity"].where(out["stockholders_equity"] > 0))
    out["net_cash"] = out["cash"] - out["long_term_debt"]
    out = out.sort_values(["ticker", "fiscal_year"])
    previous = out.groupby("ticker")["revenue"].shift(1)
    consecutive = out.groupby("ticker")["fiscal_year"].diff() == 1
    out["revenue_growth"] = _safe_ratio(out["revenue"], previous).sub(1).where(consecutive)
    return out.reset_index(drop=True)


def cagr(first: float, last: float, years: int) -> float | None:
    if years <= 0 or not first or first <= 0 or last is None or last <= 0 or pd.isna(first) or pd.isna(last):
        return None
    return (last / first) ** (1 / years) - 1


def kpi_table(ratios: pd.DataFrame) -> pd.DataFrame:
    """Latest-year KPIs per company, with revenue CAGR over the available history."""
    rows = []
    for ticker, group in ratios.groupby("ticker"):
        group = group.sort_values("fiscal_year")
        with_revenue = group.dropna(subset=["revenue"])
        latest = group.iloc[-1]
        growth_years = int(with_revenue["fiscal_year"].iloc[-1] - with_revenue["fiscal_year"].iloc[0]) if len(with_revenue) > 1 else 0
        rows.append(
            {
                "ticker": ticker,
                "company": latest["company"],
                "fiscal_year": int(latest["fiscal_year"]),
                "revenue_b": latest["revenue"] / 1e9 if pd.notna(latest["revenue"]) else None,
                "revenue_cagr": cagr(with_revenue["revenue"].iloc[0], with_revenue["revenue"].iloc[-1], growth_years)
                if growth_years
                else None,
                "cagr_years": growth_years,
                "revenue_growth": latest["revenue_growth"],
                "operating_margin": latest["operating_margin"],
                "net_margin": latest["net_margin"],
                "rd_intensity": latest["rd_intensity"],
                "return_on_assets": latest["return_on_assets"],
                "cash_conversion": latest["cash_conversion"],
                "liabilities_to_assets": latest["liabilities_to_assets"],
                "debt_to_equity": latest["debt_to_equity"],
                "net_cash_b": latest["net_cash"] / 1e9 if pd.notna(latest["net_cash"]) else None,
                "eps_diluted": latest["eps_diluted"],
            }
        )
    return pd.DataFrame(rows)


def peer_ranks(kpis: pd.DataFrame) -> pd.DataFrame:
    """Rank each company on each KPI (1 = best); missing values rank last. Adds an average rank."""
    ranks = kpis[["ticker", "company"]].copy()
    for column, higher_better in HIGHER_IS_BETTER.items():
        if column in kpis:
            ranks[column] = kpis[column].rank(ascending=not higher_better, method="min", na_option="bottom").astype(int)
    rank_columns = [column for column in HIGHER_IS_BETTER if column in ranks]
    ranks["average_rank"] = ranks[rank_columns].mean(axis=1).round(2)
    return ranks.sort_values("average_rank").reset_index(drop=True)


def indexed_series(ratios: pd.DataFrame, metric: str = "revenue") -> pd.DataFrame:
    """Each company's metric indexed to 100 at the first year they all report, for growth comparison."""
    data = ratios.dropna(subset=[metric])
    if data.empty:
        return pd.DataFrame(columns=["ticker", "fiscal_year", "index"])
    common_start = data.groupby("ticker")["fiscal_year"].min().max()
    data = data[data["fiscal_year"] >= common_start].copy()
    base = data[data["fiscal_year"] == common_start].set_index("ticker")[metric]
    data["index"] = data.apply(
        lambda row: row[metric] / base[row["ticker"]] * 100 if base.get(row["ticker"]) else None, axis=1
    )
    return data[["ticker", "fiscal_year", "index"]].dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Factor scorecard: fixed, published scoring ranges so a score means the same thing for every company.
# Each factor maps a raw value linearly onto 0–100 between "floor" (0) and "ceiling" (100), clamped.
# ---------------------------------------------------------------------------
FACTORS: dict[str, dict[str, Any]] = {
    "growth": {
        "label": "Growth",
        "measure": "Revenue CAGR",
        "formula": "(latest revenue ÷ first revenue)^(1 ÷ years) − 1",
        "floor": -0.05,
        "ceiling": 0.25,
        "format": "percent",
    },
    "profitability": {
        "label": "Profitability",
        "measure": "Net margin",
        "formula": "net income ÷ revenue (latest year)",
        "floor": 0.0,
        "ceiling": 0.30,
        "format": "percent",
    },
    "efficiency": {
        "label": "Efficiency",
        "measure": "Return on assets",
        "formula": "net income ÷ total assets (latest year)",
        "floor": 0.0,
        "ceiling": 0.20,
        "format": "percent",
    },
    "cash_quality": {
        "label": "Cash quality",
        "measure": "Cash conversion",
        "formula": "operating cash flow ÷ net income (latest year)",
        "floor": 0.5,
        "ceiling": 1.5,
        "format": "ratio",
    },
    "financial_health": {
        "label": "Financial health",
        "measure": "Liabilities ÷ assets",
        "formula": "total liabilities ÷ total assets (latest year); lower is safer",
        "floor": 0.90,
        "ceiling": 0.30,
        "format": "percent",
    },
    "consistency": {
        "label": "Consistency",
        "measure": "Years with revenue growth",
        "formula": "share of years in which revenue rose",
        "floor": 0.0,
        "ceiling": 1.0,
        "format": "percent",
    },
    "innovation": {
        "label": "Innovation",
        "measure": "R&D intensity",
        "formula": "R&D expense ÷ revenue (latest year)",
        "floor": 0.0,
        "ceiling": 0.20,
        "format": "percent",
    },
    "shareholder_return": {
        "label": "Shareholder return",
        "measure": "Annualised share-price return",
        "formula": "dividend-adjusted price CAGR over the price history",
        "floor": -0.05,
        "ceiling": 0.25,
        "format": "percent",
    },
}
DEFAULT_WEIGHTS = {key: 1 for key in FACTORS}


def factor_score(value: float | None, floor: float, ceiling: float) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(min(100.0, max(0.0, (value - floor) / (ceiling - floor) * 100)))


def growth_consistency(ratios: pd.DataFrame) -> dict[str, float | None]:
    """Share of measured years with positive revenue growth, per ticker."""
    result: dict[str, float | None] = {}
    for ticker, group in ratios.groupby("ticker"):
        growth = group["revenue_growth"].dropna()
        result[ticker] = float((growth > 0).mean()) if len(growth) >= 2 else None
    return result


def factor_scorecard(
    ratios: pd.DataFrame,
    kpis: pd.DataFrame,
    annual_returns: dict[str, float | None] | None = None,
    weights: dict[str, float] | None = None,
    not_applicable: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """One row per company: raw value and 0–100 score per factor, plus a weighted overall score.

    Missing factors (e.g. no R&D line, no price data) are left out of that company's average rather
    than counted as zero, and the number of factors used is reported alongside the score.
    `not_applicable` maps a factor to tickers it shouldn't judge (e.g. financial health for banks,
    whose deposits are liabilities by design).
    """
    not_applicable = not_applicable or {}
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    annual_returns = annual_returns or {}
    consistency = growth_consistency(ratios)
    rows = []
    for kpi in kpis.to_dict("records"):
        ticker = kpi["ticker"]
        raw = {
            "growth": kpi.get("revenue_cagr"),
            "profitability": kpi.get("net_margin"),
            "efficiency": kpi.get("return_on_assets"),
            "cash_quality": kpi.get("cash_conversion"),
            "financial_health": kpi.get("liabilities_to_assets"),
            "consistency": consistency.get(ticker),
            "innovation": kpi.get("rd_intensity"),
            "shareholder_return": annual_returns.get(ticker),
        }
        row: dict[str, Any] = {"ticker": ticker, "company": kpi.get("company")}
        weighted_total = 0.0
        weight_sum = 0.0
        used = 0
        for key, spec in FACTORS.items():
            value = raw[key]
            value = None if value is None or pd.isna(value) or ticker in not_applicable.get(key, set()) else float(value)
            score = factor_score(value, spec["floor"], spec["ceiling"])
            row[f"{key}_value"] = value
            row[f"{key}_score"] = score
            weight = float(weights.get(key, 0) or 0)
            if score is not None and weight > 0:
                weighted_total += score * weight
                weight_sum += weight
                used += 1
        row["overall_score"] = round(weighted_total / weight_sum, 1) if weight_sum else None
        row["factors_used"] = used
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("overall_score", ascending=False, na_position="last").reset_index(drop=True)
