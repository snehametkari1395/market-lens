from __future__ import annotations

from typing import Any


def build_research_shortlist(
    records_by_ticker: dict[str, dict[str, list[dict[str, Any]]]],
    minimum_revenue_growth: float,
    minimum_net_margin: float,
) -> list[dict[str, Any]]:
    """Assess companies against user-selected, transparent financial screens."""
    assessments: list[dict[str, Any]] = []
    for ticker, metrics in records_by_ticker.items():
        revenue = {row["fiscal_year"]: row for row in metrics.get("revenue", [])}
        net_income = {row["fiscal_year"]: row for row in metrics.get("net_income", [])}
        common_years = sorted(set(revenue) & set(net_income))
        company = next(
            (
                row["company"]
                for rows in metrics.values()
                for row in rows
                if row.get("company")
            ),
            ticker,
        )
        if not common_years:
            assessments.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "fiscal_year": None,
                    "revenue_growth_percent": None,
                    "net_margin_percent": None,
                    "growth_match": False,
                    "margin_match": False,
                    "sources": [],
                    "explanation": "Comparable annual revenue and net income data was not available.",
                }
            )
            continue

        fiscal_year = common_years[-1]
        revenue_value = float(revenue[fiscal_year]["value_usd"])
        net_income_value = float(net_income[fiscal_year]["value_usd"])
        net_margin = net_income_value / revenue_value * 100 if revenue_value else None
        prior_revenue = revenue.get(fiscal_year - 1)
        revenue_growth = None
        if prior_revenue is not None:
            prior_value = float(prior_revenue["value_usd"])
            if prior_value != 0:
                revenue_growth = (revenue_value / prior_value - 1) * 100

        growth_match = (
            revenue_growth is not None and revenue_growth >= minimum_revenue_growth
        )
        margin_match = net_margin is not None and net_margin >= minimum_net_margin
        reasons = []
        if growth_match:
            reasons.append(
                f"Revenue grew {revenue_growth:.1f}% from FY{fiscal_year - 1} "
                f"to FY{fiscal_year}, meeting your {minimum_revenue_growth:.1f}% minimum."
            )
        elif revenue_growth is None:
            reasons.append("Consecutive-year revenue data was unavailable, so growth could not be assessed.")
        else:
            reasons.append(
                f"Revenue growth was {revenue_growth:.1f}%, below your "
                f"{minimum_revenue_growth:.1f}% minimum."
            )

        if margin_match:
            reasons.append(
                f"Net margin was {net_margin:.1f}% in FY{fiscal_year}, meeting your "
                f"{minimum_net_margin:.1f}% minimum."
            )
        else:
            reasons.append(
                f"Net margin was {net_margin:.1f}% in FY{fiscal_year}, below your "
                f"{minimum_net_margin:.1f}% minimum."
                if net_margin is not None
                else "Net margin could not be calculated because revenue was zero."
            )

        source_rows = [revenue[fiscal_year], net_income[fiscal_year]]
        if prior_revenue is not None:
            source_rows.append(prior_revenue)
        assessments.append(
            {
                "ticker": ticker,
                "company": company,
                "fiscal_year": fiscal_year,
                "revenue_growth_percent": (
                    round(revenue_growth, 2) if revenue_growth is not None else None
                ),
                "net_margin_percent": round(net_margin, 2) if net_margin is not None else None,
                "growth_match": growth_match,
                "margin_match": margin_match,
                "sources": [
                    {"fiscal_year": year, "url": row["source_url"]}
                    for year, row in sorted(
                        {
                            (row["fiscal_year"], row["source_url"]): row
                            for row in source_rows
                            if row.get("source_url")
                        }.items()
                    )
                ],
                "explanation": " ".join(reasons),
            }
        )

    return sorted(
        assessments,
        key=lambda item: (
            -(int(item["growth_match"]) + int(item["margin_match"])),
            item["revenue_growth_percent"] is None,
            -(item["revenue_growth_percent"] or 0),
        ),
    )
