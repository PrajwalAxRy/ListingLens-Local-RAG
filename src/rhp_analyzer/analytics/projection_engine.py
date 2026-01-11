"""Projection Engine for building Base/Bull/Stress scenario forecasts.

Uses only RHP disclosures (management guidance, capacity utilization, order book,
historical CAGR) to drive forward projections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from .models import FinancialRecord, ProjectionScenario


@dataclass
class IPODetails:
    """Core IPO parameters needed for dilution and post-issue metrics."""

    price_floor: float
    price_cap: float
    face_value: float = 10.0
    shares_pre_issue: int = 0
    fresh_issue_shares: int = 0
    ofs_shares: int = 0
    esop_pool_shares: int = 0
    fresh_issue_amount_cr: float = 0.0
    ofs_amount_cr: float = 0.0
    debt_repayment_from_issue_cr: float = 0.0


@dataclass
class GuidanceInputs:
    """Management guidance and capacity data extracted from RHP.

    Provide as much as available; missing fields will fall back to historical
    trends.
    """

    revenue_growth_guidance: Optional[float] = None  # % annual
    ebitda_margin_guidance: Optional[float] = None  # % of revenue
    capacity_utilization_current: Optional[float] = None  # %
    capacity_addition_percent: Optional[float] = None  # % over 3 yrs
    order_book_months: Optional[float] = None  # Months of revenue
    depreciation_to_assets: float = 0.03  # default 3%
    tax_rate: float = 0.25  # default 25%
    capex_to_revenue: float = 0.05  # default 5%


class ProjectionEngine:
    """Build scenario-based financial forecasts from RHP disclosures only."""

    def __init__(self) -> None:
        self.logger = logger.bind(module="projection_engine")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_scenarios(
        self,
        historical: list[FinancialRecord],
        ipo: IPODetails,
        guidance: GuidanceInputs,
        *,
        projection_years: int = 3,
    ) -> list[ProjectionScenario]:
        """Generate Base, Bull, and Stress scenarios.

        Args:
            historical: Chronologically sorted normalized historical data.
            ipo: IPO pricing and share structure.
            guidance: Management guidance and capacity inputs.
            projection_years: Number of years to project (default 3).

        Returns:
            List of three ProjectionScenario objects (Base, Bull, Stress).
        """
        if not historical:
            self.logger.warning("No historical data provided; returning empty projections")
            return []

        latest = historical[-1]
        base_revenue = latest.revenue or 0.0
        base_ebitda = latest.ebitda or 0.0
        base_pat = latest.pat or 0.0
        base_equity = latest.total_equity or 0.0
        base_debt = latest.total_debt or 0.0
        base_ccc = latest.ccc_days or 60.0

        # Derive historical CAGR
        hist_cagr = self._calculate_cagr(historical, "revenue") or 10.0
        hist_margin = (base_ebitda / base_revenue * 100) if base_revenue else 10.0

        # Build growth assumptions for each scenario
        growth_base = guidance.revenue_growth_guidance or hist_cagr
        growth_bull = growth_base * 1.25
        growth_stress = growth_base * 0.6

        margin_base = guidance.ebitda_margin_guidance or hist_margin
        margin_bull = margin_base + 2.0
        margin_stress = margin_base - 3.0

        scenarios: list[ProjectionScenario] = []
        for scenario_name, growth, margin in [
            ("Base", growth_base, margin_base),
            ("Bull", growth_bull, margin_bull),
            ("Stress", growth_stress, margin_stress),
        ]:
            scenario = self._project_scenario(
                scenario_name=scenario_name,
                base_revenue=base_revenue,
                base_equity=base_equity,
                base_debt=base_debt,
                base_ccc=base_ccc,
                revenue_growth=growth,
                ebitda_margin=margin,
                tax_rate=guidance.tax_rate,
                depreciation_rate=guidance.depreciation_to_assets,
                capex_rate=guidance.capex_to_revenue,
                ipo=ipo,
                years=projection_years,
            )
            scenarios.append(scenario)

        return scenarios

    def compute_post_issue_shares(self, ipo: IPODetails) -> int:
        """Calculate fully diluted post-issue shares."""
        return ipo.shares_pre_issue + ipo.fresh_issue_shares + ipo.esop_pool_shares

    def compute_diluted_eps(
        self,
        pat_cr: float,
        post_issue_shares: int,
    ) -> float:
        """Compute diluted EPS in rupees."""
        if post_issue_shares <= 0:
            return 0.0
        return round((pat_cr * 1e7) / post_issue_shares, 2)  # Crore to rupees

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _project_scenario(
        self,
        scenario_name: str,
        base_revenue: float,
        base_equity: float,
        base_debt: float,
        base_ccc: float,
        revenue_growth: float,
        ebitda_margin: float,
        tax_rate: float,
        depreciation_rate: float,
        capex_rate: float,
        ipo: IPODetails,
        years: int,
    ) -> ProjectionScenario:
        """Build a single scenario projection."""
        # Post-issue adjustments
        fresh_net = ipo.fresh_issue_amount_cr - ipo.debt_repayment_from_issue_cr
        post_equity = base_equity + fresh_net
        post_debt = max(0.0, base_debt - ipo.debt_repayment_from_issue_cr)
        post_shares = self.compute_post_issue_shares(ipo)

        current_revenue = base_revenue
        current_equity = post_equity
        current_debt = post_debt

        projected_years: list[str] = []
        revenue_map: dict[str, float] = {}
        ebitda_map: dict[str, float] = {}
        pat_map: dict[str, float] = {}
        eps_map: dict[str, float] = {}
        roe_map: dict[str, float] = {}
        roic_map: dict[str, float] = {}
        nd_ebitda_map: dict[str, float] = {}
        ccc_map: dict[str, float] = {}

        assumptions = [
            f"Revenue growth: {revenue_growth:.1f}%",
            f"EBITDA margin: {ebitda_margin:.1f}%",
            f"Tax rate: {tax_rate * 100:.1f}%",
            f"Capex/Revenue: {capex_rate * 100:.1f}%",
        ]

        for i in range(1, years + 1):
            fy_label = f"FY{i}E"
            projected_years.append(fy_label)

            # Revenue
            current_revenue *= 1 + revenue_growth / 100
            revenue_map[fy_label] = round(current_revenue, 2)

            # EBITDA
            ebitda = current_revenue * ebitda_margin / 100
            ebitda_map[fy_label] = round(ebitda, 2)

            # Depreciation (as % of total assets proxy: revenue * 0.7)
            depreciation = current_revenue * 0.7 * depreciation_rate
            ebit = ebitda - depreciation

            # PAT
            pbt = max(0.0, ebit - current_debt * 0.10)  # 10% interest rate proxy
            pat = pbt * (1 - tax_rate)
            pat_map[fy_label] = round(pat, 2)

            # Diluted EPS
            eps = self.compute_diluted_eps(pat, post_shares)
            eps_map[fy_label] = eps

            # Retain earnings
            current_equity += pat * 0.7  # 30% dividend payout assumption

            # ROE
            roe = (pat / current_equity * 100) if current_equity > 0 else 0.0
            roe_map[fy_label] = round(roe, 2)

            # ROIC (PAT / (Equity + Debt))
            invested = current_equity + current_debt
            roic = (pat / invested * 100) if invested > 0 else 0.0
            roic_map[fy_label] = round(roic, 2)

            # Net Debt / EBITDA
            net_debt = current_debt
            nd_ebitda = (net_debt / ebitda) if ebitda > 0 else 0.0
            nd_ebitda_map[fy_label] = round(nd_ebitda, 2)

            # CCC (assume slight improvement in base, same in stress)
            if scenario_name == "Bull":
                ccc = base_ccc * 0.95
            elif scenario_name == "Stress":
                ccc = base_ccc * 1.10
            else:
                ccc = base_ccc
            ccc_map[fy_label] = round(ccc, 1)

        return ProjectionScenario(
            name=scenario_name,
            years=projected_years,
            revenue=revenue_map,
            ebitda=ebitda_map,
            pat=pat_map,
            diluted_eps=eps_map,
            roe=roe_map,
            roic=roic_map,
            net_debt_to_ebitda=nd_ebitda_map,
            cash_conversion_cycle=ccc_map,
            assumptions=assumptions,
            citations=[],
        )

    def _calculate_cagr(
        self,
        records: list[FinancialRecord],
        metric: str,
    ) -> Optional[float]:
        """Calculate CAGR for a metric."""
        values = [getattr(r, metric) for r in records if getattr(r, metric, None)]
        if len(values) < 2:
            return None
        start, end = values[0], values[-1]
        if start <= 0 or end <= 0:
            return None
        years = len(values) - 1
        return round(((end / start) ** (1 / years) - 1) * 100, 2)


__all__ = [
    "GuidanceInputs",
    "IPODetails",
    "ProjectionEngine",
]
