"""Stub period analyzer module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger


@dataclass
class StubPeriodAnalysis:
    """Stub period (interim) financial comparison."""

    stub_period: str
    comparable_prior_period: str
    stub_revenue: float
    stub_ebitda: float
    stub_pat: float
    prior_revenue: float
    prior_ebitda: float
    prior_pat: float
    revenue_growth_yoy: float
    ebitda_growth_yoy: float
    pat_growth_yoy: float
    stub_ebitda_margin: float
    prior_ebitda_margin: float
    margin_expansion: float
    annualized_revenue: float
    annualized_ebitda: float
    annualized_pat: float
    last_full_year_revenue: float
    implied_full_year_growth: float
    stub_growth_below_historical_cagr: bool
    margin_compression_in_stub: bool
    one_time_items: List[str] = field(default_factory=list)
    is_business_seasonal: bool = False
    seasonality_notes: Optional[str] = None


class StubPeriodAnalyzer:
    """Analyze disclosed stub period financials."""

    def __init__(self) -> None:
        self.logger = logger.bind(module="stub_period_analyzer")

    def analyze_stub(self, state: Dict, historical_financials: List[Dict]) -> Optional[StubPeriodAnalysis]:
        stub_data = state.get("stub_period") or state.get("stub_period_data")
        if not stub_data:
            self.logger.info("No stub period disclosures found")
            return None

        prior_data = stub_data.get("prior_period")
        if not prior_data:
            self.logger.warning("Stub period disclosure missing comparable prior period")
            return None

        months = max(int(stub_data.get("months", 0) or 0), 1)
        annualization_factor = 12 / months

        stub_revenue = float(stub_data.get("revenue", 0.0))
        stub_ebitda = float(stub_data.get("ebitda", 0.0))
        stub_pat = float(stub_data.get("pat", 0.0))

        prior_revenue = float(prior_data.get("revenue", 0.0))
        prior_ebitda = float(prior_data.get("ebitda", 0.0))
        prior_pat = float(prior_data.get("pat", 0.0))

        revenue_growth = self._growth(stub_revenue, prior_revenue)
        ebitda_growth = self._growth(stub_ebitda, prior_ebitda)
        pat_growth = self._growth(stub_pat, prior_pat)

        stub_margin = self._margin(stub_ebitda, stub_revenue)
        prior_margin = self._margin(prior_ebitda, prior_revenue)
        margin_expansion = stub_margin - prior_margin

        annualized_revenue = stub_revenue * annualization_factor
        annualized_ebitda = stub_ebitda * annualization_factor
        annualized_pat = stub_pat * annualization_factor

        last_full_year_revenue = self._get_last_full_year_revenue(historical_financials)
        implied_growth = self._growth(annualized_revenue, last_full_year_revenue)
        historical_cagr = self._calculate_historical_cagr(historical_financials)

        analysis = StubPeriodAnalysis(
            stub_period=stub_data.get("period", "Stub Period"),
            comparable_prior_period=prior_data.get("period", "Prior Period"),
            stub_revenue=stub_revenue,
            stub_ebitda=stub_ebitda,
            stub_pat=stub_pat,
            prior_revenue=prior_revenue,
            prior_ebitda=prior_ebitda,
            prior_pat=prior_pat,
            revenue_growth_yoy=revenue_growth,
            ebitda_growth_yoy=ebitda_growth,
            pat_growth_yoy=pat_growth,
            stub_ebitda_margin=stub_margin,
            prior_ebitda_margin=prior_margin,
            margin_expansion=margin_expansion,
            annualized_revenue=annualized_revenue,
            annualized_ebitda=annualized_ebitda,
            annualized_pat=annualized_pat,
            last_full_year_revenue=last_full_year_revenue,
            implied_full_year_growth=implied_growth,
            stub_growth_below_historical_cagr=(
                historical_cagr is not None and revenue_growth < historical_cagr
            ),
            margin_compression_in_stub=margin_expansion < -2,
            one_time_items=stub_data.get("one_time_items", []),
            is_business_seasonal=bool(stub_data.get("is_seasonal")),
            seasonality_notes=stub_data.get("seasonality_notes"),
        )
        return analysis

    def _growth(self, current: float, prior: float) -> float:
        if prior == 0:
            return 0.0
        return round(((current / prior) - 1) * 100, 2)

    def _margin(self, numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return round((numerator / denominator) * 100, 2)

    def _get_last_full_year_revenue(self, financials: List[Dict]) -> float:
        if not financials:
            return 0.0
        last_record = financials[-1]
        return float(last_record.get("revenue", last_record.get("Revenue", 0.0)))

    def _calculate_historical_cagr(self, financials: List[Dict]) -> Optional[float]:
        if len(financials) < 2:
            return None
        first = float(financials[0].get("revenue", financials[0].get("Revenue", 0.0)))
        last = float(financials[-1].get("revenue", financials[-1].get("Revenue", 0.0)))
        years = len(financials) - 1
        if first <= 0 or years <= 0:
            return None
        cagr = ((last / first) ** (1 / years) - 1) * 100
        return round(cagr, 2)
