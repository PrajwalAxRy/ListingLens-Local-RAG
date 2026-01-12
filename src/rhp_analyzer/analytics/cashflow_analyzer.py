"""Enhanced Cash Flow Analyzer.

Calculates Free Cash Flow (FCF), categorizes capex (maintenance vs growth),
analyzes cash burn rate and runway months for loss-making companies.
Assesses cash flow quality via CFO/EBITDA and CFO/PAT ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger


class CashBurnStatus(Enum):
    """Cash burn status classification."""

    PROFITABLE_FCF = "Profitable - Positive FCF"
    PROFITABLE_NEGATIVE_FCF = "Profitable but FCF negative"
    CASH_BURNING = "Cash Burning"
    CRITICAL_BURN = "Critical Burn Rate"


@dataclass
class CashFlowMetrics:
    """Raw cash flow metrics from financial statements."""

    fiscal_year: str

    # Core cash flows (in crores)
    cfo: float = 0.0  # Cash from Operations
    cfi: float = 0.0  # Cash from Investing (usually negative)
    cff: float = 0.0  # Cash from Financing

    # P&L items for ratios
    revenue: float = 0.0
    ebitda: float = 0.0
    pat: float = 0.0
    depreciation: float = 0.0

    # Capex (positive value, will be subtracted from CFO for FCF)
    capex: float = 0.0

    # Balance sheet items
    cash_and_equivalents: float = 0.0
    current_liabilities: float = 0.0

    # Working capital change (from CF statement)
    working_capital_change: float = 0.0


@dataclass
class CashFlowAnalysis:
    """Enhanced cash flow analysis for a single fiscal year."""

    fiscal_year: str

    # Core cash flows (crores)
    cfo: float = 0.0
    cfi: float = 0.0
    cff: float = 0.0
    net_cash_flow: float = 0.0

    # Free Cash Flow metrics
    capex: float = 0.0
    fcf: float = 0.0  # CFO - Capex
    fcf_margin: float = 0.0  # FCF / Revenue %
    fcf_yield_at_floor: Optional[float] = None  # FCF / Market Cap at floor
    fcf_yield_at_cap: Optional[float] = None  # FCF / Market Cap at cap

    # Cash burn analysis
    is_cash_burning: bool = False
    monthly_cash_burn: float = 0.0
    runway_months: float = 0.0
    burn_status: CashBurnStatus = CashBurnStatus.PROFITABLE_FCF

    # Capex analysis
    capex_to_revenue: float = 0.0  # Capex intensity %
    capex_to_depreciation: float = 0.0  # >1 = growth capex, ~1 = maintenance
    maintenance_capex_estimate: float = 0.0  # ≈ Depreciation
    growth_capex_estimate: float = 0.0  # Capex - Depreciation

    # Working capital impact
    wc_change: float = 0.0
    wc_change_to_revenue: float = 0.0  # %

    # Quality indicators
    cfo_to_ebitda: float = 0.0  # Should be >70% for quality earnings
    cfo_to_pat: float = 0.0  # Cash conversion

    # Liquidity
    cash_and_equivalents: float = 0.0
    cash_to_current_liabilities: float = 0.0

    citations: list[str] = field(default_factory=list)


@dataclass
class CashFlowTrend:
    """Multi-year cash flow trend analysis."""

    years: list[str]
    analyses: list[CashFlowAnalysis]

    # Trend summaries
    avg_fcf: float = 0.0
    avg_fcf_margin: float = 0.0
    fcf_trend: str = ""  # "Improving" / "Deteriorating" / "Volatile"

    # Quality trend
    avg_cfo_to_ebitda: float = 0.0
    earnings_quality: str = ""  # "High" / "Medium" / "Low"

    # Capex pattern
    avg_capex_intensity: float = 0.0
    capex_pattern: str = ""  # "Growth mode" / "Maintenance mode" / "Mixed"

    # Cash burn history
    consecutive_burn_years: int = 0
    total_cash_burned: float = 0.0

    # Red flags
    red_flags: list[str] = field(default_factory=list)

    overall_assessment: str = ""
    citations: list[str] = field(default_factory=list)


class EnhancedCashFlowAnalyzer:
    """Analyze cash flows with FCF, burn rate, and quality assessment.

    Provides comprehensive analysis of cash flow quality, free cash flow,
    capex categorization, and cash burn analysis for loss-making companies.
    """

    # Thresholds
    CFO_EBITDA_GOOD_THRESHOLD = 70  # %
    CFO_EBITDA_CONCERN_THRESHOLD = 50  # %
    FCF_BURN_CONCERN_MONTHS = 18  # months runway
    FCF_BURN_CRITICAL_MONTHS = 12  # months runway
    CAPEX_GROWTH_THRESHOLD = 1.5  # capex/depreciation ratio

    def __init__(self) -> None:
        """Initialize analyzer."""
        self.logger = logger.bind(module="cashflow")

    def analyze_single_year(
        self,
        metrics: CashFlowMetrics,
        market_cap_floor: Optional[float] = None,
        market_cap_cap: Optional[float] = None,
    ) -> CashFlowAnalysis:
        """Analyze cash flow for a single fiscal year.

        Args:
            metrics: Cash flow metrics for the year.
            market_cap_floor: Market cap at floor price for FCF yield calc.
            market_cap_cap: Market cap at cap price for FCF yield calc.

        Returns:
            Comprehensive cash flow analysis.
        """
        analysis = CashFlowAnalysis(fiscal_year=metrics.fiscal_year)

        # Core cash flows
        analysis.cfo = metrics.cfo
        analysis.cfi = metrics.cfi
        analysis.cff = metrics.cff
        analysis.net_cash_flow = metrics.cfo + metrics.cfi + metrics.cff

        # Free Cash Flow
        analysis.capex = metrics.capex
        analysis.fcf = metrics.cfo - metrics.capex

        # FCF margin
        if metrics.revenue > 0:
            analysis.fcf_margin = (analysis.fcf / metrics.revenue) * 100

        # FCF yield
        if market_cap_floor and market_cap_floor > 0:
            analysis.fcf_yield_at_floor = (analysis.fcf / market_cap_floor) * 100
        if market_cap_cap and market_cap_cap > 0:
            analysis.fcf_yield_at_cap = (analysis.fcf / market_cap_cap) * 100

        # Cash burn analysis
        analysis.is_cash_burning = analysis.fcf < 0
        if analysis.is_cash_burning:
            analysis.monthly_cash_burn = abs(analysis.fcf) / 12
            if analysis.monthly_cash_burn > 0:
                analysis.runway_months = (
                    metrics.cash_and_equivalents / analysis.monthly_cash_burn
                )
            else:
                analysis.runway_months = float("inf")

            # Classify burn status
            if metrics.pat > 0:
                analysis.burn_status = CashBurnStatus.PROFITABLE_NEGATIVE_FCF
            elif analysis.runway_months < self.FCF_BURN_CRITICAL_MONTHS:
                analysis.burn_status = CashBurnStatus.CRITICAL_BURN
            else:
                analysis.burn_status = CashBurnStatus.CASH_BURNING
        else:
            analysis.burn_status = CashBurnStatus.PROFITABLE_FCF

        # Capex analysis
        if metrics.revenue > 0:
            analysis.capex_to_revenue = (metrics.capex / metrics.revenue) * 100

        if metrics.depreciation > 0:
            analysis.capex_to_depreciation = metrics.capex / metrics.depreciation
        else:
            analysis.capex_to_depreciation = float("inf") if metrics.capex > 0 else 0

        # Maintenance vs Growth capex
        analysis.maintenance_capex_estimate = min(metrics.depreciation, metrics.capex)
        analysis.growth_capex_estimate = max(
            0, metrics.capex - metrics.depreciation
        )

        # Working capital impact
        analysis.wc_change = metrics.working_capital_change
        if metrics.revenue > 0:
            analysis.wc_change_to_revenue = (
                metrics.working_capital_change / metrics.revenue
            ) * 100

        # Quality indicators
        if metrics.ebitda > 0:
            analysis.cfo_to_ebitda = (metrics.cfo / metrics.ebitda) * 100
        elif metrics.ebitda < 0 and metrics.cfo > 0:
            analysis.cfo_to_ebitda = 100.0  # CFO positive despite negative EBITDA
        else:
            analysis.cfo_to_ebitda = 0.0

        if metrics.pat > 0:
            analysis.cfo_to_pat = (metrics.cfo / metrics.pat) * 100
        elif metrics.pat < 0 and metrics.cfo > 0:
            analysis.cfo_to_pat = 100.0  # CFO positive despite loss
        else:
            analysis.cfo_to_pat = 0.0

        # Liquidity
        analysis.cash_and_equivalents = metrics.cash_and_equivalents
        if metrics.current_liabilities > 0:
            analysis.cash_to_current_liabilities = (
                metrics.cash_and_equivalents / metrics.current_liabilities
            )

        return analysis

    def analyze_multi_year(
        self,
        metrics_list: list[CashFlowMetrics],
        market_cap_floor: Optional[float] = None,
        market_cap_cap: Optional[float] = None,
    ) -> CashFlowTrend:
        """Analyze cash flow trend across multiple years.

        Args:
            metrics_list: List of metrics sorted by fiscal year (earliest first).
            market_cap_floor: Market cap at floor price.
            market_cap_cap: Market cap at cap price.

        Returns:
            Multi-year cash flow trend analysis.
        """
        if not metrics_list:
            return CashFlowTrend(years=[], analyses=[])

        analyses: list[CashFlowAnalysis] = []
        years: list[str] = []

        for metrics in metrics_list:
            analysis = self.analyze_single_year(
                metrics, market_cap_floor, market_cap_cap
            )
            analyses.append(analysis)
            years.append(metrics.fiscal_year)

        trend = CashFlowTrend(years=years, analyses=analyses)

        # Calculate averages
        if analyses:
            trend.avg_fcf = sum(a.fcf for a in analyses) / len(analyses)
            trend.avg_fcf_margin = sum(a.fcf_margin for a in analyses) / len(analyses)
            trend.avg_cfo_to_ebitda = sum(a.cfo_to_ebitda for a in analyses) / len(
                analyses
            )
            trend.avg_capex_intensity = sum(a.capex_to_revenue for a in analyses) / len(
                analyses
            )

        # FCF trend
        trend.fcf_trend = self._assess_fcf_trend(analyses)

        # Earnings quality
        trend.earnings_quality = self._assess_earnings_quality(trend.avg_cfo_to_ebitda)

        # Capex pattern
        trend.capex_pattern = self._assess_capex_pattern(analyses)

        # Cash burn history
        trend.consecutive_burn_years = self._count_consecutive_burn_years(analyses)
        trend.total_cash_burned = sum(
            abs(a.fcf) for a in analyses if a.is_cash_burning
        )

        # Red flags
        trend.red_flags = self._identify_red_flags(trend, analyses)

        # Overall assessment
        trend.overall_assessment = self._generate_assessment(trend)

        self.logger.info(
            "Cash flow trend analyzed: {} years, FCF trend={}, Quality={}",
            len(years),
            trend.fcf_trend,
            trend.earnings_quality,
        )

        return trend

    def _assess_fcf_trend(self, analyses: list[CashFlowAnalysis]) -> str:
        """Assess FCF trend direction."""
        if len(analyses) < 2:
            return "Insufficient data"

        fcfs = [a.fcf for a in analyses]

        # Check if improving (more recent years better)
        first_half_avg = sum(fcfs[: len(fcfs) // 2]) / max(1, len(fcfs) // 2)
        second_half_avg = sum(fcfs[len(fcfs) // 2 :]) / max(
            1, len(fcfs) - len(fcfs) // 2
        )

        if second_half_avg > first_half_avg + 10:  # 10 Cr improvement
            return "Improving"
        elif second_half_avg < first_half_avg - 10:
            return "Deteriorating"
        else:
            # Check volatility
            avg = sum(fcfs) / len(fcfs)
            variance = sum((f - avg) ** 2 for f in fcfs) / len(fcfs)
            if variance > (abs(avg) * 0.5) ** 2:
                return "Volatile"
            return "Stable"

    def _assess_earnings_quality(self, avg_cfo_to_ebitda: float) -> str:
        """Assess earnings quality based on CFO/EBITDA ratio."""
        if avg_cfo_to_ebitda >= self.CFO_EBITDA_GOOD_THRESHOLD:
            return "High"
        elif avg_cfo_to_ebitda >= self.CFO_EBITDA_CONCERN_THRESHOLD:
            return "Medium"
        else:
            return "Low"

    def _assess_capex_pattern(self, analyses: list[CashFlowAnalysis]) -> str:
        """Assess capex pattern (growth vs maintenance)."""
        if not analyses:
            return "Unknown"

        avg_capex_to_dep = sum(a.capex_to_depreciation for a in analyses) / len(
            analyses
        )

        if avg_capex_to_dep > self.CAPEX_GROWTH_THRESHOLD:
            return "Growth mode"
        elif avg_capex_to_dep < 0.8:
            return "Under-investment"
        else:
            return "Maintenance mode"

    def _count_consecutive_burn_years(self, analyses: list[CashFlowAnalysis]) -> int:
        """Count consecutive years of cash burn from most recent."""
        count = 0
        for analysis in reversed(analyses):
            if analysis.is_cash_burning:
                count += 1
            else:
                break
        return count

    def _identify_red_flags(
        self, trend: CashFlowTrend, analyses: list[CashFlowAnalysis]
    ) -> list[str]:
        """Identify cash flow red flags."""
        flags: list[str] = []

        # Consecutive cash burn
        if trend.consecutive_burn_years >= 2:
            flags.append(
                f"Negative FCF for {trend.consecutive_burn_years} consecutive years"
            )

        # Low earnings quality
        if trend.earnings_quality == "Low":
            flags.append(
                f"Low earnings quality: CFO/EBITDA at {trend.avg_cfo_to_ebitda:.0f}%"
            )

        # Critical runway
        if analyses and analyses[-1].is_cash_burning:
            latest = analyses[-1]
            if latest.runway_months < self.FCF_BURN_CRITICAL_MONTHS:
                flags.append(
                    f"Critical cash runway: only {latest.runway_months:.0f} months"
                )
            elif latest.runway_months < self.FCF_BURN_CONCERN_MONTHS:
                flags.append(
                    f"Low cash runway: {latest.runway_months:.0f} months"
                )

        # Deteriorating FCF trend
        if trend.fcf_trend == "Deteriorating":
            flags.append("FCF trend deteriorating over time")

        # High total cash burned
        if trend.total_cash_burned > 100:  # More than 100 Cr burned
            flags.append(
                f"Total cash burned: ₹{trend.total_cash_burned:.0f} Cr over the period"
            )

        return flags

    def _generate_assessment(self, trend: CashFlowTrend) -> str:
        """Generate overall assessment text."""
        parts: list[str] = []

        # FCF assessment
        if trend.avg_fcf > 0:
            parts.append(f"Average positive FCF of ₹{trend.avg_fcf:.0f} Cr")
        else:
            parts.append(f"Average negative FCF of ₹{abs(trend.avg_fcf):.0f} Cr")

        # Quality assessment
        parts.append(f"{trend.earnings_quality} earnings quality")

        # Capex assessment
        parts.append(f"Capex in {trend.capex_pattern.lower()}")

        # Red flags
        if trend.red_flags:
            parts.append(f"Red flags: {len(trend.red_flags)}")

        return ". ".join(parts) + "."


__all__ = [
    "EnhancedCashFlowAnalyzer",
    "CashFlowAnalysis",
    "CashFlowMetrics",
    "CashFlowTrend",
    "CashBurnStatus",
]
