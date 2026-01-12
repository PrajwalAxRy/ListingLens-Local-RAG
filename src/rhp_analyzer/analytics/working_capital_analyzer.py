"""Working Capital Analyzer with Sector Benchmarks.

Calculates working capital cycle metrics (DSO, DIO, DPO, CCC) and compares
against predefined sector benchmarks. Flags significant deviations and
channel stuffing risks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


# Sector benchmarks (predefined) - CCC in days
SECTOR_BENCHMARKS: dict[str, dict[str, float]] = {
    "FMCG": {"receivable_days": 30, "inventory_days": 45, "payable_days": 35, "ccc": 40},
    "Pharma": {
        "receivable_days": 90,
        "inventory_days": 120,
        "payable_days": 30,
        "ccc": 180,
    },
    "IT Services": {
        "receivable_days": 60,
        "inventory_days": 0,
        "payable_days": 10,
        "ccc": 50,
    },
    "Auto": {"receivable_days": 45, "inventory_days": 60, "payable_days": 25, "ccc": 80},
    "Textiles": {
        "receivable_days": 60,
        "inventory_days": 90,
        "payable_days": 30,
        "ccc": 120,
    },
    "Capital Goods": {
        "receivable_days": 120,
        "inventory_days": 90,
        "payable_days": 30,
        "ccc": 180,
    },
    "Real Estate": {
        "receivable_days": 180,
        "inventory_days": 365,
        "payable_days": 145,
        "ccc": 400,
    },
    "Steel": {"receivable_days": 60, "inventory_days": 45, "payable_days": 25, "ccc": 80},
    "Cement": {"receivable_days": 30, "inventory_days": 30, "payable_days": 20, "ccc": 40},
    "Chemicals": {
        "receivable_days": 60,
        "inventory_days": 75,
        "payable_days": 25,
        "ccc": 110,
    },
    "EPC": {"receivable_days": 90, "inventory_days": 60, "payable_days": 30, "ccc": 120},
    "Defense": {
        "receivable_days": 120,
        "inventory_days": 90,
        "payable_days": 40,
        "ccc": 170,
    },
    "Infrastructure": {
        "receivable_days": 120,
        "inventory_days": 60,
        "payable_days": 60,
        "ccc": 120,
    },
    "Retail": {
        "receivable_days": 10,
        "inventory_days": 60,
        "payable_days": 45,
        "ccc": 25,
    },
    "Banking": {"receivable_days": 0, "inventory_days": 0, "payable_days": 0, "ccc": 0},
    "NBFC": {"receivable_days": 0, "inventory_days": 0, "payable_days": 0, "ccc": 0},
}


@dataclass
class WorkingCapitalMetrics:
    """Working capital metrics for a single fiscal year."""

    fiscal_year: str
    revenue: float  # in crores
    cogs: float  # Cost of Goods Sold in crores

    # Balance sheet items
    inventory: float = 0.0
    trade_receivables: float = 0.0
    trade_payables: float = 0.0
    current_assets: float = 0.0
    current_liabilities: float = 0.0

    # Calculated days
    receivable_days: float = 0.0  # DSO - Days Sales Outstanding
    inventory_days: float = 0.0  # DIO - Days Inventory Outstanding
    payable_days: float = 0.0  # DPO - Days Payable Outstanding
    cash_conversion_cycle: float = 0.0  # CCC = DSO + DIO - DPO

    # Net working capital
    net_working_capital: float = 0.0
    nwc_to_revenue: float = 0.0  # Working capital intensity %


@dataclass
class WorkingCapitalAnalysis:
    """Working capital analysis with YoY trends and sector comparison."""

    fiscal_year: str

    # Core metrics
    receivable_days: float = 0.0
    inventory_days: float = 0.0
    payable_days: float = 0.0
    cash_conversion_cycle: float = 0.0

    # Absolute values (crores)
    inventory: float = 0.0
    trade_receivables: float = 0.0
    trade_payables: float = 0.0
    net_working_capital: float = 0.0
    nwc_to_revenue: float = 0.0  # %

    # YoY trends
    receivable_days_change_yoy: float = 0.0
    inventory_days_change_yoy: float = 0.0
    payable_days_change_yoy: float = 0.0
    ccc_change_yoy: float = 0.0

    # Growth differentials
    revenue_growth: float = 0.0  # %
    receivable_growth: float = 0.0  # %
    receivable_growth_vs_revenue_growth: float = 0.0  # % difference

    # Red flags
    is_channel_stuffing_risk: bool = False  # Receivables growing faster than revenue
    is_receivable_days_worsening: bool = False  # >10 days increase YoY
    is_inventory_piling: bool = False  # >15 days increase YoY

    # Sector comparison
    sector: Optional[str] = None
    sector_avg_receivable_days: Optional[float] = None
    sector_avg_inventory_days: Optional[float] = None
    sector_avg_ccc: Optional[float] = None
    variance_vs_sector_ccc: Optional[float] = None  # % or absolute

    citations: list[str] = field(default_factory=list)


@dataclass
class WorkingCapitalTrend:
    """Multi-year working capital trend analysis."""

    years: list[str]
    analyses: list[WorkingCapitalAnalysis]

    # Trend summaries
    ccc_trend: str = ""  # "Improving" / "Worsening" / "Stable"
    avg_ccc: float = 0.0
    avg_nwc_intensity: float = 0.0

    # Red flags summary
    channel_stuffing_years: list[str] = field(default_factory=list)
    inventory_piling_years: list[str] = field(default_factory=list)

    overall_assessment: str = ""
    citations: list[str] = field(default_factory=list)


class WorkingCapitalAnalyzer:
    """Analyze working capital cycle with sector benchmarking.

    Calculates DSO, DIO, DPO, CCC and compares against sector averages.
    Flags channel stuffing and inventory piling risks.
    """

    # Red flag thresholds
    RECEIVABLE_DAYS_WORSENING_THRESHOLD = 10  # days
    INVENTORY_DAYS_PILING_THRESHOLD = 15  # days
    CHANNEL_STUFFING_THRESHOLD = 10  # percentage points
    SECTOR_VARIANCE_CONCERN_THRESHOLD = 50  # % above sector avg

    def __init__(self, sector: Optional[str] = None) -> None:
        """Initialize analyzer.

        Args:
            sector: Industry sector for benchmark comparison.
        """
        self.sector = sector
        self.sector_benchmarks = self._get_sector_benchmarks(sector)
        self.logger = logger.bind(module="working_capital")

    def analyze_single_year(
        self,
        metrics: WorkingCapitalMetrics,
        prior_metrics: Optional[WorkingCapitalMetrics] = None,
    ) -> WorkingCapitalAnalysis:
        """Analyze working capital for a single fiscal year.

        Args:
            metrics: Current year working capital metrics.
            prior_metrics: Prior year metrics for YoY comparison.

        Returns:
            Working capital analysis with trends and flags.
        """
        analysis = WorkingCapitalAnalysis(fiscal_year=metrics.fiscal_year)

        # Calculate days metrics
        analysis.receivable_days = self._calc_receivable_days(
            metrics.trade_receivables, metrics.revenue
        )
        analysis.inventory_days = self._calc_inventory_days(
            metrics.inventory, metrics.cogs
        )
        analysis.payable_days = self._calc_payable_days(
            metrics.trade_payables, metrics.cogs
        )
        analysis.cash_conversion_cycle = (
            analysis.receivable_days
            + analysis.inventory_days
            - analysis.payable_days
        )

        # Absolute values
        analysis.inventory = metrics.inventory
        analysis.trade_receivables = metrics.trade_receivables
        analysis.trade_payables = metrics.trade_payables
        analysis.net_working_capital = (
            metrics.current_assets - metrics.current_liabilities
        )
        analysis.nwc_to_revenue = (
            (analysis.net_working_capital / metrics.revenue * 100)
            if metrics.revenue > 0
            else 0.0
        )

        # YoY trends if prior data available
        if prior_metrics:
            prior_recv_days = self._calc_receivable_days(
                prior_metrics.trade_receivables, prior_metrics.revenue
            )
            prior_inv_days = self._calc_inventory_days(
                prior_metrics.inventory, prior_metrics.cogs
            )
            prior_pay_days = self._calc_payable_days(
                prior_metrics.trade_payables, prior_metrics.cogs
            )
            prior_ccc = prior_recv_days + prior_inv_days - prior_pay_days

            analysis.receivable_days_change_yoy = (
                analysis.receivable_days - prior_recv_days
            )
            analysis.inventory_days_change_yoy = (
                analysis.inventory_days - prior_inv_days
            )
            analysis.payable_days_change_yoy = (
                analysis.payable_days - prior_pay_days
            )
            analysis.ccc_change_yoy = analysis.cash_conversion_cycle - prior_ccc

            # Growth rates
            analysis.revenue_growth = self._calc_growth(
                prior_metrics.revenue, metrics.revenue
            )
            analysis.receivable_growth = self._calc_growth(
                prior_metrics.trade_receivables, metrics.trade_receivables
            )
            analysis.receivable_growth_vs_revenue_growth = (
                analysis.receivable_growth - analysis.revenue_growth
            )

            # Red flags
            analysis.is_channel_stuffing_risk = (
                analysis.receivable_growth_vs_revenue_growth
                > self.CHANNEL_STUFFING_THRESHOLD
            )
            analysis.is_receivable_days_worsening = (
                analysis.receivable_days_change_yoy
                > self.RECEIVABLE_DAYS_WORSENING_THRESHOLD
            )
            analysis.is_inventory_piling = (
                analysis.inventory_days_change_yoy
                > self.INVENTORY_DAYS_PILING_THRESHOLD
            )

        # Sector comparison
        if self.sector_benchmarks:
            analysis.sector = self.sector
            analysis.sector_avg_receivable_days = self.sector_benchmarks.get(
                "receivable_days"
            )
            analysis.sector_avg_inventory_days = self.sector_benchmarks.get(
                "inventory_days"
            )
            analysis.sector_avg_ccc = self.sector_benchmarks.get("ccc")

            if analysis.sector_avg_ccc and analysis.sector_avg_ccc > 0:
                analysis.variance_vs_sector_ccc = (
                    (analysis.cash_conversion_cycle - analysis.sector_avg_ccc)
                    / analysis.sector_avg_ccc
                    * 100
                )

        return analysis

    def analyze_multi_year(
        self,
        metrics_list: list[WorkingCapitalMetrics],
    ) -> WorkingCapitalTrend:
        """Analyze working capital trend across multiple years.

        Args:
            metrics_list: List of metrics sorted by fiscal year (earliest first).

        Returns:
            Multi-year trend analysis.
        """
        if not metrics_list:
            return WorkingCapitalTrend(years=[], analyses=[])

        analyses: list[WorkingCapitalAnalysis] = []
        years: list[str] = []

        for i, metrics in enumerate(metrics_list):
            prior = metrics_list[i - 1] if i > 0 else None
            analysis = self.analyze_single_year(metrics, prior)
            analyses.append(analysis)
            years.append(metrics.fiscal_year)

        # Calculate trend summaries
        trend = WorkingCapitalTrend(years=years, analyses=analyses)

        if len(analyses) >= 2:
            first_ccc = analyses[0].cash_conversion_cycle
            last_ccc = analyses[-1].cash_conversion_cycle

            if last_ccc < first_ccc - 10:
                trend.ccc_trend = "Improving"
            elif last_ccc > first_ccc + 10:
                trend.ccc_trend = "Worsening"
            else:
                trend.ccc_trend = "Stable"

        # Average metrics
        if analyses:
            trend.avg_ccc = sum(a.cash_conversion_cycle for a in analyses) / len(
                analyses
            )
            trend.avg_nwc_intensity = sum(a.nwc_to_revenue for a in analyses) / len(
                analyses
            )

        # Red flag years
        trend.channel_stuffing_years = [
            a.fiscal_year for a in analyses if a.is_channel_stuffing_risk
        ]
        trend.inventory_piling_years = [
            a.fiscal_year for a in analyses if a.is_inventory_piling
        ]

        # Overall assessment
        trend.overall_assessment = self._assess_trend(trend)

        self.logger.info(
            "Working capital trend analyzed: {} years, CCC trend={}",
            len(years),
            trend.ccc_trend,
        )

        return trend

    def _calc_receivable_days(self, receivables: float, revenue: float) -> float:
        """Calculate Days Sales Outstanding (DSO)."""
        if revenue <= 0:
            return 0.0
        return (receivables / revenue) * 365

    def _calc_inventory_days(self, inventory: float, cogs: float) -> float:
        """Calculate Days Inventory Outstanding (DIO)."""
        if cogs <= 0:
            return 0.0
        return (inventory / cogs) * 365

    def _calc_payable_days(self, payables: float, cogs: float) -> float:
        """Calculate Days Payable Outstanding (DPO)."""
        if cogs <= 0:
            return 0.0
        return (payables / cogs) * 365

    def _calc_growth(self, prior: float, current: float) -> float:
        """Calculate YoY growth percentage."""
        if prior <= 0:
            return 0.0
        return ((current / prior) - 1) * 100

    def _get_sector_benchmarks(
        self, sector: Optional[str]
    ) -> Optional[dict[str, float]]:
        """Get sector benchmarks for comparison."""
        if not sector:
            return None

        # Try exact match first
        if sector in SECTOR_BENCHMARKS:
            return SECTOR_BENCHMARKS[sector]

        # Try case-insensitive partial match
        sector_lower = sector.lower()
        for key in SECTOR_BENCHMARKS:
            if key.lower() in sector_lower or sector_lower in key.lower():
                return SECTOR_BENCHMARKS[key]

        return None

    def _assess_trend(self, trend: WorkingCapitalTrend) -> str:
        """Generate overall assessment text."""
        issues: list[str] = []

        if trend.ccc_trend == "Worsening":
            issues.append("Cash conversion cycle worsening over time")

        if trend.channel_stuffing_years:
            years = ", ".join(trend.channel_stuffing_years)
            issues.append(f"Channel stuffing risk detected in {years}")

        if trend.inventory_piling_years:
            years = ", ".join(trend.inventory_piling_years)
            issues.append(f"Inventory piling detected in {years}")

        # Check latest year vs sector
        if trend.analyses:
            latest = trend.analyses[-1]
            if (
                latest.variance_vs_sector_ccc
                and latest.variance_vs_sector_ccc > self.SECTOR_VARIANCE_CONCERN_THRESHOLD
            ):
                issues.append(
                    f"CCC {latest.variance_vs_sector_ccc:.0f}% above sector average"
                )

        if not issues:
            return "Working capital management appears healthy with no major concerns"

        return "Concerns: " + "; ".join(issues)


__all__ = [
    "WorkingCapitalAnalyzer",
    "WorkingCapitalAnalysis",
    "WorkingCapitalMetrics",
    "WorkingCapitalTrend",
    "SECTOR_BENCHMARKS",
]
