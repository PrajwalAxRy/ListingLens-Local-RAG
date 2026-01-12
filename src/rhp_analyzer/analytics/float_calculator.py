"""Float Calculator Module.

Calculates tradeable free float at various post-listing milestones,
builds lock-in expiry calendar, and estimates implied daily trading volume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Optional

from loguru import logger


class InvestorCategory(Enum):
    """Pre-IPO investor category classification."""

    PROMOTER = "Promoter"
    PROMOTER_GROUP = "Promoter Group"
    PE_VC = "PE/VC"
    STRATEGIC = "Strategic Investor"
    ESOP_TRUST = "ESOP Trust"
    ANCHOR = "Anchor Investor"
    HNI = "HNI"
    RETAIL = "Retail"
    EMPLOYEE = "Employee"
    OTHER = "Other"


class LockInPeriod(Enum):
    """Standard lock-in periods for different investor categories."""

    PROMOTER_3_YEAR = 1095  # 3 years in days
    PROMOTER_1_YEAR = 365  # 1 year for non-individual promoters
    PRE_IPO_1_YEAR = 365  # 1 year for pre-IPO investors
    PRE_IPO_6_MONTH = 180  # 6 months for certain pre-IPO investors
    ANCHOR_90_DAY = 90  # 90 days for anchor investors
    EMPLOYEE_1_YEAR = 365  # 1 year for employee shares
    NONE = 0  # No lock-in (retail, QIB non-anchor)


@dataclass
class ShareholderBlock:
    """Represents a block of shares held by an investor/group."""

    name: str
    category: InvestorCategory
    shares_pre_ipo: int = 0
    shares_post_ipo: int = 0
    shares_selling_ofs: int = 0
    lock_in_days: int = 0
    lock_in_expiry: Optional[date] = None
    shares_locked: int = 0
    shares_free: int = 0

    def calculate_post_ipo(self) -> None:
        """Calculate post-IPO shares and lock-in status."""
        self.shares_post_ipo = self.shares_pre_ipo - self.shares_selling_ofs
        # Locked shares typically exclude OFS shares
        self.shares_locked = min(self.shares_post_ipo, self.shares_locked)
        self.shares_free = self.shares_post_ipo - self.shares_locked


@dataclass
class LockInEvent:
    """A lock-in expiry event."""

    event_date: date
    days_from_listing: int
    investor_name: str
    category: InvestorCategory
    shares_unlocking: int
    percent_of_total: float
    cumulative_free_float_percent: float
    description: str = ""


@dataclass
class FloatAnalysis:
    """Comprehensive free float and liquidity analysis."""

    # Share capital structure
    total_shares_pre_issue: int = 0
    total_shares_post_issue: int = 0
    fresh_issue_shares: int = 0
    ofs_shares: int = 0

    # Lock-in breakdown
    promoter_locked_shares: int = 0
    promoter_locked_percent: float = 0.0
    anchor_locked_shares: int = 0
    pre_ipo_locked_shares: int = 0
    esop_unvested_shares: int = 0

    # Float calculations at different milestones
    day_1_free_float_shares: int = 0
    day_1_free_float_percent: float = 0.0
    day_90_free_float_shares: int = 0
    day_90_free_float_percent: float = 0.0
    day_180_free_float_shares: int = 0
    day_180_free_float_percent: float = 0.0
    year_1_free_float_shares: int = 0
    year_1_free_float_percent: float = 0.0

    # Quota breakdown
    retail_quota_shares: int = 0
    retail_quota_percent: float = 0.0
    nii_quota_shares: int = 0
    nii_quota_percent: float = 0.0
    qib_quota_shares: int = 0
    qib_quota_percent: float = 0.0

    # Liquidity indicators
    implied_daily_volume: float = 0.0  # Free float / 250 trading days
    low_float_warning: bool = False  # True if Day 1 float < 5%

    # Lock-in expiry calendar
    lock_in_calendar: list[LockInEvent] = field(default_factory=list)

    # Summary
    summary: str = ""
    citations: list[str] = field(default_factory=list)


@dataclass
class IPOShareDetails:
    """IPO share capital details for float calculation."""

    # Pre-issue share capital
    shares_pre_issue: int = 0

    # Fresh issue details
    fresh_issue_shares: int = 0

    # Quota allocation (typically for book-built issues)
    qib_percent: float = 50.0  # 50% for QIB
    nii_percent: float = 15.0  # 15% for NII (HNI)
    retail_percent: float = 35.0  # 35% for Retail

    # Anchor investor details (subset of QIB)
    anchor_shares: int = 0
    anchor_lock_in_days: int = 90

    # Listing date for lock-in calculation
    listing_date: Optional[date] = None


class FloatCalculator:
    """Calculate free float at various time horizons.

    Analyzes share capital structure, lock-in periods, and calculates
    tradeable float at Day 1, Day 90, 6 months, and 1 year post-listing.
    """

    # Thresholds
    LOW_FLOAT_THRESHOLD = 5.0  # % - very low liquidity risk
    ADEQUATE_FLOAT_THRESHOLD = 15.0  # % - minimum for reasonable liquidity
    TRADING_DAYS_PER_YEAR = 250

    def __init__(self) -> None:
        """Initialize calculator."""
        self.logger = logger.bind(module="float_calc")

    def calculate_float_analysis(
        self,
        ipo_details: IPOShareDetails,
        shareholders: list[ShareholderBlock],
    ) -> FloatAnalysis:
        """Calculate comprehensive float analysis.

        Args:
            ipo_details: IPO share capital details.
            shareholders: List of shareholder blocks with lock-in info.

        Returns:
            Complete float analysis.
        """
        analysis = FloatAnalysis()

        # Calculate total shares
        analysis.total_shares_pre_issue = ipo_details.shares_pre_issue
        analysis.fresh_issue_shares = ipo_details.fresh_issue_shares
        analysis.total_shares_post_issue = (
            ipo_details.shares_pre_issue + ipo_details.fresh_issue_shares
        )

        # Calculate OFS
        total_ofs = sum(sh.shares_selling_ofs for sh in shareholders)
        analysis.ofs_shares = total_ofs

        # Process shareholder blocks
        for sh in shareholders:
            sh.calculate_post_ipo()

        # Calculate locked shares by category
        promoter_locked = 0
        pre_ipo_locked = 0
        esop_locked = 0

        for sh in shareholders:
            if sh.category in (InvestorCategory.PROMOTER, InvestorCategory.PROMOTER_GROUP):
                promoter_locked += sh.shares_locked
            elif sh.category == InvestorCategory.ESOP_TRUST:
                esop_locked += sh.shares_locked
            elif sh.category == InvestorCategory.ANCHOR:
                analysis.anchor_locked_shares += sh.shares_locked
            else:
                pre_ipo_locked += sh.shares_locked

        analysis.promoter_locked_shares = promoter_locked
        analysis.promoter_locked_percent = (
            (promoter_locked / analysis.total_shares_post_issue) * 100
            if analysis.total_shares_post_issue > 0
            else 0
        )
        analysis.pre_ipo_locked_shares = pre_ipo_locked
        analysis.esop_unvested_shares = esop_locked

        # Calculate anchor locked shares
        if ipo_details.anchor_shares > 0:
            analysis.anchor_locked_shares = ipo_details.anchor_shares

        # Total locked at listing (Day 1)
        total_locked_day_1 = (
            promoter_locked
            + pre_ipo_locked
            + esop_locked
            + analysis.anchor_locked_shares
        )

        # Day 1 free float
        analysis.day_1_free_float_shares = (
            analysis.total_shares_post_issue - total_locked_day_1
        )
        analysis.day_1_free_float_percent = (
            (analysis.day_1_free_float_shares / analysis.total_shares_post_issue) * 100
            if analysis.total_shares_post_issue > 0
            else 0
        )

        # Day 90 free float (anchor unlocks)
        analysis.day_90_free_float_shares = (
            analysis.day_1_free_float_shares + analysis.anchor_locked_shares
        )
        analysis.day_90_free_float_percent = (
            (analysis.day_90_free_float_shares / analysis.total_shares_post_issue) * 100
            if analysis.total_shares_post_issue > 0
            else 0
        )

        # Day 180 (6 months) - some pre-IPO investors unlock
        shares_unlocking_180 = sum(
            sh.shares_locked
            for sh in shareholders
            if sh.lock_in_days > 0 and sh.lock_in_days <= 180
            and sh.category not in (
                InvestorCategory.PROMOTER,
                InvestorCategory.PROMOTER_GROUP,
                InvestorCategory.ANCHOR,
            )
        )
        analysis.day_180_free_float_shares = (
            analysis.day_90_free_float_shares + shares_unlocking_180
        )
        analysis.day_180_free_float_percent = (
            (analysis.day_180_free_float_shares / analysis.total_shares_post_issue) * 100
            if analysis.total_shares_post_issue > 0
            else 0
        )

        # Year 1 - most pre-IPO investors unlock (except 3-year promoter lock)
        shares_unlocking_365 = sum(
            sh.shares_locked
            for sh in shareholders
            if sh.lock_in_days > 180 and sh.lock_in_days <= 365
            and sh.category not in (
                InvestorCategory.PROMOTER,
                InvestorCategory.PROMOTER_GROUP,
            )
        )
        analysis.year_1_free_float_shares = (
            analysis.day_180_free_float_shares + shares_unlocking_365
        )
        analysis.year_1_free_float_percent = (
            (analysis.year_1_free_float_shares / analysis.total_shares_post_issue) * 100
            if analysis.total_shares_post_issue > 0
            else 0
        )

        # Calculate quota allocations
        fresh = ipo_details.fresh_issue_shares
        analysis.retail_quota_shares = int(fresh * ipo_details.retail_percent / 100)
        analysis.retail_quota_percent = ipo_details.retail_percent
        analysis.nii_quota_shares = int(fresh * ipo_details.nii_percent / 100)
        analysis.nii_quota_percent = ipo_details.nii_percent
        analysis.qib_quota_shares = int(fresh * ipo_details.qib_percent / 100)
        analysis.qib_quota_percent = ipo_details.qib_percent

        # Liquidity indicators
        analysis.implied_daily_volume = (
            analysis.day_1_free_float_shares / self.TRADING_DAYS_PER_YEAR
        )
        analysis.low_float_warning = (
            analysis.day_1_free_float_percent < self.LOW_FLOAT_THRESHOLD
        )

        # Build lock-in calendar
        if ipo_details.listing_date:
            analysis.lock_in_calendar = self._build_lock_in_calendar(
                shareholders,
                ipo_details,
                analysis.total_shares_post_issue,
            )

        # Generate summary
        analysis.summary = self._generate_summary(analysis)

        self.logger.info(
            "Float calculated: Day 1={:.1f}%, Day 90={:.1f}%, Year 1={:.1f}%",
            analysis.day_1_free_float_percent,
            analysis.day_90_free_float_percent,
            analysis.year_1_free_float_percent,
        )

        return analysis

    def _build_lock_in_calendar(
        self,
        shareholders: list[ShareholderBlock],
        ipo_details: IPOShareDetails,
        total_shares: int,
    ) -> list[LockInEvent]:
        """Build chronological lock-in expiry calendar."""
        events: list[LockInEvent] = []
        listing_date = ipo_details.listing_date

        if not listing_date:
            return events

        cumulative_free_percent = 0.0

        # Calculate initial free float
        total_locked = sum(sh.shares_locked for sh in shareholders)
        if ipo_details.anchor_shares > 0:
            total_locked += ipo_details.anchor_shares

        initial_free = total_shares - total_locked
        cumulative_free_percent = (initial_free / total_shares) * 100 if total_shares > 0 else 0

        # Add anchor unlock at Day 90
        if ipo_details.anchor_shares > 0:
            unlock_date = listing_date + timedelta(days=90)
            anchor_percent = (
                (ipo_details.anchor_shares / total_shares) * 100
                if total_shares > 0
                else 0
            )
            cumulative_free_percent += anchor_percent
            events.append(
                LockInEvent(
                    event_date=unlock_date,
                    days_from_listing=90,
                    investor_name="Anchor Investors",
                    category=InvestorCategory.ANCHOR,
                    shares_unlocking=ipo_details.anchor_shares,
                    percent_of_total=anchor_percent,
                    cumulative_free_float_percent=cumulative_free_percent,
                    description="Anchor lock-in expires",
                )
            )

        # Add unlock events for each shareholder with lock-in
        for sh in shareholders:
            if sh.shares_locked > 0 and sh.lock_in_days > 0 and sh.lock_in_expiry:
                days_from_listing = (sh.lock_in_expiry - listing_date).days
                sh_percent = (
                    (sh.shares_locked / total_shares) * 100
                    if total_shares > 0
                    else 0
                )
                cumulative_free_percent += sh_percent
                events.append(
                    LockInEvent(
                        event_date=sh.lock_in_expiry,
                        days_from_listing=days_from_listing,
                        investor_name=sh.name,
                        category=sh.category,
                        shares_unlocking=sh.shares_locked,
                        percent_of_total=sh_percent,
                        cumulative_free_float_percent=min(100, cumulative_free_percent),
                        description=f"{sh.category.value} lock-in expires",
                    )
                )

        # Sort by date
        events.sort(key=lambda e: e.event_date)

        # Recalculate cumulative after sorting
        running_total = (initial_free / total_shares) * 100 if total_shares > 0 else 0
        for event in events:
            running_total += event.percent_of_total
            event.cumulative_free_float_percent = min(100, running_total)

        return events

    def _generate_summary(self, analysis: FloatAnalysis) -> str:
        """Generate human-readable summary."""
        parts: list[str] = []

        # Day 1 float assessment
        if analysis.day_1_free_float_percent < self.LOW_FLOAT_THRESHOLD:
            parts.append(
                f"Very low Day-1 float of {analysis.day_1_free_float_percent:.1f}% "
                "may cause high volatility"
            )
        elif analysis.day_1_free_float_percent < self.ADEQUATE_FLOAT_THRESHOLD:
            parts.append(
                f"Moderate Day-1 float of {analysis.day_1_free_float_percent:.1f}%"
            )
        else:
            parts.append(
                f"Adequate Day-1 float of {analysis.day_1_free_float_percent:.1f}%"
            )

        # Promoter lock-in
        parts.append(
            f"Promoters holding {analysis.promoter_locked_percent:.1f}% "
            f"({analysis.promoter_locked_shares:,} shares) locked for 3 years"
        )

        # Float evolution
        if analysis.day_90_free_float_percent > analysis.day_1_free_float_percent + 5:
            parts.append(
                f"Significant float increase to {analysis.day_90_free_float_percent:.1f}% "
                "at Day 90 (anchor unlock)"
            )

        # Daily volume estimate
        parts.append(
            f"Estimated avg daily trading volume: {analysis.implied_daily_volume:,.0f} shares"
        )

        return ". ".join(parts) + "."


__all__ = [
    "FloatCalculator",
    "FloatAnalysis",
    "IPOShareDetails",
    "ShareholderBlock",
    "LockInEvent",
    "InvestorCategory",
    "LockInPeriod",
]
