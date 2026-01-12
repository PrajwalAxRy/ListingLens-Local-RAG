"""Risk & Litigation Quantification Module.

Parses "Outstanding Litigation", "Material Developments", and "Contingent
Liabilities" sections of the RHP. Aggregates litigation by entity and type,
calculates percentage of post-issue net worth, and flags timeline risks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger


class EntityType(str, Enum):
    """Entity types for litigation classification."""

    COMPANY = "Company"
    PROMOTER = "Promoter"
    DIRECTOR = "Director"
    SUBSIDIARY = "Subsidiary"
    GROUP_COMPANY = "Group Company"


class CaseType(str, Enum):
    """Case type classification."""

    CIVIL = "Civil"
    CRIMINAL = "Criminal"
    TAX = "Tax"
    REGULATORY = "Regulatory"
    LABOR = "Labor"
    ENVIRONMENTAL = "Environmental"
    OTHER = "Other"


class Severity(str, Enum):
    """Risk severity levels."""

    CRITICAL = "Critical"
    MAJOR = "Major"
    MINOR = "Minor"


class Probability(str, Enum):
    """Probability of liability crystallizing."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


# Probability weights for risk-weighting contingent liabilities
PROBABILITY_WEIGHTS = {
    Probability.HIGH: 0.75,
    Probability.MEDIUM: 0.50,
    Probability.LOW: 0.25,
}


@dataclass
class LitigationItem:
    """Individual litigation or claim item."""

    entity_type: EntityType
    entity_name: str
    case_type: CaseType
    description: str
    amount_cr: float  # Amount in crores
    count: int = 1
    hearing_date: Optional[str] = None
    status: str = "Pending"
    page_reference: Optional[int] = None
    citation: str = ""


@dataclass
class ContingentItem:
    """Individual contingent liability item."""

    category: str  # Tax, Customs, Bank Guarantee, Legal, etc.
    description: str
    amount_cr: float
    probability: Probability = Probability.MEDIUM
    page_reference: Optional[int] = None
    citation: str = ""


@dataclass
class RiskExposure:
    """Quantified risk item for reporting."""

    entity: str  # Company/Promoter/Subsidiary/Director
    category: str  # Civil/Criminal/Tax/Regulatory
    count: int
    amount_cr: float
    percent_networth: float
    severity: Severity
    next_hearing: Optional[str] = None
    citation: str = ""


@dataclass
class LitigationSummary:
    """Aggregated litigation summary by entity."""

    entity_type: EntityType
    total_count: int = 0
    civil_count: int = 0
    criminal_count: int = 0
    tax_count: int = 0
    regulatory_count: int = 0
    other_count: int = 0
    total_amount_cr: float = 0.0
    percent_networth: float = 0.0
    items: list[LitigationItem] = field(default_factory=list)


@dataclass
class ContingentLiabilityAnalysis:
    """Categorized contingent liability analysis."""

    total_contingent: float = 0.0
    tax_disputes: float = 0.0
    customs_excise_disputes: float = 0.0
    legal_claims: float = 0.0
    bank_guarantees: float = 0.0
    labor_disputes: float = 0.0
    environmental: float = 0.0
    regulatory_fines: float = 0.0
    other_categories: dict[str, float] = field(default_factory=dict)

    # Risk-weighted amounts
    high_probability_exposure: float = 0.0
    medium_probability_exposure: float = 0.0
    low_probability_exposure: float = 0.0
    risk_weighted_total: float = 0.0

    # Post-IPO analysis
    percent_of_post_ipo_networth: float = 0.0
    earmarked_in_objects: bool = False
    earmarked_amount: float = 0.0

    # Detailed items
    items: list[ContingentItem] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


@dataclass
class RiskQuantificationResult:
    """Complete risk quantification output."""

    # Litigation by entity
    company_litigation: LitigationSummary
    promoter_litigation: LitigationSummary
    director_litigation: LitigationSummary
    subsidiary_litigation: LitigationSummary

    # Totals
    total_litigation_count: int = 0
    total_litigation_amount_cr: float = 0.0
    total_litigation_percent_networth: float = 0.0

    # Criminal flags
    criminal_cases_count: int = 0
    criminal_cases_against_promoters: int = 0

    # Timeline risks
    matters_with_hearing_in_12_months: int = 0
    amount_at_risk_in_12_months: float = 0.0

    # Contingent liabilities
    contingent_analysis: ContingentLiabilityAnalysis = field(
        default_factory=ContingentLiabilityAnalysis
    )

    # Risk exposures for reporting
    risk_exposures: list[RiskExposure] = field(default_factory=list)

    # Overall assessment
    severity: Severity = Severity.MINOR
    veto_flag: bool = False
    veto_reason: str = ""

    citations: list[str] = field(default_factory=list)


class RiskLitigationQuantifier:
    """Quantify and categorize all legal and contingent risks.

    Parses litigation tables, aggregates by entity/type, calculates
    percentage of net worth, and flags high-risk items.
    """

    # Veto thresholds
    LITIGATION_NETWORTH_VETO_THRESHOLD = 10.0  # %
    CRIMINAL_PROMOTER_VETO = True

    def __init__(self, post_issue_networth: float = 0.0) -> None:
        """Initialize quantifier.

        Args:
            post_issue_networth: Post-issue net worth in crores.
        """
        self.post_issue_networth = post_issue_networth
        self.logger = logger.bind(module="risk_quant")

    def quantify_risks(
        self,
        litigation_items: list[LitigationItem],
        contingent_items: list[ContingentItem],
        *,
        objects_earmarked: float = 0.0,
    ) -> RiskQuantificationResult:
        """Quantify all legal and contingent risks.

        Args:
            litigation_items: List of litigation items parsed from RHP.
            contingent_items: List of contingent liability items.
            objects_earmarked: Amount earmarked for settlements in Objects.

        Returns:
            Comprehensive risk quantification result.
        """
        # Aggregate litigation by entity
        company_lit = self._aggregate_litigation(
            litigation_items, EntityType.COMPANY
        )
        promoter_lit = self._aggregate_litigation(
            litigation_items, EntityType.PROMOTER
        )
        director_lit = self._aggregate_litigation(
            litigation_items, EntityType.DIRECTOR
        )
        subsidiary_lit = self._aggregate_litigation(
            litigation_items, EntityType.SUBSIDIARY
        )

        # Calculate totals
        total_lit_count = (
            company_lit.total_count
            + promoter_lit.total_count
            + director_lit.total_count
            + subsidiary_lit.total_count
        )
        total_lit_amount = (
            company_lit.total_amount_cr
            + promoter_lit.total_amount_cr
            + director_lit.total_amount_cr
            + subsidiary_lit.total_amount_cr
        )
        total_lit_pct = self._calc_percent_networth(total_lit_amount)

        # Criminal cases
        criminal_count = (
            company_lit.criminal_count
            + promoter_lit.criminal_count
            + director_lit.criminal_count
            + subsidiary_lit.criminal_count
        )
        criminal_promoter = promoter_lit.criminal_count

        # Timeline risks
        hearing_12m_count, hearing_12m_amount = self._calc_timeline_risk(
            litigation_items
        )

        # Analyze contingent liabilities
        contingent_analysis = self._analyze_contingent(
            contingent_items, objects_earmarked
        )

        # Build risk exposures for reporting
        risk_exposures = self._build_risk_exposures(
            [company_lit, promoter_lit, director_lit, subsidiary_lit]
        )

        # Determine severity and veto
        severity, veto_flag, veto_reason = self._assess_severity(
            total_lit_pct,
            criminal_promoter,
            contingent_analysis.percent_of_post_ipo_networth,
        )

        # Collect citations
        all_citations: list[str] = []
        for item in litigation_items:
            if item.citation:
                all_citations.append(item.citation)
        for item in contingent_items:
            if item.citation:
                all_citations.append(item.citation)

        result = RiskQuantificationResult(
            company_litigation=company_lit,
            promoter_litigation=promoter_lit,
            director_litigation=director_lit,
            subsidiary_litigation=subsidiary_lit,
            total_litigation_count=total_lit_count,
            total_litigation_amount_cr=total_lit_amount,
            total_litigation_percent_networth=total_lit_pct,
            criminal_cases_count=criminal_count,
            criminal_cases_against_promoters=criminal_promoter,
            matters_with_hearing_in_12_months=hearing_12m_count,
            amount_at_risk_in_12_months=hearing_12m_amount,
            contingent_analysis=contingent_analysis,
            risk_exposures=risk_exposures,
            severity=severity,
            veto_flag=veto_flag,
            veto_reason=veto_reason,
            citations=list(set(all_citations)),
        )

        self.logger.info(
            "Risk quantification complete: {} litigation items, {} contingent items, severity={}",
            total_lit_count,
            len(contingent_items),
            severity.value,
        )

        return result

    def _aggregate_litigation(
        self,
        items: list[LitigationItem],
        entity_type: EntityType,
    ) -> LitigationSummary:
        """Aggregate litigation items for a specific entity type."""
        filtered = [i for i in items if i.entity_type == entity_type]

        summary = LitigationSummary(entity_type=entity_type, items=filtered)

        for item in filtered:
            summary.total_count += item.count
            summary.total_amount_cr += item.amount_cr

            if item.case_type == CaseType.CIVIL:
                summary.civil_count += item.count
            elif item.case_type == CaseType.CRIMINAL:
                summary.criminal_count += item.count
            elif item.case_type == CaseType.TAX:
                summary.tax_count += item.count
            elif item.case_type == CaseType.REGULATORY:
                summary.regulatory_count += item.count
            else:
                summary.other_count += item.count

        summary.percent_networth = self._calc_percent_networth(
            summary.total_amount_cr
        )

        return summary

    def _analyze_contingent(
        self,
        items: list[ContingentItem],
        objects_earmarked: float,
    ) -> ContingentLiabilityAnalysis:
        """Analyze and categorize contingent liabilities."""
        analysis = ContingentLiabilityAnalysis(
            earmarked_in_objects=objects_earmarked > 0,
            earmarked_amount=objects_earmarked,
            items=items,
        )

        for item in items:
            category_lower = item.category.lower()
            analysis.total_contingent += item.amount_cr

            # Categorize
            if "tax" in category_lower or "income tax" in category_lower:
                analysis.tax_disputes += item.amount_cr
            elif "customs" in category_lower or "excise" in category_lower:
                analysis.customs_excise_disputes += item.amount_cr
            elif "legal" in category_lower or "civil" in category_lower:
                analysis.legal_claims += item.amount_cr
            elif "bank" in category_lower or "guarantee" in category_lower:
                analysis.bank_guarantees += item.amount_cr
            elif "labor" in category_lower or "employee" in category_lower:
                analysis.labor_disputes += item.amount_cr
            elif "environment" in category_lower:
                analysis.environmental += item.amount_cr
            elif "regulatory" in category_lower or "sebi" in category_lower:
                analysis.regulatory_fines += item.amount_cr
            else:
                key = item.category
                analysis.other_categories[key] = (
                    analysis.other_categories.get(key, 0.0) + item.amount_cr
                )

            # Risk-weight by probability
            weight = PROBABILITY_WEIGHTS.get(item.probability, 0.5)
            weighted_amount = item.amount_cr * weight

            if item.probability == Probability.HIGH:
                analysis.high_probability_exposure += item.amount_cr
            elif item.probability == Probability.MEDIUM:
                analysis.medium_probability_exposure += item.amount_cr
            else:
                analysis.low_probability_exposure += item.amount_cr

            analysis.risk_weighted_total += weighted_amount

            if item.citation:
                analysis.citations.append(item.citation)

        analysis.percent_of_post_ipo_networth = self._calc_percent_networth(
            analysis.total_contingent
        )

        return analysis

    def _calc_timeline_risk(
        self,
        items: list[LitigationItem],
    ) -> tuple[int, float]:
        """Calculate matters with hearings in next 12 months.

        Returns:
            Tuple of (count, amount_at_risk).
        """
        # In a real implementation, would parse hearing dates
        # For now, check if hearing_date is provided
        count = 0
        amount = 0.0

        for item in items:
            if item.hearing_date:
                # Simplified: assume any item with hearing_date is within 12 months
                # A real implementation would parse and compare dates
                count += item.count
                amount += item.amount_cr

        return count, amount

    def _build_risk_exposures(
        self,
        summaries: list[LitigationSummary],
    ) -> list[RiskExposure]:
        """Build risk exposures for reporting."""
        exposures: list[RiskExposure] = []

        for summary in summaries:
            if summary.total_count == 0:
                continue

            # One exposure per case type if significant
            for case_type, count_attr in [
                (CaseType.CRIMINAL, "criminal_count"),
                (CaseType.CIVIL, "civil_count"),
                (CaseType.TAX, "tax_count"),
                (CaseType.REGULATORY, "regulatory_count"),
            ]:
                count = getattr(summary, count_attr, 0)
                if count > 0:
                    # Estimate amount per case type (simplified)
                    items_of_type = [
                        i for i in summary.items if i.case_type == case_type
                    ]
                    amount = sum(i.amount_cr for i in items_of_type)
                    pct = self._calc_percent_networth(amount)

                    # Determine severity
                    if case_type == CaseType.CRIMINAL:
                        severity = Severity.CRITICAL
                    elif pct > 5.0:
                        severity = Severity.CRITICAL
                    elif pct > 2.0:
                        severity = Severity.MAJOR
                    else:
                        severity = Severity.MINOR

                    # Get next hearing
                    next_hearing = None
                    for i in items_of_type:
                        if i.hearing_date:
                            next_hearing = i.hearing_date
                            break

                    exposures.append(
                        RiskExposure(
                            entity=summary.entity_type.value,
                            category=case_type.value,
                            count=count,
                            amount_cr=amount,
                            percent_networth=pct,
                            severity=severity,
                            next_hearing=next_hearing,
                        )
                    )

        return exposures

    def _assess_severity(
        self,
        litigation_pct: float,
        criminal_promoter: int,
        contingent_pct: float,
    ) -> tuple[Severity, bool, str]:
        """Assess overall severity and veto flags."""
        veto_flag = False
        veto_reason = ""

        # Check veto conditions
        if litigation_pct > self.LITIGATION_NETWORTH_VETO_THRESHOLD:
            veto_flag = True
            veto_reason = (
                f"Litigation exposure ({litigation_pct:.1f}%) exceeds "
                f"{self.LITIGATION_NETWORTH_VETO_THRESHOLD}% of net worth"
            )

        if self.CRIMINAL_PROMOTER_VETO and criminal_promoter > 0:
            veto_flag = True
            if veto_reason:
                veto_reason += "; "
            veto_reason += f"Criminal cases against promoters: {criminal_promoter}"

        # Determine severity
        combined_pct = litigation_pct + contingent_pct
        if veto_flag or combined_pct > 15.0 or criminal_promoter > 0:
            severity = Severity.CRITICAL
        elif combined_pct > 5.0:
            severity = Severity.MAJOR
        else:
            severity = Severity.MINOR

        return severity, veto_flag, veto_reason

    def _calc_percent_networth(self, amount: float) -> float:
        """Calculate amount as percentage of post-issue net worth."""
        if self.post_issue_networth <= 0:
            return 0.0
        return (amount / self.post_issue_networth) * 100.0


__all__ = [
    "RiskLitigationQuantifier",
    "RiskQuantificationResult",
    "LitigationItem",
    "LitigationSummary",
    "ContingentItem",
    "ContingentLiabilityAnalysis",
    "RiskExposure",
    "EntityType",
    "CaseType",
    "Severity",
    "Probability",
]
