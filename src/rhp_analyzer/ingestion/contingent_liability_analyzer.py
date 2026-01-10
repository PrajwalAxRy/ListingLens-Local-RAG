"""Contingent liability categorizer module for RHP Analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

PROBABILITY_WEIGHTS = {
    "high": 0.75,
    "medium": 0.5,
    "low": 0.25,
}

CATEGORY_KEYWORDS = {
    "tax": {"tax", "gst", "sales", "vat", "income"},
    "customs_excise": {"customs", "excise"},
    "civil": {"civil", "suit", "arbitration"},
    "labor": {"labor", "employee", "employment"},
    "bank_guarantee": {"guarantee", "lc", "letter of credit"},
    "environmental": {"environment", "pollution"},
}


@dataclass
class ContingentLiabilityItem:
    """Individual contingent liability disclosure."""

    entity: str
    category: str
    amount_cr: float
    probability: str
    risk_weighted_amount: float
    next_hearing: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ContingentLiabilityAnalysis:
    """Aggregated view of contingent liabilities."""

    total_contingent: float = 0.0
    total_as_percent_networth: float = 0.0
    category_totals: Dict[str, float] = field(default_factory=dict)
    probability_totals: Dict[str, float] = field(default_factory=dict)
    risk_weighted_total: float = 0.0
    amount_earmarked_from_ipo: float = 0.0
    matters_with_hearing_in_12_months: int = 0
    amount_at_risk_in_12_months: float = 0.0
    items: List[ContingentLiabilityItem] = field(default_factory=list)


class ContingentLiabilityCategorizer:
    """Categorize and risk-weight contingent liabilities."""

    def __init__(self) -> None:
        self.logger = logger.bind(module="contingent_liability_categorizer")

    def categorize_contingencies(self, state: Dict, post_ipo_networth: float) -> ContingentLiabilityAnalysis:
        """Build an exposure summary using structured state inputs."""
        analysis = ContingentLiabilityAnalysis()
        disclosures = state.get("contingent_liabilities", [])
        if not disclosures:
            self.logger.warning("No contingent liabilities provided in state")
            return analysis

        for entry in disclosures:
            amount = float(entry.get("amount_cr") or entry.get("amount", 0.0))
            if amount <= 0:
                continue

            category = self._normalize_category(entry.get("category", "other"))
            probability = self._normalize_probability(entry.get("probability"))
            risk_weight = PROBABILITY_WEIGHTS[probability]
            risk_amount = amount * risk_weight

            item = ContingentLiabilityItem(
                entity=entry.get("entity", "Company"),
                category=category,
                amount_cr=amount,
                probability=probability,
                risk_weighted_amount=risk_amount,
                next_hearing=entry.get("next_hearing"),
                description=entry.get("description"),
            )
            analysis.items.append(item)

            analysis.total_contingent += amount
            analysis.category_totals[category] = analysis.category_totals.get(category, 0.0) + amount
            analysis.probability_totals[probability] = analysis.probability_totals.get(probability, 0.0) + amount
            analysis.risk_weighted_total += risk_amount

            if item.next_hearing and self._is_within_next_12_months(item.next_hearing):
                analysis.matters_with_hearing_in_12_months += 1
                analysis.amount_at_risk_in_12_months += amount

        analysis.total_as_percent_networth = self._safe_percent(analysis.total_contingent, post_ipo_networth)
        analysis.amount_earmarked_from_ipo = self._extract_objects_allocation(state)
        return analysis

    def _normalize_category(self, raw_category: str) -> str:
        lowered = (raw_category or "other").lower()
        for label, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return label
        if "legal" in lowered or "litigation" in lowered:
            return "civil"
        return "other"

    def _normalize_probability(self, value: Optional[str]) -> str:
        if not value:
            return "medium"
        lowered = value.lower()
        return lowered if lowered in PROBABILITY_WEIGHTS else "medium"

    def _is_within_next_12_months(self, value: str) -> bool:
        try:
            hearing_date = datetime.strptime(value, "%Y-%m-%d")
        except (TypeError, ValueError):
            return False
        return hearing_date <= datetime.today() + timedelta(days=365)

    def _safe_percent(self, numerator: float, denominator: float) -> float:
        if not denominator:
            return 0.0
        return round((numerator / denominator) * 100, 2)

    def _extract_objects_allocation(self, state: Dict) -> float:
        objects_data = state.get("objects_of_issue", {})
        return float(objects_data.get("litigation_settlement_amount", 0.0))
