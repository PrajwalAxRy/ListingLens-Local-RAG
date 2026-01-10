"""Objects of the Issue tracker module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger


@dataclass
class ObjectsOfIssueAnalysis:
    """Normalized view of the use of IPO proceeds."""

    total_issue_size: float = 0.0
    fresh_issue: float = 0.0
    ofs: float = 0.0
    fresh_issue_percent: float = 0.0
    ofs_percent: float = 0.0
    capex_amount: float = 0.0
    debt_repayment_amount: float = 0.0
    working_capital_amount: float = 0.0
    acquisition_amount: float = 0.0
    general_corporate_purposes: float = 0.0
    issue_expenses: float = 0.0
    deployment_schedule: List[Dict[str, str]] = field(default_factory=list)
    land_acquired_for_capex: bool = False
    approvals_in_place: bool = False
    capex_already_incurred: float = 0.0
    orders_placed_for_equipment: bool = False
    has_monitoring_agency: bool = False
    monitoring_agency_name: Optional[str] = None
    is_growth_oriented: bool = False
    is_exit_oriented: bool = False
    is_deleveraging: bool = False
    gcp_exceeds_25_percent: bool = False
    vague_deployment_timeline: bool = False


class ObjectsOfIssueTracker:
    """Analyze the Objects of the Issue section."""

    CATEGORY_ALIASES = {
        "capex": {"capex", "capital expenditure", "expansion"},
        "debt_repayment": {"debt", "loan", "repayment"},
        "working_capital": {"working", "wc"},
        "acquisition": {"acquisition", "investment"},
        "gcp": {"general", "gcp"},
        "issue_expenses": {"issue expense", "offer expense", "ofer"},
    }

    def __init__(self) -> None:
        self.logger = logger.bind(module="objects_of_issue_tracker")

    def analyze_objects(self, state: Dict) -> ObjectsOfIssueAnalysis:
        result = ObjectsOfIssueAnalysis()
        ipo_details = state.get("ipo_details", {})
        objects_data = state.get("objects_of_issue", {})
        uses = objects_data.get("uses", [])

        result.fresh_issue = float(ipo_details.get("fresh_issue_cr") or objects_data.get("fresh_issue_cr", 0.0))
        result.ofs = float(ipo_details.get("ofs_cr") or objects_data.get("ofs_cr", 0.0))
        result.total_issue_size = float(ipo_details.get("issue_size_cr") or (result.fresh_issue + result.ofs))

        self._apply_use_breakdown(result, uses)
        self._apply_percentages(result)
        self._apply_readiness_flags(result, objects_data)
        self._apply_monitoring_info(result, objects_data)
        self._evaluate_posture(result)
        return result

    def _apply_use_breakdown(self, result: ObjectsOfIssueAnalysis, uses: List[Dict[str, float]]) -> None:
        for use in uses:
            category = self._normalize_category(use.get("category", "other"))
            amount = float(use.get("amount_cr", 0.0))
            if category == "capex":
                result.capex_amount += amount
            elif category == "debt_repayment":
                result.debt_repayment_amount += amount
            elif category == "working_capital":
                result.working_capital_amount += amount
            elif category == "acquisition":
                result.acquisition_amount += amount
            elif category == "gcp":
                result.general_corporate_purposes += amount
            elif category == "issue_expenses":
                result.issue_expenses += amount

        result.deployment_schedule = uses or []
        result.vague_deployment_timeline = not bool(result.deployment_schedule)

    def _apply_percentages(self, result: ObjectsOfIssueAnalysis) -> None:
        if result.total_issue_size:
            result.fresh_issue_percent = round((result.fresh_issue / result.total_issue_size) * 100, 2)
            result.ofs_percent = round((result.ofs / result.total_issue_size) * 100, 2)
        if result.fresh_issue:
            result.gcp_exceeds_25_percent = (
                (result.general_corporate_purposes / result.fresh_issue) * 100
            ) > 25

    def _apply_readiness_flags(self, result: ObjectsOfIssueAnalysis, objects_data: Dict) -> None:
        readiness = objects_data.get("readiness", {})
        result.land_acquired_for_capex = bool(readiness.get("land_acquired"))
        result.approvals_in_place = bool(readiness.get("approvals_in_place"))
        result.capex_already_incurred = float(readiness.get("capex_incurred_cr", 0.0))
        result.orders_placed_for_equipment = bool(readiness.get("orders_placed"))

    def _apply_monitoring_info(self, result: ObjectsOfIssueAnalysis, objects_data: Dict) -> None:
        monitoring = objects_data.get("monitoring_agency")
        if monitoring:
            result.has_monitoring_agency = True
            if isinstance(monitoring, dict):
                result.monitoring_agency_name = monitoring.get("name")
            elif isinstance(monitoring, str):
                result.monitoring_agency_name = monitoring

    def _evaluate_posture(self, result: ObjectsOfIssueAnalysis) -> None:
        result.is_growth_oriented = result.capex_amount > result.debt_repayment_amount
        result.is_exit_oriented = result.ofs > result.fresh_issue
        result.is_deleveraging = result.debt_repayment_amount > (result.capex_amount + result.working_capital_amount)

    def _normalize_category(self, label: str) -> str:
        lowered = label.lower()
        for key, aliases in self.CATEGORY_ALIASES.items():
            if any(alias in lowered for alias in aliases):
                return key
        return "other"
