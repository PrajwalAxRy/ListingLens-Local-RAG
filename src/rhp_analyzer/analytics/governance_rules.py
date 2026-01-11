"""Governance rulebook engine for SEBI-aligned red-flag detection.

Evaluates RHP inputs against configurable rules and emits structured
RuleViolation findings for downstream agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

from .models import RuleViolation


@dataclass
class RuleInput:
    """Aggregated inputs for rule evaluation."""

    # Promoter metrics
    promoter_post_issue_holding_pct: float = 0.0
    promoter_pledge_pct: float = 0.0
    ofs_pct_of_issue: float = 0.0
    fresh_issue_pct_of_issue: float = 0.0

    # Related party
    rpt_revenue_pct: float = 0.0
    rpt_expense_pct: float = 0.0

    # Audit quality
    auditor_resigned: bool = False
    modified_audit_opinion: bool = False
    caro_qualifications: int = 0

    # Litigation
    total_litigation_cr: float = 0.0
    post_issue_networth_cr: float = 0.0

    # Working capital
    receivable_days_growth: float = 0.0  # YoY change
    revenue_cagr_3yr: float = 0.0

    # Citations (optional)
    citations: dict[str, str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.citations is None:
            self.citations = {}


# ------------------------------------------------------------------
# Default rule definitions (used if no YAML provided)
# ------------------------------------------------------------------
DEFAULT_RULES: dict[str, Any] = {
    "rules": [
        # Shareholder skin-in-the-game
        {
            "id": "SKIN-001",
            "family": "shareholder_skin_in_game",
            "description": "Promoter post-issue holding below 51%",
            "severity": "critical",
            "condition": "promoter_post_issue_holding_pct < 51",
            "threshold": "≥51%",
        },
        {
            "id": "SKIN-002",
            "family": "shareholder_skin_in_game",
            "description": "OFS exceeds fresh issue",
            "severity": "major",
            "condition": "ofs_pct_of_issue > fresh_issue_pct_of_issue",
            "threshold": "OFS ≤ Fresh Issue",
        },
        # Pledge & encumbrance
        {
            "id": "PLEDGE-001",
            "family": "pledge_encumbrance",
            "description": "Promoter pledge above 25%",
            "severity": "critical",
            "condition": "promoter_pledge_pct > 25",
            "threshold": "≤25%",
        },
        {
            "id": "PLEDGE-002",
            "family": "pledge_encumbrance",
            "description": "Any promoter pledge exists",
            "severity": "major",
            "condition": "promoter_pledge_pct > 0",
            "threshold": "0%",
        },
        # Related party concentration
        {
            "id": "RPT-001",
            "family": "related_party",
            "description": "RPT revenue exceeds 20% of total",
            "severity": "major",
            "condition": "rpt_revenue_pct > 20",
            "threshold": "≤20%",
        },
        {
            "id": "RPT-002",
            "family": "related_party",
            "description": "RPT expenses exceed 20% of total",
            "severity": "major",
            "condition": "rpt_expense_pct > 20",
            "threshold": "≤20%",
        },
        # Audit quality
        {
            "id": "AUDIT-001",
            "family": "audit_quality",
            "description": "Auditor resignation disclosed",
            "severity": "critical",
            "condition": "auditor_resigned == True",
            "threshold": "No resignation",
        },
        {
            "id": "AUDIT-002",
            "family": "audit_quality",
            "description": "Modified audit opinion",
            "severity": "critical",
            "condition": "modified_audit_opinion == True",
            "threshold": "Unqualified opinion",
        },
        {
            "id": "AUDIT-003",
            "family": "audit_quality",
            "description": "CARO/NCF qualifications present",
            "severity": "major",
            "condition": "caro_qualifications > 0",
            "threshold": "No qualifications",
        },
        # Litigation materiality
        {
            "id": "LITIG-001",
            "family": "litigation",
            "description": "Litigation exceeds 10% of post-issue net worth",
            "severity": "critical",
            "condition": "post_issue_networth_cr > 0 and (total_litigation_cr / post_issue_networth_cr * 100) > 10",
            "threshold": "≤10%",
        },
        # Working capital stress
        {
            "id": "WC-001",
            "family": "working_capital",
            "description": "Receivable days growth exceeds revenue CAGR by 10pp",
            "severity": "major",
            "condition": "(receivable_days_growth - revenue_cagr_3yr) > 10",
            "threshold": "Gap ≤10pp",
        },
    ]
}


class GovernanceRulebook:
    """Rule engine for governance and forensic pre-flagging."""

    def __init__(self, rules_path: Optional[Path] = None) -> None:
        """Initialize rulebook.

        Args:
            rules_path: Optional path to YAML file with custom rules.
                        Falls back to DEFAULT_RULES if not provided.
        """
        self.logger = logger.bind(module="governance_rulebook")
        self.rules: list[dict[str, Any]] = self._load_rules(rules_path)
        self.logger.info("Loaded {} governance rules", len(self.rules))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(self, inputs: RuleInput) -> list[RuleViolation]:
        """Evaluate all rules against given inputs.

        Args:
            inputs: RuleInput dataclass with company metrics.

        Returns:
            List of RuleViolation for any triggered rules.
        """
        violations: list[RuleViolation] = []

        for rule in self.rules:
            triggered = self._evaluate_condition(rule["condition"], inputs)
            if triggered:
                actual = self._get_actual_value(rule, inputs)
                citation = inputs.citations.get(rule["id"], "")
                violation = RuleViolation(
                    rule_id=rule["id"],
                    description=rule["description"],
                    severity=rule["severity"],
                    actual_value=actual,
                    threshold=rule["threshold"],
                    metric=rule["family"],
                    citation=citation,
                )
                violations.append(violation)
                self.logger.warning(
                    "Rule {} triggered: {} (actual={})",
                    rule["id"],
                    rule["description"],
                    actual,
                )

        self.logger.info(
            "Rule evaluation complete: {} violations ({} critical, {} major)",
            len(violations),
            sum(1 for v in violations if v.severity == "critical"),
            sum(1 for v in violations if v.severity == "major"),
        )
        return violations

    def get_critical_violations(
        self,
        violations: list[RuleViolation],
    ) -> list[RuleViolation]:
        """Filter for critical severity violations."""
        return [v for v in violations if v.severity == "critical"]

    def get_veto_flags(self, violations: list[RuleViolation]) -> list[str]:
        """Return veto flag descriptions from critical violations."""
        critical = self.get_critical_violations(violations)
        return [v.description for v in critical]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_rules(self, rules_path: Optional[Path]) -> list[dict[str, Any]]:
        """Load rules from YAML or use defaults."""
        if rules_path and rules_path.exists():
            with open(rules_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("rules", [])
        return DEFAULT_RULES["rules"]

    def _evaluate_condition(self, condition: str, inputs: RuleInput) -> bool:
        """Safely evaluate a condition string against inputs."""
        # Build evaluation namespace from inputs
        namespace: dict[str, Any] = {
            "promoter_post_issue_holding_pct": inputs.promoter_post_issue_holding_pct,
            "promoter_pledge_pct": inputs.promoter_pledge_pct,
            "ofs_pct_of_issue": inputs.ofs_pct_of_issue,
            "fresh_issue_pct_of_issue": inputs.fresh_issue_pct_of_issue,
            "rpt_revenue_pct": inputs.rpt_revenue_pct,
            "rpt_expense_pct": inputs.rpt_expense_pct,
            "auditor_resigned": inputs.auditor_resigned,
            "modified_audit_opinion": inputs.modified_audit_opinion,
            "caro_qualifications": inputs.caro_qualifications,
            "total_litigation_cr": inputs.total_litigation_cr,
            "post_issue_networth_cr": inputs.post_issue_networth_cr,
            "receivable_days_growth": inputs.receivable_days_growth,
            "revenue_cagr_3yr": inputs.revenue_cagr_3yr,
            "True": True,
            "False": False,
        }

        try:
            # Safe eval with restricted namespace
            return bool(eval(condition, {"__builtins__": {}}, namespace))
        except Exception as e:
            self.logger.error("Error evaluating rule condition '{}': {}", condition, e)
            return False

    def _get_actual_value(self, rule: dict[str, Any], inputs: RuleInput) -> str:
        """Extract actual value for display in violation."""
        rule_id = rule["id"]

        if rule_id.startswith("SKIN-001"):
            return f"{inputs.promoter_post_issue_holding_pct:.1f}%"
        if rule_id.startswith("SKIN-002"):
            return f"OFS={inputs.ofs_pct_of_issue:.1f}%, Fresh={inputs.fresh_issue_pct_of_issue:.1f}%"
        if rule_id.startswith("PLEDGE"):
            return f"{inputs.promoter_pledge_pct:.1f}%"
        if rule_id.startswith("RPT-001"):
            return f"{inputs.rpt_revenue_pct:.1f}%"
        if rule_id.startswith("RPT-002"):
            return f"{inputs.rpt_expense_pct:.1f}%"
        if rule_id.startswith("AUDIT-001"):
            return "Yes"
        if rule_id.startswith("AUDIT-002"):
            return "Yes"
        if rule_id.startswith("AUDIT-003"):
            return f"{inputs.caro_qualifications} qualifications"
        if rule_id.startswith("LITIG"):
            if inputs.post_issue_networth_cr > 0:
                pct = inputs.total_litigation_cr / inputs.post_issue_networth_cr * 100
                return f"{pct:.1f}% (₹{inputs.total_litigation_cr:.1f} Cr)"
            return f"₹{inputs.total_litigation_cr:.1f} Cr"
        if rule_id.startswith("WC"):
            gap = inputs.receivable_days_growth - inputs.revenue_cagr_3yr
            return f"{gap:.1f}pp gap"

        return "N/A"


__all__ = [
    "GovernanceRulebook",
    "RuleInput",
    "DEFAULT_RULES",
]
