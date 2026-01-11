"""Shared dataclasses for analytics modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FinancialRecord:
    """Normalized financial metrics for a fiscal period.

    All values are stored in crore rupees to keep units consistent across
    downstream analytics modules.
    """

    fiscal_year: str
    revenue: Optional[float] = None
    ebitda: Optional[float] = None
    pat: Optional[float] = None
    total_assets: Optional[float] = None
    total_equity: Optional[float] = None
    total_debt: Optional[float] = None
    cfo: Optional[float] = None
    ccc_days: Optional[float] = None
    statement_type: str = "consolidated"
    source: Optional[str] = None


@dataclass
class ProjectionScenario:
    """Forward looking financial projections derived from RHP disclosures."""

    name: str
    years: list[str]
    revenue: dict[str, float]
    ebitda: dict[str, float]
    pat: dict[str, float]
    diluted_eps: dict[str, float]
    roe: dict[str, float]
    roic: dict[str, float]
    net_debt_to_ebitda: dict[str, float]
    cash_conversion_cycle: dict[str, float]
    assumptions: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


@dataclass
class PeerComparable:
    """Peer valuation snapshot extracted from the RHP tables."""

    name: str
    fiscal_year: Optional[str]
    market_cap_cr: Optional[float]
    pat_cr: Optional[float]
    net_worth_cr: Optional[float]
    ebitda_cr: Optional[float]
    net_debt_cr: Optional[float]
    enterprise_value_cr: Optional[float]
    pe: Optional[float]
    pb: Optional[float]
    ev_ebitda: Optional[float]
    peg: Optional[float]
    citation: Optional[str] = None


@dataclass
class ValuationSummary:
    """Issuer valuation relative to normalized peer group."""

    peer_multiples: list[PeerComparable]
    peer_medians: dict[str, float]
    issuer_floor_metrics: dict[str, float]
    issuer_cap_metrics: dict[str, float]
    premium_discount_vs_peers: dict[str, float]
    missing_peers: list[str]


@dataclass
class RuleViolation:
    """Represents a governance rule breach identified by the rulebook."""

    rule_id: str
    description: str
    severity: str
    actual_value: Optional[float]
    threshold: Optional[float]
    metric: str
    citation: Optional[str] = None


__all__ = [
    "FinancialRecord",
    "ProjectionScenario",
    "PeerComparable",
    "ValuationSummary",
    "RuleViolation",
]
