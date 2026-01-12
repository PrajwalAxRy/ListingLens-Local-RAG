"""Analytics modules for RHP financial modeling and valuation."""

from .governance_rules import GovernanceRulebook, RuleInput
from .models import (
    FinancialRecord,
    PeerComparable,
    ProjectionScenario,
    RuleViolation,
    ValuationSummary,
)
from .normalizer import HistoricalNormalizer
from .projection_engine import GuidanceInputs, IPODetails, ProjectionEngine
from .valuation_module import IssuerMetrics, ValuationNormalization

# New modules added for Milestone 3.5A
from .citation_manager import CitationManager, CitationRecord
from .risk_quant import (
    RiskLitigationQuantifier,
    RiskQuantificationResult,
    LitigationItem,
    LitigationSummary,
    ContingentItem,
    ContingentLiabilityAnalysis,
    RiskExposure,
    EntityType,
    CaseType,
    Severity,
    Probability,
)
from .working_capital_analyzer import (
    WorkingCapitalAnalyzer,
    WorkingCapitalAnalysis,
    WorkingCapitalMetrics,
    WorkingCapitalTrend,
)
from .cashflow_analyzer import (
    EnhancedCashFlowAnalyzer,
    CashFlowAnalysis,
    CashFlowMetrics,
    CashFlowTrend,
    CashBurnStatus,
)
from .float_calculator import (
    FloatCalculator,
    FloatAnalysis,
    IPOShareDetails,
    ShareholderBlock,
    LockInEvent,
    InvestorCategory,
    LockInPeriod,
)

__all__ = [
    # Models
    "FinancialRecord",
    "PeerComparable",
    "ProjectionScenario",
    "RuleViolation",
    "ValuationSummary",
    # Normalizer
    "HistoricalNormalizer",
    # Projection Engine
    "GuidanceInputs",
    "IPODetails",
    "ProjectionEngine",
    # Valuation Module
    "IssuerMetrics",
    "ValuationNormalization",
    # Governance Rulebook
    "GovernanceRulebook",
    "RuleInput",
    # Citation Manager
    "CitationManager",
    "CitationRecord",
    # Risk & Litigation Quantification
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
    # Working Capital Analyzer
    "WorkingCapitalAnalyzer",
    "WorkingCapitalAnalysis",
    "WorkingCapitalMetrics",
    "WorkingCapitalTrend",
    # Cash Flow Analyzer
    "EnhancedCashFlowAnalyzer",
    "CashFlowAnalysis",
    "CashFlowMetrics",
    "CashFlowTrend",
    "CashBurnStatus",
    # Float Calculator
    "FloatCalculator",
    "FloatAnalysis",
    "IPOShareDetails",
    "ShareholderBlock",
    "LockInEvent",
    "InvestorCategory",
    "LockInPeriod",
]
