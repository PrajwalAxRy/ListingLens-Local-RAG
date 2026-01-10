"""Unit tests for specialized ingestion analyzers."""

from datetime import datetime, timedelta

import pytest

from rhp_analyzer.ingestion.contingent_liability_analyzer import (
    ContingentLiabilityCategorizer,
)
from rhp_analyzer.ingestion.objects_tracker import ObjectsOfIssueTracker
from rhp_analyzer.ingestion.stub_analyzer import StubPeriodAnalyzer


def test_contingent_liability_categorizer_basic():
    categorizer = ContingentLiabilityCategorizer()
    state = {
        "contingent_liabilities": [
            {
                "entity": "Company",
                "category": "tax dispute",
                "amount_cr": 50,
                "probability": "high",
                "next_hearing": (datetime.today() + timedelta(days=120)).strftime("%Y-%m-%d"),
            },
            {
                "entity": "Promoter",
                "category": "civil suit",
                "amount_cr": 20,
                "probability": "low",
            },
        ],
        "objects_of_issue": {"litigation_settlement_amount": 5},
    }
    analysis = categorizer.categorize_contingencies(state, post_ipo_networth=200)

    assert pytest.approx(analysis.total_contingent, rel=1e-3) == 70
    assert analysis.category_totals["tax"] == 50
    assert pytest.approx(analysis.total_as_percent_networth, rel=1e-3) == 35
    assert analysis.matters_with_hearing_in_12_months == 1
    assert pytest.approx(analysis.risk_weighted_total, rel=1e-3) == 50 * 0.75 + 20 * 0.25
    assert analysis.amount_earmarked_from_ipo == 5
    assert len(analysis.items) == 2


def test_objects_of_issue_tracker_breakdown():
    tracker = ObjectsOfIssueTracker()
    state = {
        "ipo_details": {"fresh_issue_cr": 120, "ofs_cr": 80, "issue_size_cr": 200},
        "objects_of_issue": {
            "uses": [
                {"category": "capital expenditure", "amount_cr": 60},
                {"category": "debt repayment", "amount_cr": 40},
                {"category": "working capital", "amount_cr": 20},
                {"category": "general corporate purposes", "amount_cr": 10},
            ],
            "readiness": {
                "land_acquired": True,
                "approvals_in_place": False,
                "capex_incurred_cr": 5,
                "orders_placed": True,
            },
            "monitoring_agency": {"name": "Big Four LLP"},
        },
    }
    analysis = tracker.analyze_objects(state)

    assert analysis.total_issue_size == 200
    assert analysis.capex_amount == 60
    assert analysis.debt_repayment_amount == 40
    assert analysis.general_corporate_purposes == 10
    assert analysis.has_monitoring_agency is True
    assert analysis.monitoring_agency_name == "Big Four LLP"
    assert analysis.land_acquired_for_capex is True
    assert analysis.is_growth_oriented is True
    assert analysis.is_exit_oriented is False
    assert analysis.gcp_exceeds_25_percent is False


def test_stub_period_analyzer_comparisons():
    analyzer = StubPeriodAnalyzer()
    state = {
        "stub_period": {
            "period": "6 months ended Sep 30 2025",
            "months": 6,
            "revenue": 400,
            "ebitda": 80,
            "pat": 50,
            "prior_period": {
                "period": "6 months ended Sep 30 2024",
                "revenue": 320,
                "ebitda": 60,
                "pat": 40,
            },
            "is_seasonal": True,
            "seasonality_notes": "H1 captures festive demand",
        }
    }
    historical = [
        {"fiscal_year": "FY22", "revenue": 500},
        {"fiscal_year": "FY23", "revenue": 620},
    ]

    analysis = analyzer.analyze_stub(state, historical)
    assert analysis is not None
    assert pytest.approx(analysis.revenue_growth_yoy, rel=1e-3) == ((400 / 320) - 1) * 100
    assert analysis.is_business_seasonal is True
    assert analysis.margin_compression_in_stub is False
    assert analysis.stub_growth_below_historical_cagr is False
    assert pytest.approx(analysis.annualized_revenue, rel=1e-3) == 800
