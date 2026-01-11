"""Unit tests for analytics modules (Milestone 2.5E.1).

Tests cover:
- HistoricalNormalizer (2.5A)
- ProjectionEngine (2.5B)
- ValuationNormalization (2.5C)
- GovernanceRulebook (2.5D)
"""

from __future__ import annotations

import pytest
from pathlib import Path

from rhp_analyzer.analytics import (
    FinancialRecord,
    GuidanceInputs,
    GovernanceRulebook,
    HistoricalNormalizer,
    IPODetails,
    IssuerMetrics,
    PeerComparable,
    ProjectionEngine,
    ProjectionScenario,
    RuleInput,
    RuleViolation,
    ValuationNormalization,
    ValuationSummary,
)


# ============================================================
# Milestone 2.5A: HistoricalNormalizer Tests
# ============================================================
class TestHistoricalNormalizer:
    """Tests for HistoricalNormalizer."""

    def test_normalize_crore_passthrough(self):
        """Crore values should pass through unchanged."""
        normalizer = HistoricalNormalizer()
        records = [
            {"fiscal_year": "FY24", "values": {"revenue": 100.0}, "unit": "crore"},
        ]
        normalized = normalizer.normalize(records)
        assert len(normalized) == 1
        # FinancialRecord uses 'revenue' not 'revenue_cr'
        assert normalized[0].revenue == 100.0

    def test_normalize_lakh_conversion(self):
        """Lakh values should convert to crore (÷100)."""
        normalizer = HistoricalNormalizer()
        records = [
            {"fiscal_year": "FY24", "values": {"revenue": 10000.0}, "unit": "lakh"},
        ]
        normalized = normalizer.normalize(records)
        assert normalized[0].revenue == 100.0  # 10000 lakh = 100 cr

    def test_normalize_million_conversion(self):
        """Million values should convert to crore (÷10)."""
        normalizer = HistoricalNormalizer()
        records = [
            {"fiscal_year": "FY24", "values": {"revenue": 500.0}, "unit": "million"},
        ]
        normalized = normalizer.normalize(records)
        assert normalized[0].revenue == 50.0  # 500 million = 50 cr

    def test_normalize_multiple_years(self):
        """Multiple year records should all be normalized."""
        normalizer = HistoricalNormalizer()
        records = [
            {"fiscal_year": "FY22", "values": {"revenue": 80.0}, "unit": "crore"},
            {"fiscal_year": "FY23", "values": {"revenue": 100.0}, "unit": "crore"},
            {"fiscal_year": "FY24", "values": {"revenue": 120.0}, "unit": "crore"},
        ]
        normalized = normalizer.normalize(records)
        assert len(normalized) == 3
        assert normalized[0].fiscal_year == "FY22"
        assert normalized[2].fiscal_year == "FY24"

    def test_calculate_cagr_positive(self):
        """CAGR calculation for positive growth."""
        normalizer = HistoricalNormalizer()
        # FinancialRecord uses field names without _cr suffix
        records = [
            FinancialRecord(fiscal_year="FY22", revenue=100.0),
            FinancialRecord(fiscal_year="FY23", revenue=120.0),
            FinancialRecord(fiscal_year="FY24", revenue=144.0),
        ]
        cagr = normalizer.calculate_cagr(records, "revenue")
        # (144/100)^(1/2) - 1 = 0.2 = 20%
        assert abs(cagr - 20.0) < 0.1

    def test_calculate_cagr_insufficient_data(self):
        """CAGR returns None with insufficient data."""
        normalizer = HistoricalNormalizer()
        records = [FinancialRecord(fiscal_year="FY24", revenue=100.0)]
        cagr = normalizer.calculate_cagr(records, "revenue")
        assert cagr is None

    def test_normalize_all_fields(self):
        """All financial fields should be normalized."""
        normalizer = HistoricalNormalizer()
        records = [
            {
                "fiscal_year": "FY24",
                "values": {
                    "revenue": 100.0,
                    "ebitda": 20.0,
                    "pat": 15.0,
                    "total_assets": 200.0,
                    "total_equity": 80.0,
                    "total_debt": 50.0,
                    "cfo": 18.0,
                },
                "unit": "crore",
            }
        ]
        normalized = normalizer.normalize(records)
        rec = normalized[0]
        # FinancialRecord uses field names without _cr suffix
        assert rec.revenue == 100.0
        assert rec.ebitda == 20.0
        assert rec.pat == 15.0
        assert rec.total_assets == 200.0
        assert rec.total_equity == 80.0
        assert rec.total_debt == 50.0
        assert rec.cfo == 18.0


# ============================================================
# Milestone 2.5B: ProjectionEngine Tests
# ============================================================
class TestProjectionEngine:
    """Tests for ProjectionEngine."""

    @pytest.fixture
    def ipo_details(self):
        # IPODetails uses actual field names from projection_engine.py
        return IPODetails(
            price_floor=100.0,
            price_cap=110.0,
            shares_pre_issue=50_000_000,
            fresh_issue_shares=27_272_727,  # ~300 Cr / 110 Rs
            fresh_issue_amount_cr=300.0,
            ofs_amount_cr=200.0,
            face_value=10.0,
        )

    @pytest.fixture
    def guidance(self):
        # GuidanceInputs uses actual field names
        return GuidanceInputs(
            revenue_growth_guidance=15.0,
            ebitda_margin_guidance=22.0,
            capacity_utilization_current=70.0,
            capacity_addition_percent=20.0,
            order_book_months=18.0,
        )

    @pytest.fixture
    def historical(self):
        return [
            FinancialRecord(fiscal_year="FY24", revenue=400.0, pat=48.0, total_equity=200.0, total_debt=100.0),
        ]

    def test_build_scenarios_returns_three(self, ipo_details, guidance, historical):
        """Should return Base, Bull, and Stress scenarios."""
        engine = ProjectionEngine()
        # Correct argument order: historical, ipo, guidance
        scenarios = engine.build_scenarios(historical, ipo_details, guidance)
        assert len(scenarios) == 3
        names = [s.name for s in scenarios]  # Field is 'name' not 'scenario_name'
        assert "Base" in names
        assert "Bull" in names
        assert "Stress" in names

    def test_base_scenario_uses_guidance(self, ipo_details, guidance, historical):
        """Base scenario should use guidance growth rates."""
        engine = ProjectionEngine()
        scenarios = engine.build_scenarios(historical, ipo_details, guidance)
        base = next(s for s in scenarios if s.name == "Base")
        
        # FY1E revenue should be FY24 * (1 + 15%)
        expected_fy1 = 400.0 * 1.15
        # Field is 'revenue' dict, keys are FY1E, FY2E, FY3E
        assert abs(base.revenue["FY1E"] - expected_fy1) < 1.0

    def test_bull_scenario_higher_growth(self, ipo_details, guidance, historical):
        """Bull scenario should have higher growth than Base."""
        engine = ProjectionEngine()
        scenarios = engine.build_scenarios(historical, ipo_details, guidance)
        base = next(s for s in scenarios if s.name == "Base")
        bull = next(s for s in scenarios if s.name == "Bull")
        
        assert bull.revenue["FY2E"] > base.revenue["FY2E"]

    def test_stress_scenario_lower_growth(self, ipo_details, guidance, historical):
        """Stress scenario should have lower growth than Base."""
        engine = ProjectionEngine()
        scenarios = engine.build_scenarios(historical, ipo_details, guidance)
        base = next(s for s in scenarios if s.name == "Base")
        stress = next(s for s in scenarios if s.name == "Stress")
        
        assert stress.revenue["FY2E"] < base.revenue["FY2E"]

    def test_compute_post_issue_shares(self, ipo_details):
        """Post-issue shares should include fresh issue shares."""
        engine = ProjectionEngine()
        post_shares = engine.compute_post_issue_shares(ipo_details)
        
        # Total = pre-issue + fresh issue shares
        expected = 50_000_000 + 27_272_727
        assert abs(post_shares - expected) < 1

    def test_compute_diluted_eps(self, ipo_details):
        """EPS calculation should be PAT / shares."""
        engine = ProjectionEngine()
        post_shares = engine.compute_post_issue_shares(ipo_details)
        eps = engine.compute_diluted_eps(100.0, post_shares)
        
        expected = 100.0 * 10_000_000 / post_shares  # PAT in Cr to Rs
        assert abs(eps - expected) < 0.01


# ============================================================
# Milestone 2.5C: ValuationNormalization Tests
# ============================================================
class TestValuationNormalization:
    """Tests for ValuationNormalization."""

    @pytest.fixture
    def issuer_metrics(self):
        return IssuerMetrics(
            name="Test Co",
            pat_cr=50.0,
            net_worth_cr=200.0,
            ebitda_cr=80.0,
            net_debt_cr=50.0,
            pat_cagr_3yr=15.0,
            shares_post_issue=50_000_000,
            price_floor=100.0,
            price_cap=110.0,
        )

    @pytest.fixture
    def peer_data(self):
        return [
            {"name": "Peer A", "pe": 20.0, "pb": 3.0, "ev_ebitda": 12.0},
            {"name": "Peer B", "pe": 25.0, "pb": 4.0, "ev_ebitda": 15.0},
            {"name": "Peer C", "pe": 22.0, "pb": 3.5, "ev_ebitda": 13.0},
        ]

    def test_normalize_peers_returns_summary(self, issuer_metrics, peer_data):
        """Should return ValuationSummary."""
        module = ValuationNormalization()
        # API: normalize_peers(raw_peers, issuer, *, industry_peers=None)
        summary = module.normalize_peers(peer_data, issuer_metrics)
        
        assert isinstance(summary, ValuationSummary)

    def test_peer_median_calculation(self, issuer_metrics, peer_data):
        """Median calculations should be correct."""
        module = ValuationNormalization()
        summary = module.normalize_peers(peer_data, issuer_metrics)
        
        # PE median of [20, 22, 25] = 22
        assert summary.peer_medians["pe"] == 22.0
        # PB median of [3, 3.5, 4] = 3.5
        assert summary.peer_medians["pb"] == 3.5

    def test_issuer_pe_at_cap(self, issuer_metrics, peer_data):
        """PE at cap price should be market_cap / PAT."""
        module = ValuationNormalization()
        summary = module.normalize_peers(peer_data, issuer_metrics)
        
        # Market cap at cap = 50M shares * 110 / 1e7 = 550 Cr
        # PE = 550 / 50 = 11.0
        assert summary.issuer_cap_metrics["pe"] == 11.0

    def test_premium_discount_calculation(self, issuer_metrics, peer_data):
        """Premium/discount should be calculated vs median."""
        module = ValuationNormalization()
        summary = module.normalize_peers(peer_data, issuer_metrics)
        
        # At cap: PE = 11.0, median = 22
        # Premium = (11 / 22 - 1) * 100 = -50%
        issuer_pe = summary.issuer_cap_metrics["pe"]
        median_pe = summary.peer_medians["pe"]
        expected_premium = (issuer_pe / median_pe - 1) * 100
        assert abs(summary.premium_discount_vs_peers["pe"] - expected_premium) < 0.1

    def test_peg_ratio_calculation(self, issuer_metrics, peer_data):
        """PEG should be PE / growth rate."""
        module = ValuationNormalization()
        summary = module.normalize_peers(peer_data, issuer_metrics)
        
        # PEG at cap = PE / growth = 11.0 / 15 = 0.73
        issuer_pe = summary.issuer_cap_metrics["pe"]
        expected_peg = issuer_pe / 15.0
        assert abs(summary.issuer_cap_metrics["peg"] - expected_peg) < 0.1

    def test_missing_peers_identification(self, issuer_metrics, peer_data):
        """Should identify peers mentioned in industry but not in pricing."""
        module = ValuationNormalization()
        industry_peer_list = ["Peer A", "Peer B", "Peer C", "Peer D", "Peer E"]
        summary = module.normalize_peers(
            peer_data, issuer_metrics, industry_peers=industry_peer_list
        )
        
        assert "Peer D" in summary.missing_peers
        assert "Peer E" in summary.missing_peers


# ============================================================
# Milestone 2.5D: GovernanceRulebook Tests
# ============================================================
class TestGovernanceRulebook:
    """Tests for GovernanceRulebook."""

    def test_no_violations_clean_input(self):
        """Clean inputs should trigger no violations."""
        rulebook = GovernanceRulebook()
        inputs = RuleInput(
            promoter_post_issue_holding_pct=60.0,
            promoter_pledge_pct=0.0,
            ofs_pct_of_issue=30.0,
            fresh_issue_pct_of_issue=70.0,
            rpt_revenue_pct=5.0,
            rpt_expense_pct=5.0,
            auditor_resigned=False,
            modified_audit_opinion=False,
            caro_qualifications=0,
            total_litigation_cr=10.0,
            post_issue_networth_cr=500.0,
            receivable_days_growth=5.0,
            revenue_cagr_3yr=10.0,
        )
        violations = rulebook.evaluate(inputs)
        assert len(violations) == 0

    def test_promoter_holding_below_51_critical(self):
        """Promoter holding <51% should trigger critical violation."""
        rulebook = GovernanceRulebook()
        inputs = RuleInput(
            promoter_post_issue_holding_pct=45.0,
            ofs_pct_of_issue=30.0,
            fresh_issue_pct_of_issue=70.0,
        )
        violations = rulebook.evaluate(inputs)
        
        skin_violations = [v for v in violations if v.rule_id == "SKIN-001"]
        assert len(skin_violations) == 1
        assert skin_violations[0].severity == "critical"

    def test_pledge_above_25_critical(self):
        """Pledge >25% should trigger critical violation."""
        rulebook = GovernanceRulebook()
        inputs = RuleInput(
            promoter_post_issue_holding_pct=60.0,
            promoter_pledge_pct=30.0,
            ofs_pct_of_issue=30.0,
            fresh_issue_pct_of_issue=70.0,
        )
        violations = rulebook.evaluate(inputs)
        
        pledge_critical = [v for v in violations if v.rule_id == "PLEDGE-001"]
        assert len(pledge_critical) == 1
        assert pledge_critical[0].severity == "critical"

    def test_any_pledge_triggers_major(self):
        """Any pledge >0% should trigger major violation."""
        rulebook = GovernanceRulebook()
        inputs = RuleInput(
            promoter_post_issue_holding_pct=60.0,
            promoter_pledge_pct=5.0,
            ofs_pct_of_issue=30.0,
            fresh_issue_pct_of_issue=70.0,
        )
        violations = rulebook.evaluate(inputs)
        
        pledge_major = [v for v in violations if v.rule_id == "PLEDGE-002"]
        assert len(pledge_major) == 1
        assert pledge_major[0].severity == "major"

    def test_rpt_revenue_above_20_major(self):
        """RPT revenue >20% should trigger major violation."""
        rulebook = GovernanceRulebook()
        inputs = RuleInput(
            promoter_post_issue_holding_pct=60.0,
            ofs_pct_of_issue=30.0,
            fresh_issue_pct_of_issue=70.0,
            rpt_revenue_pct=25.0,
        )
        violations = rulebook.evaluate(inputs)
        
        rpt_violations = [v for v in violations if v.rule_id == "RPT-001"]
        assert len(rpt_violations) == 1

    def test_auditor_resigned_critical(self):
        """Auditor resignation should trigger critical violation."""
        rulebook = GovernanceRulebook()
        inputs = RuleInput(
            promoter_post_issue_holding_pct=60.0,
            ofs_pct_of_issue=30.0,
            fresh_issue_pct_of_issue=70.0,
            auditor_resigned=True,
        )
        violations = rulebook.evaluate(inputs)
        
        audit_violations = [v for v in violations if v.rule_id == "AUDIT-001"]
        assert len(audit_violations) == 1
        assert audit_violations[0].severity == "critical"

    def test_litigation_above_10_pct_networth_critical(self):
        """Litigation >10% of net worth should be critical."""
        rulebook = GovernanceRulebook()
        inputs = RuleInput(
            promoter_post_issue_holding_pct=60.0,
            ofs_pct_of_issue=30.0,
            fresh_issue_pct_of_issue=70.0,
            total_litigation_cr=60.0,
            post_issue_networth_cr=500.0,  # 12% of net worth
        )
        violations = rulebook.evaluate(inputs)
        
        litig_violations = [v for v in violations if v.rule_id == "LITIG-001"]
        assert len(litig_violations) == 1
        assert litig_violations[0].severity == "critical"

    def test_working_capital_stress_major(self):
        """Receivables growing faster than revenue should trigger major."""
        rulebook = GovernanceRulebook()
        inputs = RuleInput(
            promoter_post_issue_holding_pct=60.0,
            ofs_pct_of_issue=30.0,
            fresh_issue_pct_of_issue=70.0,
            receivable_days_growth=25.0,
            revenue_cagr_3yr=10.0,  # 15pp gap
        )
        violations = rulebook.evaluate(inputs)
        
        wc_violations = [v for v in violations if v.rule_id == "WC-001"]
        assert len(wc_violations) == 1

    def test_get_critical_violations_filter(self):
        """Should filter only critical violations."""
        rulebook = GovernanceRulebook()
        inputs = RuleInput(
            promoter_post_issue_holding_pct=45.0,  # Critical
            promoter_pledge_pct=10.0,  # Major only
            ofs_pct_of_issue=30.0,
            fresh_issue_pct_of_issue=70.0,
        )
        violations = rulebook.evaluate(inputs)
        critical = rulebook.get_critical_violations(violations)
        
        assert all(v.severity == "critical" for v in critical)

    def test_get_veto_flags(self):
        """Should return descriptions of critical violations as veto flags."""
        rulebook = GovernanceRulebook()
        inputs = RuleInput(
            promoter_post_issue_holding_pct=45.0,
            ofs_pct_of_issue=30.0,
            fresh_issue_pct_of_issue=70.0,
        )
        violations = rulebook.evaluate(inputs)
        veto_flags = rulebook.get_veto_flags(violations)
        
        assert len(veto_flags) > 0
        assert "Promoter post-issue holding below 51%" in veto_flags

    def test_multiple_violations(self):
        """Multiple rule violations should all be captured."""
        rulebook = GovernanceRulebook()
        inputs = RuleInput(
            promoter_post_issue_holding_pct=45.0,  # SKIN-001
            promoter_pledge_pct=30.0,  # PLEDGE-001, PLEDGE-002
            ofs_pct_of_issue=80.0,  # SKIN-002 (OFS > fresh)
            fresh_issue_pct_of_issue=20.0,
            rpt_revenue_pct=25.0,  # RPT-001
            auditor_resigned=True,  # AUDIT-001
        )
        violations = rulebook.evaluate(inputs)
        
        # Should have at least 5 violations
        assert len(violations) >= 5


# ============================================================
# Integration: Module Import Tests
# ============================================================
class TestModuleImports:
    """Verify all analytics modules can be imported."""

    def test_import_all_from_package(self):
        """All exports should be importable from package."""
        from rhp_analyzer.analytics import (
            FinancialRecord,
            GuidanceInputs,
            GovernanceRulebook,
            HistoricalNormalizer,
            IPODetails,
            IssuerMetrics,
            PeerComparable,
            ProjectionEngine,
            ProjectionScenario,
            RuleInput,
            RuleViolation,
            ValuationNormalization,
            ValuationSummary,
        )
        
        # All should be class types
        assert FinancialRecord is not None
        assert HistoricalNormalizer is not None
        assert ProjectionEngine is not None
        assert ValuationNormalization is not None
        assert GovernanceRulebook is not None
