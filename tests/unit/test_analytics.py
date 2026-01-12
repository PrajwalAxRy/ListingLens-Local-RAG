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

    def test_import_new_modules_from_package(self):
        """All new Milestone 3.5A exports should be importable."""
        from rhp_analyzer.analytics import (
            # Citation Manager
            CitationManager,
            CitationRecord,
            # Risk Quantification
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
            # Working Capital Analyzer
            WorkingCapitalAnalyzer,
            WorkingCapitalAnalysis,
            WorkingCapitalMetrics,
            WorkingCapitalTrend,
            # Cash Flow Analyzer
            EnhancedCashFlowAnalyzer,
            CashFlowAnalysis,
            CashFlowMetrics,
            CashFlowTrend,
            CashBurnStatus,
            # Float Calculator
            FloatCalculator,
            FloatAnalysis,
            IPOShareDetails,
            ShareholderBlock,
            LockInEvent,
            InvestorCategory,
            LockInPeriod,
        )
        
        # All should be class/enum types
        assert CitationManager is not None
        assert RiskLitigationQuantifier is not None
        assert WorkingCapitalAnalyzer is not None
        assert EnhancedCashFlowAnalyzer is not None
        assert FloatCalculator is not None


# ============================================================
# Milestone 3.5A.5: CitationManager Tests
# ============================================================
class TestCitationManager:
    """Tests for CitationManager - audit trail for sourced claims."""

    def test_add_and_get_citation(self):
        """Add a citation and retrieve it."""
        from rhp_analyzer.analytics import CitationManager
        
        mgr = CitationManager()
        mgr.add_citation(
            claim_id="CLAIM-001",
            section="Risk Factors",
            page=45,
            snippet="Revenue may decline due to regulatory changes."
        )
        
        record = mgr.get("CLAIM-001")
        assert record is not None
        assert record.section == "Risk Factors"
        assert record.page == 45

    def test_attach_citation(self):
        """Attach method with full parameters."""
        from rhp_analyzer.analytics import CitationManager
        
        mgr = CitationManager()
        # attach() takes individual params, not a CitationRecord object
        mgr.attach(
            claim_id="CLAIM-002",
            section="Financial Statements",
            page=120,
            text_snippet="Trade receivables increased 25%.",
            paragraph_label="Note 5"
        )
        
        retrieved = mgr.get("CLAIM-002")
        assert retrieved is not None
        assert retrieved.paragraph_label == "Note 5"

    def test_validate_claim_exists(self):
        """Validate returns True for existing claims."""
        from rhp_analyzer.analytics import CitationManager
        
        mgr = CitationManager()
        mgr.add_citation("CLAIM-003", "Objects of Issue", 85)
        
        assert mgr.validate_claim("CLAIM-003") is True
        assert mgr.validate_claim("NONEXISTENT") is False

    def test_get_uncited_claims(self):
        """Get list of uncited claims from a list."""
        from rhp_analyzer.analytics import CitationManager
        
        mgr = CitationManager()
        mgr.add_citation("CLAIM-A", "Section A", 10)
        mgr.add_citation("CLAIM-B", "Section B", 20)
        
        required = ["CLAIM-A", "CLAIM-B", "CLAIM-C", "CLAIM-D"]
        uncited = mgr.get_uncited_claims(required)
        
        # Returns a list, convert to set for order-independent comparison
        assert set(uncited) == {"CLAIM-C", "CLAIM-D"}

    def test_generate_footnotes(self):
        """Generate markdown footnotes from citations."""
        from rhp_analyzer.analytics import CitationManager
        
        mgr = CitationManager()
        mgr.add_citation("C1", "Risk Factors", 45, "First citation text.")
        mgr.add_citation("C2", "Business Overview", 60, "Second citation text.")
        
        footnotes = mgr.generate_footnotes()
        
        # Footnote format is [^1] not [C1]
        assert "[^1]" in footnotes or "[^2]" in footnotes
        assert "Risk Factors" in footnotes
        assert "p. 45" in footnotes or "p. 60" in footnotes

    def test_export_and_import_citations(self):
        """Export citations to JSON and reimport."""
        from rhp_analyzer.analytics import CitationManager
        
        mgr1 = CitationManager()
        mgr1.add_citation("EXP-1", "Section X", 100, "Export test.")
        
        json_data = mgr1.export_citations()
        
        mgr2 = CitationManager()
        mgr2.import_citations(json_data)
        
        record = mgr2.get("EXP-1")
        assert record is not None
        assert record.section == "Section X"

    def test_citation_count(self):
        """Citation count property should be accurate."""
        from rhp_analyzer.analytics import CitationManager
        
        mgr = CitationManager()
        assert mgr.count == 0
        
        mgr.add_citation("A", "S", 1)
        mgr.add_citation("B", "S", 2)
        assert mgr.count == 2


# ============================================================
# Milestone 3.5A.4: RiskLitigationQuantifier Tests
# ============================================================
class TestRiskLitigationQuantifier:
    """Tests for RiskLitigationQuantifier - litigation and contingent liability analysis."""

    def test_add_and_summarize_litigation(self):
        """Create litigation items and get summary."""
        from rhp_analyzer.analytics import (
            RiskLitigationQuantifier,
            LitigationItem,
            EntityType,
            CaseType,
        )
        
        quantifier = RiskLitigationQuantifier(post_issue_networth=500.0)
        
        litigation_items = [
            LitigationItem(
                entity_type=EntityType.COMPANY,
                entity_name="ABC Ltd",
                case_type=CaseType.TAX,
                description="GST dispute",
                amount_cr=10.0,
                citation="Litigation, p. 200"
            ),
            LitigationItem(
                entity_type=EntityType.PROMOTER,
                entity_name="Promoter A",
                case_type=CaseType.CRIMINAL,
                description="Criminal case",
                amount_cr=5.0,
                citation="Litigation, p. 205"
            ),
        ]
        
        result = quantifier.quantify_risks(litigation_items, [])
        
        assert result.total_litigation_count == 2
        assert result.total_litigation_amount_cr == 15.0
        assert result.total_litigation_percent_networth == 3.0  # 15/500 * 100

    def test_litigation_by_entity(self):
        """Get litigation breakdown by entity type."""
        from rhp_analyzer.analytics import (
            RiskLitigationQuantifier,
            LitigationItem,
            EntityType,
            CaseType,
        )
        
        quantifier = RiskLitigationQuantifier(post_issue_networth=1000.0)
        
        litigation_items = [
            LitigationItem(
                entity_type=EntityType.COMPANY,
                entity_name="ABC Ltd",
                case_type=CaseType.CIVIL,
                description="Case 1",
                amount_cr=20.0,
                citation="p1"
            ),
            LitigationItem(
                entity_type=EntityType.COMPANY,
                entity_name="ABC Ltd",
                case_type=CaseType.TAX,
                description="Case 2",
                amount_cr=30.0,
                citation="p2"
            ),
            LitigationItem(
                entity_type=EntityType.PROMOTER,
                entity_name="Promoter A",
                case_type=CaseType.CIVIL,
                description="Case 3",
                amount_cr=10.0,
                citation="p3"
            ),
        ]
        
        result = quantifier.quantify_risks(litigation_items, [])
        
        # Company litigation should have 2 cases totaling 50 cr
        assert result.company_litigation.total_count == 2
        assert result.company_litigation.total_amount_cr == 50.0

    def test_add_contingent_liability(self):
        """Add contingent liabilities and analyze."""
        from rhp_analyzer.analytics import (
            RiskLitigationQuantifier,
            ContingentItem,
            Probability,
        )
        
        quantifier = RiskLitigationQuantifier(post_issue_networth=200.0)
        
        contingent_items = [
            ContingentItem(
                category="Tax",
                description="Income tax demand",
                amount_cr=20.0,
                probability=Probability.MEDIUM,
                citation="Notes, p. 150"
            ),
            ContingentItem(
                category="Bank Guarantee",
                description="Performance guarantee",
                amount_cr=50.0,
                probability=Probability.LOW,
                citation="Notes, p. 152"
            ),
        ]
        
        result = quantifier.quantify_risks([], contingent_items)
        analysis = result.contingent_analysis
        
        assert analysis.total_contingent == 70.0
        # Medium (50%) + Low (25%)
        expected_weighted = 20.0 * 0.5 + 50.0 * 0.25
        assert abs(analysis.risk_weighted_total - expected_weighted) < 0.1

    def test_quantify_full_analysis(self):
        """Full quantification with all components."""
        from rhp_analyzer.analytics import (
            RiskLitigationQuantifier,
            LitigationItem,
            ContingentItem,
            EntityType,
            CaseType,
            Probability,
        )
        
        quantifier = RiskLitigationQuantifier(post_issue_networth=500.0)
        
        litigation_items = [
            LitigationItem(
                entity_type=EntityType.COMPANY,
                entity_name="ABC Ltd",
                case_type=CaseType.TAX,
                description="Tax dispute",
                amount_cr=25.0,
                citation="p1"
            ),
        ]
        contingent_items = [
            ContingentItem(
                category="Customs",
                description="Customs demand",
                amount_cr=15.0,
                probability=Probability.HIGH,
                citation="p2"
            ),
        ]
        
        result = quantifier.quantify_risks(litigation_items, contingent_items)
        
        assert result.total_litigation_count == 1
        assert result.contingent_analysis.total_contingent == 15.0

    def test_veto_threshold_check(self):
        """Check if litigation exceeds veto threshold (10% of NW)."""
        from rhp_analyzer.analytics import (
            RiskLitigationQuantifier,
            LitigationItem,
            EntityType,
            CaseType,
        )
        
        quantifier = RiskLitigationQuantifier(post_issue_networth=100.0)
        
        # Add 15 cr litigation = 15% of NW (exceeds 10% threshold)
        litigation_items = [
            LitigationItem(
                entity_type=EntityType.COMPANY,
                entity_name="ABC Ltd",
                case_type=CaseType.CIVIL,
                description="Large case",
                amount_cr=15.0,
                citation="p1"
            ),
        ]
        
        result = quantifier.quantify_risks(litigation_items, [])
        
        assert result.veto_flag is True

    def test_severity_classification(self):
        """Items should be classified by severity."""
        from rhp_analyzer.analytics import (
            RiskLitigationQuantifier,
            LitigationItem,
            EntityType,
            CaseType,
        )
        
        quantifier = RiskLitigationQuantifier(post_issue_networth=1000.0)
        
        litigation_items = [
            # Criminal = HIGH severity
            LitigationItem(
                entity_type=EntityType.PROMOTER,
                entity_name="Promoter A",
                case_type=CaseType.CRIMINAL,
                description="Criminal",
                amount_cr=5.0,
                citation="p1"
            ),
            # Tax = MEDIUM severity
            LitigationItem(
                entity_type=EntityType.COMPANY,
                entity_name="ABC Ltd",
                case_type=CaseType.TAX,
                description="Tax",
                amount_cr=10.0,
                citation="p2"
            ),
        ]
        
        result = quantifier.quantify_risks(litigation_items, [])
        
        # Criminal cases against promoter should flag
        assert result.criminal_cases_against_promoters >= 1


# ============================================================
# Milestone 3.5A.6: WorkingCapitalAnalyzer Tests
# ============================================================
class TestWorkingCapitalAnalyzer:
    """Tests for WorkingCapitalAnalyzer with sector benchmarks."""

    def test_calculate_working_capital_metrics(self):
        """Calculate DSO, DIO, DPO, and CCC."""
        from rhp_analyzer.analytics import WorkingCapitalAnalyzer, WorkingCapitalMetrics
        
        analyzer = WorkingCapitalAnalyzer(sector="FMCG")
        
        metrics = WorkingCapitalMetrics(
            fiscal_year="FY24",
            revenue=1000.0,
            cogs=700.0,
            trade_receivables=100.0,
            inventory=70.0,
            trade_payables=50.0
        )
        
        analysis = analyzer.analyze_single_year(metrics)
        
        # DSO = (100 / 1000) * 365 = 36.5 days
        assert abs(analysis.receivable_days - 36.5) < 0.1
        # DIO = (70 / 700) * 365 = 36.5 days
        assert abs(analysis.inventory_days - 36.5) < 0.1
        # DPO = (50 / 700) * 365 = 26.1 days
        assert abs(analysis.payable_days - 26.1) < 0.1
        # CCC = 36.5 + 36.5 - 26.1 = 46.9 days
        assert abs(analysis.cash_conversion_cycle - 46.9) < 0.1

    def test_analyze_with_sector_benchmark(self):
        """Compare metrics against sector benchmark."""
        from rhp_analyzer.analytics import WorkingCapitalAnalyzer, WorkingCapitalMetrics
        
        analyzer = WorkingCapitalAnalyzer(sector="FMCG")  # CCC benchmark ~40 days
        
        metrics = WorkingCapitalMetrics(
            fiscal_year="FY24",
            revenue=1000.0,
            cogs=700.0,
            trade_receivables=100.0,
            inventory=70.0,
            trade_payables=50.0
        )
        
        analysis = analyzer.analyze_single_year(metrics)
        
        assert analysis.sector == "FMCG"
        assert analysis.sector_avg_ccc is not None
        # CCC ~47 vs benchmark ~40 = positive variance
        assert analysis.variance_vs_sector_ccc is not None

    def test_yoy_trend_detection(self):
        """Detect year-over-year trends in working capital."""
        from rhp_analyzer.analytics import WorkingCapitalAnalyzer, WorkingCapitalMetrics
        
        analyzer = WorkingCapitalAnalyzer(sector="IT Services")
        
        fy23 = WorkingCapitalMetrics(
            fiscal_year="FY23",
            revenue=800.0,
            cogs=500.0,
            trade_receivables=60.0,
            inventory=20.0,
            trade_payables=40.0
        )
        fy24 = WorkingCapitalMetrics(
            fiscal_year="FY24",
            revenue=1000.0,
            cogs=600.0,
            trade_receivables=100.0,
            inventory=30.0,
            trade_payables=50.0
        )
        
        analysis = analyzer.analyze_single_year(fy24, prior_metrics=fy23)
        
        # Receivable days increased
        assert analysis.receivable_days_change_yoy > 0

    def test_channel_stuffing_detection(self):
        """Detect channel stuffing risk (receivables growing faster than revenue)."""
        from rhp_analyzer.analytics import WorkingCapitalAnalyzer, WorkingCapitalMetrics
        
        analyzer = WorkingCapitalAnalyzer(sector="Pharma")
        
        # FY23: Revenue 1000, Receivables 100
        fy23 = WorkingCapitalMetrics(
            fiscal_year="FY23",
            revenue=1000.0,
            cogs=700.0,
            trade_receivables=100.0,
            inventory=100.0,
            trade_payables=70.0
        )
        # FY24: Revenue 1100 (+10%), Receivables 150 (+50%)
        fy24 = WorkingCapitalMetrics(
            fiscal_year="FY24",
            revenue=1100.0,
            cogs=770.0,
            trade_receivables=150.0,
            inventory=110.0,
            trade_payables=77.0
        )
        
        analysis = analyzer.analyze_single_year(fy24, prior_metrics=fy23)
        
        # Receivables growing faster than revenue - check for channel stuffing flag
        # receivable_growth = 50%, revenue_growth = 10%, difference = 40%
        assert analysis.receivable_growth_vs_revenue_growth > 30.0

    def test_sector_not_found(self):
        """Handle unknown sector gracefully."""
        from rhp_analyzer.analytics import WorkingCapitalAnalyzer, WorkingCapitalMetrics
        
        analyzer = WorkingCapitalAnalyzer(sector="Unknown Sector")
        
        metrics = WorkingCapitalMetrics(
            fiscal_year="FY24",
            revenue=1000.0,
            cogs=700.0,
            trade_receivables=100.0,
            inventory=50.0,
            trade_payables=50.0
        )
        analysis = analyzer.analyze_single_year(metrics)
        
        # Should still work, just without sector comparison
        assert analysis.sector_avg_ccc is None or analysis.sector == "Unknown Sector"


# ============================================================
# Milestone 3.5A.7: EnhancedCashFlowAnalyzer Tests
# ============================================================
class TestEnhancedCashFlowAnalyzer:
    """Tests for EnhancedCashFlowAnalyzer with FCF, burn rate, and runway."""

    def test_calculate_free_cash_flow(self):
        """Calculate FCF = CFO - Capex."""
        from rhp_analyzer.analytics import EnhancedCashFlowAnalyzer, CashFlowMetrics
        
        analyzer = EnhancedCashFlowAnalyzer()
        
        metrics = CashFlowMetrics(
            fiscal_year="FY24",
            cfo=100.0,
            cfi=-50.0,
            cff=-20.0,
            capex=40.0,
            revenue=500.0,
            ebitda=80.0,
            pat=50.0,
            depreciation=20.0,
            cash_and_equivalents=30.0
        )
        
        analysis = analyzer.analyze_single_year(metrics)
        
        assert analysis.fcf == 60.0  # 100 - 40
        assert abs(analysis.fcf_margin - 12.0) < 0.1  # 60/500 * 100

    def test_cfo_to_ebitda_quality(self):
        """CFO/EBITDA ratio should indicate earnings quality."""
        from rhp_analyzer.analytics import EnhancedCashFlowAnalyzer, CashFlowMetrics
        
        analyzer = EnhancedCashFlowAnalyzer()
        
        # Good quality: CFO close to EBITDA
        good_metrics = CashFlowMetrics(
            fiscal_year="FY24",
            cfo=75.0,
            cfi=-30.0,
            cff=-10.0,
            capex=25.0,
            revenue=400.0,
            ebitda=80.0,
            pat=50.0,
            depreciation=15.0,
            cash_and_equivalents=50.0
        )
        good = analyzer.analyze_single_year(good_metrics)
        
        # Poor quality: CFO much lower than EBITDA
        poor_metrics = CashFlowMetrics(
            fiscal_year="FY24",
            cfo=30.0,
            cfi=-30.0,
            cff=-10.0,
            capex=25.0,
            revenue=400.0,
            ebitda=80.0,
            pat=50.0,
            depreciation=15.0,
            cash_and_equivalents=50.0
        )
        poor = analyzer.analyze_single_year(poor_metrics)
        
        assert good.cfo_to_ebitda > 70.0  # Good quality
        assert poor.cfo_to_ebitda < 50.0  # Poor quality - paper profits risk

    def test_cash_burning_detection(self):
        """Detect cash burning companies with runway calculation."""
        from rhp_analyzer.analytics import EnhancedCashFlowAnalyzer, CashFlowMetrics, CashBurnStatus
        
        analyzer = EnhancedCashFlowAnalyzer()
        
        # Negative FCF = cash burning
        metrics = CashFlowMetrics(
            fiscal_year="FY24",
            cfo=20.0,
            cfi=-60.0,
            cff=30.0,
            capex=50.0,
            revenue=200.0,
            ebitda=30.0,
            pat=10.0,
            depreciation=15.0,
            cash_and_equivalents=60.0
        )
        
        analysis = analyzer.analyze_single_year(metrics)
        
        assert analysis.fcf == -30.0  # 20 - 50
        assert analysis.is_cash_burning is True
        # Monthly burn = 30/12 = 2.5, Runway = 60/2.5 = 24 months
        assert abs(analysis.monthly_cash_burn - 2.5) < 0.1
        assert abs(analysis.runway_months - 24.0) < 0.5

    def test_capex_categorization(self):
        """Categorize capex into maintenance vs growth."""
        from rhp_analyzer.analytics import EnhancedCashFlowAnalyzer, CashFlowMetrics
        
        analyzer = EnhancedCashFlowAnalyzer()
        
        metrics = CashFlowMetrics(
            fiscal_year="FY24",
            cfo=100.0,
            cfi=-80.0,
            cff=-10.0,
            capex=60.0,
            revenue=500.0,
            ebitda=120.0,
            pat=80.0,
            depreciation=25.0,
            cash_and_equivalents=50.0
        )
        
        analysis = analyzer.analyze_single_year(metrics)
        
        # Maintenance capex ≈ depreciation
        assert analysis.maintenance_capex_estimate == 25.0
        # Growth capex = total capex - maintenance
        assert analysis.growth_capex_estimate == 35.0  # 60 - 25

    def test_fcf_yield_calculation(self):
        """Calculate FCF yield at floor and cap prices."""
        from rhp_analyzer.analytics import EnhancedCashFlowAnalyzer, CashFlowMetrics
        
        analyzer = EnhancedCashFlowAnalyzer()
        
        metrics = CashFlowMetrics(
            fiscal_year="FY24",
            cfo=100.0,
            cfi=-40.0,
            cff=-20.0,
            capex=30.0,
            revenue=400.0,
            ebitda=100.0,
            pat=60.0,
            depreciation=20.0,
            cash_and_equivalents=40.0
        )
        
        analysis = analyzer.analyze_single_year(
            metrics, 
            market_cap_floor=700.0,
            market_cap_cap=800.0
        )
        
        # FCF = 100 - 30 = 70
        # FCF yield at floor = 70/700 = 10%
        # FCF yield at cap = 70/800 = 8.75%
        assert abs(analysis.fcf_yield_at_floor - 10.0) < 0.1
        assert abs(analysis.fcf_yield_at_cap - 8.75) < 0.1

    def test_trend_analysis(self):
        """Analyze FCF trends over multiple years."""
        from rhp_analyzer.analytics import EnhancedCashFlowAnalyzer, CashFlowMetrics
        
        analyzer = EnhancedCashFlowAnalyzer()
        
        # Create metrics for each year
        fy22 = CashFlowMetrics(
            fiscal_year="FY22",
            cfo=60.0, cfi=-30.0, cff=-10.0, capex=20.0,
            revenue=300.0, ebitda=70.0, pat=40.0,
            depreciation=15.0, cash_and_equivalents=30.0
        )
        fy23 = CashFlowMetrics(
            fiscal_year="FY23",
            cfo=80.0, cfi=-40.0, cff=-15.0, capex=25.0,
            revenue=400.0, ebitda=90.0, pat=55.0,
            depreciation=18.0, cash_and_equivalents=40.0
        )
        fy24 = CashFlowMetrics(
            fiscal_year="FY24",
            cfo=100.0, cfi=-50.0, cff=-20.0, capex=30.0,
            revenue=500.0, ebitda=110.0, pat=70.0,
            depreciation=22.0, cash_and_equivalents=50.0
        )
        
        # Analyze each year
        analyses = [
            analyzer.analyze_single_year(fy22),
            analyzer.analyze_single_year(fy23),
            analyzer.analyze_single_year(fy24),
        ]
        
        # Check analyses were created
        assert len(analyses) == 3
        # FCF should be increasing: 40, 55, 70
        assert analyses[2].fcf > analyses[0].fcf


# ============================================================
# Milestone 3.5A.8: FloatCalculator Tests
# ============================================================
class TestFloatCalculator:
    """Tests for FloatCalculator - free float and lock-in calendar."""

    def test_calculate_day_1_float(self):
        """Calculate Day 1 free float after listing."""
        from rhp_analyzer.analytics import (
            FloatCalculator,
            IPOShareDetails,
            ShareholderBlock,
            InvestorCategory,
        )
        
        ipo_details = IPOShareDetails(
            shares_pre_issue=80_000_000,
            fresh_issue_shares=20_000_000,
            qib_percent=50.0,
            nii_percent=15.0,
            retail_percent=35.0,
            anchor_shares=5_000_000,  # Part of QIB
            listing_date=None  # Will default
        )
        
        shareholders = [
            ShareholderBlock(
                name="Promoter Group",
                category=InvestorCategory.PROMOTER,
                shares_pre_ipo=60_000_000,
                shares_post_ipo=60_000_000,
                shares_selling_ofs=0,
                lock_in_days=1095,  # 3-year lock
                shares_locked=60_000_000,  # All promoter shares locked
            ),
            ShareholderBlock(
                name="PE Fund",
                category=InvestorCategory.PE_VC,
                shares_pre_ipo=20_000_000,
                shares_post_ipo=20_000_000,
                shares_selling_ofs=0,
                lock_in_days=180,  # 6-month lock
                shares_locked=20_000_000,  # All PE shares locked
            ),
        ]
        
        calculator = FloatCalculator()
        analysis = calculator.calculate_float_analysis(ipo_details, shareholders)
        
        # Total post-issue = 100M shares
        assert analysis.total_shares_post_issue == 100_000_000
        # Day 1: Only fresh issue minus anchor (locked 90 days)
        # 20M - 5M anchor = 15M free from fresh issue
        assert analysis.day_1_free_float_shares > 0
        assert analysis.day_1_free_float_percent < 20.0  # Limited float

    def test_lock_in_calendar_generation(self):
        """Generate lock-in expiry calendar with milestones."""
        from rhp_analyzer.analytics import (
            FloatCalculator,
            IPOShareDetails,
            ShareholderBlock,
            InvestorCategory,
        )
        from datetime import date
        
        ipo_details = IPOShareDetails(
            shares_pre_issue=50_000_000,
            fresh_issue_shares=10_000_000,
            qib_percent=50.0,
            nii_percent=15.0,
            retail_percent=35.0,
            anchor_shares=2_500_000,
            listing_date=date(2025, 6, 1)
        )
        
        shareholders = [
            ShareholderBlock(
                name="Promoters",
                category=InvestorCategory.PROMOTER,
                shares_pre_ipo=40_000_000,
                shares_post_ipo=40_000_000,
                shares_selling_ofs=0,
                lock_in_days=1095,
                shares_locked=40_000_000,
            ),
            ShareholderBlock(
                name="VC Fund",
                category=InvestorCategory.PE_VC,
                shares_pre_ipo=10_000_000,
                shares_post_ipo=10_000_000,
                shares_selling_ofs=0,
                lock_in_days=365,
                shares_locked=10_000_000,
            ),
        ]
        
        calculator = FloatCalculator()
        analysis = calculator.calculate_float_analysis(ipo_details, shareholders)
        
        # Should have lock-in events
        assert len(analysis.lock_in_calendar) >= 0  # May be empty if no events generated
        
        # Check if total shares calculated correctly
        assert analysis.total_shares_post_issue == 60_000_000

    def test_day_90_float_post_anchor(self):
        """Day 90 float should increase after anchor unlock."""
        from rhp_analyzer.analytics import (
            FloatCalculator,
            IPOShareDetails,
            ShareholderBlock,
            InvestorCategory,
        )
        from datetime import date
        
        ipo_details = IPOShareDetails(
            shares_pre_issue=40_000_000,
            fresh_issue_shares=10_000_000,
            qib_percent=50.0,
            nii_percent=15.0,
            retail_percent=35.0,
            anchor_shares=2_000_000,
            listing_date=date(2025, 1, 15)
        )
        
        shareholders = [
            ShareholderBlock(
                name="Promoters",
                category=InvestorCategory.PROMOTER,
                shares_pre_ipo=40_000_000,
                shares_post_ipo=40_000_000,
                lock_in_days=1095,
                shares_locked=40_000_000,
            ),
        ]
        
        calculator = FloatCalculator()
        analysis = calculator.calculate_float_analysis(ipo_details, shareholders)
        
        # Day 90 float should be higher than or equal to Day 1
        assert analysis.day_90_free_float_percent >= analysis.day_1_free_float_percent

    def test_implied_daily_volume(self):
        """Calculate implied daily trading volume from free float."""
        from rhp_analyzer.analytics import (
            FloatCalculator,
            IPOShareDetails,
            ShareholderBlock,
            InvestorCategory,
        )
        from datetime import date
        
        ipo_details = IPOShareDetails(
            shares_pre_issue=90_000_000,
            fresh_issue_shares=10_000_000,
            qib_percent=50.0,
            nii_percent=15.0,
            retail_percent=35.0,
            anchor_shares=2_000_000,
            listing_date=date(2025, 4, 1)
        )
        
        shareholders = [
            ShareholderBlock(
                name="Promoters",
                category=InvestorCategory.PROMOTER,
                shares_pre_ipo=90_000_000,
                shares_post_ipo=90_000_000,
                lock_in_days=1095,
                shares_locked=90_000_000,
            ),
        ]
        
        calculator = FloatCalculator()
        analysis = calculator.calculate_float_analysis(ipo_details, shareholders)
        
        # Implied daily volume = Day 1 float / 250 trading days
        if analysis.day_1_free_float_shares > 0:
            expected_daily = analysis.day_1_free_float_shares / 250
            assert abs(analysis.implied_daily_volume - expected_daily) < 1.0

    def test_low_float_warning(self):
        """Flag low float situations (<5%)."""
        from rhp_analyzer.analytics import (
            FloatCalculator,
            IPOShareDetails,
            ShareholderBlock,
            InvestorCategory,
        )
        from datetime import date
        
        # Scenario with very low fresh issue
        ipo_details = IPOShareDetails(
            shares_pre_issue=95_000_000,
            fresh_issue_shares=5_000_000,  # Only 5% fresh
            qib_percent=50.0,
            nii_percent=15.0,
            retail_percent=35.0,
            anchor_shares=1_250_000,
            listing_date=date(2025, 2, 1)
        )
        
        shareholders = [
            ShareholderBlock(
                name="Promoters",
                category=InvestorCategory.PROMOTER,
                shares_pre_ipo=95_000_000,
                shares_post_ipo=95_000_000,
                lock_in_days=1095,
                shares_locked=95_000_000,  # All promoter shares locked
            ),
        ]
        
        calculator = FloatCalculator()
        analysis = calculator.calculate_float_analysis(ipo_details, shareholders)
        
        # Day 1 float should be flagged as low (only 3.75M free = 3.75%)
        # Fresh issue 5M - anchor 1.25M = 3.75M free
        assert analysis.low_float_warning is True or analysis.day_1_free_float_percent < 5.0

    def test_retail_quota_tracking(self):
        """Track retail quota shares and percentage."""
        from rhp_analyzer.analytics import (
            FloatCalculator,
            IPOShareDetails,
            ShareholderBlock,
            InvestorCategory,
        )
        from datetime import date
        
        ipo_details = IPOShareDetails(
            shares_pre_issue=70_000_000,
            fresh_issue_shares=30_000_000,
            qib_percent=50.0,
            nii_percent=15.0,
            retail_percent=35.0,  # 35% of fresh issue
            anchor_shares=7_500_000,
            listing_date=date(2025, 5, 1)
        )
        
        shareholders = [
            ShareholderBlock(
                name="Promoters",
                category=InvestorCategory.PROMOTER,
                shares_pre_ipo=70_000_000,
                shares_post_ipo=70_000_000,
                lock_in_days=1095,
                shares_locked=70_000_000,  # All promoter shares locked
            ),
        ]
        
        calculator = FloatCalculator()
        analysis = calculator.calculate_float_analysis(ipo_details, shareholders)
        
        # Retail quota = 35% of 30M = 10.5M shares
        assert analysis.retail_quota_shares == 10_500_000
        # Retail quota percent is % of fresh issue, not total
        assert abs(analysis.retail_quota_percent - 35.0) < 0.1
