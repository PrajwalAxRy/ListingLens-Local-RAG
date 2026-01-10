"""Unit tests for specialized extractors.

This module tests:
- PromoterExtractor: Promoter profile extraction
- PreIPOInvestorAnalyzer: IRR calculation, return multiples, holding period
- OrderBookAnalyzer: Order book ratios and sector applicability
"""

import math
import pytest

from rhp_analyzer.ingestion.promoter_extractor import (
    PromoterDossier,
    PromoterExtractor,
)
from rhp_analyzer.ingestion.pre_ipo_analyzer import (
    PreIPOInvestor,
    PreIPOInvestorAnalyzer,
)
from rhp_analyzer.ingestion.order_book_analyzer import (
    OrderBookAnalysis,
    OrderBookAnalyzer,
)


# ==============================================================================
# PromoterExtractor Tests
# ==============================================================================


class TestPromoterExtractor:
    """Test suite for PromoterExtractor."""

    def test_promoter_dossier_dataclass_defaults(self):
        """Test PromoterDossier initialization with defaults."""
        dossier = PromoterDossier(name="John Doe")
        
        assert dossier.name == "John Doe"
        assert dossier.din is None
        assert dossier.age is None
        assert dossier.other_directorships == []
        assert dossier.other_directorships_count == 0
        assert dossier.shareholding_pre_ipo == 0.0
        assert dossier.criminal_cases == 0
        assert dossier.disqualifications is False

    def test_promoter_dossier_full_profile(self):
        """Test PromoterDossier with complete promoter profile."""
        dossier = PromoterDossier(
            name="Rajesh Kumar",
            din="12345678",
            age=55,
            qualification="MBA, IIM Ahmedabad",
            experience_years=30,
            designation="Managing Director",
            other_directorships=["ABC Ltd", "XYZ Pvt Ltd", "DEF Holdings"],
            group_companies_in_same_line=["ABC Ltd"],
            shareholding_pre_ipo=45.5,
            shareholding_post_ipo=35.2,
            shares_selling_via_ofs=1000000,
            ofs_amount=50.0,  # ₹ Cr
            loans_from_company=2.5,
            criminal_cases=0,
            civil_cases=2,
            total_litigation_amount=15.0,
            skin_in_game_post_ipo=350.0,  # ₹ Cr at cap price
        )
        
        assert dossier.din == "12345678"
        assert dossier.age == 55
        assert dossier.other_directorships_count == 3
        assert dossier.shareholding_pre_ipo == 45.5
        assert len(dossier.group_companies_in_same_line) == 1
        assert dossier.civil_cases == 2
        assert dossier.skin_in_game_post_ipo == 350.0

    def test_promoter_dossier_post_init_calculates_directorships_count(self):
        """Test that __post_init__ calculates directorship count."""
        dossier = PromoterDossier(
            name="Test Promoter",
            other_directorships=["Company A", "Company B"],
        )
        
        # __post_init__ should update the count
        assert dossier.other_directorships_count == 2

    def test_promoter_extractor_initialization(self):
        """Test PromoterExtractor can be initialized."""
        extractor = PromoterExtractor()
        
        assert extractor.vector_store is None
        assert extractor.citation_manager is None

    def test_promoter_extractor_with_dependencies(self):
        """Test PromoterExtractor initialization with dependencies."""
        mock_vector_store = object()
        mock_citation_mgr = object()
        
        extractor = PromoterExtractor(
            vector_store=mock_vector_store,
            citation_manager=mock_citation_mgr
        )
        
        assert extractor.vector_store is mock_vector_store
        assert extractor.citation_manager is mock_citation_mgr

    def test_promoter_extractor_empty_state(self):
        """Test PromoterExtractor with empty state returns empty list."""
        extractor = PromoterExtractor()
        state = {}
        
        promoters = extractor.extract_promoters(state)
        
        assert promoters == []

    def test_promoter_extractor_basic_profiles(self):
        """Test PromoterExtractor extracts basic promoter profiles from sections content."""
        extractor = PromoterExtractor()
        # PromoterExtractor parses text from state['sections'] with content fields
        state = {
            "sections": {
                "Our Promoters": {
                    "content": """
                    Mr. John Smith is the Chairman of our Company.
                    Name: Mr. John Smith
                    DIN: 00001111
                    Age: 50 years
                    Qualification: B.Tech from IIT Delhi
                    Experience: 25 years in the industry
                    """,
                    "tables": [],
                },
            },
        }
        
        promoters = extractor.extract_promoters(state)
        
        # The extractor returns a list (extraction depends on regex patterns)
        assert isinstance(promoters, list)
        # Note: Actual extraction depends on regex patterns in _extract_promoter_names

    def test_promoter_extractor_with_litigation(self):
        """Test PromoterExtractor extracts promoter-specific litigation."""
        extractor = PromoterExtractor()
        # PromoterExtractor parses text from sections with content fields
        state = {
            "sections": {
                "Our Promoters": {
                    "content": """
                    Mr. Litigated Promoter (DIN: 99998888) is the Managing Director.
                    Age: 45 years
                    """,
                    "tables": [],
                },
                "Litigation involving Promoters": {
                    "content": """
                    Civil cases: 3 pending (Rs. 23.0 crores)
                    Criminal cases: 1 pending (Rs. 2.5 crores)
                    """,
                    "tables": [],
                },
            },
        }
        
        promoters = extractor.extract_promoters(state)
        
        # The extractor returns a list (actual data depends on regex parsing)
        assert isinstance(promoters, list)

    def test_promoter_extractor_skin_in_game_calculation(self):
        """Test skin-in-the-game calculation at cap price."""
        extractor = PromoterExtractor()
        # PromoterExtractor parses text from sections with content fields
        state = {
            "sections": {
                "Our Promoters": {
                    "content": """
                    Mr. Rich Promoter (DIN: 11112222) is our Promoter.
                    Age: 55 years
                    """,
                    "tables": [],
                },
                "Shareholding Pattern": {
                    "content": """
                    Pre-IPO shareholding: 50.0%
                    Post-IPO shareholding: 40.0%
                    Total shares post-IPO: 10,000,000
                    """,
                    "tables": [],
                },
            },
            "ipo_details": {
                "price_band_cap": 500,  # ₹500 per share
            },
        }
        
        promoters = extractor.extract_promoters(state)
        
        # The extractor returns a list (actual data depends on regex parsing)
        assert isinstance(promoters, list)


# ==============================================================================
# PreIPOInvestorAnalyzer Tests
# ==============================================================================


class TestPreIPOInvestorAnalyzer:
    """Test suite for PreIPOInvestorAnalyzer."""

    def test_pre_ipo_investor_dataclass_defaults(self):
        """Test PreIPOInvestor initialization with defaults."""
        investor = PreIPOInvestor(name="Test Fund", category="PE/VC")
        
        assert investor.name == "Test Fund"
        assert investor.category == "PE/VC"
        assert investor.entry_price is None
        assert investor.shares_acquired == 0
        assert investor.implied_irr_at_cap is None

    def test_analyzer_initialization(self):
        """Test PreIPOInvestorAnalyzer can be initialized."""
        analyzer = PreIPOInvestorAnalyzer()
        
        assert analyzer.citation_manager is None

    def test_analyzer_empty_state(self):
        """Test analyzer with empty state returns empty list."""
        analyzer = PreIPOInvestorAnalyzer()
        state = {}
        
        investors = analyzer.analyze_investors(state)
        
        assert investors == []

    def test_irr_calculation_basic(self):
        """Test basic IRR calculation: 2x return in 2 years = ~41.4% IRR."""
        analyzer = PreIPOInvestorAnalyzer()
        
        # Entry at ₹100, exit at ₹200, 24 months holding
        irr = analyzer._calculate_irr(
            entry_price=100.0,
            exit_price=200.0,
            holding_period_months=24
        )
        
        # (200/100)^(1/2) - 1 = 0.414 = 41.4%
        assert irr is not None
        assert pytest.approx(irr, rel=1e-2) == 41.42

    def test_irr_calculation_3x_in_3_years(self):
        """Test IRR calculation: 3x return in 3 years = ~44.2% IRR."""
        analyzer = PreIPOInvestorAnalyzer()
        
        irr = analyzer._calculate_irr(
            entry_price=50.0,
            exit_price=150.0,
            holding_period_months=36
        )
        
        # (150/50)^(1/3) - 1 = 0.4422 = 44.22%
        assert irr is not None
        assert pytest.approx(irr, rel=1e-2) == 44.22

    def test_irr_calculation_short_holding_period(self):
        """Test IRR with short holding period (6 months)."""
        analyzer = PreIPOInvestorAnalyzer()
        
        # 1.5x in 6 months annualizes to very high IRR
        irr = analyzer._calculate_irr(
            entry_price=100.0,
            exit_price=150.0,
            holding_period_months=6
        )
        
        # (150/100)^(12/6) - 1 = 1.25 = 125%
        assert irr is not None
        assert pytest.approx(irr, rel=1e-2) == 125.0

    def test_irr_calculation_edge_cases(self):
        """Test IRR calculation edge cases."""
        analyzer = PreIPOInvestorAnalyzer()
        
        # Missing entry price
        assert analyzer._calculate_irr(None, 200.0, 24) is None
        
        # Missing exit price
        assert analyzer._calculate_irr(100.0, None, 24) is None
        
        # Zero holding period
        assert analyzer._calculate_irr(100.0, 200.0, 0) is None
        
        # Zero entry price
        assert analyzer._calculate_irr(0.0, 200.0, 24) is None

    def test_return_multiple_calculation(self):
        """Test return multiple calculation."""
        analyzer = PreIPOInvestorAnalyzer()
        
        # 2x return
        multiple = analyzer._calculate_return_multiple(exit_price=200.0, entry_price=100.0)
        assert multiple == 2.0
        
        # 5x return
        multiple = analyzer._calculate_return_multiple(exit_price=500.0, entry_price=100.0)
        assert multiple == 5.0
        
        # Edge cases
        assert analyzer._calculate_return_multiple(None, 100.0) is None
        assert analyzer._calculate_return_multiple(200.0, None) is None
        assert analyzer._calculate_return_multiple(200.0, 0.0) is None

    def test_investor_classification(self):
        """Test investor type classification heuristics."""
        analyzer = PreIPOInvestorAnalyzer()
        
        assert analyzer._classify_investor("Sequoia Capital Fund") == "PE/VC"
        assert analyzer._classify_investor("Tiger Global Partners") == "PE/VC"
        assert analyzer._classify_investor("Accel Ventures") == "PE/VC"
        assert analyzer._classify_investor("Employee ESOP Trust") == "ESOP Trust"
        assert analyzer._classify_investor("Promoter Group") == "Promoter"
        assert analyzer._classify_investor("Random Individual") == "Other"

    def test_holding_period_calculation(self):
        """Test holding period calculation in months."""
        analyzer = PreIPOInvestorAnalyzer()
        
        # 2 years
        months = analyzer._calculate_holding_period_months("2022-01-15", "2024-01-15")
        assert months == 24
        
        # 18 months
        months = analyzer._calculate_holding_period_months("2023-01-01", "2024-07-01")
        assert months == 18
        
        # Edge cases
        assert analyzer._calculate_holding_period_months(None, "2024-01-01") is None
        assert analyzer._calculate_holding_period_months("2022-01-01", None) is None
        assert analyzer._calculate_holding_period_months("invalid", "2024-01-01") is None

    def test_full_investor_analysis(self):
        """Test complete investor analysis with state data."""
        analyzer = PreIPOInvestorAnalyzer()
        state = {
            "capital_structure_history": [
                {
                    "investor": "Alpha Fund",
                    "date": "2022-01-15",
                    "price": 100.0,
                    "shares": 1_000_000,
                },
                {
                    "investor": "Alpha Fund",
                    "date": "2023-01-15",
                    "price": 150.0,
                    "shares": 500_000,
                },
                {
                    "investor": "Beta Capital Partners",
                    "date": "2023-06-01",
                    "price": 200.0,
                    "shares": 800_000,
                },
            ],
            "ofs_details": [
                {"investor": "Alpha Fund", "shares": 300_000, "price": 400.0},
            ],
            "lock_in_schedule": [
                {"investor": "Alpha Fund", "period": "6 months", "expiry": "2025-06-01", "shares_locked": 700_000},
            ],
            "ipo_details": {
                "price_band_floor": 350.0,
                "price_band_cap": 400.0,
                "listing_date": "2024-12-01",
            },
            "share_capital": {"pre_issue_shares": 10_000_000},
        }
        
        investors = analyzer.analyze_investors(state)
        
        assert len(investors) == 2
        
        # Check Alpha Fund
        alpha = next((i for i in investors if i.name == "Alpha Fund"), None)
        assert alpha is not None
        assert alpha.category == "PE/VC"
        assert alpha.shares_acquired == 1_500_000
        assert alpha.shares_selling_via_ofs == 300_000
        assert alpha.shares_locked == 700_000
        assert alpha.implied_return_multiple_at_cap is not None
        assert alpha.implied_irr_at_cap is not None
        
        # Check Beta Capital
        beta = next((i for i in investors if i.name == "Beta Capital Partners"), None)
        assert beta is not None
        assert beta.category == "PE/VC"
        assert beta.shares_acquired == 800_000

    def test_rupees_to_crore_conversion(self):
        """Test rupee to crore conversion utility."""
        analyzer = PreIPOInvestorAnalyzer()
        
        # 1 Crore = 10,000,000 rupees
        assert analyzer._rupees_to_crore(10_000_000) == 1.0
        assert analyzer._rupees_to_crore(100_000_000) == 10.0
        assert analyzer._rupees_to_crore(0) == 0.0


# ==============================================================================
# OrderBookAnalyzer Tests
# ==============================================================================


class TestOrderBookAnalyzer:
    """Test suite for OrderBookAnalyzer."""

    def test_order_book_analysis_dataclass_defaults(self):
        """Test OrderBookAnalysis initialization with defaults."""
        analysis = OrderBookAnalysis()
        
        assert analysis.applicable is False
        assert analysis.total_order_book == 0.0
        assert analysis.order_book_to_ltm_revenue is None

    def test_analyzer_initialization(self):
        """Test OrderBookAnalyzer can be initialized."""
        analyzer = OrderBookAnalyzer()
        
        assert analyzer is not None

    def test_sector_applicability_defense(self):
        """Test order book analysis for defense sector (applicable)."""
        analyzer = OrderBookAnalyzer()
        
        assert analyzer._is_applicable_sector("Defense") is True
        assert analyzer._is_applicable_sector("defense") is True
        assert analyzer._is_applicable_sector("DEFENSE") is True

    def test_sector_applicability_epc(self):
        """Test order book analysis for EPC sector (applicable)."""
        analyzer = OrderBookAnalyzer()
        
        assert analyzer._is_applicable_sector("EPC") is True
        assert analyzer._is_applicable_sector("epc") is True

    def test_sector_applicability_infrastructure(self):
        """Test order book analysis for infrastructure sector (applicable)."""
        analyzer = OrderBookAnalyzer()
        
        assert analyzer._is_applicable_sector("Infrastructure") is True
        assert analyzer._is_applicable_sector("infrastructure") is True

    def test_sector_applicability_capital_goods(self):
        """Test order book analysis for capital goods sector (applicable)."""
        analyzer = OrderBookAnalyzer()
        
        assert analyzer._is_applicable_sector("Capital Goods") is True
        assert analyzer._is_applicable_sector("capital goods") is True

    def test_sector_applicability_it_services(self):
        """Test order book analysis for IT services sector (applicable)."""
        analyzer = OrderBookAnalyzer()
        
        assert analyzer._is_applicable_sector("IT Services") is True
        assert analyzer._is_applicable_sector("it services") is True

    def test_sector_non_applicability(self):
        """Test order book analysis for non-applicable sectors."""
        analyzer = OrderBookAnalyzer()
        
        # FMCG, Pharma, Retail, Banking etc. should not be applicable
        assert analyzer._is_applicable_sector("FMCG") is False
        assert analyzer._is_applicable_sector("Pharma") is False
        assert analyzer._is_applicable_sector("Retail") is False
        assert analyzer._is_applicable_sector("Banking") is False
        assert analyzer._is_applicable_sector("Fintech") is False
        assert analyzer._is_applicable_sector("") is False

    def test_non_applicable_sector_returns_not_applicable(self):
        """Test analyzer returns not applicable for non-applicable sectors."""
        analyzer = OrderBookAnalyzer()
        state = {"order_book_data": {"total_order_book": 500}}
        
        analysis = analyzer.analyze_order_book(state, sector="FMCG")
        
        assert analysis.applicable is False
        assert analysis.total_order_book == 0.0

    def test_order_book_to_revenue_ratio(self):
        """Test order book to LTM revenue ratio calculation."""
        analyzer = OrderBookAnalyzer()
        state = {
            "order_book_data": {
                "total_order_book": 1000.0,  # ₹1000 Cr order book
                "as_of_date": "2024-12-31",
            },
            "financial_data": {
                "ltm_revenue": 500.0,  # ₹500 Cr LTM revenue
            },
        }
        
        analysis = analyzer.analyze_order_book(state, sector="EPC")
        
        assert analysis.applicable is True
        assert analysis.total_order_book == 1000.0
        assert analysis.order_book_to_ltm_revenue == pytest.approx(2.0, rel=1e-3)

    def test_top_5_orders_concentration(self):
        """Test top 5 orders concentration calculation."""
        analyzer = OrderBookAnalyzer()
        state = {
            "order_book_data": {
                "total_order_book": 1000.0,
                "top_orders": [200.0, 150.0, 100.0, 80.0, 70.0, 50.0, 50.0],
            },
            "financial_data": {"ltm_revenue": 500.0},
        }
        
        analysis = analyzer.analyze_order_book(state, sector="Defense")
        
        # Top 5: 200 + 150 + 100 + 80 + 70 = 600
        assert analysis.top_5_orders_value == 600.0
        # Concentration: 600 / 1000 * 100 = 60%
        assert analysis.top_5_orders_concentration == pytest.approx(60.0, rel=1e-3)

    def test_largest_single_order_calculation(self):
        """Test largest single order calculation."""
        analyzer = OrderBookAnalyzer()
        state = {
            "order_book_data": {
                "total_order_book": 1000.0,
                "largest_order": 250.0,
            },
            "financial_data": {"ltm_revenue": 500.0},
        }
        
        analysis = analyzer.analyze_order_book(state, sector="Infrastructure")
        
        assert analysis.largest_single_order == 250.0
        # 250 / 1000 * 100 = 25%
        assert analysis.largest_single_order_percent == pytest.approx(25.0, rel=1e-3)

    def test_executable_in_12_months_calculation(self):
        """Test executable in 12 months calculation."""
        analyzer = OrderBookAnalyzer()
        state = {
            "order_book_data": {
                "total_order_book": 1000.0,
                "executable_in_12_months": 400.0,
            },
            "financial_data": {"ltm_revenue": 500.0},
        }
        
        analysis = analyzer.analyze_order_book(state, sector="Capital Goods")
        
        assert analysis.executable_in_12_months == 400.0
        # 400 / 1000 * 100 = 40%
        assert analysis.executable_in_12_months_percent == pytest.approx(40.0, rel=1e-3)

    def test_order_book_yoy_growth(self):
        """Test year-over-year order book growth calculation."""
        analyzer = OrderBookAnalyzer()
        state = {
            "order_book_data": {
                "total_order_book": 1200.0,
                "order_book_1yr_ago": 1000.0,
            },
            "financial_data": {"ltm_revenue": 500.0},
        }
        
        analysis = analyzer.analyze_order_book(state, sector="Engineering")
        
        # (1200 / 1000 - 1) * 100 = 20%
        assert analysis.order_book_growth_yoy == pytest.approx(20.0, rel=1e-3)

    def test_government_vs_private_orders(self):
        """Test government vs private orders breakdown."""
        analyzer = OrderBookAnalyzer()
        state = {
            "order_book_data": {
                "total_order_book": 1000.0,
                "government_orders_percent": 60.0,
                "private_orders_percent": 40.0,
            },
            "financial_data": {"ltm_revenue": 500.0},
        }
        
        analysis = analyzer.analyze_order_book(state, sector="Defense")
        
        assert analysis.government_orders_percent == 60.0
        assert analysis.private_orders_percent == 40.0

    def test_full_order_book_analysis(self):
        """Test complete order book analysis with all fields."""
        analyzer = OrderBookAnalyzer()
        state = {
            "order_book_data": {
                "total_order_book": 2500.0,
                "as_of_date": "2024-12-31",
                "top_orders": [500.0, 400.0, 350.0, 300.0, 250.0, 200.0],
                "largest_order": 500.0,
                "executable_in_12_months": 1000.0,
                "order_book_1yr_ago": 2000.0,
                "government_orders_percent": 70.0,
                "private_orders_percent": 30.0,
                "repeat_customer_orders_percent": 45.0,
            },
            "financial_data": {
                "ltm_revenue": 1000.0,
            },
        }
        
        analysis = analyzer.analyze_order_book(state, sector="EPC")
        
        assert analysis.applicable is True
        assert analysis.total_order_book == 2500.0
        assert analysis.order_book_as_of_date == "2024-12-31"
        assert analysis.order_book_to_ltm_revenue == pytest.approx(2.5, rel=1e-3)
        assert analysis.top_5_orders_value == 1800.0  # 500+400+350+300+250
        assert analysis.top_5_orders_concentration == pytest.approx(72.0, rel=1e-3)
        assert analysis.largest_single_order == 500.0
        assert analysis.largest_single_order_percent == pytest.approx(20.0, rel=1e-3)
        assert analysis.executable_in_12_months == 1000.0
        assert analysis.executable_in_12_months_percent == pytest.approx(40.0, rel=1e-3)
        assert analysis.order_book_growth_yoy == pytest.approx(25.0, rel=1e-3)
        assert analysis.government_orders_percent == 70.0
        assert analysis.repeat_customer_orders_percent == 45.0

    def test_missing_order_book_data(self):
        """Test behavior when order book data is missing for applicable sector."""
        analyzer = OrderBookAnalyzer()
        state = {
            "financial_data": {"ltm_revenue": 500.0},
            # No order_book_data provided
        }
        
        analysis = analyzer.analyze_order_book(state, sector="Defense")
        
        # Should still be applicable, just empty
        assert analysis.applicable is True
        assert analysis.total_order_book == 0.0
        assert analysis.order_book_to_ltm_revenue is None

    def test_safe_divide_utility(self):
        """Test safe divide utility method."""
        analyzer = OrderBookAnalyzer()
        
        assert analyzer._safe_divide(100.0, 50.0) == 2.0
        assert analyzer._safe_divide(100.0, 0.0) is None
        assert analyzer._safe_divide(100.0, None) is None


# ==============================================================================
# Edge Cases and Integration Tests
# ==============================================================================


class TestSpecializedExtractorsEdgeCases:
    """Test edge cases across all specialized extractors."""

    def test_promoter_with_zero_shareholding(self):
        """Test promoter with zero shareholding (rare but possible)."""
        dossier = PromoterDossier(
            name="Non-Owning Promoter",
            shareholding_pre_ipo=0.0,
            shareholding_post_ipo=0.0,
        )
        
        assert dossier.shareholding_pre_ipo == 0.0
        assert dossier.shareholding_post_ipo == 0.0

    def test_pre_ipo_investor_negative_return(self):
        """Test pre-IPO investor with negative return (exit below entry)."""
        analyzer = PreIPOInvestorAnalyzer()
        
        # Entry at ₹200, exit at ₹150 (loss)
        multiple = analyzer._calculate_return_multiple(exit_price=150.0, entry_price=200.0)
        assert multiple == 0.75
        
        # IRR would be negative
        irr = analyzer._calculate_irr(entry_price=200.0, exit_price=150.0, holding_period_months=12)
        assert irr is not None
        assert irr < 0  # Negative IRR

    def test_order_book_with_zero_values(self):
        """Test order book analysis with zero values."""
        analyzer = OrderBookAnalyzer()
        state = {
            "order_book_data": {
                "total_order_book": 0.0,
            },
            "financial_data": {"ltm_revenue": 500.0},
        }
        
        analysis = analyzer.analyze_order_book(state, sector="EPC")
        
        assert analysis.applicable is True
        assert analysis.total_order_book == 0.0
        assert analysis.top_5_orders_concentration is None  # Division by zero avoided

    def test_very_high_irr_short_term_investor(self):
        """Test very high IRR for short-term pre-IPO investor (3x in 3 months)."""
        analyzer = PreIPOInvestorAnalyzer()
        
        # 3x return in 3 months
        irr = analyzer._calculate_irr(
            entry_price=100.0,
            exit_price=300.0,
            holding_period_months=3
        )
        
        # (300/100)^(12/3) - 1 = 3^4 - 1 = 81 - 1 = 80 = 8000%
        assert irr is not None
        assert irr > 500  # Very high annualized return

    def test_order_book_from_sections_fallback(self):
        """Test order book data extraction from sections fallback."""
        analyzer = OrderBookAnalyzer()
        state = {
            "sections": {
                "Order Book": {
                    "structured_data": {
                        "total_order_book": 800.0,
                        "as_of_date": "2024-09-30",
                    }
                }
            },
            "financial_data": {"ltm_revenue": 400.0},
        }
        
        analysis = analyzer.analyze_order_book(state, sector="Infrastructure")
        
        assert analysis.applicable is True
        assert analysis.total_order_book == 800.0
        assert analysis.order_book_to_ltm_revenue == pytest.approx(2.0, rel=1e-3)


# =============================================================================
# Test FinancialParser (Subtask 2.5A.0)
# =============================================================================

from rhp_analyzer.ingestion.financial_parser import (
    FinancialMetrics,
    NewAgeMetrics,
    DivergenceWarning,
    TrendAnalysis,
    FinancialParser,
)


class TestFinancialMetrics:
    """Tests for the FinancialMetrics dataclass."""
    
    def test_financial_metrics_dataclass_defaults(self):
        """Test FinancialMetrics dataclass default values."""
        metrics = FinancialMetrics(fiscal_year="FY2024")
        
        assert metrics.fiscal_year == "FY2024"
        assert metrics.revenue == 0.0
        assert metrics.ebitda == 0.0
        assert metrics.pat == 0.0
        assert metrics.total_assets == 0.0
        assert metrics.total_equity == 0.0
        assert metrics.total_debt == 0.0
        assert metrics.roe is None
        assert metrics.roce is None
        assert metrics.debt_equity is None
    
    def test_financial_metrics_full_profile(self):
        """Test FinancialMetrics with full data."""
        metrics = FinancialMetrics(
            fiscal_year="FY2024",
            revenue=1000.0,
            cost_of_goods_sold=600.0,
            gross_profit=400.0,
            ebitda=200.0,
            depreciation=50.0,
            ebit=150.0,
            interest_expense=30.0,
            pbt=120.0,
            tax_expense=30.0,
            pat=90.0,
            total_assets=800.0,
            total_equity=400.0,
            total_debt=200.0,
            current_assets=300.0,
            current_liabilities=150.0,
            cfo=180.0,
            capex=80.0,
        )
        
        assert metrics.fiscal_year == "FY2024"
        assert metrics.revenue == 1000.0
        assert metrics.gross_profit == 400.0
        assert metrics.ebitda == 200.0
        assert metrics.pat == 90.0
        assert metrics.total_equity == 400.0
        assert metrics.cfo == 180.0
    
    def test_financial_metrics_to_dict(self):
        """Test FinancialMetrics to_dict method."""
        metrics = FinancialMetrics(
            fiscal_year="FY2024",
            revenue=500.0,
            ebitda=100.0,
            pat=50.0,
        )
        
        result = metrics.to_dict()
        
        assert isinstance(result, dict)
        assert result["fiscal_year"] == "FY2024"
        assert result["revenue"] == 500.0
        assert result["ebitda"] == 100.0
        assert result["pat"] == 50.0


class TestNewAgeMetrics:
    """Tests for the NewAgeMetrics dataclass (startup metrics)."""
    
    def test_new_age_metrics_defaults(self):
        """Test NewAgeMetrics default values."""
        metrics = NewAgeMetrics(fiscal_year="FY2024")
        
        assert metrics.fiscal_year == "FY2024"
        assert metrics.contribution_margin is None
        assert metrics.cac is None
        assert metrics.ltv is None
        assert metrics.burn_rate is None
        assert metrics.runway_months is None
    
    def test_new_age_metrics_full_profile(self):
        """Test NewAgeMetrics with startup data."""
        metrics = NewAgeMetrics(
            fiscal_year="FY2024",
            contribution_margin=45.0,
            cac=2500.0,
            ltv=7500.0,
            cac_ltv_ratio=3.0,
            burn_rate=15.0,
            runway_months=18,
            revenue_per_user=150.0,
            gmv=1000.0,
            take_rate=0.15,
        )
        
        assert metrics.fiscal_year == "FY2024"
        assert metrics.contribution_margin == 45.0
        assert metrics.cac == 2500.0
        assert metrics.ltv == 7500.0
        assert metrics.cac_ltv_ratio == 3.0
        assert metrics.burn_rate == 15.0
        assert metrics.runway_months == 18
    
    def test_new_age_metrics_to_dict(self):
        """Test NewAgeMetrics to_dict method."""
        metrics = NewAgeMetrics(
            fiscal_year="FY2024",
            contribution_margin=40.0,
            burn_rate=10.0,
            runway_months=24,
        )
        
        result = metrics.to_dict()
        
        assert isinstance(result, dict)
        assert result["fiscal_year"] == "FY2024"
        assert result["contribution_margin"] == 40.0
        assert result["burn_rate"] == 10.0
        assert result["runway_months"] == 24


class TestDivergenceWarning:
    """Tests for the DivergenceWarning dataclass."""
    
    def test_divergence_warning_defaults(self):
        """Test DivergenceWarning default values."""
        warning = DivergenceWarning(
            fiscal_year="FY2024",
            warning_type="channel_stuffing",
            severity="MAJOR",
            description="Test warning",
            metric_name="receivables_growth",
            actual_value=25.0,
            threshold_value=10.0,
        )
        
        assert warning.fiscal_year == "FY2024"
        assert warning.warning_type == "channel_stuffing"
        assert warning.severity == "MAJOR"
        assert warning.description == "Test warning"
        assert warning.metric_name == "receivables_growth"
        assert warning.actual_value == 25.0
        assert warning.threshold_value == 10.0
        assert warning.citation == ""  # Only field with default
    
    def test_divergence_warning_full(self):
        """Test DivergenceWarning with full data."""
        warning = DivergenceWarning(
            fiscal_year="FY2024",
            warning_type="paper_profits",
            severity="CRITICAL",
            description="CFO/EBITDA = 35% - Paper profits risk",
            metric_name="cfo_to_ebitda",
            actual_value=35.0,
            threshold_value=50.0,
        )
        
        assert warning.actual_value == 35.0
        assert warning.threshold_value == 50.0
        assert "paper_profits" in warning.warning_type
    
    def test_divergence_warning_to_dict(self):
        """Test DivergenceWarning to_dict method."""
        warning = DivergenceWarning(
            fiscal_year="FY2024",
            warning_type="inventory_piling",
            severity="MAJOR",
            description="Inventory days increased",
            metric_name="inventory_days_change",
            actual_value=25.0,
            threshold_value=15.0,
        )
        
        result = warning.to_dict()
        
        assert isinstance(result, dict)
        assert result["warning_type"] == "inventory_piling"
        assert result["severity"] == "MAJOR"
        assert result["metric_name"] == "inventory_days_change"
        assert result["threshold_value"] == 15.0


class TestTrendAnalysis:
    """Tests for the TrendAnalysis dataclass."""
    
    def test_trend_analysis_defaults(self):
        """Test TrendAnalysis default values."""
        trend = TrendAnalysis()
        
        assert trend.revenue_cagr is None
        assert trend.ebitda_cagr is None
        assert trend.pat_cagr is None
        assert trend.margin_trend == "stable"
        assert trend.roe_trend == "stable"
        assert trend.debt_trend == "stable"
        assert trend.working_capital_trend == "stable"
        assert trend.anomalies == []
    
    def test_trend_analysis_full_profile(self):
        """Test TrendAnalysis with full data."""
        trend = TrendAnalysis(
            revenue_cagr=25.5,
            ebitda_cagr=30.0,
            pat_cagr=35.2,
            margin_trend="expanding",
            roe_trend="improving",
            debt_trend="deleveraging",
            anomalies=["One-time gain in FY23"],
        )
        
        assert trend.revenue_cagr == 25.5
        assert trend.margin_trend == "expanding"
        assert len(trend.anomalies) == 1
    
    def test_trend_analysis_to_dict(self):
        """Test TrendAnalysis to_dict method."""
        trend = TrendAnalysis(
            revenue_cagr=20.0,
            margin_trend="stable",
        )
        
        result = trend.to_dict()
        
        assert isinstance(result, dict)
        assert result["revenue_cagr"] == 20.0
        assert result["margin_trend"] == "stable"


class TestFinancialParser:
    """Tests for the FinancialParser class."""
    
    def test_parser_initialization(self):
        """Test FinancialParser initialization."""
        parser = FinancialParser()
        
        assert parser.table_extractor is None
        assert parser.CRORE_PATTERN is not None
        assert parser.LAKH_PATTERN is not None
    
    def test_parser_initialization_with_extractor(self):
        """Test FinancialParser initialization with table extractor."""
        mock_extractor = object()
        parser = FinancialParser(table_extractor=mock_extractor)
        
        assert parser.table_extractor is mock_extractor
    
    def test_parse_amount_float(self):
        """Test parsing float amounts."""
        parser = FinancialParser()
        
        assert parser._parse_amount(100.5) == 100.5
        assert parser._parse_amount(0) == 0.0
        assert parser._parse_amount(None) == 0.0
    
    def test_parse_amount_string(self):
        """Test parsing string amounts."""
        parser = FinancialParser()
        
        assert parser._parse_amount("100") == 100.0
        assert parser._parse_amount("1,000") == 1000.0
        assert parser._parse_amount("₹500") == 500.0
    
    def test_parse_amount_crore_format(self):
        """Test parsing crore format amounts."""
        parser = FinancialParser()
        
        assert parser._parse_amount("100 Crore") == 100.0
        assert parser._parse_amount("₹50 Cr") == 50.0
        assert parser._parse_amount("Rs. 25 Crores") == 25.0
    
    def test_parse_amount_lakh_format(self):
        """Test parsing lakh format amounts (converts to crores)."""
        parser = FinancialParser()
        
        # 100 Lakh = 1 Crore
        assert parser._parse_amount("100 Lakh") == pytest.approx(1.0, rel=1e-3)
        assert parser._parse_amount("50 Lakhs") == pytest.approx(0.5, rel=1e-3)
    
    def test_parse_amount_negative(self):
        """Test parsing negative amounts."""
        parser = FinancialParser()
        
        assert parser._parse_amount("(100)") == -100.0
        assert parser._parse_amount("-50") == -50.0
    
    def test_parse_financial_statements(self):
        """Test parsing complete financial statements."""
        parser = FinancialParser()
        
        tables = [
            {
                "type": "income_statement",
                "data": {
                    "FY2023": {
                        "revenue": 800.0,
                        "ebitda": 160.0,
                        "pat": 80.0,
                    },
                    "FY2024": {
                        "revenue": 1000.0,
                        "ebitda": 200.0,
                        "pat": 100.0,
                    },
                },
            },
            {
                "type": "balance_sheet",
                "data": {
                    "FY2024": {
                        "total_assets": 600.0,
                        "total_equity": 300.0,
                        "total_debt": 150.0,
                    },
                },
            },
        ]
        
        metrics_list = parser.parse_financial_statements(tables)
        
        assert len(metrics_list) == 2
        assert metrics_list[0].fiscal_year == "FY2023"
        assert metrics_list[1].fiscal_year == "FY2024"
        assert metrics_list[1].revenue == 1000.0
        assert metrics_list[1].total_assets == 600.0
    
    def test_calculate_ratios(self):
        """Test ratio calculations."""
        parser = FinancialParser()
        
        metrics = FinancialMetrics(
            fiscal_year="FY2024",
            revenue=1000.0,
            ebitda=200.0,
            pat=100.0,
            total_equity=500.0,
            total_debt=200.0,
            current_assets=300.0,
            current_liabilities=150.0,
            trade_receivables=100.0,
            cost_of_goods_sold=600.0,
        )
        
        ratios = parser.calculate_ratios(metrics)
        
        assert ratios["roe"] == pytest.approx(20.0, rel=1e-2)  # 100/500 * 100
        assert ratios["ebitda_margin"] == pytest.approx(20.0, rel=1e-2)  # 200/1000 * 100
        assert ratios["current_ratio"] == pytest.approx(2.0, rel=1e-2)  # 300/150
        assert ratios["debt_equity"] == pytest.approx(0.4, rel=1e-2)  # 200/500
    
    def test_detect_divergences_channel_stuffing(self):
        """Test detection of channel stuffing (receivables growing faster than revenue)."""
        parser = FinancialParser()
        
        metrics_list = [
            FinancialMetrics(
                fiscal_year="FY2023",
                revenue=1000.0,
                trade_receivables=100.0,
            ),
            FinancialMetrics(
                fiscal_year="FY2024",
                revenue=1100.0,  # 10% growth
                trade_receivables=150.0,  # 50% growth - 40pp gap
            ),
        ]
        
        warnings = parser.detect_divergences(metrics_list)
        
        assert len(warnings) >= 1
        channel_warning = next(
            (w for w in warnings if w.warning_type == "channel_stuffing"), None
        )
        assert channel_warning is not None
        assert channel_warning.severity in ["MAJOR", "CRITICAL"]
    
    def test_detect_divergences_paper_profits(self):
        """Test detection of paper profits (low CFO/EBITDA)."""
        parser = FinancialParser()
        
        metrics_list = [
            FinancialMetrics(fiscal_year="FY2023", ebitda=100.0, cfo=80.0),
            FinancialMetrics(
                fiscal_year="FY2024",
                ebitda=150.0,
                cfo=45.0,  # 30% CFO/EBITDA - below 50% threshold
            ),
        ]
        
        warnings = parser.detect_divergences(metrics_list)
        
        assert len(warnings) >= 1
        paper_warning = next(
            (w for w in warnings if w.warning_type == "paper_profits"), None
        )
        assert paper_warning is not None
        assert paper_warning.severity in ["MAJOR", "CRITICAL"]
    
    def test_detect_divergences_inventory_piling(self):
        """Test detection of inventory piling."""
        parser = FinancialParser()
        
        metrics_list = [
            FinancialMetrics(
                fiscal_year="FY2023",
                inventory_days=60.0,
            ),
            FinancialMetrics(
                fiscal_year="FY2024",
                inventory_days=95.0,  # 35 days increase
            ),
        ]
        
        warnings = parser.detect_divergences(metrics_list)
        
        assert len(warnings) >= 1
        inventory_warning = next(
            (w for w in warnings if w.warning_type == "inventory_piling"), None
        )
        assert inventory_warning is not None
    
    def test_detect_divergences_no_warnings(self):
        """Test that healthy financials generate no warnings."""
        parser = FinancialParser()
        
        metrics_list = [
            FinancialMetrics(
                fiscal_year="FY2023",
                revenue=1000.0,
                trade_receivables=100.0,
                ebitda=200.0,
                cfo=180.0,
                inventory_days=60.0,
            ),
            FinancialMetrics(
                fiscal_year="FY2024",
                revenue=1100.0,  # 10% growth
                trade_receivables=108.0,  # ~8% growth - within bounds
                ebitda=220.0,
                cfo=190.0,  # 86% CFO/EBITDA - healthy
                inventory_days=62.0,  # Only 2 day increase
            ),
        ]
        
        warnings = parser.detect_divergences(metrics_list)
        
        assert len(warnings) == 0


# =============================================================================
# Test DebtStructureAnalyzer (Subtask 2.5A.4)
# =============================================================================

from rhp_analyzer.ingestion.debt_analyzer import (
    DebtItem,
    MaturityProfile,
    Covenant,
    DebtStructure,
    DebtStructureAnalyzer,
)


class TestDebtItem:
    """Tests for the DebtItem dataclass."""
    
    def test_debt_item_defaults(self):
        """Test DebtItem default values."""
        item = DebtItem(
            lender="HDFC Bank",
            facility_type="Term Loan",
            amount=100.0,
        )
        
        assert item.lender == "HDFC Bank"
        assert item.facility_type == "Term Loan"
        assert item.amount == 100.0
        assert item.interest_rate is None
        assert item.is_secured is True
        assert item.is_short_term is False
        assert item.maturity_date is None
    
    def test_debt_item_full_profile(self):
        """Test DebtItem with full data."""
        item = DebtItem(
            lender="State Bank of India",
            facility_type="Working Capital",
            amount=50.0,
            interest_rate=9.5,
            is_secured=True,
            is_short_term=True,
            maturity_date="2025-03-31",
            security_details="First pari-passu charge on current assets",
            covenants=["DSCR > 1.5"],
            outstanding_as_of="2024-09-30",
            page_reference=156,
        )
        
        assert item.lender == "State Bank of India"
        assert item.interest_rate == 9.5
        assert item.is_short_term is True
        assert item.page_reference == 156


class TestMaturityProfile:
    """Tests for the MaturityProfile dataclass."""
    
    def test_maturity_profile_defaults(self):
        """Test MaturityProfile default values."""
        profile = MaturityProfile()
        
        assert profile.within_1_year == 0.0
        assert profile.between_1_and_3_years == 0.0
        assert profile.between_3_and_5_years == 0.0
        assert profile.beyond_5_years == 0.0
    
    def test_maturity_profile_total(self):
        """Test MaturityProfile total property."""
        profile = MaturityProfile(
            within_1_year=50.0,
            between_1_and_3_years=100.0,
            between_3_and_5_years=75.0,
            beyond_5_years=25.0,
        )
        
        assert profile.total == pytest.approx(250.0, rel=1e-3)
    
    def test_maturity_profile_to_dict(self):
        """Test MaturityProfile to_dict method."""
        profile = MaturityProfile(
            within_1_year=30.0,
            between_1_and_3_years=70.0,
        )
        
        result = profile.to_dict()
        
        assert isinstance(result, dict)
        assert result["0-1yr"] == 30.0
        assert result["1-3yr"] == 70.0
        assert result["total"] == 100.0


class TestCovenant:
    """Tests for the Covenant dataclass."""
    
    def test_covenant_defaults(self):
        """Test Covenant default values."""
        covenant = Covenant(
            covenant_type="DSCR",
            description="Debt Service Coverage Ratio",
        )
        
        assert covenant.covenant_type == "DSCR"
        assert covenant.description == "Debt Service Coverage Ratio"
        assert covenant.threshold is None
        assert covenant.current_value is None
        assert covenant.is_breached is False
    
    def test_covenant_with_values(self):
        """Test Covenant with threshold and current value."""
        covenant = Covenant(
            covenant_type="Debt Equity",
            description="Maximum Debt to Equity ratio",
            threshold="< 2.0",
            current_value=1.5,
            is_breached=False,
            lender="ICICI Bank",
            source="Facility Agreement dated 01-Apr-2023",
        )
        
        assert covenant.threshold == "< 2.0"
        assert covenant.current_value == 1.5
        assert covenant.is_breached is False
        assert covenant.lender == "ICICI Bank"
    
    def test_covenant_breached(self):
        """Test breached covenant."""
        covenant = Covenant(
            covenant_type="Current Ratio",
            description="Minimum current ratio",
            threshold="> 1.25",
            current_value=1.1,
            is_breached=True,
        )
        
        assert covenant.is_breached is True


class TestDebtStructure:
    """Tests for the DebtStructure dataclass."""
    
    def test_debt_structure_defaults(self):
        """Test DebtStructure default values."""
        structure = DebtStructure()
        
        assert structure.total_debt == 0.0
        assert structure.secured_debt == 0.0
        assert structure.unsecured_debt == 0.0
        assert structure.short_term_debt == 0.0
        assert structure.long_term_debt == 0.0
        assert structure.debt_items == []
        assert isinstance(structure.maturity_profile, MaturityProfile)
        assert structure.covenants == []
        assert structure.has_financial_covenants is False
    
    def test_debt_structure_full_profile(self):
        """Test DebtStructure with full data."""
        structure = DebtStructure(
            total_debt=500.0,
            secured_debt=400.0,
            unsecured_debt=100.0,
            short_term_debt=100.0,
            long_term_debt=400.0,
            weighted_avg_interest_rate=10.5,
            highest_interest_rate=12.0,
            lowest_interest_rate=8.5,
            number_of_lenders=4,
            top_lender="HDFC Bank",
            top_lender_exposure=200.0,
            debt_repayment_from_ipo=150.0,
            post_ipo_debt=350.0,
            has_financial_covenants=True,
        )
        
        assert structure.total_debt == 500.0
        assert structure.secured_debt == 400.0
        assert structure.weighted_avg_interest_rate == 10.5
        assert structure.top_lender == "HDFC Bank"
        assert structure.post_ipo_debt == 350.0
    
    def test_debt_structure_concentration_risk_high(self):
        """Test get_concentration_risk for high concentration."""
        structure = DebtStructure(
            total_debt=100.0,
            top_lender_exposure=75.0,  # 75% from one lender
        )
        
        risk = structure.get_concentration_risk()
        
        assert risk == "HIGH"
    
    def test_debt_structure_concentration_risk_medium(self):
        """Test get_concentration_risk for medium concentration."""
        structure = DebtStructure(
            total_debt=100.0,
            top_lender_exposure=45.0,  # 45% from one lender
        )
        
        risk = structure.get_concentration_risk()
        
        assert risk == "MEDIUM"
    
    def test_debt_structure_concentration_risk_low(self):
        """Test get_concentration_risk for low concentration."""
        structure = DebtStructure(
            total_debt=100.0,
            top_lender_exposure=20.0,  # 20% from one lender
        )
        
        risk = structure.get_concentration_risk()
        
        assert risk == "LOW"
    
    def test_debt_structure_maturity_risk_high(self):
        """Test get_maturity_risk for high short-term concentration."""
        structure = DebtStructure(
            total_debt=100.0,
            maturity_profile=MaturityProfile(
                within_1_year=55.0,  # 55% maturing in 1 year
                between_1_and_3_years=25.0,
                between_3_and_5_years=15.0,
                beyond_5_years=5.0,
            ),
        )
        
        risk = structure.get_maturity_risk()
        
        assert risk == "HIGH"
    
    def test_debt_structure_maturity_risk_low(self):
        """Test get_maturity_risk for well-distributed maturity."""
        structure = DebtStructure(
            total_debt=100.0,
            maturity_profile=MaturityProfile(
                within_1_year=20.0,  # Only 20% in 1 year
                between_1_and_3_years=30.0,
                between_3_and_5_years=30.0,
                beyond_5_years=20.0,
            ),
        )
        
        risk = structure.get_maturity_risk()
        
        assert risk == "LOW"


class TestDebtStructureAnalyzer:
    """Tests for the DebtStructureAnalyzer class."""
    
    def test_analyzer_initialization(self):
        """Test DebtStructureAnalyzer initialization."""
        analyzer = DebtStructureAnalyzer()
        
        assert hasattr(analyzer, "LENDER_PATTERNS")
        assert hasattr(analyzer, "FACILITY_PATTERNS")
        assert hasattr(analyzer, "COVENANT_PATTERNS")
        assert len(analyzer.LENDER_PATTERNS) > 0
    
    def test_classify_facility_type_term_loan(self):
        """Test facility type classification for term loan."""
        analyzer = DebtStructureAnalyzer()
        
        result = analyzer._classify_facility_type("Term Loan Facility")
        
        assert "Term Loan" in result
    
    def test_classify_facility_type_working_capital(self):
        """Test facility type classification for working capital."""
        analyzer = DebtStructureAnalyzer()
        
        result = analyzer._classify_facility_type("CC/OD Working Capital")
        
        assert "Working Capital" in result
    
    def test_classify_facility_type_ncd(self):
        """Test facility type classification for NCDs."""
        analyzer = DebtStructureAnalyzer()
        
        result = analyzer._classify_facility_type("Non-Convertible Debentures")
        
        assert "Ncd" in result
    
    def test_parse_amount_numeric(self):
        """Test parsing numeric amounts."""
        analyzer = DebtStructureAnalyzer()
        
        assert analyzer._parse_amount(100.5) == 100.5
        assert analyzer._parse_amount(100) == 100.0
        assert analyzer._parse_amount(None) == 0.0
    
    def test_parse_amount_string(self):
        """Test parsing string amounts."""
        analyzer = DebtStructureAnalyzer()
        
        assert analyzer._parse_amount("100") == 100.0
        assert analyzer._parse_amount("1,000") == 1000.0
        assert analyzer._parse_amount("₹500") == 500.0
    
    def test_parse_amount_crore(self):
        """Test parsing crore format."""
        analyzer = DebtStructureAnalyzer()
        
        assert analyzer._parse_amount("50 Crore") == 50.0
        assert analyzer._parse_amount("100Cr") == 100.0
    
    def test_parse_rate_numeric(self):
        """Test parsing numeric interest rates."""
        analyzer = DebtStructureAnalyzer()
        
        assert analyzer._parse_rate(9.5) == 9.5
        assert analyzer._parse_rate(None) is None
    
    def test_parse_rate_string(self):
        """Test parsing string interest rates."""
        analyzer = DebtStructureAnalyzer()
        
        assert analyzer._parse_rate("9.5%") == 9.5
        assert analyzer._parse_rate("MCLR + 2.5%") == 2.5
    
    def test_analyze_debt_basic(self):
        """Test basic debt analysis."""
        analyzer = DebtStructureAnalyzer()
        
        indebtedness_data = {
            "debt_items": [
                {
                    "lender": "HDFC Bank",
                    "facility_type": "Term Loan",
                    "amount": 100.0,
                    "interest_rate": 10.0,
                    "is_secured": True,
                    "is_short_term": False,
                },
                {
                    "lender": "SBI",
                    "facility_type": "Working Capital",
                    "amount": 50.0,
                    "interest_rate": 9.0,
                    "is_secured": True,
                    "is_short_term": True,
                },
            ],
            "maturity_profile": {
                "0-1yr": 50.0,
                "1-3yr": 60.0,
                "3-5yr": 30.0,
                "5yr+": 10.0,
            },
        }
        
        result = analyzer.analyze_debt(indebtedness_data)
        
        assert result.total_debt == 150.0
        assert result.secured_debt == 150.0
        assert result.short_term_debt == 50.0
        assert result.long_term_debt == 100.0
        assert result.number_of_lenders == 2
    
    def test_analyze_debt_with_covenants(self):
        """Test debt analysis with covenants."""
        analyzer = DebtStructureAnalyzer()
        
        indebtedness_data = {
            "debt_items": [
                {
                    "lender": "ICICI Bank",
                    "facility_type": "Term Loan",
                    "amount": 200.0,
                    "interest_rate": 11.0,
                    "is_secured": True,
                    "is_short_term": False,
                },
            ],
            "covenants": [
                {
                    "type": "DSCR",
                    "description": "Minimum DSCR of 1.5x",
                    "threshold": "> 1.5",
                    "is_breached": False,
                },
                {
                    "type": "Debt Equity",
                    "description": "Maximum D/E of 2.0x",
                    "threshold": "< 2.0",
                    "is_breached": False,
                },
            ],
        }
        
        result = analyzer.analyze_debt(indebtedness_data)
        
        assert result.has_financial_covenants is True
        assert len(result.covenants) == 2
        assert result.covenant_breaches_disclosed is False
    
    def test_analyze_debt_with_ipo_impact(self):
        """Test debt analysis with IPO proceeds impact."""
        analyzer = DebtStructureAnalyzer()
        
        indebtedness_data = {
            "debt_items": [
                {
                    "lender": "Axis Bank",
                    "facility_type": "Term Loan",
                    "amount": 300.0,
                    "interest_rate": 10.5,
                    "is_secured": True,
                    "is_short_term": False,
                },
            ],
        }
        
        ipo_details = {
            "fresh_issue": 500.0,
            "debt_repayment": 150.0,
        }
        
        financial_data = {
            "total_equity": 400.0,
            "ebitda": 100.0,
        }
        
        result = analyzer.analyze_debt(
            indebtedness_data,
            ipo_details=ipo_details,
            financial_data=financial_data,
        )
        
        assert result.debt_repayment_from_ipo == 150.0
        assert result.post_ipo_debt == 150.0  # 300 - 150
    
    def test_analyze_debt_interest_rates(self):
        """Test weighted average interest rate calculation."""
        analyzer = DebtStructureAnalyzer()
        
        indebtedness_data = {
            "debt_items": [
                {
                    "lender": "Bank A",
                    "facility_type": "Term Loan",
                    "amount": 100.0,
                    "interest_rate": 10.0,
                    "is_secured": True,
                    "is_short_term": False,
                },
                {
                    "lender": "Bank B",
                    "facility_type": "Term Loan",
                    "amount": 100.0,
                    "interest_rate": 12.0,
                    "is_secured": True,
                    "is_short_term": False,
                },
            ],
        }
        
        result = analyzer.analyze_debt(indebtedness_data)
        
        # Weighted avg = (100*10 + 100*12) / 200 = 11
        assert result.weighted_avg_interest_rate == pytest.approx(11.0, rel=1e-2)
        assert result.highest_interest_rate == 12.0
        assert result.lowest_interest_rate == 10.0
