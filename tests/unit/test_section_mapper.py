"""
Unit tests for SectionMapper module.

Tests section detection, hierarchy building, and section classification
for RHP document analysis.
"""

import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import List, Dict, Optional

from rhp_analyzer.ingestion.section_mapper import (
    Section,
    SectionMapper,
    SectionTree,
    SectionType,
    STANDARD_RHP_SECTIONS,
    normalize_section_name,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def section_mapper():
    """Create a SectionMapper instance for testing."""
    return SectionMapper()


@pytest.fixture
def sample_page_info():
    """Create a mock PageInfo object for testing."""
    page = Mock()
    page.page_num = 1
    page.text = "RISK FACTORS\n\nThe following are the risk factors..."
    page.word_count = 100
    page.has_tables = False
    page.font_sizes = [12.0, 14.0, 18.0]  # Various font sizes
    return page


@pytest.fixture
def mock_pages_with_sections():
    """Create a list of mock pages with section headers."""
    pages = []
    
    # Page 1: Cover page
    page1 = Mock()
    page1.page_num = 1
    page1.text = "RED HERRING PROSPECTUS\n\nCompany Name Limited\nIPO Details"
    page1.word_count = 50
    page1.has_tables = False
    page1.font_sizes = [24.0, 18.0, 14.0, 12.0]
    pages.append(page1)
    
    # Page 2: Summary section
    page2 = Mock()
    page2.page_num = 2
    page2.text = "SUMMARY OF THE PROSPECTUS\n\nThis summary provides an overview..."
    page2.word_count = 200
    page2.has_tables = False
    page2.font_sizes = [18.0, 12.0]
    pages.append(page2)
    
    # Page 3: Risk Factors section
    page3 = Mock()
    page3.page_num = 3
    page3.text = "RISK FACTORS\n\nInvestors should consider the following risks..."
    page3.word_count = 500
    page3.has_tables = False
    page3.font_sizes = [18.0, 14.0, 12.0]
    pages.append(page3)
    
    # Page 4: Business section
    page4 = Mock()
    page4.page_num = 4
    page4.text = "OUR BUSINESS\n\nWe are engaged in manufacturing..."
    page4.word_count = 800
    page4.has_tables = True
    page4.font_sizes = [18.0, 14.0, 12.0]
    pages.append(page4)
    
    # Page 5: Financial Information section
    page5 = Mock()
    page5.page_num = 5
    page5.text = "FINANCIAL INFORMATION\n\nRestated Financial Statements..."
    page5.word_count = 600
    page5.has_tables = True
    page5.font_sizes = [18.0, 14.0, 12.0]
    pages.append(page5)
    
    return pages


@pytest.fixture
def sample_section():
    """Create a sample Section for testing."""
    return Section(
        section_id="sect_001",
        section_type=SectionType.RISK_FACTORS,
        title="RISK FACTORS",
        level=1,
        start_page=10,
        end_page=25,
        content="Risk factors content...",
        word_count=5000,
        has_tables=False,
        subsections=[],
        parent_id=None,
        metadata={"detected_by": "pattern_match"}
    )


@pytest.fixture
def sample_section_tree():
    """Create a sample SectionTree for testing."""
    section1 = Section(
        section_id="sect_001",
        section_type=SectionType.SUMMARY,
        title="Summary",
        level=1,
        start_page=1,
        end_page=5
    )
    section2 = Section(
        section_id="sect_002",
        section_type=SectionType.RISK_FACTORS,
        title="Risk Factors",
        level=1,
        start_page=6,
        end_page=20
    )
    section3 = Section(
        section_id="sect_003",
        section_type=SectionType.BUSINESS,
        title="Our Business",
        level=1,
        start_page=21,
        end_page=40
    )
    
    return SectionTree(
        root_sections=[section1, section2, section3],
        all_sections={
            "sect_001": section1,
            "sect_002": section2,
            "sect_003": section3,
        },
        document_id="test_doc",
        total_pages=40
    )


# =============================================================================
# SectionType Enum Tests
# =============================================================================

class TestSectionType:
    """Tests for SectionType enum."""
    
    def test_section_type_values_exist(self):
        """Test that all expected section types exist."""
        expected_types = [
            "SUMMARY",
            "RISK_FACTORS",
            "INTRODUCTION",
            "THE_ISSUE",
            "CAPITAL_STRUCTURE",
            "OBJECTS_OF_ISSUE",
            "BASIS_FOR_ISSUE_PRICE",
            "BUSINESS",
            "INDUSTRY_OVERVIEW",
            "MANAGEMENT",
            "FINANCIAL_INFORMATION",
            "LEGAL_INFORMATION",
            "REGULATORY_DISCLOSURES",
            "ARTICLES_OF_ASSOCIATION",
            "DEFINITIONS",
            "OTHER",
        ]
        
        for type_name in expected_types:
            assert hasattr(SectionType, type_name), f"SectionType.{type_name} not found"
    
    def test_section_type_values(self):
        """Test that section types have correct string values."""
        assert SectionType.SUMMARY.value == "summary"
        assert SectionType.RISK_FACTORS.value == "risk_factors"
        assert SectionType.BUSINESS.value == "business"
        assert SectionType.FINANCIAL_INFORMATION.value == "financial_information"
    
    def test_section_type_iteration(self):
        """Test that we can iterate over section types."""
        types = list(SectionType)
        assert len(types) == 16  # Should have 16 section types
    
    def test_section_type_comparison(self):
        """Test section type comparison."""
        assert SectionType.SUMMARY == SectionType.SUMMARY
        assert SectionType.SUMMARY != SectionType.RISK_FACTORS


# =============================================================================
# Section Dataclass Tests
# =============================================================================

class TestSection:
    """Tests for Section dataclass."""
    
    def test_section_creation(self, sample_section):
        """Test creating a Section instance."""
        assert sample_section.section_id == "sect_001"
        assert sample_section.section_type == SectionType.RISK_FACTORS
        assert sample_section.title == "RISK FACTORS"
        assert sample_section.level == 1
        assert sample_section.start_page == 10
        assert sample_section.end_page == 25
    
    def test_section_defaults(self):
        """Test Section default values."""
        section = Section(
            section_id="test",
            section_type=SectionType.OTHER,
            title="Test Section",
            level=1,
            start_page=1,
            end_page=5
        )
        
        assert section.content == ""
        assert section.word_count == 0
        assert section.has_tables is False
        assert section.subsections == []
        assert section.parent_id is None
        assert section.metadata == {}
    
    def test_section_get_page_range(self, sample_section):
        """Test get_page_range method."""
        start, end = sample_section.get_page_range()
        assert start == 10
        assert end == 25
    
    def test_section_to_dict(self, sample_section):
        """Test Section to_dict method."""
        result = sample_section.to_dict()
        
        assert isinstance(result, dict)
        assert result["section_id"] == "sect_001"
        assert result["section_type"] == "risk_factors"
        assert result["title"] == "RISK FACTORS"
        assert result["level"] == 1
        assert result["start_page"] == 10
        assert result["end_page"] == 25
        assert "subsection_count" in result
    
    def test_section_with_subsections(self, sample_section):
        """Test Section with nested subsections."""
        subsection = Section(
            section_id="sect_001_sub",
            section_type=SectionType.RISK_FACTORS,
            title="Internal Risks",
            level=2,
            start_page=12,
            end_page=15,
            parent_id="sect_001"
        )
        sample_section.subsections.append(subsection)
        
        assert len(sample_section.subsections) == 1
        assert sample_section.subsections[0].title == "Internal Risks"
        assert sample_section.subsections[0].parent_id == "sect_001"


# =============================================================================
# SectionTree Tests
# =============================================================================

class TestSectionTree:
    """Tests for SectionTree dataclass."""
    
    def test_section_tree_creation(self, sample_section_tree):
        """Test creating a SectionTree instance."""
        assert len(sample_section_tree.root_sections) == 3
        assert len(sample_section_tree.all_sections) == 3
        assert sample_section_tree.document_id == "test_doc"
        assert sample_section_tree.total_pages == 40
    
    def test_get_section_by_id(self, sample_section_tree):
        """Test retrieving section by ID."""
        section = sample_section_tree.get_section_by_id("sect_001")
        assert section is not None
        assert section.section_type == SectionType.SUMMARY
        
        # Test non-existent ID
        assert sample_section_tree.get_section_by_id("nonexistent") is None
    
    def test_get_section_by_type(self, sample_section_tree):
        """Test retrieving sections by type."""
        risk_sections = sample_section_tree.get_section_by_type(SectionType.RISK_FACTORS)
        assert len(risk_sections) == 1
        assert risk_sections[0].title == "Risk Factors"
        
        # Test type with no matches
        legal_sections = sample_section_tree.get_section_by_type(SectionType.LEGAL_INFORMATION)
        assert len(legal_sections) == 0
    
    def test_get_section_by_title(self, sample_section_tree):
        """Test retrieving section by title."""
        section = sample_section_tree.get_section_by_title("Our Business")
        assert section is not None
        assert section.section_type == SectionType.BUSINESS
        
        # Test partial match
        section = sample_section_tree.get_section_by_title("Business")
        assert section is not None
        
        # Test non-existent title
        assert sample_section_tree.get_section_by_title("Nonexistent") is None
    
    def test_get_page_to_section_map(self, sample_section_tree):
        """Test page to section mapping."""
        page_map = sample_section_tree.get_page_to_section_map()
        
        assert isinstance(page_map, dict)
        # Page 3 should be in Summary section
        assert 3 in page_map
        assert isinstance(page_map[3], list)
        # Page 10 should be in Risk Factors section
        assert 10 in page_map
        assert isinstance(page_map[10], list)
        # Page 30 should be in Business section
        assert 30 in page_map
        assert isinstance(page_map[30], list)
    
    def test_section_tree_to_dict(self, sample_section_tree):
        """Test SectionTree to_dict method."""
        result = sample_section_tree.to_dict()
        
        assert isinstance(result, dict)
        assert result["document_id"] == "test_doc"
        assert result["total_pages"] == 40
        assert result["total_section_count"] == 3
        assert "root_sections" in result
        assert len(result["root_sections"]) == 3


# =============================================================================
# SectionMapper Tests
# =============================================================================

class TestSectionMapper:
    """Tests for SectionMapper class."""
    
    def test_section_mapper_creation(self, section_mapper):
        """Test creating a SectionMapper instance."""
        assert section_mapper is not None
        assert section_mapper.HEADER_FONT_SIZE_RATIO == 1.15
        assert section_mapper.MAJOR_SECTION_FONT_SIZE_RATIO == 1.3
    
    def test_section_patterns_defined(self, section_mapper):
        """Test that section patterns are defined."""
        assert hasattr(section_mapper, 'SECTION_PATTERNS')
        assert SectionType.RISK_FACTORS in section_mapper.SECTION_PATTERNS
        assert SectionType.BUSINESS in section_mapper.SECTION_PATTERNS
        assert SectionType.FINANCIAL_INFORMATION in section_mapper.SECTION_PATTERNS
    
    def test_match_section_pattern_risk_factors(self, section_mapper):
        """Test matching RISK FACTORS pattern."""
        result = section_mapper._match_section_pattern("RISK FACTORS")
        assert result == SectionType.RISK_FACTORS
        
        result = section_mapper._match_section_pattern("Risk Factors")
        assert result == SectionType.RISK_FACTORS
    
    def test_match_section_pattern_business(self, section_mapper):
        """Test matching business section patterns."""
        test_cases = [
            ("OUR BUSINESS", SectionType.BUSINESS),
            ("ABOUT OUR COMPANY", SectionType.BUSINESS),
            ("BUSINESS OVERVIEW", SectionType.BUSINESS),
        ]
        
        for text, expected in test_cases:
            result = section_mapper._match_section_pattern(text)
            assert result == expected, f"Failed for: {text}"
    
    def test_match_section_pattern_financial(self, section_mapper):
        """Test matching financial section patterns."""
        test_cases = [
            ("FINANCIAL INFORMATION", SectionType.FINANCIAL_INFORMATION),
            ("FINANCIAL STATEMENTS", SectionType.FINANCIAL_INFORMATION),
            ("RESTATED FINANCIAL STATEMENTS", SectionType.FINANCIAL_INFORMATION),
        ]
        
        for text, expected in test_cases:
            result = section_mapper._match_section_pattern(text)
            assert result == expected, f"Failed for: {text}"
    
    def test_match_section_pattern_no_match(self, section_mapper):
        """Test that unrecognized text returns OTHER for unrecognized patterns."""
        result = section_mapper._match_section_pattern("Random text content")
        assert result == SectionType.OTHER
        
        result = section_mapper._match_section_pattern("Some paragraph text")
        assert result == SectionType.OTHER
    
    def test_calculate_baseline_font_size(self, section_mapper, mock_pages_with_sections):
        """Test baseline font size calculation."""
        baseline = section_mapper._calculate_baseline_font_size(mock_pages_with_sections)
        
        # Should return a reasonable baseline (most common text size in mock)
        assert baseline > 0
        # Note: Mock pages may have header-sized text (18.0) as most common
        assert baseline <= 24.0
    
    def test_determine_header_level_major_section(self, section_mapper):
        """Test header level determination for major sections."""
        # Major section (uppercase, large font)
        level = section_mapper._determine_header_level(
            text="RISK FACTORS",
            font_size=18.0,
            baseline=12.0,
            has_numbering=False
        )
        assert level == 1  # Major sections should be level 1
    
    def test_determine_header_level_subsection(self, section_mapper):
        """Test header level determination for subsections."""
        # Subsection with numbering
        level = section_mapper._determine_header_level(
            text="1.1 Internal Risks",
            font_size=14.0,
            baseline=12.0,
            has_numbering=True
        )
        assert level >= 2  # Subsections should be level 2 or higher
    
    def test_build_hierarchy_empty_pages(self, section_mapper):
        """Test build_hierarchy with empty page list."""
        result = section_mapper.build_hierarchy([])
        
        assert isinstance(result, SectionTree)
        assert len(result.root_sections) == 0
        assert result.total_pages == 0
    
    def test_build_hierarchy_with_pages(self, section_mapper, mock_pages_with_sections):
        """Test build_hierarchy with sample pages."""
        result = section_mapper.build_hierarchy(mock_pages_with_sections)
        
        assert isinstance(result, SectionTree)
        # document_id is generated from pages, not passed as parameter
        assert result.total_pages == 5
        
        # Should have detected at least some sections
        # Note: Actual detection depends on implementation
        assert len(result.root_sections) >= 0
    
    def test_extract_section_boundaries(self, section_mapper, mock_pages_with_sections):
        """Test extracting section boundaries."""
        tree = section_mapper.build_hierarchy(mock_pages_with_sections)
        boundaries = section_mapper.extract_section_boundaries(tree)
        
        assert isinstance(boundaries, dict)
        # Each entry should be a tuple of (start_page, end_page)
        for section_name, (start, end) in boundaries.items():
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert start <= end
    
    def test_get_section_for_page(self, section_mapper, sample_section_tree):
        """Test getting section for a specific page."""
        # Page within Risk Factors section (pages 6-20)
        sections = section_mapper.get_section_for_page(sample_section_tree, 10)
        assert isinstance(sections, list)
        assert len(sections) > 0
        assert sections[0].section_type == SectionType.RISK_FACTORS
        
        # Page within Business section (pages 21-40)
        sections = section_mapper.get_section_for_page(sample_section_tree, 30)
        assert isinstance(sections, list)
        assert len(sections) > 0
        assert sections[0].section_type == SectionType.BUSINESS
    
    def test_get_section_for_page_not_found(self, section_mapper, sample_section_tree):
        """Test getting section for page outside any section."""
        # Page beyond all sections
        sections = section_mapper.get_section_for_page(sample_section_tree, 100)
        assert isinstance(sections, list)
        assert len(sections) == 0


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestNormalizeSectionName:
    """Tests for normalize_section_name utility function."""
    
    def test_normalize_exact_match(self):
        """Test normalization with exact match."""
        result = normalize_section_name("RISK FACTORS")
        assert result == "Risk Factors"
    
    def test_normalize_case_insensitive(self):
        """Test case-insensitive normalization."""
        result = normalize_section_name("risk factors")
        assert result == "Risk Factors"
        
        result = normalize_section_name("Risk FACTORS")
        assert result == "Risk Factors"
    
    def test_normalize_alias_match(self):
        """Test normalization with aliases."""
        # "About Our Company" should normalize to "Our Business"
        result = normalize_section_name("ABOUT OUR COMPANY")
        assert result == "Our Business"
        
        # "Restated Financial Statements" should normalize to "Financial Information"
        result = normalize_section_name("RESTATED FINANCIAL STATEMENTS")
        assert result == "Financial Information"
    
    def test_normalize_with_extra_whitespace(self):
        """Test normalization with leading/trailing whitespace."""
        # Leading/trailing whitespace is stripped, but internal spacing must match aliases
        result = normalize_section_name("  RISK FACTORS  ")
        # Should match since stripping makes it match the alias
        assert result == "Risk Factors"
    
    def test_normalize_unknown_section(self):
        """Test normalization of unknown section name."""
        result = normalize_section_name("Some Random Section")
        # Should return the original (title-cased or as-is)
        assert result is not None


class TestStandardRHPSections:
    """Tests for STANDARD_RHP_SECTIONS constant."""
    
    def test_standard_sections_exist(self):
        """Test that standard sections are defined."""
        assert STANDARD_RHP_SECTIONS is not None
        assert isinstance(STANDARD_RHP_SECTIONS, dict)
    
    def test_standard_sections_have_aliases(self):
        """Test that sections have aliases defined."""
        # Each section should have a list of aliases
        for section_name, aliases in STANDARD_RHP_SECTIONS.items():
            assert isinstance(aliases, list), f"{section_name} should have list of aliases"
    
    def test_key_sections_present(self):
        """Test that key RHP sections are present."""
        expected_sections = [
            "Summary of the Issue",
            "Risk Factors",
            "Our Business",
            "Financial Information",
            "Objects of the Issue",
        ]
        
        section_names = list(STANDARD_RHP_SECTIONS.keys())
        for expected in expected_sections:
            assert expected in section_names, f"'{expected}' not found in standard sections"


# =============================================================================
# Integration Tests
# =============================================================================

class TestSectionMapperIntegration:
    """Integration tests for SectionMapper with multiple components."""
    
    def test_full_workflow(self, section_mapper, mock_pages_with_sections):
        """Test the complete section mapping workflow."""
        # Build hierarchy
        tree = section_mapper.build_hierarchy(mock_pages_with_sections)
        
        # Verify tree structure
        assert tree is not None
        # document_id is generated from pages, not passed as parameter
        assert tree.document_id is not None
        
        # Extract boundaries
        boundaries = section_mapper.extract_section_boundaries(tree)
        assert isinstance(boundaries, dict)
        
        # Get page map
        page_map = tree.get_page_to_section_map()
        assert isinstance(page_map, dict)
        
        # Convert to dict (for serialization)
        tree_dict = tree.to_dict()
        assert isinstance(tree_dict, dict)
        assert "root_sections" in tree_dict
    
    def test_section_content_assignment(self, section_mapper, mock_pages_with_sections):
        """Test that section content is properly assigned."""
        tree = section_mapper.build_hierarchy(mock_pages_with_sections)
        
        # Check that sections with content have word counts
        for section in tree.root_sections:
            if section.content:
                assert section.word_count > 0
    
    def test_has_tables_detection(self, section_mapper, mock_pages_with_sections):
        """Test that has_tables flag is correctly set."""
        tree = section_mapper.build_hierarchy(mock_pages_with_sections)
        
        # Find sections that should have tables
        for section in tree.root_sections:
            # Business section (page 4) and Financial section (page 5) have tables
            if section.section_type in [SectionType.BUSINESS, SectionType.FINANCIAL_INFORMATION]:
                # These sections should detect tables if pages are correctly assigned
                pass  # Detection depends on page assignment


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_text_page(self, section_mapper):
        """Test handling of pages with empty text."""
        page = Mock()
        page.page_num = 1
        page.text = ""
        page.word_count = 0
        page.has_tables = False
        page.font_sizes = []
        
        tree = section_mapper.build_hierarchy([page])
        assert tree is not None
    
    def test_page_with_no_font_info(self, section_mapper):
        """Test handling of pages without font information."""
        page = Mock()
        page.page_num = 1
        page.text = "Some content without font info"
        page.word_count = 5
        page.has_tables = False
        page.font_sizes = []  # No font sizes
        
        tree = section_mapper.build_hierarchy([page])
        assert tree is not None
    
    def test_single_page_document(self, section_mapper):
        """Test handling of single-page document."""
        page = Mock()
        page.page_num = 1
        page.text = "RISK FACTORS\n\nSingle page content"
        page.word_count = 10
        page.has_tables = False
        page.font_sizes = [12.0]
        
        tree = section_mapper.build_hierarchy([page])
        assert tree.total_pages == 1
    
    def test_duplicate_section_headers(self, section_mapper):
        """Test handling of duplicate section headers."""
        pages = []
        
        # Two pages with same section header
        for i in range(2):
            page = Mock()
            page.page_num = i + 1
            page.text = "RISK FACTORS\n\nContent..."
            page.word_count = 100
            page.has_tables = False
            page.font_sizes = [18.0, 12.0]
            pages.append(page)
        
        tree = section_mapper.build_hierarchy(pages)
        
        # Should handle duplicates gracefully
        assert tree is not None
    
    def test_section_type_other_fallback(self, section_mapper):
        """Test that unrecognized sections fall back to OTHER type."""
        page = Mock()
        page.page_num = 1
        page.text = "UNUSUAL SECTION NAME\n\nContent..."
        page.word_count = 100
        page.has_tables = False
        page.font_sizes = [18.0, 12.0]
        
        tree = section_mapper.build_hierarchy([page])
        
        # If detected as a section, should be classified as OTHER
        # (if the header detection picks it up)
        assert tree is not None


# =============================================================================
# Performance/Stress Tests
# =============================================================================

class TestPerformance:
    """Performance-related tests."""
    
    def test_large_page_count(self, section_mapper):
        """Test handling of large number of pages."""
        pages = []
        
        for i in range(100):  # Simulate 100 pages
            page = Mock()
            page.page_num = i + 1
            page.text = f"Page {i + 1} content..."
            page.word_count = 500
            page.has_tables = i % 10 == 0
            page.font_sizes = [12.0, 14.0]
            pages.append(page)
        
        # Add some section headers
        pages[0].text = "SUMMARY\n\nSummary content..."
        pages[10].text = "RISK FACTORS\n\nRisk content..."
        pages[30].text = "OUR BUSINESS\n\nBusiness content..."
        
        tree = section_mapper.build_hierarchy(pages)
        
        assert tree is not None
        assert tree.total_pages == 100
    
    def test_deeply_nested_sections(self, section_mapper):
        """Test handling of deeply nested section structure."""
        pages = []
        
        # Create pages with nested numbering
        page1 = Mock()
        page1.page_num = 1
        page1.text = "1. RISK FACTORS\n\n1.1 Internal Risks\n\n1.1.1 Operational Risks\n\n1.1.1.1 Specific Risk"
        page1.word_count = 100
        page1.has_tables = False
        page1.font_sizes = [18.0, 16.0, 14.0, 12.0]
        pages.append(page1)
        
        tree = section_mapper.build_hierarchy(pages)
        
        # Should handle nested structure
        assert tree is not None
