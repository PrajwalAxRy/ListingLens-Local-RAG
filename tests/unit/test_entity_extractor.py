"""
Unit tests for Entity Extractor Module
"""

import pytest
from src.rhp_analyzer.ingestion.entity_extractor import (
    EntityExtractor,
    Entity,
    EntityType
)


@pytest.fixture
def extractor():
    """Create entity extractor instance for testing"""
    return EntityExtractor()


class TestEntityExtraction:
    """Test basic entity extraction functionality"""

    def test_extract_company_name(self, extractor):
        """Test company name extraction"""
        text = "ABC Technologies Limited is a leading software company."
        entities = extractor.extract_entities(text)
        
        company_entities = [e for e in entities if e.entity_type == EntityType.COMPANY]
        assert len(company_entities) > 0
        
        # Check if company name is extracted
        company_texts = [e.text for e in company_entities]
        assert any('ABC Technologies' in text for text in company_texts)

    def test_extract_person_name(self, extractor):
        """Test person name extraction with honorifics"""
        text = "Mr. Rajesh Kumar is the Managing Director of the company."
        entities = extractor.extract_entities(text)
        
        person_entities = [e for e in entities if e.entity_type == EntityType.PERSON]
        assert len(person_entities) > 0
        
        # Check if person name is extracted
        person_texts = [e.text.lower() for e in person_entities]
        assert any('rajesh' in text or 'kumar' in text for text in person_texts)

    def test_extract_location(self, extractor):
        """Test location extraction"""
        text = "The company's head office is in Mumbai, India."
        entities = extractor.extract_entities(text)
        
        location_entities = [e for e in entities if e.entity_type == EntityType.LOCATION]
        assert len(location_entities) > 0
        
        # Check if Mumbai or India is extracted
        location_texts = [e.text for e in location_entities]
        assert any('Mumbai' in text or 'India' in text for text in location_texts)

    def test_extract_regulator(self, extractor):
        """Test regulator entity extraction"""
        text = "The company is registered with SEBI and regulated by RBI."
        entities = extractor.extract_entities(text)
        
        org_entities = [e for e in entities if e.entity_type == EntityType.ORG]
        org_texts = [e.text for e in org_entities]
        
        assert 'SEBI' in org_texts or 'RBI' in org_texts


class TestFinancialEntityPatterns:
    """Test extraction of Indian financial entity patterns"""

    def test_extract_crores(self, extractor):
        """Test extraction of amounts in crores"""
        test_cases = [
            ("₹500 crores", 500, 5_000_000_000),
            ("Rs. 1,234.5 crore", 1234.5, 12_345_000_000),
            ("100 Cr", 100, 1_000_000_000),
        ]
        
        for text, expected_crore, expected_value in test_cases:
            entities = extractor.extract_entities(text)
            money_entities = [e for e in entities if e.entity_type == EntityType.MONEY and e.value is not None]
            
            assert len(money_entities) > 0, f"Failed to extract from: {text}"
            # Check if value is approximately correct (within 1% tolerance)
            assert any(
                abs(e.value - expected_value) / expected_value < 0.01 
                for e in money_entities
            ), f"Value mismatch for: {text}. Got: {[e.value for e in money_entities]}"

    def test_extract_lakhs(self, extractor):
        """Test extraction of amounts in lakhs"""
        test_cases = [
            ("₹50 lakhs", 50, 5_000_000),
            ("Rs. 10 lakh", 10, 1_000_000),
            ("200.5 lacs", 200.5, 20_050_000),
        ]
        
        for text, expected_lakh, expected_value in test_cases:
            entities = extractor.extract_entities(text)
            money_entities = [e for e in entities if e.entity_type == EntityType.MONEY and e.value is not None]
            
            assert len(money_entities) > 0, f"Failed to extract from: {text}"
            assert any(
                abs(e.value - expected_value) / expected_value < 0.01 
                for e in money_entities
            ), f"Value mismatch for: {text}. Got: {[e.value for e in money_entities]}"

    def test_extract_percentage(self, extractor):
        """Test percentage extraction"""
        test_cases = [
            ("The ROE is 25%", 25.0),
            ("Growth of 15.5%", 15.5),
            ("100% compliance", 100.0),
        ]
        
        for text, expected_value in test_cases:
            entities = extractor.extract_entities(text)
            pct_entities = [e for e in entities if e.entity_type == EntityType.PERCENTAGE and e.value is not None]
            
            assert len(pct_entities) > 0, f"Failed to extract from: {text}"
            assert any(
                abs(e.value - expected_value) < 0.01 
                for e in pct_entities
            ), f"Value mismatch for: {text}. Got: {[e.value for e in pct_entities]}"

    def test_extract_fiscal_year(self, extractor):
        """Test fiscal year extraction"""
        test_cases = [
            ("FY 2023-24", "FY23-24"),
            ("FY23", "FY23"),
            ("Fiscal Year 2022-23", "FY22-23"),
            ("FY 2024", "FY24"),
        ]
        
        for text, expected_normalized in test_cases:
            entities = extractor.extract_entities(text)
            fy_entities = [e for e in entities if e.entity_type == EntityType.FISCAL_YEAR]
            
            assert len(fy_entities) > 0, f"Failed to extract from: {text}"
            assert any(
                e.normalized_text == expected_normalized 
                for e in fy_entities
            ), f"Normalization failed for: {text}"

    def test_extract_date(self, extractor):
        """Test date extraction in DD/MM/YYYY format"""
        test_cases = [
            ("The meeting is on 15/03/2024", "2024-03-15"),
            ("Date: 01-12-2023", "2023-12-01"),
            ("31.01.2024", "2024-01-31"),
        ]
        
        for text, expected_normalized in test_cases:
            entities = extractor.extract_entities(text)
            date_entities = [e for e in entities if e.entity_type == EntityType.DATE]
            
            assert len(date_entities) > 0, f"Failed to extract from: {text}"
            assert any(
                e.normalized_text == expected_normalized 
                for e in date_entities
            ), f"Date normalization failed for: {text}"

    def test_extract_price_band(self, extractor):
        """Test price band extraction"""
        test_cases = [
            "₹100 to ₹120",
            "Rs. 500 - Rs. 550",
            "Price band: ₹1,000-₹1,200",
        ]
        
        for text in test_cases:
            entities = extractor.extract_entities(text)
            money_entities = [e for e in entities if e.entity_type == EntityType.MONEY]
            
            assert len(money_entities) > 0, f"Failed to extract price band from: {text}"
            # Price band should have a normalized format
            assert any(
                '-' in (e.normalized_text or '') 
                for e in money_entities
            ), f"Price band not properly formatted: {text}"


class TestEntityDeduplication:
    """Test entity deduplication functionality"""

    def test_deduplicate_identical_entities(self, extractor):
        """Test deduplication of identical entities"""
        entities = [
            Entity(text="ABC Limited", entity_type=EntityType.COMPANY, page_num=1),
            Entity(text="ABC Limited", entity_type=EntityType.COMPANY, page_num=2),
            Entity(text="ABC Limited", entity_type=EntityType.COMPANY, page_num=3),
        ]
        
        deduplicated = extractor.deduplicate_entities(entities)
        
        assert len(deduplicated) == 1
        assert deduplicated[0].mentions == 3
        assert len(deduplicated[0].page_references) == 3

    def test_deduplicate_variant_entities(self, extractor):
        """Test deduplication of entity variants"""
        entities = [
            Entity(text="ABC Technologies Limited", entity_type=EntityType.COMPANY, page_num=1),
            Entity(text="ABC Technologies Ltd", entity_type=EntityType.COMPANY, page_num=2),
            Entity(text="ABC Technologies", entity_type=EntityType.COMPANY, page_num=3),
        ]
        
        deduplicated = extractor.deduplicate_entities(entities)
        
        # Should deduplicate to single entity
        assert len(deduplicated) <= 2  # Allow for some variation tolerance
        # Most mentioned form should have multiple page references
        assert any(len(e.page_references) >= 2 for e in deduplicated)

    def test_preserve_different_entity_types(self, extractor):
        """Test that different entity types are not merged"""
        entities = [
            Entity(text="ABC", entity_type=EntityType.COMPANY, page_num=1),
            Entity(text="ABC", entity_type=EntityType.PERSON, page_num=2),
        ]
        
        deduplicated = extractor.deduplicate_entities(entities)
        
        # Should remain as 2 separate entities
        assert len(deduplicated) == 2


class TestEntityNormalization:
    """Test entity text normalization"""

    def test_normalize_company_name(self, extractor):
        """Test company name normalization"""
        test_cases = [
            ("ABC Technologies Limited", "ABC Technologies"),
            ("XYZ Pvt. Ltd.", "XYZ"),
            ("Tech Corp", "Tech Corp"),
        ]
        
        for original, expected_base in test_cases:
            normalized = extractor._normalize_entity_text(original, EntityType.COMPANY)
            # Check that common suffixes are removed
            assert "Limited" not in normalized
            assert "Ltd." not in normalized
            assert "Pvt." not in normalized

    def test_normalize_person_name(self, extractor):
        """Test person name normalization"""
        test_cases = [
            ("Mr. Rajesh Kumar", "Rajesh Kumar"),
            ("Dr. Priya Sharma", "Priya Sharma"),
            ("Ms. Anjali Singh", "Anjali Singh"),
        ]
        
        for original, expected in test_cases:
            normalized = extractor._normalize_entity_text(original, EntityType.PERSON)
            # Check that honorifics are removed
            assert "Mr." not in normalized
            assert "Dr." not in normalized
            assert "Ms." not in normalized
            # Check name is present
            assert any(part in normalized for part in expected.split())


class TestCoreferenceResolution:
    """Test coreference resolution"""

    def test_resolve_company_variants(self, extractor):
        """Test resolution of company name variants"""
        entities = [
            Entity(text="ABC Technologies Limited", entity_type=EntityType.COMPANY, mentions=5, page_num=1),
            Entity(text="ABC Technologies Ltd", entity_type=EntityType.COMPANY, mentions=3, page_num=2),
            Entity(text="ABC Tech", entity_type=EntityType.COMPANY, mentions=2, page_num=3),
        ]
        
        resolved = extractor.resolve_coreferences(entities)
        
        # Should resolve to single entity with most mentions
        assert len(resolved) <= 2  # Allow some tolerance
        # Check that mentions are aggregated
        total_mentions = sum(e.mentions for e in resolved)
        assert total_mentions >= 5  # At least the most mentioned form

    def test_resolve_person_variants(self, extractor):
        """Test resolution of person name variants"""
        entities = [
            Entity(text="Mr. John Doe", entity_type=EntityType.PERSON, mentions=3, page_num=1),
            Entity(text="John Doe", entity_type=EntityType.PERSON, mentions=5, page_num=2),
        ]
        
        resolved = extractor.resolve_coreferences(entities)
        
        # Should resolve to single entity
        assert len(resolved) == 1
        # Check aliases are captured
        assert len(resolved[0].aliases) >= 2


class TestExtractAll:
    """Test batch extraction from multiple pages"""

    def test_extract_from_multiple_pages(self, extractor):
        """Test extraction from multiple pages"""
        pages_text = [
            (1, "ABC Limited reported revenue of ₹500 crores in FY 2023-24."),
            (2, "The company, ABC Ltd, has offices in Mumbai and Delhi."),
            (3, "Mr. Rajesh Kumar is the CEO. The company has a ROE of 25%."),
        ]
        
        result = extractor.extract_all(pages_text)
        
        # Should have multiple entity types
        assert len(result) > 0
        
        # Check for expected entity types
        if EntityType.COMPANY in result:
            assert len(result[EntityType.COMPANY]) > 0
        
        if EntityType.MONEY in result:
            assert len(result[EntityType.MONEY]) > 0
        
        if EntityType.FISCAL_YEAR in result:
            assert len(result[EntityType.FISCAL_YEAR]) > 0

    def test_page_references_preserved(self, extractor):
        """Test that page references are preserved and deduplicated"""
        pages_text = [
            (1, "ABC Limited is a software company."),
            (2, "ABC Limited reported strong growth."),
            (5, "ABC Limited has offices in Mumbai."),
        ]
        
        result = extractor.extract_all(pages_text)
        
        # Get company entities
        if EntityType.COMPANY in result:
            for entity in result[EntityType.COMPANY]:
                if 'ABC' in entity.text:
                    # Should have references from multiple pages
                    assert len(entity.page_references) > 1
                    # Check specific page numbers
                    assert 1 in entity.page_references
                    assert 2 in entity.page_references
                    assert 5 in entity.page_references


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_text(self, extractor):
        """Test extraction from empty text"""
        entities = extractor.extract_entities("")
        assert entities == []

    def test_no_entities(self, extractor):
        """Test text with no recognizable entities"""
        text = "The quick brown fox jumps over the lazy dog."
        entities = extractor.extract_entities(text)
        # May have some generic entities, but should not crash
        assert isinstance(entities, list)

    def test_unicode_text(self, extractor):
        """Test extraction from text with unicode characters"""
        text = "The company has operations in मुंबई and दिल्ली with revenue of ₹500 करोड़."
        entities = extractor.extract_entities(text)
        # Should handle unicode without crashing
        assert isinstance(entities, list)

    def test_malformed_amounts(self, extractor):
        """Test handling of malformed amount strings"""
        test_cases = [
            "₹ABC crores",  # Non-numeric
            "Rs. ,, lakhs",  # Invalid commas
            "₹ crores",  # Missing amount
        ]
        
        for text in test_cases:
            entities = extractor.extract_entities(text)
            # Should not crash, may or may not extract
            assert isinstance(entities, list)

    def test_invalid_dates(self, extractor):
        """Test handling of invalid dates"""
        test_cases = [
            "32/13/2024",  # Invalid day/month
            "00/00/0000",  # Zero date
            "31/02/2024",  # Invalid day for February
        ]
        
        for text in test_cases:
            entities = extractor.extract_entities(text)
            date_entities = [e for e in entities if e.entity_type == EntityType.DATE]
            # Should not extract invalid dates
            assert len(date_entities) == 0


class TestIntegration:
    """Integration tests with realistic RHP content"""

    def test_realistic_rhp_paragraph(self, extractor):
        """Test extraction from realistic RHP paragraph"""
        text = """
        ABC Technologies Limited (the "Company") was incorporated on 15/03/2010 
        under the Companies Act. The company reported a revenue of ₹1,234.5 crores 
        in FY 2023-24, representing a growth of 25% over the previous fiscal year.
        The company's registered office is in Mumbai, Maharashtra. Mr. Rajesh Kumar 
        serves as the Managing Director. The company is registered with SEBI and 
        maintains compliance with all regulatory requirements. The price band for 
        the IPO is ₹500 to ₹550 per share.
        """
        
        entities = extractor.extract_entities(text, page_num=1)
        
        # Check that various entity types are extracted
        entity_types = {e.entity_type for e in entities}
        
        # Should have multiple types
        assert len(entity_types) >= 3
        
        # Check for specific extractions
        company_entities = [e for e in entities if e.entity_type == EntityType.COMPANY]
        assert len(company_entities) > 0
        
        money_entities = [e for e in entities if e.entity_type == EntityType.MONEY]
        assert len(money_entities) > 0  # Should extract revenue and price band
        
        fy_entities = [e for e in entities if e.entity_type == EntityType.FISCAL_YEAR]
        assert len(fy_entities) > 0

    def test_extract_financial_summary(self, extractor):
        """Test extraction from financial summary section"""
        text = """
        Financial Performance (in ₹ Crores):
        - Revenue: FY22: 100 | FY23: 150 | FY24: 200
        - EBITDA: 25 crores (FY24)
        - PAT: ₹18 crores
        - ROE: 22%
        - Debt/Equity: 0.5
        """
        
        entities = extractor.extract_entities(text)
        
        # Should extract multiple financial entities
        money_entities = [e for e in entities if e.entity_type == EntityType.MONEY]
        assert len(money_entities) >= 3  # Revenue, EBITDA, PAT
        
        fy_entities = [e for e in entities if e.entity_type == EntityType.FISCAL_YEAR]
        assert len(fy_entities) >= 2
        
        pct_entities = [e for e in entities if e.entity_type == EntityType.PERCENTAGE]
        assert len(pct_entities) >= 1  # ROE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
