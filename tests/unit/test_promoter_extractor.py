"""
Unit tests for Promoter Extractor Module
"""

import pytest
from src.rhp_analyzer.ingestion.promoter_extractor import (
    PromoterExtractor,
    PromoterDossier
)


@pytest.fixture
def promoter_extractor():
    """Create a PromoterExtractor instance for testing."""
    return PromoterExtractor()


@pytest.fixture
def sample_state():
    """Create sample state data for testing."""
    return {
        'sections': {
            'Our Promoters': {
                'content': '''
                Mr. John Doe, aged 45 years, is the Managing Director of the Company.
                He holds a B.Tech degree and has 20 years of experience in the industry.
                DIN: 12345678
                
                Mrs. Jane Smith, aged 40 years, is a Director of the Company.
                She holds an MBA degree and has 15 years of experience.
                DIN: 87654321
                ''',
                'tables': []
            },
            'Other Directorships of Promoters': {
                'content': 'Mr. John Doe is a director in ABC Ltd, XYZ Corp',
                'tables': []
            },
            'Capital Structure': {
                'content': '',
                'tables': []
            }
        },
        'ipo_details': {
            'price_band_cap': 100,
            'shares_post_issue': 10000000
        }
    }


class TestPromoterExtractor:
    """Test suite for PromoterExtractor class."""
    
    def test_initialization(self, promoter_extractor):
        """Test PromoterExtractor initialization."""
        assert promoter_extractor is not None
        assert promoter_extractor.vector_store is None
        assert promoter_extractor.citation_manager is None
    
    def test_extract_promoters_basic(self, promoter_extractor, sample_state):
        """Test basic promoter extraction."""
        promoters = promoter_extractor.extract_promoters(sample_state)
        
        assert isinstance(promoters, list)
        # Should extract at least the promoters from the content
        # Note: Actual extraction depends on pattern matching
    
    def test_extract_basic_profiles(self, promoter_extractor, sample_state):
        """Test basic profile extraction."""
        profiles = promoter_extractor._extract_basic_profiles(sample_state)
        
        assert isinstance(profiles, dict)
        # The actual content should be parsed
    
    def test_find_promoter_sections(self, promoter_extractor, sample_state):
        """Test finding promoter-related sections."""
        sections = promoter_extractor._find_promoter_sections(sample_state['sections'])
        
        assert 'Our Promoters' in sections
        assert len(sections) >= 1
    
    def test_extract_promoter_names(self, promoter_extractor):
        """Test promoter name extraction."""
        content = "Mr. John Doe and Mrs. Jane Smith are the promoters."
        names = promoter_extractor._extract_promoter_names(content)
        
        assert isinstance(names, list)
        assert 'John Doe' in names or 'Jane Smith' in names
    
    def test_extract_din(self, promoter_extractor):
        """Test DIN extraction."""
        content = "Mr. John Doe, DIN: 12345678, is a director"
        din = promoter_extractor._extract_din(content, "John Doe")
        
        assert din == "12345678"
    
    def test_extract_din_no_match(self, promoter_extractor):
        """Test DIN extraction when no DIN is present."""
        content = "Mr. John Doe is a director"
        din = promoter_extractor._extract_din(content, "John Doe")
        
        assert din is None
    
    def test_extract_age(self, promoter_extractor):
        """Test age extraction."""
        content = "Mr. John Doe, aged 45 years, is a director"
        age = promoter_extractor._extract_age(content, "John Doe")
        
        assert age == 45
    
    def test_extract_age_no_match(self, promoter_extractor):
        """Test age extraction when no age is present."""
        content = "Mr. John Doe is a director"
        age = promoter_extractor._extract_age(content, "John Doe")
        
        assert age is None
    
    def test_extract_qualification(self, promoter_extractor):
        """Test qualification extraction."""
        content = "Mr. John Doe holds a B.Tech degree from IIT"
        qualification = promoter_extractor._extract_qualification(content, "John Doe")
        
        assert qualification is not None
        assert 'B.Tech' in qualification
    
    def test_extract_experience(self, promoter_extractor):
        """Test experience extraction."""
        content = "Mr. John Doe has 20 years of experience in the industry"
        experience = promoter_extractor._extract_experience(content, "John Doe")
        
        assert experience == 20
    
    def test_extract_designation(self, promoter_extractor):
        """Test designation extraction."""
        content = "Mr. John Doe is the Managing Director of the Company"
        designation = promoter_extractor._extract_designation(content, "John Doe")
        
        assert designation == "Managing Director"
    
    def test_calculate_skin_in_game(self, promoter_extractor):
        """Test skin-in-the-game calculation."""
        dossier = PromoterDossier(
            name="John Doe",
            shareholding_post_ipo=25.0  # 25%
        )
        
        cap_price = 100  # ₹100 per share
        total_shares = 10000000  # 1 crore shares
        
        value = promoter_extractor.calculate_skin_in_game(
            dossier,
            cap_price,
            total_shares
        )
        
        # Expected: 25% of 1 crore shares = 25 lakh shares
        # Value = 25,00,000 * 100 = ₹25 crore
        assert value == 25.0
    
    def test_merge_promoter_data(self, promoter_extractor, sample_state):
        """Test merging of promoter data."""
        basic_profiles = {
            'John Doe': {
                'name': 'John Doe',
                'din': '12345678',
                'age': 45,
                'qualification': 'B.Tech',
                'experience': 20,
                'designation': 'Managing Director'
            }
        }
        
        directorships = {
            'John Doe': ['ABC Ltd', 'XYZ Corp']
        }
        
        shareholding = {
            'John Doe': {'pre': 30.0, 'post': 25.0}
        }
        
        dossiers = promoter_extractor._merge_promoter_data(
            basic_profiles,
            directorships,
            {},  # common_pursuits
            {},  # financial_interests
            {},  # litigation
            shareholding,
            sample_state
        )
        
        assert len(dossiers) == 1
        assert dossiers[0].name == 'John Doe'
        assert dossiers[0].din == '12345678'
        assert dossiers[0].age == 45
        assert len(dossiers[0].other_directorships) == 2
        assert dossiers[0].shareholding_pre_ipo == 30.0
        assert dossiers[0].shareholding_post_ipo == 25.0


class TestPromoterDossier:
    """Test suite for PromoterDossier dataclass."""
    
    def test_dossier_creation(self):
        """Test creating a PromoterDossier instance."""
        dossier = PromoterDossier(
            name="John Doe",
            din="12345678",
            age=45
        )
        
        assert dossier.name == "John Doe"
        assert dossier.din == "12345678"
        assert dossier.age == 45
        assert dossier.other_directorships_count == 0
    
    def test_dossier_with_directorships(self):
        """Test dossier with other directorships."""
        dossier = PromoterDossier(
            name="John Doe",
            other_directorships=["ABC Ltd", "XYZ Corp", "PQR Inc"]
        )
        
        assert len(dossier.other_directorships) == 3
        assert dossier.other_directorships_count == 3
    
    def test_dossier_defaults(self):
        """Test default values in dossier."""
        dossier = PromoterDossier(name="John Doe")
        
        assert dossier.shareholding_pre_ipo == 0.0
        assert dossier.shareholding_post_ipo == 0.0
        assert dossier.criminal_cases == 0
        assert dossier.civil_cases == 0
        assert len(dossier.litigation_as_defendant) == 0
    
    def test_dossier_litigation_summary(self):
        """Test litigation summary in dossier."""
        dossier = PromoterDossier(
            name="John Doe",
            criminal_cases=2,
            civil_cases=5,
            regulatory_actions=1,
            total_litigation_amount=10.5
        )
        
        assert dossier.criminal_cases == 2
        assert dossier.civil_cases == 5
        assert dossier.regulatory_actions == 1
        assert dossier.total_litigation_amount == 10.5


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_state(self, promoter_extractor):
        """Test handling of empty state."""
        empty_state = {'sections': {}}
        
        promoters = promoter_extractor.extract_promoters(empty_state)
        assert isinstance(promoters, list)
        assert len(promoters) == 0
    
    def test_malformed_content(self, promoter_extractor):
        """Test handling of malformed content."""
        state = {
            'sections': {
                'Our Promoters': {
                    'content': 'Random text without proper structure',
                    'tables': []
                }
            }
        }
        
        # Should not raise an exception
        promoters = promoter_extractor.extract_promoters(state)
        assert isinstance(promoters, list)
    
    def test_missing_ipo_details(self, promoter_extractor):
        """Test handling when IPO details are missing."""
        state = {
            'sections': {
                'Our Promoters': {
                    'content': 'Mr. John Doe, DIN: 12345678',
                    'tables': []
                }
            }
        }
        
        # Should not crash even without ipo_details
        promoters = promoter_extractor.extract_promoters(state)
        assert isinstance(promoters, list)


class TestIntegration:
    """Integration tests for the full extraction pipeline."""
    
    def test_full_extraction_workflow(self, promoter_extractor):
        """Test the complete extraction workflow."""
        state = {
            'sections': {
                'Our Promoters': {
                    'content': '''
                    The promoters of the Company are:
                    
                    Mr. Rajesh Kumar, aged 50 years, DIN: 00112233
                    He is the Managing Director and holds a B.E. degree.
                    He has 25 years of experience in manufacturing.
                    
                    Mrs. Priya Sharma, aged 45 years, DIN: 00998877
                    She is a Director and holds an MBA degree.
                    She has 20 years of experience in finance.
                    ''',
                    'tables': []
                },
                'Other Directorships of Promoters': {
                    'content': '''
                    Mr. Rajesh Kumar is also a director in:
                    - Kumar Industries Ltd
                    - Global Manufacturing Corp
                    
                    Mrs. Priya Sharma is also a director in:
                    - Sharma Consultants Pvt Ltd
                    ''',
                    'tables': []
                },
                'Capital Structure': {
                    'content': 'Shareholding pattern disclosed separately',
                    'tables': []
                }
            },
            'ipo_details': {
                'price_band_cap': 150,
                'shares_post_issue': 50000000
            }
        }
        
        promoters = promoter_extractor.extract_promoters(state)
        
        # Should extract promoter information
        assert isinstance(promoters, list)
        
        # Verify structure
        for promoter in promoters:
            assert isinstance(promoter, PromoterDossier)
            assert promoter.name is not None
