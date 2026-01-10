"""
Promoter Extractor Module

Extracts comprehensive promoter profiles from RHP documents including:
- Basic information (name, DIN, age, qualification)
- Directorships and conflicts of interest
- Financial interests and shareholding
- Litigation exposure
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from loguru import logger
import re


@dataclass
class PromoterDossier:
    """Comprehensive promoter profile extracted from RHP"""
    name: str
    din: Optional[str] = None
    age: Optional[int] = None
    qualification: Optional[str] = None
    experience_years: Optional[int] = None
    designation: Optional[str] = None  # MD, Chairman, etc.
    
    # Directorships & Conflicts
    other_directorships: List[str] = field(default_factory=list)
    other_directorships_count: int = 0
    group_companies_in_same_line: List[str] = field(default_factory=list)
    common_pursuits: List[str] = field(default_factory=list)
    
    # Financial Interest
    shareholding_pre_ipo: float = 0.0  # Percentage
    shareholding_post_ipo: float = 0.0  # Percentage
    shares_selling_via_ofs: int = 0  # Number of shares
    ofs_amount: float = 0.0  # ₹ Cr
    loans_from_company: float = 0.0  # ₹ Cr
    guarantees_given: float = 0.0  # ₹ Cr
    remuneration_last_3_years: List[float] = field(default_factory=list)  # ₹ Cr per year
    other_benefits: List[str] = field(default_factory=list)
    
    # Litigation specific to promoter
    litigation_as_defendant: List[Dict[str, Any]] = field(default_factory=list)
    criminal_cases: int = 0
    civil_cases: int = 0
    regulatory_actions: int = 0
    total_litigation_amount: float = 0.0  # ₹ Cr
    
    # Track record signals
    past_ventures_mentioned: List[str] = field(default_factory=list)
    disqualifications: bool = False
    
    # Computed metrics
    skin_in_game_post_ipo: float = 0.0  # Post-IPO holding value at cap price (₹ Cr)
    
    def __post_init__(self):
        """Calculate derived metrics after initialization"""
        self.other_directorships_count = len(self.other_directorships)


class PromoterExtractor:
    """
    Extract comprehensive promoter profiles from RHP documents.
    
    This extractor parses multiple sections including:
    - Our Promoters
    - Common Pursuits and Interests
    - Other Directorships
    - Interest of Promoters
    - Payment of Benefits to Promoters
    - Outstanding Litigation
    """
    
    def __init__(self, vector_store=None, citation_manager=None):
        """
        Initialize promoter extractor.
        
        Args:
            vector_store: Optional vector store for RAG retrieval
            citation_manager: Optional citation manager for tracking sources
        """
        self.vector_store = vector_store
        self.citation_manager = citation_manager
        self.logger = logger.bind(module="promoter_extractor")
    
    def extract_promoters(self, state: Dict) -> List[PromoterDossier]:
        """
        Extract all promoters with comprehensive details from RHP.
        
        Args:
            state: Analysis state containing document data
            
        Returns:
            List of PromoterDossier objects with complete profiles
        """
        self.logger.info("Starting promoter extraction")
        promoters = []
        
        try:
            # 1. Get basic promoter profiles from "Our Promoters" section
            self.logger.debug("Extracting basic promoter profiles")
            basic_profiles = self._extract_basic_profiles(state)
            
            # 2. Extract directorships
            self.logger.debug("Extracting other directorships")
            directorships = self._extract_directorships(state)
            
            # 3. Extract common pursuits (conflict of interest check)
            self.logger.debug("Extracting common pursuits")
            common_pursuits = self._extract_common_pursuits(state)
            
            # 4. Extract financial interests
            self.logger.debug("Extracting financial interests")
            financial_interests = self._extract_financial_interests(state)
            
            # 5. Extract promoter-specific litigation
            self.logger.debug("Extracting promoter litigation")
            litigation = self._extract_promoter_litigation(state)
            
            # 6. Extract shareholding
            self.logger.debug("Extracting shareholding pattern")
            shareholding = self._extract_shareholding(state)
            
            # 7. Merge all data into dossiers
            self.logger.debug("Merging promoter data")
            promoters = self._merge_promoter_data(
                basic_profiles,
                directorships,
                common_pursuits,
                financial_interests,
                litigation,
                shareholding,
                state
            )
            
            self.logger.info(f"Successfully extracted {len(promoters)} promoter profiles")
            
        except Exception as e:
            self.logger.error(f"Error extracting promoters: {e}")
            raise
        
        return promoters
    
    def _extract_basic_profiles(self, state: Dict) -> Dict[str, Dict]:
        """
        Extract basic promoter information from "Our Promoters" section.
        
        Returns:
            Dict mapping promoter name to their basic profile
        """
        profiles = {}
        
        # Get sections data
        sections = state.get('sections', {})
        
        # Look for promoter-related sections
        promoter_sections = self._find_promoter_sections(sections)
        
        for section_name, section_data in promoter_sections.items():
            content = section_data.get('content', '')
            
            # Extract promoter names using common patterns
            promoter_names = self._extract_promoter_names(content)
            
            for name in promoter_names:
                if name not in profiles:
                    profiles[name] = {
                        'name': name,
                        'din': self._extract_din(content, name),
                        'age': self._extract_age(content, name),
                        'qualification': self._extract_qualification(content, name),
                        'experience': self._extract_experience(content, name),
                        'designation': self._extract_designation(content, name)
                    }
        
        return profiles
    
    def _extract_directorships(self, state: Dict) -> Dict[str, List[str]]:
        """
        Extract other directorships for each promoter.
        
        Returns:
            Dict mapping promoter name to list of other directorships
        """
        directorships = {}
        sections = state.get('sections', {})
        
        # Look for "Other Directorships" section
        for section_name, section_data in sections.items():
            if 'directorship' in section_name.lower():
                content = section_data.get('content', '')
                tables = section_data.get('tables', [])
                
                # Parse tables if available
                if tables:
                    directorships.update(self._parse_directorship_tables(tables))
                else:
                    # Parse from text
                    directorships.update(self._parse_directorship_text(content))
        
        return directorships
    
    def _extract_common_pursuits(self, state: Dict) -> Dict[str, List[str]]:
        """
        Extract group companies in same line of business (conflict of interest).
        
        Returns:
            Dict mapping promoter name to list of conflicting companies
        """
        pursuits = {}
        sections = state.get('sections', {})
        
        # Look for "Common Pursuits" or "Interest of Promoters" sections
        for section_name, section_data in sections.items():
            if any(kw in section_name.lower() for kw in ['common pursuit', 'interest of promoter']):
                content = section_data.get('content', '')
                pursuits.update(self._parse_common_pursuits(content))
        
        return pursuits
    
    def _extract_financial_interests(self, state: Dict) -> Dict[str, Dict]:
        """
        Extract loans, remuneration, guarantees from company to promoters.
        
        Returns:
            Dict mapping promoter name to financial interests
        """
        interests = {}
        sections = state.get('sections', {})
        
        for section_name, section_data in sections.items():
            if any(kw in section_name.lower() for kw in ['payment of benefit', 'remuneration', 'interest']):
                content = section_data.get('content', '')
                tables = section_data.get('tables', [])
                
                if tables:
                    interests.update(self._parse_financial_tables(tables))
                else:
                    interests.update(self._parse_financial_text(content))
        
        return interests
    
    def _extract_promoter_litigation(self, state: Dict) -> Dict[str, List[Dict]]:
        """
        Extract litigation specific to promoters.
        
        Returns:
            Dict mapping promoter name to list of litigation cases
        """
        litigation = {}
        sections = state.get('sections', {})
        
        for section_name, section_data in sections.items():
            if 'litigation' in section_name.lower() or 'legal proceeding' in section_name.lower():
                content = section_data.get('content', '')
                tables = section_data.get('tables', [])
                
                if tables:
                    litigation.update(self._parse_litigation_tables(tables, filter_promoters=True))
                else:
                    litigation.update(self._parse_litigation_text(content, filter_promoters=True))
        
        return litigation
    
    def _extract_shareholding(self, state: Dict) -> Dict[str, Dict]:
        """
        Extract pre/post IPO shareholding for promoters.
        
        Returns:
            Dict with pre and post IPO shareholding percentages
        """
        shareholding = {}
        sections = state.get('sections', {})
        
        for section_name, section_data in sections.items():
            if 'capital structure' in section_name.lower() or 'shareholding' in section_name.lower():
                tables = section_data.get('tables', [])
                
                if tables:
                    shareholding.update(self._parse_shareholding_tables(tables))
        
        return shareholding
    
    def _merge_promoter_data(
        self,
        basic_profiles: Dict,
        directorships: Dict,
        common_pursuits: Dict,
        financial_interests: Dict,
        litigation: Dict,
        shareholding: Dict,
        state: Dict
    ) -> List[PromoterDossier]:
        """
        Merge all extracted data into PromoterDossier objects.
        
        Args:
            All the extracted data dictionaries
            state: Full state for additional calculations
            
        Returns:
            List of complete PromoterDossier objects
        """
        dossiers = []
        
        for name, profile in basic_profiles.items():
            # Get OFS details if available
            ofs_data = financial_interests.get(name, {}).get('ofs', {})
            
            # Get litigation data
            lit_data = litigation.get(name, [])
            criminal = sum(1 for case in lit_data if case.get('type') == 'criminal')
            civil = sum(1 for case in lit_data if case.get('type') == 'civil')
            regulatory = sum(1 for case in lit_data if case.get('type') == 'regulatory')
            total_lit_amount = sum(case.get('amount', 0) for case in lit_data)
            
            # Create dossier
            dossier = PromoterDossier(
                name=name,
                din=profile.get('din'),
                age=profile.get('age'),
                qualification=profile.get('qualification'),
                experience_years=profile.get('experience'),
                designation=profile.get('designation'),
                
                other_directorships=directorships.get(name, []),
                group_companies_in_same_line=common_pursuits.get(name, []),
                
                shareholding_pre_ipo=shareholding.get(name, {}).get('pre', 0.0),
                shareholding_post_ipo=shareholding.get(name, {}).get('post', 0.0),
                shares_selling_via_ofs=ofs_data.get('shares', 0),
                ofs_amount=ofs_data.get('amount', 0.0),
                
                loans_from_company=financial_interests.get(name, {}).get('loans', 0.0),
                guarantees_given=financial_interests.get(name, {}).get('guarantees', 0.0),
                remuneration_last_3_years=financial_interests.get(name, {}).get('remuneration', []),
                
                litigation_as_defendant=lit_data,
                criminal_cases=criminal,
                civil_cases=civil,
                regulatory_actions=regulatory,
                total_litigation_amount=total_lit_amount
            )
            
            # Calculate skin in the game
            ipo_details = state.get('ipo_details', {})
            cap_price = ipo_details.get('price_band_cap', 0)
            shares_post_ipo = ipo_details.get('shares_post_issue', 0)
            
            if cap_price and shares_post_ipo:
                holding_shares = (dossier.shareholding_post_ipo / 100) * shares_post_ipo
                dossier.skin_in_game_post_ipo = (holding_shares * cap_price) / 10000000  # Convert to Cr
            
            dossiers.append(dossier)
        
        return dossiers
    
    # Helper methods for parsing
    
    def _find_promoter_sections(self, sections: Dict) -> Dict:
        """Find sections related to promoters."""
        promoter_keywords = ['our promoter', 'promoter and promoter group', 'promoters']
        found = {}
        
        for section_name, section_data in sections.items():
            if any(kw in section_name.lower() for kw in promoter_keywords):
                found[section_name] = section_data
        
        return found
    
    def _extract_promoter_names(self, content: str) -> List[str]:
        """Extract promoter names from content."""
        # Common patterns for promoter names in RHPs
        # This is a simplified implementation - in production, use NER
        names = []
        
        # Pattern: "Mr./Ms./Mrs. [Name]" or names in bold/caps
        name_patterns = [
            r'(?:Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\n([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\n',  # Standalone names
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, content)
            names.extend(matches)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(names))
    
    def _extract_din(self, content: str, name: str) -> Optional[str]:
        """Extract DIN for a promoter."""
        # Pattern: DIN: XXXXXXXX or DIN XXXXXXXX
        pattern = rf'{re.escape(name)}.*?DIN[:\s]+(\d{{8}})'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        return match.group(1) if match else None
    
    def _extract_age(self, content: str, name: str) -> Optional[int]:
        """Extract age for a promoter."""
        # Pattern: Age: XX years or aged XX
        pattern = rf'{re.escape(name)}.*?(?:age[d]?[:\s]+|years old)(\d{{2}})'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        return int(match.group(1)) if match else None
    
    def _extract_qualification(self, content: str, name: str) -> Optional[str]:
        """Extract qualification for a promoter."""
        # Look for degree mentions near the name
        degrees = ['B.E.', 'B.Tech', 'M.E.', 'M.Tech', 'MBA', 'CA', 'CS', 'PhD', 'Bachelor', 'Master']
        pattern = rf'{re.escape(name)}.*?({"|".join(re.escape(d) for d in degrees)}[^.]*)'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_experience(self, content: str, name: str) -> Optional[int]:
        """Extract years of experience."""
        # Pattern: XX years of experience
        pattern = rf'{re.escape(name)}.*?(\d+)\s+years?\s+of\s+experience'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        return int(match.group(1)) if match else None
    
    def _extract_designation(self, content: str, name: str) -> Optional[str]:
        """Extract designation (MD, Chairman, etc.)."""
        designations = ['Managing Director', 'Chairman', 'Director', 'CEO', 'CFO', 'MD']
        pattern = rf'{re.escape(name)}.*?({"|".join(re.escape(d) for d in designations)})'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        return match.group(1) if match else None
    
    def _parse_directorship_tables(self, tables: List) -> Dict[str, List[str]]:
        """Parse directorship information from tables."""
        directorships = {}
        # Implementation depends on table structure
        # This is a placeholder
        return directorships
    
    def _parse_directorship_text(self, content: str) -> Dict[str, List[str]]:
        """Parse directorship information from text."""
        directorships = {}
        # Implementation depends on text format
        return directorships
    
    def _parse_common_pursuits(self, content: str) -> Dict[str, List[str]]:
        """Parse common pursuits from content."""
        pursuits = {}
        # Implementation to identify group companies in same business
        return pursuits
    
    def _parse_financial_tables(self, tables: List) -> Dict[str, Dict]:
        """Parse financial interest tables."""
        interests = {}
        # Parse remuneration, loans, etc. from tables
        return interests
    
    def _parse_financial_text(self, content: str) -> Dict[str, Dict]:
        """Parse financial interests from text."""
        interests = {}
        return interests
    
    def _parse_litigation_tables(self, tables: List, filter_promoters: bool = False) -> Dict[str, List[Dict]]:
        """Parse litigation tables, optionally filtering for promoters only."""
        litigation = {}
        # Parse litigation tables
        return litigation
    
    def _parse_litigation_text(self, content: str, filter_promoters: bool = False) -> Dict[str, List[Dict]]:
        """Parse litigation from text content."""
        litigation = {}
        return litigation
    
    def _parse_shareholding_tables(self, tables: List) -> Dict[str, Dict]:
        """Parse shareholding pattern tables."""
        shareholding = {}
        # Extract pre and post IPO shareholding percentages
        return shareholding
    
    def calculate_skin_in_game(
        self,
        dossier: PromoterDossier,
        cap_price: float,
        total_shares_post_ipo: int
    ) -> float:
        """
        Calculate skin-in-the-game (holding value) at cap price.
        
        Args:
            dossier: Promoter dossier
            cap_price: IPO cap price per share
            total_shares_post_ipo: Total shares after IPO
            
        Returns:
            Holding value in ₹ Crores
        """
        holding_percent = dossier.shareholding_post_ipo
        holding_shares = (holding_percent / 100) * total_shares_post_ipo
        value_in_rupees = holding_shares * cap_price
        value_in_crores = value_in_rupees / 10000000
        
        return value_in_crores
