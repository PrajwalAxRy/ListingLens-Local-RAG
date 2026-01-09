"""
Entity Extraction Module for RHP Analysis

This module handles extraction of named entities from RHP documents including:
- Companies, persons, locations
- Financial entities (amounts in crores/lakhs, percentages)
- Dates (fiscal years, issue dates)
- Organizations (regulators, underwriters)

Uses spaCy for base NER with custom patterns for Indian financial terminology.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
from datetime import datetime

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
from loguru import logger


class EntityType(Enum):
    """Enumeration of entity types to extract"""
    COMPANY = "COMPANY"
    PERSON = "PERSON"
    MONEY = "MONEY"
    DATE = "DATE"
    LOCATION = "LOCATION"
    ORG = "ORG"
    PERCENTAGE = "PERCENTAGE"
    FISCAL_YEAR = "FISCAL_YEAR"


@dataclass
class Entity:
    """Represents an extracted entity with metadata"""
    text: str
    entity_type: EntityType
    value: Optional[float] = None  # For MONEY and PERCENTAGE
    normalized_text: Optional[str] = None  # Canonical form
    start_char: int = 0
    end_char: int = 0
    page_num: Optional[int] = None
    confidence: float = 1.0
    mentions: int = 1
    contexts: List[str] = field(default_factory=list)
    page_references: List[int] = field(default_factory=list)
    aliases: Set[str] = field(default_factory=set)

    def __hash__(self):
        """Make Entity hashable for use in sets"""
        return hash((self.normalized_text or self.text, self.entity_type))

    def __eq__(self, other):
        """Equality based on normalized text and type"""
        if not isinstance(other, Entity):
            return False
        return (
            (self.normalized_text or self.text) == (other.normalized_text or other.text)
            and self.entity_type == other.entity_type
        )


class EntityExtractor:
    """
    Extract and normalize entities from RHP documents.
    
    Uses spaCy for base NER and custom patterns for Indian financial entities.
    Handles deduplication and coreference resolution.
    """

    # Indian currency conversion constants
    CRORE = 10_000_000  # 1 crore = 10 million
    LAKH = 100_000      # 1 lakh = 100 thousand

    # Regex patterns for Indian financial entities
    MONEY_PATTERNS = {
        'crore': re.compile(
            r'[₹Rs\.?\s]*'  # Optional currency symbol
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*'  # Amount with optional commas and decimals
            r'(?:crores?|cr\.?|Cr\.?)',  # "crore" variants
            re.IGNORECASE
        ),
        'lakh': re.compile(
            r'[₹Rs\.?\s]*'
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*'
            r'(?:lakhs?|lacs?)',
            re.IGNORECASE
        ),
        'basic': re.compile(
            r'[₹]\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            re.IGNORECASE
        ),
        'rs': re.compile(
            r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            re.IGNORECASE
        )
    }

    # Fiscal year pattern (FY 2023-24, FY23, FY 2023)
    FISCAL_YEAR_PATTERN = re.compile(
        r'\b(?:FY|F\.Y\.?|Fiscal Year)\s*'
        r'(?:(?:20)?(\d{2})(?:-(?:20)?(\d{2}))?)'
        r'|\b(?:20)?(\d{2})(?:-(?:20)?(\d{2}))\b',
        re.IGNORECASE
    )

    # Percentage pattern
    PERCENTAGE_PATTERN = re.compile(
        r'(\d+(?:\.\d+)?)\s*%'
    )

    # Date pattern (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY)
    DATE_PATTERN = re.compile(
        r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b'
    )

    # Price band pattern
    PRICE_BAND_PATTERN = re.compile(
        r'[₹Rs\.?\s]*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:to|-)\s*[₹Rs\.?\s]*(\d+(?:,\d+)*(?:\.\d+)?)',
        re.IGNORECASE
    )

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize entity extractor with spaCy model.
        
        Args:
            model_name: spaCy model to use (default: en_core_web_sm)
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(
                f"spaCy model '{model_name}' not found. "
                f"Run: python -m spacy download {model_name}"
            )
            raise

        # Entity cache for deduplication
        self.entity_cache: Dict[Tuple[str, EntityType], Entity] = {}
        
        # Add custom patterns to matcher
        self.matcher = Matcher(self.nlp.vocab)
        self._add_custom_patterns()

    def _add_custom_patterns(self):
        """Add custom patterns to spaCy matcher for Indian entities"""
        
        # Pattern for "Mr./Ms./Dr." + Name
        honorific_pattern = [
            {"LOWER": {"IN": ["mr", "ms", "mrs", "dr", "prof"]}},
            {"IS_PUNCT": True, "OP": "?"},
            {"IS_ALPHA": True, "LENGTH": {">": 1}},
            {"IS_ALPHA": True, "LENGTH": {">": 1}, "OP": "?"}
        ]
        self.matcher.add("PERSON_HONORIFIC", [honorific_pattern])

        # Pattern for company suffixes
        company_suffix_pattern = [
            {"IS_ALPHA": True, "LENGTH": {">": 2}},
            {"LOWER": {"IN": ["limited", "ltd", "ltd.", "pvt", "pvt.", "private", "inc", "inc.", "corporation", "corp"]}}
        ]
        self.matcher.add("COMPANY_SUFFIX", [company_suffix_pattern])

        # Pattern for regulators
        regulator_pattern = [
            {"TEXT": {"IN": ["SEBI", "RBI", "NCLT", "NCLAT", "ROC", "MCA"]}}
        ]
        self.matcher.add("REGULATOR", [regulator_pattern])

    def extract_entities(self, text: str, page_num: Optional[int] = None) -> List[Entity]:
        """
        Extract all entities from text.
        
        Args:
            text: Input text to extract entities from
            page_num: Optional page number for reference
            
        Returns:
            List of extracted Entity objects
        """
        entities = []

        # Extract using spaCy NER
        doc = self.nlp(text)
        
        # Extract standard entities
        entities.extend(self._extract_spacy_entities(doc, page_num))
        
        # Extract financial entities using custom patterns
        entities.extend(self._extract_money_entities(text, page_num))
        entities.extend(self._extract_percentage_entities(text, page_num))
        entities.extend(self._extract_fiscal_year_entities(text, page_num))
        entities.extend(self._extract_date_entities(text, page_num))
        
        # Apply custom matcher patterns
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            entity_type = self._match_id_to_entity_type(self.nlp.vocab.strings[match_id])
            if entity_type:
                entity = Entity(
                    text=span.text,
                    entity_type=entity_type,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    page_num=page_num
                )
                entities.append(entity)

        logger.debug(f"Extracted {len(entities)} entities from page {page_num}")
        return entities

    def _extract_spacy_entities(self, doc: Doc, page_num: Optional[int]) -> List[Entity]:
        """Extract entities using spaCy's base NER"""
        entities = []
        
        for ent in doc.ents:
            entity_type = self._map_spacy_label(ent.label_)
            if entity_type:
                entity = Entity(
                    text=ent.text,
                    entity_type=entity_type,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    page_num=page_num,
                    confidence=0.8  # Base confidence for spaCy entities
                )
                entities.append(entity)
        
        return entities

    def _map_spacy_label(self, label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to our EntityType enum"""
        mapping = {
            'ORG': EntityType.COMPANY,
            'PERSON': EntityType.PERSON,
            'GPE': EntityType.LOCATION,
            'LOC': EntityType.LOCATION,
            'DATE': EntityType.DATE,
            'MONEY': EntityType.MONEY,
            'PERCENT': EntityType.PERCENTAGE,
        }
        return mapping.get(label)

    def _extract_money_entities(self, text: str, page_num: Optional[int]) -> List[Entity]:
        """Extract money amounts in crores, lakhs, and rupees"""
        entities = []

        # Extract crores
        for match in self.MONEY_PATTERNS['crore'].finditer(text):
            amount_str = match.group(1).replace(',', '')
            try:
                amount = float(amount_str)
                value_in_units = amount * self.CRORE
                entity = Entity(
                    text=match.group(0),
                    entity_type=EntityType.MONEY,
                    value=value_in_units,
                    normalized_text=f"₹{amount} Cr",
                    start_char=match.start(),
                    end_char=match.end(),
                    page_num=page_num,
                    confidence=0.95
                )
                entities.append(entity)
            except ValueError:
                logger.warning(f"Could not parse amount: {match.group(1)}")

        # Extract lakhs
        for match in self.MONEY_PATTERNS['lakh'].finditer(text):
            amount_str = match.group(1).replace(',', '')
            try:
                amount = float(amount_str)
                value_in_units = amount * self.LAKH
                entity = Entity(
                    text=match.group(0),
                    entity_type=EntityType.MONEY,
                    value=value_in_units,
                    normalized_text=f"₹{amount} Lakh",
                    start_char=match.start(),
                    end_char=match.end(),
                    page_num=page_num,
                    confidence=0.95
                )
                entities.append(entity)
            except ValueError:
                logger.warning(f"Could not parse amount: {match.group(1)}")

        # Extract price bands
        for match in self.PRICE_BAND_PATTERN.finditer(text):
            lower_str = match.group(1).replace(',', '')
            upper_str = match.group(2).replace(',', '')
            try:
                lower = float(lower_str)
                upper = float(upper_str)
                entity = Entity(
                    text=match.group(0),
                    entity_type=EntityType.MONEY,
                    value=(lower + upper) / 2,  # Average as representative value
                    normalized_text=f"₹{lower}-₹{upper}",
                    start_char=match.start(),
                    end_char=match.end(),
                    page_num=page_num,
                    confidence=0.9
                )
                entities.append(entity)
            except ValueError:
                logger.warning(f"Could not parse price band: {match.group(0)}")

        return entities

    def _extract_percentage_entities(self, text: str, page_num: Optional[int]) -> List[Entity]:
        """Extract percentage values"""
        entities = []

        for match in self.PERCENTAGE_PATTERN.finditer(text):
            try:
                value = float(match.group(1))
                entity = Entity(
                    text=match.group(0),
                    entity_type=EntityType.PERCENTAGE,
                    value=value,
                    normalized_text=f"{value}%",
                    start_char=match.start(),
                    end_char=match.end(),
                    page_num=page_num,
                    confidence=0.95
                )
                entities.append(entity)
            except ValueError:
                logger.warning(f"Could not parse percentage: {match.group(1)}")

        return entities

    def _extract_fiscal_year_entities(self, text: str, page_num: Optional[int]) -> List[Entity]:
        """Extract fiscal year references (FY 2023-24, FY23, etc.)"""
        entities = []

        for match in self.FISCAL_YEAR_PATTERN.finditer(text):
            # Extract year components
            year1 = match.group(1) or match.group(3)
            year2 = match.group(2) or match.group(4)
            
            if year1:
                # Normalize to full year format
                if len(year1) == 2:
                    year1_full = f"20{year1}"
                else:
                    year1_full = year1
                
                if year2:
                    if len(year2) == 2:
                        year2_full = f"20{year2}"
                    else:
                        year2_full = year2
                    normalized = f"FY{year1_full[-2:]}-{year2_full[-2:]}"
                else:
                    normalized = f"FY{year1_full[-2:]}"
                
                entity = Entity(
                    text=match.group(0),
                    entity_type=EntityType.FISCAL_YEAR,
                    normalized_text=normalized,
                    start_char=match.start(),
                    end_char=match.end(),
                    page_num=page_num,
                    confidence=0.95
                )
                entities.append(entity)

        return entities

    def _extract_date_entities(self, text: str, page_num: Optional[int]) -> List[Entity]:
        """Extract dates in DD/MM/YYYY format"""
        entities = []

        for match in self.DATE_PATTERN.finditer(text):
            try:
                day = int(match.group(1))
                month = int(match.group(2))
                year = int(match.group(3))
                
                # Validate date
                date_obj = datetime(year, month, day)
                normalized = date_obj.strftime("%Y-%m-%d")
                
                entity = Entity(
                    text=match.group(0),
                    entity_type=EntityType.DATE,
                    normalized_text=normalized,
                    start_char=match.start(),
                    end_char=match.end(),
                    page_num=page_num,
                    confidence=0.9
                )
                entities.append(entity)
            except (ValueError, OverflowError):
                # Invalid date, skip
                continue

        return entities

    def _match_id_to_entity_type(self, match_id: str) -> Optional[EntityType]:
        """Map matcher pattern IDs to entity types"""
        mapping = {
            'PERSON_HONORIFIC': EntityType.PERSON,
            'COMPANY_SUFFIX': EntityType.COMPANY,
            'REGULATOR': EntityType.ORG,
        }
        return mapping.get(match_id)

    def deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate entities and merge information.
        
        Args:
            entities: List of entities to deduplicate
            
        Returns:
            Deduplicated list of entities with merged information
        """
        # Clear cache for new deduplication
        self.entity_cache.clear()

        for entity in entities:
            # Normalize the entity text
            normalized = self._normalize_entity_text(entity.text, entity.entity_type)
            entity.normalized_text = normalized

            # Create cache key
            cache_key = (normalized, entity.entity_type)

            if cache_key in self.entity_cache:
                # Merge with existing entity
                existing = self.entity_cache[cache_key]
                existing.mentions += 1
                existing.aliases.add(entity.text)
                if entity.page_num and entity.page_num not in existing.page_references:
                    existing.page_references.append(entity.page_num)
                # Update confidence (average)
                existing.confidence = (existing.confidence + entity.confidence) / 2
            else:
                # Add new entity to cache
                entity.aliases.add(entity.text)
                if entity.page_num:
                    entity.page_references.append(entity.page_num)
                self.entity_cache[cache_key] = entity

        deduplicated = list(self.entity_cache.values())
        logger.info(f"Deduplicated {len(entities)} entities to {len(deduplicated)}")
        return deduplicated

    def _normalize_entity_text(self, text: str, entity_type: EntityType) -> str:
        """
        Normalize entity text for deduplication.
        
        Args:
            text: Entity text to normalize
            entity_type: Type of entity
            
        Returns:
            Normalized text
        """
        normalized = text.strip()

        if entity_type == EntityType.COMPANY:
            # Remove common suffixes and normalize
            # Handle all combinations: Private Limited, Pvt Ltd, Pvt. Ltd., etc.
            normalized = re.sub(r'\s+(Private\s+Limited|Pvt\.?\s*Ltd\.?|Private|Pvt\.?|Limited|Ltd\.?|Inc\.?|Corporation|Corp\.?)\s*$', '', normalized, flags=re.IGNORECASE)
            normalized = re.sub(r'\s+', ' ', normalized).strip()  # Normalize whitespace
            normalized = normalized.title()  # Title case
        
        elif entity_type == EntityType.PERSON:
            # Remove honorifics
            normalized = re.sub(r'^(Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Prof\.?)\s+', '', normalized, flags=re.IGNORECASE)
            normalized = normalized.title()
        
        elif entity_type == EntityType.LOCATION:
            normalized = normalized.title()

        return normalized

    def resolve_coreferences(self, entities: List[Entity]) -> List[Entity]:
        """
        Resolve entity coreferences and abbreviations.
        
        Rules:
        - "XYZ Ltd" = "XYZ Limited" = "XYZ"
        - "Mr. John Doe" = "John Doe"
        - First mention establishes canonical name
        
        Args:
            entities: List of entities to resolve
            
        Returns:
            List of entities with resolved coreferences
        """
        # Group by entity type
        by_type: Dict[EntityType, List[Entity]] = {}
        for entity in entities:
            if entity.entity_type not in by_type:
                by_type[entity.entity_type] = []
            by_type[entity.entity_type].append(entity)

        # Resolve within each type
        resolved = []
        for entity_type, entity_list in by_type.items():
            if entity_type in [EntityType.COMPANY, EntityType.PERSON]:
                resolved.extend(self._resolve_name_entities(entity_list, entity_type))
            else:
                resolved.extend(entity_list)

        logger.info(f"Resolved coreferences for {len(entities)} entities")
        return resolved

    def _resolve_name_entities(self, entities: List[Entity], entity_type: EntityType) -> List[Entity]:
        """Resolve coreferences for name-based entities (COMPANY, PERSON)"""
        if not entities:
            return []

        # Sort by number of mentions (descending) to prefer most common form
        entities.sort(key=lambda e: e.mentions, reverse=True)

        # Use most mentioned entity as canonical
        canonical = entities[0]
        # Add canonical's own text to aliases
        canonical.aliases.add(canonical.text)
        
        # Merge aliases from other entities
        for entity in entities[1:]:
            # Check if this is a variant of canonical
            if self._is_variant(canonical.text, entity.text, entity_type):
                # Add the variant text itself to aliases (not just nested aliases)
                canonical.aliases.add(entity.text)
                canonical.aliases.update(entity.aliases)
                canonical.mentions += entity.mentions
                canonical.page_references.extend(entity.page_references)
                canonical.page_references = list(set(canonical.page_references))  # Deduplicate

        return [canonical]

    def _is_variant(self, canonical: str, candidate: str, entity_type: EntityType) -> bool:
        """Check if candidate is a variant of canonical entity"""
        # Normalize for comparison
        canon_norm = self._normalize_entity_text(canonical, entity_type).lower()
        cand_norm = self._normalize_entity_text(candidate, entity_type).lower()

        # Exact match after normalization
        if canon_norm == cand_norm:
            return True

        # Substring match (abbreviation)
        if cand_norm in canon_norm or canon_norm in cand_norm:
            return True

        # Check if words overlap significantly
        canon_words = set(canon_norm.split())
        cand_words = set(cand_norm.split())
        overlap = len(canon_words & cand_words)
        
        # At least 80% overlap
        min_len = min(len(canon_words), len(cand_words))
        if min_len > 0 and overlap / min_len >= 0.8:
            return True

        return False

    def extract_all(self, pages_text: List[Tuple[int, str]]) -> Dict[EntityType, List[Entity]]:
        """
        Extract entities from multiple pages and organize by type.
        
        Args:
            pages_text: List of (page_number, text) tuples
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        all_entities = []

        # Extract from all pages
        for page_num, text in pages_text:
            entities = self.extract_entities(text, page_num)
            all_entities.extend(entities)

        # Deduplicate
        deduplicated = self.deduplicate_entities(all_entities)
        
        # Resolve coreferences
        resolved = self.resolve_coreferences(deduplicated)

        # Organize by type
        by_type: Dict[EntityType, List[Entity]] = {}
        for entity in resolved:
            if entity.entity_type not in by_type:
                by_type[entity.entity_type] = []
            by_type[entity.entity_type].append(entity)

        # Log summary
        for entity_type, entity_list in by_type.items():
            logger.info(f"Extracted {len(entity_list)} {entity_type.value} entities")

        return by_type
