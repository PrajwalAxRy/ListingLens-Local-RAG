"""
Section Mapping Module for RHP Analyzer.

This module builds a hierarchical structure of RHP document sections
by analyzing font sizes, styles, and regex patterns for common section headers.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger


class SectionType(Enum):
    """Classification of RHP section types."""

    SUMMARY = "summary"
    RISK_FACTORS = "risk_factors"
    INTRODUCTION = "introduction"
    THE_ISSUE = "the_issue"
    CAPITAL_STRUCTURE = "capital_structure"
    OBJECTS_OF_ISSUE = "objects_of_issue"
    BASIS_FOR_ISSUE_PRICE = "basis_for_issue_price"
    BUSINESS = "business"
    INDUSTRY_OVERVIEW = "industry_overview"
    MANAGEMENT = "management"
    FINANCIAL_INFORMATION = "financial_information"
    LEGAL_INFORMATION = "legal_information"
    REGULATORY_DISCLOSURES = "regulatory_disclosures"
    ARTICLES_OF_ASSOCIATION = "articles_of_association"
    DEFINITIONS = "definitions"
    OTHER = "other"


@dataclass
class Section:
    """Represents a document section in the RHP."""

    section_id: str
    section_type: SectionType
    title: str
    level: int  # 1 = top-level, 2 = subsection, 3 = sub-subsection
    start_page: int
    end_page: int
    content: str = ""
    word_count: int = 0
    has_tables: bool = False
    subsections: list["Section"] = field(default_factory=list)
    parent_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert section to dictionary representation."""
        return {
            "section_id": self.section_id,
            "section_type": self.section_type.value,
            "title": self.title,
            "level": self.level,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "word_count": self.word_count,
            "has_tables": self.has_tables,
            "subsection_count": len(self.subsections),
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }

    def get_page_range(self) -> tuple[int, int]:
        """Return (start_page, end_page) tuple."""
        return (self.start_page, self.end_page)


@dataclass
class SectionTree:
    """Hierarchical tree structure of document sections."""

    root_sections: list[Section] = field(default_factory=list)
    all_sections: dict[str, Section] = field(default_factory=dict)
    document_id: str = ""
    total_pages: int = 0

    def get_section_by_id(self, section_id: str) -> Optional[Section]:
        """Get a section by its ID."""
        return self.all_sections.get(section_id)

    def get_section_by_type(self, section_type: SectionType) -> list[Section]:
        """Get all sections of a specific type."""
        return [s for s in self.all_sections.values() if s.section_type == section_type]

    def get_section_by_title(self, title: str, fuzzy: bool = True) -> Optional[Section]:
        """Get a section by title (with optional fuzzy matching)."""
        title_lower = title.lower().strip()
        for section in self.all_sections.values():
            section_title = section.title.lower().strip()
            if fuzzy:
                if title_lower in section_title or section_title in title_lower:
                    return section
            else:
                if section_title == title_lower:
                    return section
        return None

    def get_page_to_section_map(self) -> dict[int, list[Section]]:
        """Create a mapping from page numbers to sections."""
        page_map: dict[int, list[Section]] = {}
        for section in self.all_sections.values():
            for page_num in range(section.start_page, section.end_page + 1):
                if page_num not in page_map:
                    page_map[page_num] = []
                page_map[page_num].append(section)
        return page_map

    def to_dict(self) -> dict[str, Any]:
        """Convert tree to dictionary representation."""
        return {
            "document_id": self.document_id,
            "total_pages": self.total_pages,
            "root_section_count": len(self.root_sections),
            "total_section_count": len(self.all_sections),
            "root_sections": [s.to_dict() for s in self.root_sections],
        }


class SectionMapper:
    """
    Creates hierarchical document structure from extracted pages.

    Uses font size analysis, regex patterns, and ToC parsing to identify
    section boundaries and build a section tree.
    """

    # Section title patterns for Indian RHP documents
    SECTION_PATTERNS: dict[SectionType, list[str]] = {
        SectionType.SUMMARY: [
            r"^\s*SUMMARY\s+OF\s+(?:THE\s+)?(?:OFFER|PROSPECTUS)\s*$",
            r"^\s*SUMMARY\s+OF\s+BUSINESS\s*$",
            r"^\s*SUMMARY\s*$",
        ],
        SectionType.DEFINITIONS: [
            r"^\s*DEFINITIONS\s+AND\s+ABBREVIATIONS?\s*$",
            r"^\s*CONVENTIONAL\s+AND\s+GENERAL\s+TERMS?\s*$",
            r"^\s*DEFINITIONS?\s*$",
        ],
        SectionType.RISK_FACTORS: [
            r"^\s*RISK\s+FACTORS?\s*$",
            r"^\s*RISKS?\s+RELATING\s+TO\s*",
        ],
        SectionType.INTRODUCTION: [
            r"^\s*INTRODUCTION\s*$",
            r"^\s*GENERAL\s+INFORMATION\s*$",
        ],
        SectionType.THE_ISSUE: [
            r"^\s*THE\s+ISSUE\s*$",
            r"^\s*ISSUE\s+DETAILS?\s*$",
            r"^\s*TERMS\s+OF\s+THE\s+ISSUE\s*$",
        ],
        SectionType.CAPITAL_STRUCTURE: [
            r"^\s*CAPITAL\s+STRUCTURE\s*$",
            r"^\s*SHARE\s+CAPITAL\s*$",
        ],
        SectionType.OBJECTS_OF_ISSUE: [
            r"^\s*OBJECTS?\s+OF\s+THE\s+(?:ISSUE|OFFER)\s*$",
            r"^\s*USE\s+OF\s+(?:ISSUE\s+)?PROCEEDS?\s*$",
        ],
        SectionType.BASIS_FOR_ISSUE_PRICE: [
            r"^\s*BASIS\s+FOR\s+(?:ISSUE\s+)?(?:PRICE|PRICING)\s*$",
            r"^\s*PRICE\s+JUSTIFICATION\s*$",
        ],
        SectionType.BUSINESS: [
            r"^\s*(?:OUR\s+)?BUSINESS\s*$",
            r"^\s*(?:ABOUT\s+)?OUR\s+COMPANY\s*$",
            r"^\s*DESCRIPTION\s+OF\s+(?:OUR\s+)?BUSINESS\s*$",
            r"^\s*BUSINESS\s+OVERVIEW\s*$",
        ],
        SectionType.INDUSTRY_OVERVIEW: [
            r"^\s*INDUSTRY\s+(?:OVERVIEW|ANALYSIS)\s*$",
            r"^\s*MARKET\s+OVERVIEW\s*$",
            r"^\s*SECTOR\s+OVERVIEW\s*$",
        ],
        SectionType.MANAGEMENT: [
            r"^\s*(?:OUR\s+)?MANAGEMENT\s*$",
            r"^\s*BOARD\s+OF\s+DIRECTORS?\s*$",
            r"^\s*KEY\s+MANAGERIAL?\s+PERSONNEL\s*$",
            r"^\s*(?:OUR\s+)?PROMOTERS?\s*$",
        ],
        SectionType.FINANCIAL_INFORMATION: [
            r"^\s*FINANCIAL\s+(?:INFORMATION|STATEMENTS?)\s*$",
            r"^\s*RESTATED\s+(?:CONSOLIDATED\s+)?FINANCIAL\s+(?:INFORMATION|STATEMENTS?)\s*$",
            r"^\s*AUDITOR(?:'?S)?\s+REPORT\s*$",
        ],
        SectionType.LEGAL_INFORMATION: [
            r"^\s*LEGAL\s+(?:AND\s+OTHER\s+)?INFORMATION\s*$",
            r"^\s*OUTSTANDING\s+LITIGATION\s*$",
            r"^\s*LEGAL\s+PROCEEDINGS?\s*$",
            r"^\s*MATERIAL\s+DEVELOPMENTS?\s*$",
        ],
        SectionType.REGULATORY_DISCLOSURES: [
            r"^\s*(?:OTHER\s+)?REGULATORY\s+(?:AND\s+STATUTORY\s+)?DISCLOSURES?\s*$",
            r"^\s*STATUTORY\s+(?:AND\s+OTHER\s+)?INFORMATION\s*$",
        ],
        SectionType.ARTICLES_OF_ASSOCIATION: [
            r"^\s*MAIN\s+PROVISIONS\s+OF\s+(?:THE\s+)?ARTICLES?\s*",
            r"^\s*ARTICLES?\s+OF\s+ASSOCIATION\s*$",
            r"^\s*MEMORANDUM\s+(?:AND|&)\s+ARTICLES?\s*",
        ],
    }

    # Minimum font size to be considered a header (relative to body text)
    HEADER_FONT_SIZE_RATIO = 1.15

    # Minimum font size for major section headers
    MAJOR_SECTION_FONT_SIZE_RATIO = 1.3

    def __init__(self, min_section_words: int = 50):
        """
        Initialize the section mapper.

        Args:
            min_section_words: Minimum words for a valid section.
        """
        self.min_section_words = min_section_words
        self._section_counter = 0
        self._compiled_patterns: dict[SectionType, list[re.Pattern]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile all regex patterns for efficiency."""
        for section_type, patterns in self.SECTION_PATTERNS.items():
            self._compiled_patterns[section_type] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns
            ]

    def _generate_section_id(self) -> str:
        """Generate a unique section ID."""
        self._section_counter += 1
        return f"sec_{self._section_counter:04d}"

    def build_hierarchy(self, pages: list[Any]) -> SectionTree:
        """
        Build document section tree from extracted pages.

        Args:
            pages: List of PageInfo objects from PDFProcessor.

        Returns:
            SectionTree with hierarchical section structure.
        """
        logger.info(f"Building section hierarchy from {len(pages)} pages")
        self._section_counter = 0

        # Step 1: Calculate baseline font size
        baseline_font_size = self._calculate_baseline_font_size(pages)
        logger.debug(f"Baseline font size: {baseline_font_size:.1f}")

        # Step 2: Detect section headers from each page
        section_candidates = self._detect_section_headers(pages, baseline_font_size)
        logger.debug(f"Found {len(section_candidates)} section header candidates")

        # Step 3: Build initial sections with boundaries
        sections = self._build_sections(section_candidates, pages)
        logger.debug(f"Built {len(sections)} sections")

        # Step 4: Assign content and metadata to sections
        self._assign_section_content(sections, pages)

        # Step 5: Build hierarchy tree
        tree = self._build_tree(sections, pages)

        logger.info(
            f"Section mapping complete: {len(tree.root_sections)} root sections, "
            f"{len(tree.all_sections)} total sections"
        )

        return tree

    def _calculate_baseline_font_size(self, pages: list[Any]) -> float:
        """
        Calculate the most common (body text) font size across pages.

        Args:
            pages: List of PageInfo objects.

        Returns:
            The baseline/body text font size.
        """
        font_size_counts: dict[float, int] = {}

        for page in pages:
            if hasattr(page, "font_sizes") and page.font_sizes:
                for size in page.font_sizes:
                    # Round to nearest 0.5 for grouping
                    rounded = round(size * 2) / 2
                    font_size_counts[rounded] = font_size_counts.get(rounded, 0) + 1

        if not font_size_counts:
            return 10.0  # Default fallback

        # Return most common font size (assumed to be body text)
        return max(font_size_counts, key=font_size_counts.get)

    def _detect_section_headers(
        self, pages: list[Any], baseline_font_size: float
    ) -> list[dict[str, Any]]:
        """
        Detect potential section headers from pages.

        Args:
            pages: List of PageInfo objects.
            baseline_font_size: The baseline body text font size.

        Returns:
            List of section header candidates with metadata.
        """
        candidates = []
        header_threshold = baseline_font_size * self.HEADER_FONT_SIZE_RATIO
        major_threshold = baseline_font_size * self.MAJOR_SECTION_FONT_SIZE_RATIO

        for page in pages:
            page_num = page.page_num if hasattr(page, "page_num") else 0
            text = page.text if hasattr(page, "text") else ""
            font_sizes = page.font_sizes if hasattr(page, "font_sizes") else []
            max_font_size = max(font_sizes) if font_sizes else baseline_font_size

            # Extract potential headers from text
            lines = text.split("\n")
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                # Skip very short or very long lines
                if len(line_stripped) < 3 or len(line_stripped) > 150:
                    continue

                # Check for section pattern matches
                section_type = self._match_section_pattern(line_stripped)

                # Determine header level based on patterns and characteristics
                is_uppercase = line_stripped.isupper()
                is_title_case = line_stripped.istitle()
                has_numbering = bool(re.match(r"^\d+\.?\s+|^[IVXLCDM]+\.?\s+", line_stripped))

                # Calculate header score
                header_score = 0.0

                if section_type != SectionType.OTHER:
                    header_score += 3.0  # Pattern match is strong signal

                if is_uppercase and len(line_stripped) < 80:
                    header_score += 2.0

                if has_numbering:
                    header_score += 1.0

                if is_title_case and not is_uppercase:
                    header_score += 0.5

                if max_font_size >= major_threshold:
                    header_score += 1.5
                elif max_font_size >= header_threshold:
                    header_score += 1.0

                # Only consider if score is above threshold
                if header_score >= 2.0:
                    level = self._determine_header_level(
                        line_stripped, max_font_size, baseline_font_size, has_numbering
                    )

                    candidates.append({
                        "title": line_stripped,
                        "page_num": page_num,
                        "section_type": section_type,
                        "level": level,
                        "score": header_score,
                        "is_uppercase": is_uppercase,
                        "font_size": max_font_size,
                    })

        # Remove duplicates and sort by page number
        candidates = self._deduplicate_candidates(candidates)
        candidates.sort(key=lambda x: (x["page_num"], -x["score"]))

        return candidates

    def _match_section_pattern(self, text: str) -> SectionType:
        """
        Match text against known section patterns.

        Args:
            text: Text to match.

        Returns:
            Matched SectionType or OTHER if no match.
        """
        for section_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return section_type
        return SectionType.OTHER

    def _determine_header_level(
        self,
        text: str,
        font_size: float,
        baseline: float,
        has_numbering: bool,
    ) -> int:
        """
        Determine the hierarchical level of a header.

        Args:
            text: Header text.
            font_size: Font size of the header.
            baseline: Baseline body font size.
            has_numbering: Whether header has numbering.

        Returns:
            Level (1 = top-level, 2 = subsection, 3 = sub-subsection).
        """
        # Check for nested numbering pattern (e.g., "1.2.3")
        nested_match = re.match(r"^(\d+(?:\.\d+)*)", text)
        if nested_match:
            dots = nested_match.group(1).count(".")
            return min(dots + 1, 3)

        # Check font size ratio
        size_ratio = font_size / baseline if baseline > 0 else 1.0

        if size_ratio >= self.MAJOR_SECTION_FONT_SIZE_RATIO:
            return 1
        elif size_ratio >= self.HEADER_FONT_SIZE_RATIO:
            return 2
        else:
            return 3

    def _deduplicate_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Remove duplicate header candidates.

        Args:
            candidates: List of header candidates.

        Returns:
            Deduplicated list.
        """
        seen = set()
        unique = []

        for candidate in candidates:
            # Create a key based on normalized title and page
            key = (
                re.sub(r"\s+", " ", candidate["title"].lower().strip()),
                candidate["page_num"],
            )

            if key not in seen:
                seen.add(key)
                unique.append(candidate)

        return unique

    def _build_sections(
        self, candidates: list[dict[str, Any]], pages: list[Any]
    ) -> list[Section]:
        """
        Build Section objects from header candidates.

        Args:
            candidates: List of header candidates.
            pages: List of PageInfo objects.

        Returns:
            List of Section objects with boundaries.
        """
        sections = []
        total_pages = len(pages)

        for i, candidate in enumerate(candidates):
            # Determine end page (start of next section - 1)
            if i + 1 < len(candidates):
                end_page = candidates[i + 1]["page_num"] - 1
                if end_page < candidate["page_num"]:
                    end_page = candidate["page_num"]
            else:
                end_page = total_pages

            section = Section(
                section_id=self._generate_section_id(),
                section_type=candidate["section_type"],
                title=candidate["title"],
                level=candidate["level"],
                start_page=candidate["page_num"],
                end_page=end_page,
                metadata={
                    "detection_score": candidate["score"],
                    "is_uppercase": candidate["is_uppercase"],
                    "font_size": candidate.get("font_size", 0),
                },
            )

            sections.append(section)

        return sections

    def _assign_section_content(
        self, sections: list[Section], pages: list[Any]
    ) -> None:
        """
        Assign content and calculate metadata for each section.

        Args:
            sections: List of Section objects.
            pages: List of PageInfo objects.
        """
        # Create page lookup
        page_map = {p.page_num: p for p in pages if hasattr(p, "page_num")}

        for section in sections:
            content_parts = []
            total_words = 0
            has_tables = False

            for page_num in range(section.start_page, section.end_page + 1):
                page = page_map.get(page_num)
                if page:
                    if hasattr(page, "text"):
                        content_parts.append(page.text)
                    if hasattr(page, "word_count"):
                        total_words += page.word_count
                    if hasattr(page, "has_tables") and page.has_tables:
                        has_tables = True

            section.content = "\n\n".join(content_parts)
            section.word_count = total_words
            section.has_tables = has_tables

    def _build_tree(self, sections: list[Section], pages: list[Any]) -> SectionTree:
        """
        Build hierarchical tree structure from flat sections list.

        Args:
            sections: List of Section objects.
            pages: List of PageInfo objects.

        Returns:
            SectionTree with hierarchy.
        """
        tree = SectionTree(
            document_id="",
            total_pages=len(pages),
        )

        if not sections:
            return tree

        # Build parent-child relationships based on levels
        root_sections = []
        section_stack: list[Section] = []

        for section in sections:
            # Add to all_sections
            tree.all_sections[section.section_id] = section

            # Find parent based on level
            while section_stack and section_stack[-1].level >= section.level:
                section_stack.pop()

            if section_stack:
                # This section is a child of the top of the stack
                parent = section_stack[-1]
                section.parent_id = parent.section_id
                parent.subsections.append(section)
            else:
                # This is a root section
                root_sections.append(section)

            section_stack.append(section)

        tree.root_sections = root_sections

        return tree

    def extract_section_boundaries(self, tree: SectionTree) -> dict[str, tuple[int, int]]:
        """
        Get page ranges for each section.

        Args:
            tree: SectionTree to extract from.

        Returns:
            Dict mapping section titles to (start_page, end_page) tuples.
        """
        boundaries = {}

        for section in tree.all_sections.values():
            boundaries[section.title] = (section.start_page, section.end_page)

        return boundaries

    def get_section_for_page(
        self, tree: SectionTree, page_num: int
    ) -> list[Section]:
        """
        Get all sections that contain a specific page.

        Args:
            tree: SectionTree to search.
            page_num: Page number to find.

        Returns:
            List of sections containing the page.
        """
        matching = []

        for section in tree.all_sections.values():
            if section.start_page <= page_num <= section.end_page:
                matching.append(section)

        # Sort by level (most specific first)
        matching.sort(key=lambda s: s.level, reverse=True)

        return matching


# Standard RHP section names with aliases for normalization
STANDARD_RHP_SECTIONS: dict[str, list[str]] = {
    "Summary of the Issue": [
        "SUMMARY OF PROSPECTUS",
        "SUMMARY OF THE OFFER",
        "SUMMARY",
    ],
    "Definitions and Abbreviations": [
        "DEFINITIONS AND ABBREVIATIONS",
        "CONVENTIONAL AND GENERAL TERMS",
        "DEFINITIONS",
    ],
    "Risk Factors": [
        "RISK FACTORS",
        "RISKS",
    ],
    "Introduction": [
        "INTRODUCTION",
        "GENERAL INFORMATION",
    ],
    "The Issue": [
        "THE ISSUE",
        "ISSUE DETAILS",
        "TERMS OF THE ISSUE",
    ],
    "Capital Structure": [
        "CAPITAL STRUCTURE",
        "SHARE CAPITAL",
    ],
    "Objects of the Issue": [
        "OBJECTS OF THE ISSUE",
        "OBJECTS OF THE OFFER",
        "USE OF PROCEEDS",
    ],
    "Basis for Issue Price": [
        "BASIS FOR ISSUE PRICE",
        "BASIS FOR PRICING",
        "PRICE JUSTIFICATION",
    ],
    "Our Business": [
        "OUR BUSINESS",
        "ABOUT OUR COMPANY",
        "DESCRIPTION OF BUSINESS",
        "BUSINESS",
    ],
    "Industry Overview": [
        "INDUSTRY OVERVIEW",
        "INDUSTRY ANALYSIS",
        "MARKET OVERVIEW",
        "SECTOR OVERVIEW",
    ],
    "Our Management": [
        "OUR MANAGEMENT",
        "BOARD OF DIRECTORS",
        "KEY MANAGERIAL PERSONNEL",
        "MANAGEMENT",
    ],
    "Our Promoters": [
        "OUR PROMOTERS",
        "PROMOTERS",
        "OUR PROMOTER AND PROMOTER GROUP",
    ],
    "Financial Information": [
        "FINANCIAL INFORMATION",
        "FINANCIAL STATEMENTS",
        "RESTATED FINANCIAL INFORMATION",
        "AUDITOR'S REPORT",
    ],
    "Legal and Other Information": [
        "LEGAL AND OTHER INFORMATION",
        "OUTSTANDING LITIGATION",
        "LEGAL PROCEEDINGS",
        "MATERIAL DEVELOPMENTS",
    ],
    "Other Regulatory Disclosures": [
        "OTHER REGULATORY DISCLOSURES",
        "REGULATORY AND STATUTORY DISCLOSURES",
        "STATUTORY INFORMATION",
    ],
    "Main Provisions of Articles": [
        "MAIN PROVISIONS OF THE ARTICLES",
        "ARTICLES OF ASSOCIATION",
        "MEMORANDUM AND ARTICLES",
    ],
}


def normalize_section_name(section_title: str) -> str:
    """
    Normalize a section title to its standard name.

    Args:
        section_title: Raw section title from document.

    Returns:
        Normalized standard section name or original if no match.
    """
    title_upper = section_title.upper().strip()

    for standard_name, aliases in STANDARD_RHP_SECTIONS.items():
        for alias in aliases:
            if alias in title_upper or title_upper in alias:
                return standard_name

    return section_title
