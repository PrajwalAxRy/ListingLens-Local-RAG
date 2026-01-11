"""
Semantic Chunking Module for RHP Analyzer.

This module provides intelligent text chunking that preserves semantic meaning
by respecting section boundaries, paragraph structures, and handling tables specially.

Chunking Strategy:
1. First split by section boundaries
2. Then split by paragraphs within sections
3. Merge small paragraphs, split large ones
4. Target 500-1000 tokens per chunk
5. Add 100-token overlap between chunks
6. Keep tables whole or create summaries

Reference: milestones.md Milestone 3.1, blueprint.md Section 3.2.1
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger

from rhp_analyzer.ingestion.pdf_processor import PageInfo
from rhp_analyzer.ingestion.section_mapper import Section, SectionTree
from rhp_analyzer.ingestion.table_extractor import Table


class ChunkType(Enum):
    """Classification of chunk content types."""

    NARRATIVE = "narrative"  # Regular text content
    TABLE = "table"  # Table content (kept whole or summarized)
    MIXED = "mixed"  # Contains both text and table elements
    LIST = "list"  # Bulleted or numbered list


@dataclass
class Chunk:
    """
    Represents a semantic chunk of document content.

    Chunks are the primary unit for embedding and retrieval operations.
    Each chunk maintains rich metadata for filtering and citation.

    Attributes:
        chunk_id: Unique identifier for the chunk
        document_id: ID of the source document
        text: The actual text content of the chunk
        chunk_type: Classification (narrative, table, mixed, list)
        section_name: Name of the containing section
        section_type: Type of the containing section
        page_start: First page number containing this chunk
        page_end: Last page number containing this chunk
        start_char: Character offset where chunk starts in source
        end_char: Character offset where chunk ends in source
        token_count: Approximate token count (chars / 4)
        char_count: Exact character count
        preceding_chunk_id: ID of the previous chunk (for context)
        following_chunk_id: ID of the next chunk (for context)
        has_table: Whether chunk contains table content
        table_id: ID of associated table if applicable
        metadata: Additional metadata (e.g., font info, headers)
        embedding: Vector embedding (set later by embedding module)
    """

    chunk_id: str
    document_id: str
    text: str
    chunk_type: ChunkType
    section_name: str
    section_type: str
    page_start: int
    page_end: int
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    char_count: int = 0
    preceding_chunk_id: Optional[str] = None
    following_chunk_id: Optional[str] = None
    has_table: bool = False
    table_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.char_count == 0:
            self.char_count = len(self.text)
        if self.token_count == 0:
            # Rough estimate: ~4 characters per token for English text
            self.token_count = len(self.text) // 4

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_type": self.chunk_type.value,
            "section_name": self.section_name,
            "section_type": self.section_type,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "preceding_chunk_id": self.preceding_chunk_id,
            "following_chunk_id": self.following_chunk_id,
            "has_table": self.has_table,
            "table_id": self.table_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Create a Chunk from dictionary."""
        data["chunk_type"] = ChunkType(data["chunk_type"])
        # Remove embedding if present (not needed for reconstruction)
        data.pop("embedding", None)
        return cls(**data)


@dataclass
class ChunkingConfig:
    """Configuration for the chunking process."""

    # Token-based settings (1 token ≈ 4 chars)
    target_chunk_size: int = 750  # Target tokens per chunk (middle of 500-1000 range)
    min_chunk_size: int = 200  # Minimum tokens (don't create tiny chunks)
    max_chunk_size: int = 1500  # Maximum tokens (hard limit)
    chunk_overlap: int = 100  # Overlap tokens between consecutive chunks

    # Table handling
    max_table_tokens: int = 500  # Tables above this get summarized
    keep_small_tables_whole: bool = True  # Don't chunk small tables

    # Paragraph handling
    paragraph_separator: str = "\n\n"  # How paragraphs are separated
    sentence_separators: tuple[str, ...] = (".", "!", "?", "।")  # For sentence splitting

    def to_chars(self, tokens: int) -> int:
        """Convert token count to approximate character count."""
        return tokens * 4

    def to_tokens(self, chars: int) -> int:
        """Convert character count to approximate token count."""
        return chars // 4


class SemanticChunker:
    """
    Smart text chunking that preserves semantic meaning.

    This chunker:
    - Respects section boundaries (never crosses sections)
    - Preserves paragraph structure where possible
    - Handles tables specially (keeps whole or summarizes)
    - Adds overlap between consecutive chunks for context
    - Attaches rich metadata for filtering and citation
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the semantic chunker.

        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        self.config = config or ChunkingConfig()
        logger.debug(
            f"SemanticChunker initialized with target_chunk_size={self.config.target_chunk_size}"
        )

    def chunk_document(
        self,
        document_id: str,
        pages: list[PageInfo],
        sections: SectionTree,
        tables: Optional[list[Table]] = None,
    ) -> list[Chunk]:
        """
        Chunk an entire document respecting section boundaries.

        This is the main entry point for chunking a document.

        Args:
            document_id: Unique identifier for the document
            pages: List of PageInfo objects from PDF processing
            sections: Section tree from section mapping
            tables: Optional list of extracted tables

        Returns:
            List of Chunk objects ready for embedding
        """
        logger.info(f"Starting chunking for document {document_id}")

        all_chunks: list[Chunk] = []
        tables = tables or []

        # Build page text lookup for efficient access
        page_texts = {p.page_num: p.text for p in pages}

        # Create table lookup by page
        tables_by_page: dict[int, list[Table]] = {}
        for table in tables:
            if table.page_num not in tables_by_page:
                tables_by_page[table.page_num] = []
            tables_by_page[table.page_num].append(table)

        # Process each root section
        for section in sections.root_sections:
            section_chunks = self._chunk_section(
                document_id=document_id,
                section=section,
                page_texts=page_texts,
                tables_by_page=tables_by_page,
            )
            all_chunks.extend(section_chunks)

            # Recursively process subsections
            for subsection in section.subsections:
                subsection_chunks = self._chunk_section(
                    document_id=document_id,
                    section=subsection,
                    page_texts=page_texts,
                    tables_by_page=tables_by_page,
                )
                all_chunks.extend(subsection_chunks)

        # Link chunks (set preceding/following IDs)
        self._link_chunks(all_chunks)

        logger.info(
            f"Chunking complete: {len(all_chunks)} chunks created for document {document_id}"
        )

        return all_chunks

    def _chunk_section(
        self,
        document_id: str,
        section: Section,
        page_texts: dict[int, str],
        tables_by_page: dict[int, list[Table]],
    ) -> list[Chunk]:
        """
        Chunk a single section into smaller pieces.

        Args:
            document_id: Document identifier
            section: Section to chunk
            page_texts: Mapping of page numbers to text content
            tables_by_page: Mapping of page numbers to tables

        Returns:
            List of chunks for this section
        """
        chunks: list[Chunk] = []

        # Get text content for this section's page range
        section_text = section.content
        if not section_text:
            # Fallback: concatenate page texts
            section_text = ""
            for page_num in range(section.start_page, section.end_page + 1):
                if page_num in page_texts:
                    section_text += page_texts[page_num] + "\n\n"

        if not section_text.strip():
            logger.debug(f"Section '{section.title}' has no text content, skipping")
            return chunks

        # First, handle any tables in this section's pages
        for page_num in range(section.start_page, section.end_page + 1):
            if page_num in tables_by_page:
                for table in tables_by_page[page_num]:
                    table_chunk = self._create_table_chunk(
                        document_id=document_id,
                        section=section,
                        table=table,
                    )
                    if table_chunk:
                        chunks.append(table_chunk)

        # Split narrative text into paragraphs
        paragraphs = self._split_into_paragraphs(section_text)

        # Group paragraphs into chunks
        narrative_chunks = self._create_narrative_chunks(
            document_id=document_id,
            section=section,
            paragraphs=paragraphs,
        )
        chunks.extend(narrative_chunks)

        return chunks

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """
        Split text into paragraphs.

        Args:
            text: Text to split

        Returns:
            List of paragraph strings
        """
        # Split on double newlines (paragraph separator)
        raw_paragraphs = re.split(r"\n\s*\n", text)

        # Clean and filter empty paragraphs
        paragraphs = []
        for para in raw_paragraphs:
            cleaned = para.strip()
            if cleaned:
                paragraphs.append(cleaned)

        return paragraphs

    def _create_narrative_chunks(
        self,
        document_id: str,
        section: Section,
        paragraphs: list[str],
    ) -> list[Chunk]:
        """
        Create chunks from narrative paragraphs.

        Strategy:
        1. Accumulate paragraphs until target size reached
        2. If a single paragraph exceeds max size, split it
        3. Add overlap from previous chunk

        Args:
            document_id: Document identifier
            section: Parent section
            paragraphs: List of paragraph texts

        Returns:
            List of narrative chunks
        """
        chunks: list[Chunk] = []
        current_text = ""
        current_start_char = 0
        char_position = 0

        target_chars = self.config.to_chars(self.config.target_chunk_size)
        min_chars = self.config.to_chars(self.config.min_chunk_size)
        max_chars = self.config.to_chars(self.config.max_chunk_size)
        overlap_chars = self.config.to_chars(self.config.chunk_overlap)

        for i, para in enumerate(paragraphs):
            para_len = len(para)

            # Check if single paragraph exceeds max size
            if para_len > max_chars:
                # First, flush any accumulated text
                if current_text.strip():
                    chunk = self._create_chunk(
                        document_id=document_id,
                        section=section,
                        text=current_text,
                        chunk_type=ChunkType.NARRATIVE,
                        start_char=current_start_char,
                    )
                    chunks.append(chunk)
                    current_text = ""

                # Split large paragraph by sentences
                para_chunks = self._split_large_paragraph(
                    document_id=document_id,
                    section=section,
                    paragraph=para,
                    start_char=char_position,
                )
                chunks.extend(para_chunks)

            elif len(current_text) + para_len + 2 > target_chars and current_text.strip():
                # Current text + new paragraph exceeds target, flush current
                chunk = self._create_chunk(
                    document_id=document_id,
                    section=section,
                    text=current_text,
                    chunk_type=ChunkType.NARRATIVE,
                    start_char=current_start_char,
                )
                chunks.append(chunk)

                # Start new chunk with overlap from end of previous
                overlap_text = self._get_overlap_text(current_text, overlap_chars)
                current_text = overlap_text + "\n\n" + para if overlap_text else para
                current_start_char = char_position - len(overlap_text) if overlap_text else char_position

            else:
                # Accumulate paragraph
                if current_text:
                    current_text += "\n\n" + para
                else:
                    current_text = para
                    current_start_char = char_position

            char_position += para_len + 2  # +2 for paragraph separator

        # Flush remaining text
        if current_text.strip():
            # Check if remaining text is too small and can be merged with last chunk
            if len(current_text) < min_chars and chunks:
                # Merge with last chunk if it won't exceed max
                last_chunk = chunks[-1]
                if len(last_chunk.text) + len(current_text) + 2 <= max_chars:
                    chunks[-1] = self._create_chunk(
                        document_id=document_id,
                        section=section,
                        text=last_chunk.text + "\n\n" + current_text,
                        chunk_type=ChunkType.NARRATIVE,
                        start_char=last_chunk.start_char,
                    )
                else:
                    # Create final chunk even if small
                    chunk = self._create_chunk(
                        document_id=document_id,
                        section=section,
                        text=current_text,
                        chunk_type=ChunkType.NARRATIVE,
                        start_char=current_start_char,
                    )
                    chunks.append(chunk)
            else:
                chunk = self._create_chunk(
                    document_id=document_id,
                    section=section,
                    text=current_text,
                    chunk_type=ChunkType.NARRATIVE,
                    start_char=current_start_char,
                )
                chunks.append(chunk)

        return chunks

    def _split_large_paragraph(
        self,
        document_id: str,
        section: Section,
        paragraph: str,
        start_char: int,
    ) -> list[Chunk]:
        """
        Split a large paragraph that exceeds max chunk size.

        Splits by sentences while respecting chunk size limits.

        Args:
            document_id: Document identifier
            section: Parent section
            paragraph: Large paragraph to split
            start_char: Starting character offset

        Returns:
            List of chunks from the split paragraph
        """
        chunks: list[Chunk] = []

        # Split into sentences
        sentences = self._split_into_sentences(paragraph)

        target_chars = self.config.to_chars(self.config.target_chunk_size)
        overlap_chars = self.config.to_chars(self.config.chunk_overlap)

        current_text = ""
        current_start = start_char

        for sentence in sentences:
            sentence_len = len(sentence)

            if len(current_text) + sentence_len + 1 > target_chars and current_text.strip():
                # Create chunk and start new one
                chunk = self._create_chunk(
                    document_id=document_id,
                    section=section,
                    text=current_text.strip(),
                    chunk_type=ChunkType.NARRATIVE,
                    start_char=current_start,
                )
                chunks.append(chunk)

                # Start new with overlap
                overlap_text = self._get_overlap_text(current_text, overlap_chars)
                current_text = overlap_text + " " + sentence if overlap_text else sentence
                current_start = current_start + len(chunk.text) - len(overlap_text)
            else:
                if current_text:
                    current_text += " " + sentence
                else:
                    current_text = sentence

        # Flush remaining
        if current_text.strip():
            chunk = self._create_chunk(
                document_id=document_id,
                section=section,
                text=current_text.strip(),
                chunk_type=ChunkType.NARRATIVE,
                start_char=current_start,
            )
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Handles common sentence terminators including Hindi (।).

        Args:
            text: Text to split

        Returns:
            List of sentence strings
        """
        # Pattern matches sentence terminators followed by space or end of string
        pattern = r"(?<=[.!?।])\s+"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_text(self, text: str, overlap_chars: int) -> str:
        """
        Get the last N characters of text for overlap.

        Tries to break at word boundary.

        Args:
            text: Source text
            overlap_chars: Number of characters for overlap

        Returns:
            Overlap text string
        """
        if len(text) <= overlap_chars:
            return text

        overlap = text[-overlap_chars:]

        # Try to start at word boundary
        space_idx = overlap.find(" ")
        if space_idx > 0 and space_idx < len(overlap) // 2:
            overlap = overlap[space_idx + 1:]

        return overlap

    def _create_table_chunk(
        self,
        document_id: str,
        section: Section,
        table: Table,
    ) -> Optional[Chunk]:
        """
        Create a chunk from a table.

        Small tables are kept whole. Large tables get a summary.

        Args:
            document_id: Document identifier
            section: Parent section
            table: Table to convert to chunk

        Returns:
            Table chunk or None if table has no content
        """
        table_text = self._table_to_text(table)
        if not table_text.strip():
            return None

        token_count = len(table_text) // 4
        max_table_tokens = self.config.max_table_tokens

        # Decide whether to keep whole or summarize
        if token_count <= max_table_tokens and self.config.keep_small_tables_whole:
            # Keep table as-is
            chunk_text = table_text
            chunk_type = ChunkType.TABLE
        else:
            # Create summary for large tables
            chunk_text = self._summarize_table(table)
            chunk_type = ChunkType.TABLE

        return self._create_chunk(
            document_id=document_id,
            section=section,
            text=chunk_text,
            chunk_type=chunk_type,
            start_char=0,
            table_id=table.table_id,
            has_table=True,
            metadata={
                "table_type": table.table_type.value,
                "row_count": table.row_count,
                "col_count": table.col_count,
                "extraction_method": table.extraction_method,
            },
        )

    def _table_to_text(self, table: Table) -> str:
        """
        Convert a table to text representation.

        Creates a markdown-style table format.

        Args:
            table: Table to convert

        Returns:
            Text representation of the table
        """
        lines = []

        # Add headers if present
        if table.headers:
            lines.append("| " + " | ".join(table.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(table.headers)) + " |")

        # Add rows
        for row in table.rows:
            # Ensure row has consistent number of cells
            cells = [str(cell) if cell else "" for cell in row]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def _summarize_table(self, table: Table) -> str:
        """
        Create a text summary of a large table.

        Includes table type, dimensions, and sample content.

        Args:
            table: Table to summarize

        Returns:
            Summary text
        """
        summary_parts = [
            f"[TABLE: {table.table_type.value}]",
            f"Rows: {table.row_count}, Columns: {table.col_count}",
        ]

        # Add headers
        if table.headers:
            summary_parts.append(f"Headers: {', '.join(table.headers[:5])}")
            if len(table.headers) > 5:
                summary_parts[-1] += f" (+{len(table.headers) - 5} more)"

        # Add first few rows as sample
        sample_rows = min(3, len(table.rows))
        if sample_rows > 0:
            summary_parts.append("Sample data:")
            for row in table.rows[:sample_rows]:
                row_text = " | ".join(str(cell)[:30] if cell else "" for cell in row[:5])
                summary_parts.append(f"  {row_text}")

        return "\n".join(summary_parts)

    def _create_chunk(
        self,
        document_id: str,
        section: Section,
        text: str,
        chunk_type: ChunkType,
        start_char: int,
        table_id: Optional[str] = None,
        has_table: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Chunk:
        """
        Create a Chunk object with generated ID and metadata.

        Args:
            document_id: Document identifier
            section: Parent section
            text: Chunk text content
            chunk_type: Type of chunk content
            start_char: Starting character offset
            table_id: Associated table ID if applicable
            has_table: Whether chunk contains table
            metadata: Additional metadata

        Returns:
            Chunk object
        """
        # Generate deterministic chunk ID based on content
        content_hash = hashlib.md5(
            f"{document_id}:{section.section_id}:{start_char}:{text[:100]}".encode()
        ).hexdigest()[:12]
        chunk_id = f"{document_id}_chunk_{content_hash}"

        return Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            chunk_type=chunk_type,
            section_name=section.title,
            section_type=section.section_type.value,
            page_start=section.start_page,
            page_end=section.end_page,
            start_char=start_char,
            end_char=start_char + len(text),
            has_table=has_table,
            table_id=table_id,
            metadata=metadata or {},
        )

    def _link_chunks(self, chunks: list[Chunk]) -> None:
        """
        Link chunks with preceding/following IDs.

        Modifies chunks in place to set the chain.

        Args:
            chunks: List of chunks to link
        """
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.preceding_chunk_id = chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                chunk.following_chunk_id = chunks[i + 1].chunk_id

    def chunk_text(
        self,
        document_id: str,
        text: str,
        section_name: str = "document",
        section_type: str = "other",
        page_start: int = 1,
        page_end: int = 1,
    ) -> list[Chunk]:
        """
        Chunk raw text without section structure.

        Simplified chunking for cases where full section tree is not available.

        Args:
            document_id: Document identifier
            text: Text to chunk
            section_name: Name to use for the section
            section_type: Type to use for the section
            page_start: Starting page number
            page_end: Ending page number

        Returns:
            List of chunks
        """
        from rhp_analyzer.ingestion.section_mapper import SectionType

        # Create a dummy section
        dummy_section = Section(
            section_id=f"{document_id}_section_0",
            section_type=SectionType(section_type) if section_type in [e.value for e in SectionType] else SectionType.OTHER,
            title=section_name,
            level=1,
            start_page=page_start,
            end_page=page_end,
            content=text,
        )

        paragraphs = self._split_into_paragraphs(text)
        chunks = self._create_narrative_chunks(
            document_id=document_id,
            section=dummy_section,
            paragraphs=paragraphs,
        )

        self._link_chunks(chunks)
        return chunks


# Convenience function for simple use cases
def chunk_text(
    text: str,
    document_id: str = "doc",
    chunk_size: int = 750,
    overlap: int = 100,
) -> list[Chunk]:
    """
    Simple function to chunk text with default settings.

    Args:
        text: Text to chunk
        document_id: Document identifier
        chunk_size: Target tokens per chunk
        overlap: Overlap tokens between chunks

    Returns:
        List of chunks
    """
    config = ChunkingConfig(
        target_chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunker = SemanticChunker(config)
    return chunker.chunk_text(document_id, text)
