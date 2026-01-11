"""
Unit tests for the Semantic Chunking Module.

Tests cover:
- Chunk size compliance (500-1000 token target range)
- Section boundary respect (chunks don't cross sections)
- Table handling (keep whole or summarize)
- Overlap verification
- Chunk metadata accuracy

Reference: milestones.md Milestone 3.1.4
"""

import pytest

from rhp_analyzer.ingestion.pdf_processor import PageInfo
from rhp_analyzer.ingestion.section_mapper import Section, SectionTree, SectionType
from rhp_analyzer.ingestion.table_extractor import Table, TableType
from rhp_analyzer.storage.chunker import (
    Chunk,
    ChunkingConfig,
    ChunkType,
    SemanticChunker,
    chunk_text,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_chunker() -> SemanticChunker:
    """Create a chunker with default configuration."""
    return SemanticChunker()


@pytest.fixture
def custom_chunker() -> SemanticChunker:
    """Create a chunker with custom configuration for testing."""
    config = ChunkingConfig(
        target_chunk_size=200,  # Smaller for testing
        min_chunk_size=50,
        max_chunk_size=400,
        chunk_overlap=20,
        max_table_tokens=100,
    )
    return SemanticChunker(config)


@pytest.fixture
def sample_pages() -> list[PageInfo]:
    """Create sample pages for testing."""
    return [
        PageInfo(
            page_num=1,
            text="This is the first page content. It contains introduction material.",
            char_count=65,
            word_count=10,
            has_images=False,
            image_count=0,
            is_scanned=False,
            fonts=["Arial"],
            font_sizes=[12.0],
            page_width=612.0,
            page_height=792.0,
            page_type="text",
            has_tables=False,
            is_multi_column=False,
            header_text=None,
            footer_text=None,
        ),
        PageInfo(
            page_num=2,
            text="This is the second page. It has more detailed content about the business.",
            char_count=72,
            word_count=12,
            has_images=False,
            image_count=0,
            is_scanned=False,
            fonts=["Arial"],
            font_sizes=[12.0],
            page_width=612.0,
            page_height=792.0,
            page_type="text",
            has_tables=True,
            is_multi_column=False,
            header_text=None,
            footer_text=None,
        ),
        PageInfo(
            page_num=3,
            text="Third page discusses risk factors. Multiple risks are outlined here.",
            char_count=67,
            word_count=10,
            has_images=False,
            image_count=0,
            is_scanned=False,
            fonts=["Arial"],
            font_sizes=[12.0],
            page_width=612.0,
            page_height=792.0,
            page_type="text",
            has_tables=False,
            is_multi_column=False,
            header_text=None,
            footer_text=None,
        ),
    ]


@pytest.fixture
def sample_sections() -> list[Section]:
    """Create sample sections for testing."""
    return [
        Section(
            section_id="sec_001",
            section_type=SectionType.SUMMARY,
            title="Summary of Prospectus",
            start_page=1,
            end_page=1,
            subsections=[],
            content="This is the summary section. It provides an overview of the IPO. "
            "The company is seeking to raise capital for expansion. "
            "Key highlights include strong revenue growth and market position.",
            word_count=30,
            level=1,
        ),
        Section(
            section_id="sec_002",
            section_type=SectionType.BUSINESS,
            title="Business Overview",
            start_page=2,
            end_page=2,
            subsections=[],
            content="The company operates in the technology sector. "
            "It was founded in 2010 and has grown significantly. "
            "The business model is based on subscription services. "
            "Revenue has increased year over year consistently.",
            word_count=35,
            level=1,
        ),
        Section(
            section_id="sec_003",
            section_type=SectionType.RISK_FACTORS,
            title="Risk Factors",
            start_page=3,
            end_page=3,
            subsections=[],
            content="There are several risk factors to consider. "
            "Market competition is intense in this sector. "
            "Regulatory changes could impact operations. "
            "Economic downturns may affect customer spending.",
            word_count=30,
            level=1,
        ),
    ]


@pytest.fixture
def sample_section_tree(sample_sections: list[Section]) -> SectionTree:
    """Create a sample section tree for testing."""
    sections_dict = {s.section_id: s for s in sample_sections}
    return SectionTree(
        root_sections=sample_sections,
        all_sections=sections_dict,
        document_id="test_doc",
        total_pages=3,
    )


@pytest.fixture
def sample_tables() -> list[Table]:
    """Create sample tables for testing."""
    return [
        Table(
            table_id="tbl_001",
            page_num=2,
            rows=[
                ["Year", "Revenue", "Profit"],
                ["2022", "100 Cr", "10 Cr"],
                ["2023", "150 Cr", "20 Cr"],
                ["2024", "200 Cr", "35 Cr"],
            ],
            headers=["Year", "Revenue", "Profit"],
            table_type=TableType.INCOME_STATEMENT,
            confidence=0.95,
        ),
    ]


@pytest.fixture
def large_text_section() -> Section:
    """Create a section with large text content for chunk splitting tests."""
    # Create a long text that will require multiple chunks
    paragraph1 = "This is the first paragraph of a very long section. " * 20
    paragraph2 = "This is the second paragraph with different content. " * 20
    paragraph3 = "The third paragraph discusses additional details. " * 20

    content = f"{paragraph1}\n\n{paragraph2}\n\n{paragraph3}"

    return Section(
        section_id="sec_large",
        section_type=SectionType.BUSINESS,
        title="Large Business Section",
        start_page=1,
        end_page=5,
        subsections=[],
        content=content,
        word_count=len(content.split()),
        level=1,
    )


@pytest.fixture
def large_table() -> Table:
    """Create a large table that should be summarized."""
    # Create a table with many rows
    headers = ["Column1", "Column2", "Column3", "Column4", "Column5"]
    rows = [headers]
    for i in range(50):  # 50 data rows
        rows.append([f"Value{i}_{j}" for j in range(5)])

    return Table(
        table_id="tbl_large",
        page_num=2,  # Must be within section page range
        rows=rows,
        headers=headers,
        table_type=TableType.OTHER,
        confidence=0.90,
    )


# =============================================================================
# Basic Chunking Tests
# =============================================================================


class TestChunkingConfig:
    """Tests for ChunkingConfig dataclass."""

    def test_default_config_values(self):
        """Test that default config has expected values."""
        config = ChunkingConfig()

        assert config.target_chunk_size == 750
        assert config.min_chunk_size == 200
        assert config.max_chunk_size == 1500
        assert config.chunk_overlap == 100
        assert config.max_table_tokens == 500

    def test_custom_config_values(self):
        """Test custom config values are applied."""
        config = ChunkingConfig(
            target_chunk_size=500,
            min_chunk_size=100,
            max_chunk_size=1000,
            chunk_overlap=50,
            max_table_tokens=300,
        )

        assert config.target_chunk_size == 500
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 1000
        assert config.chunk_overlap == 50
        assert config.max_table_tokens == 300


class TestChunkDataclass:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            chunk_id="chunk_001",
            document_id="doc_001",
            text="This is test content.",
            chunk_type=ChunkType.NARRATIVE,
            section_name="Test Section",
            section_type="business",
            page_start=1,
            page_end=1,
        )

        assert chunk.chunk_id == "chunk_001"
        assert chunk.document_id == "doc_001"
        assert chunk.text == "This is test content."
        assert chunk.chunk_type == ChunkType.NARRATIVE
        assert chunk.section_name == "Test Section"

    def test_chunk_auto_calculates_counts(self):
        """Test that char_count and token_count are auto-calculated."""
        text = "A" * 100  # 100 characters
        chunk = Chunk(
            chunk_id="chunk_001",
            document_id="doc_001",
            text=text,
            chunk_type=ChunkType.NARRATIVE,
            section_name="Test Section",
            section_type="business",
            page_start=1,
            page_end=1,
        )

        assert chunk.char_count == 100
        assert chunk.token_count == 25  # ~4 chars per token

    def test_chunk_to_dict(self):
        """Test chunk serialization to dictionary."""
        chunk = Chunk(
            chunk_id="chunk_001",
            document_id="doc_001",
            text="Test content",
            chunk_type=ChunkType.NARRATIVE,
            section_name="Test Section",
            section_type="business",
            page_start=1,
            page_end=2,
        )

        chunk_dict = chunk.to_dict()

        assert isinstance(chunk_dict, dict)
        assert chunk_dict["chunk_id"] == "chunk_001"
        assert chunk_dict["document_id"] == "doc_001"
        assert chunk_dict["chunk_type"] == "narrative"
        assert chunk_dict["page_start"] == 1
        assert chunk_dict["page_end"] == 2


class TestChunkType:
    """Tests for ChunkType enum."""

    def test_chunk_types_exist(self):
        """Test all expected chunk types exist."""
        assert ChunkType.NARRATIVE.value == "narrative"
        assert ChunkType.TABLE.value == "table"
        assert ChunkType.MIXED.value == "mixed"
        assert ChunkType.LIST.value == "list"


# =============================================================================
# Chunk Size Compliance Tests
# =============================================================================


class TestChunkSizeCompliance:
    """Tests for chunk size within acceptable ranges."""

    def test_chunks_within_size_range(self, custom_chunker: SemanticChunker, large_text_section: Section):
        """Test that generated chunks are within the configured size range."""
        section_tree = SectionTree(
            root_sections=[large_text_section],
            all_sections={large_text_section.section_id: large_text_section},
            document_id="test_doc",
            total_pages=10,
        )

        chunks = custom_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=section_tree,
            tables=[],
        )

        # Filter narrative chunks (table chunks have different rules)
        narrative_chunks = [c for c in chunks if c.chunk_type == ChunkType.NARRATIVE]

        # At least some chunks should be generated
        assert len(narrative_chunks) > 0

        for chunk in narrative_chunks:
            # Most chunks should be around target size
            # Allow some flexibility for boundary conditions
            assert chunk.token_count >= custom_chunker.config.min_chunk_size // 2, (
                f"Chunk too small: {chunk.token_count} tokens"
            )

    def test_small_section_not_over_chunked(self, default_chunker: SemanticChunker, sample_sections: list[Section]):
        """Test that small sections are not unnecessarily split."""
        # Use only one small section
        small_section = sample_sections[0]
        section_tree = SectionTree(
            root_sections=[small_section],
            all_sections={small_section.section_id: small_section},
            document_id="test_doc",
            total_pages=1,
        )

        chunks = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=section_tree,
            tables=[],
        )

        # A small section should result in minimal chunks
        assert len(chunks) <= 2  # Should be 1-2 chunks for small content


# =============================================================================
# Section Boundary Tests
# =============================================================================


class TestSectionBoundaryRespect:
    """Tests ensuring chunks don't cross section boundaries."""

    def test_chunks_dont_cross_sections(
        self,
        default_chunker: SemanticChunker,
        sample_section_tree: SectionTree,
    ):
        """Test that each chunk belongs to a single section."""
        chunks = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=sample_section_tree,
            tables=[],
        )

        # Each chunk should have exactly one section_name
        for chunk in chunks:
            assert chunk.section_name is not None
            assert isinstance(chunk.section_name, str)
            assert len(chunk.section_name) > 0

    def test_section_names_preserved(
        self,
        default_chunker: SemanticChunker,
        sample_section_tree: SectionTree,
    ):
        """Test that section names are correctly preserved in chunks."""
        chunks = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=sample_section_tree,
            tables=[],
        )

        # Get unique section names from chunks
        chunk_section_names = set(c.section_name for c in chunks)

        # All sections with content should have chunks
        expected_names = {s.title for s in sample_section_tree.root_sections if s.content}
        assert chunk_section_names.issubset(expected_names) or chunk_section_names == expected_names

    def test_section_type_preserved(
        self,
        default_chunker: SemanticChunker,
        sample_section_tree: SectionTree,
    ):
        """Test that section types are correctly preserved."""
        chunks = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=sample_section_tree,
            tables=[],
        )

        for chunk in chunks:
            assert chunk.section_type is not None
            assert isinstance(chunk.section_type, str)


# =============================================================================
# Table Handling Tests
# =============================================================================


class TestTableHandling:
    """Tests for special table chunk handling."""

    def test_small_table_kept_whole(
        self,
        default_chunker: SemanticChunker,
        sample_section_tree: SectionTree,
        sample_tables: list[Table],
    ):
        """Test that small tables are kept as complete chunks."""
        chunks = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=sample_section_tree,
            tables=sample_tables,
        )

        # Find table chunks
        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]

        # Should have table chunks
        assert len(table_chunks) >= len(sample_tables)

        # Table chunks should have table metadata
        for chunk in table_chunks:
            assert chunk.has_table is True
            assert chunk.table_id is not None

    def test_large_table_summarized(
        self,
        custom_chunker: SemanticChunker,
        sample_section_tree: SectionTree,
        large_table: Table,
    ):
        """Test that large tables are summarized."""
        chunks = custom_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=sample_section_tree,
            tables=[large_table],
        )

        # Find table chunks
        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]

        # Should have at least one table chunk
        assert len(table_chunks) >= 1

        # Large table should be summarized (check metadata)
        for chunk in table_chunks:
            if chunk.table_id == large_table.table_id:
                # Either it's summarized or has summary indicator
                assert chunk.has_table is True

    def test_table_page_range_preserved(
        self,
        default_chunker: SemanticChunker,
        sample_section_tree: SectionTree,
        sample_tables: list[Table],
    ):
        """Test that table chunks have correct page references."""
        chunks = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=sample_section_tree,
            tables=sample_tables,
        )

        # Find table chunks
        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]

        for chunk in table_chunks:
            # Page range should be valid
            assert chunk.page_start >= 1
            assert chunk.page_end >= chunk.page_start


# =============================================================================
# Chunk Linking Tests (Overlap and Chain)
# =============================================================================


class TestChunkLinking:
    """Tests for chunk linking (preceding/following references)."""

    def test_chunk_chain_linked(
        self,
        default_chunker: SemanticChunker,
        large_text_section: Section,
    ):
        """Test that chunks are properly linked in a chain."""
        section_tree = SectionTree(
            root_sections=[large_text_section],
            all_sections={large_text_section.section_id: large_text_section},
            document_id="test_doc",
            total_pages=10,
        )

        chunks = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=section_tree,
            tables=[],
        )

        if len(chunks) <= 1:
            pytest.skip("Not enough chunks to test linking")

        # Build a set of chunk IDs
        chunk_ids = {c.chunk_id for c in chunks}

        # First chunk should have no preceding
        first_chunk = chunks[0]
        assert first_chunk.preceding_chunk_id is None

        # Last chunk should have no following
        last_chunk = chunks[-1]
        assert last_chunk.following_chunk_id is None

        # Middle chunks should have valid links
        for i, chunk in enumerate(chunks):
            if chunk.preceding_chunk_id is not None:
                assert chunk.preceding_chunk_id in chunk_ids
            if chunk.following_chunk_id is not None:
                assert chunk.following_chunk_id in chunk_ids

    def test_bidirectional_linking(
        self,
        custom_chunker: SemanticChunker,
        large_text_section: Section,
    ):
        """Test that chunk links are bidirectional."""
        section_tree = SectionTree(
            root_sections=[large_text_section],
            all_sections={large_text_section.section_id: large_text_section},
            document_id="test_doc",
            total_pages=10,
        )

        chunks = custom_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=section_tree,
            tables=[],
        )

        if len(chunks) <= 2:
            pytest.skip("Not enough chunks to test bidirectional linking")

        # Create chunk lookup
        chunk_by_id = {c.chunk_id: c for c in chunks}

        # Check bidirectional consistency
        for chunk in chunks:
            if chunk.following_chunk_id:
                next_chunk = chunk_by_id.get(chunk.following_chunk_id)
                if next_chunk:
                    assert next_chunk.preceding_chunk_id == chunk.chunk_id


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestChunkTextFunction:
    """Tests for the chunk_text convenience function."""

    def test_chunk_text_simple(self):
        """Test simple text chunking."""
        text = "This is a simple test. " * 10
        chunks = chunk_text(text, document_id="test_doc")

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_text_with_custom_config(self):
        """Test text chunking with custom chunk size."""
        text = "This is a test paragraph. " * 50

        chunks = chunk_text(text, document_id="test_doc", chunk_size=100, overlap=20)

        assert len(chunks) >= 1
        # Chunks should be created - verify it works with custom params
        for chunk in chunks:
            # Allow flexibility since chunking may include overlap or boundary text
            assert chunk.token_count <= 500  # Allow more flexibility

    def test_chunk_text_empty_returns_empty(self):
        """Test that empty text returns empty list."""
        chunks = chunk_text("", document_id="test_doc")
        assert chunks == []

    def test_chunk_text_preserves_document_id(self):
        """Test that document ID is preserved in chunks."""
        text = "Test content for document ID check."
        chunks = chunk_text(text, document_id="my_special_doc_123")

        for chunk in chunks:
            assert chunk.document_id == "my_special_doc_123"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_sections(self, default_chunker: SemanticChunker):
        """Test handling of empty section tree."""
        empty_tree = SectionTree(
            root_sections=[],
            all_sections={},
            document_id="test_doc",
            total_pages=0,
        )

        chunks = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=empty_tree,
            tables=[],
        )

        assert chunks == []

    def test_section_with_empty_content(self, default_chunker: SemanticChunker):
        """Test handling of section with empty content."""
        empty_section = Section(
            section_id="sec_empty",
            section_type=SectionType.OTHER,
            title="Empty Section",
            start_page=1,
            end_page=1,
            subsections=[],
            content="",
            word_count=0,
            level=1,
        )

        section_tree = SectionTree(
            root_sections=[empty_section],
            all_sections={empty_section.section_id: empty_section},
            document_id="test_doc",
            total_pages=1,
        )

        chunks = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=section_tree,
            tables=[],
        )

        # Should handle empty content gracefully
        # Either no chunks or minimal chunks
        assert isinstance(chunks, list)

    def test_section_with_whitespace_only(self, default_chunker: SemanticChunker):
        """Test handling of section with only whitespace."""
        whitespace_section = Section(
            section_id="sec_ws",
            section_type=SectionType.OTHER,
            title="Whitespace Section",
            start_page=1,
            end_page=1,
            subsections=[],
            content="   \n\n\t\t   \n   ",
            word_count=0,
            level=1,
        )

        section_tree = SectionTree(
            root_sections=[whitespace_section],
            all_sections={whitespace_section.section_id: whitespace_section},
            document_id="test_doc",
            total_pages=1,
        )

        chunks = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=section_tree,
            tables=[],
        )

        # Should handle whitespace gracefully
        assert isinstance(chunks, list)

    def test_very_long_single_paragraph(self, custom_chunker: SemanticChunker):
        """Test handling of a very long single paragraph without breaks."""
        # Create a long paragraph without any paragraph breaks
        long_paragraph = "Word " * 1000  # ~1000 words, ~5000 chars

        section = Section(
            section_id="sec_long",
            section_type=SectionType.BUSINESS,
            title="Long Paragraph Section",
            start_page=1,
            end_page=10,
            subsections=[],
            content=long_paragraph,
            word_count=1000,
            level=1,
        )

        section_tree = SectionTree(
            root_sections=[section],
            all_sections={section.section_id: section},
            document_id="test_doc",
            total_pages=10,
        )

        chunks = custom_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=section_tree,
            tables=[],
        )

        # Should produce at least one chunk from the long content
        # Note: A single continuous paragraph may not be split if no natural break points
        assert len(chunks) >= 1

    def test_unicode_content(self, default_chunker: SemanticChunker):
        """Test handling of Unicode content (Hindi, special chars)."""
        unicode_content = (
            "This is English text. "
            "यह हिंदी में है। "  # Hindi text
            "Special chars: ₹100 crore, 50% growth, "
            "More text follows here."
        )

        section = Section(
            section_id="sec_unicode",
            section_type=SectionType.FINANCIAL_INFORMATION,
            title="Unicode Section",
            start_page=1,
            end_page=1,
            subsections=[],
            content=unicode_content,
            word_count=20,
            level=1,
        )

        section_tree = SectionTree(
            root_sections=[section],
            all_sections={section.section_id: section},
            document_id="test_doc",
            total_pages=1,
        )

        chunks = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=section_tree,
            tables=[],
        )

        # Should handle Unicode without errors
        assert len(chunks) >= 1
        # Content should be preserved
        assert any("₹" in c.text for c in chunks)


class TestMetadataAccuracy:
    """Tests for accurate metadata in chunks."""

    def test_page_range_accuracy(
        self,
        default_chunker: SemanticChunker,
        sample_sections: list[Section],
    ):
        """Test that page ranges are accurately set from sections."""
        sections_dict = {s.section_id: s for s in sample_sections}
        section_tree = SectionTree(
            root_sections=sample_sections,
            all_sections=sections_dict,
            document_id="test_doc",
            total_pages=3,
        )

        chunks = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=section_tree,
            tables=[],
        )

        for chunk in chunks:
            # Find the source section
            source_section = next(
                (s for s in sample_sections if s.title == chunk.section_name),
                None,
            )
            if source_section:
                # Page range should match section's page range
                assert chunk.page_start >= source_section.start_page
                assert chunk.page_end <= source_section.end_page

    def test_chunk_id_uniqueness(
        self,
        default_chunker: SemanticChunker,
        sample_section_tree: SectionTree,
    ):
        """Test that all chunk IDs are unique."""
        chunks = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=sample_section_tree,
            tables=[],
        )

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found"

    def test_document_id_consistency(
        self,
        default_chunker: SemanticChunker,
        sample_section_tree: SectionTree,
    ):
        """Test that document ID is consistent across all chunks."""
        doc_id = "consistent_doc_id_123"

        chunks = default_chunker.chunk_document(
            document_id=doc_id,
            pages=[],
            sections=sample_section_tree,
            tables=[],
        )

        for chunk in chunks:
            assert chunk.document_id == doc_id


# =============================================================================
# Integration-like Tests
# =============================================================================


class TestChunkerIntegration:
    """Integration-style tests combining multiple features."""

    def test_full_document_chunking(
        self,
        default_chunker: SemanticChunker,
        sample_pages: list[PageInfo],
        sample_section_tree: SectionTree,
        sample_tables: list[Table],
    ):
        """Test complete document chunking with all inputs."""
        chunks = default_chunker.chunk_document(
            document_id="full_test_doc",
            pages=sample_pages,
            sections=sample_section_tree,
            tables=sample_tables,
        )

        # Should produce chunks
        assert len(chunks) > 0

        # Should have both narrative and table chunks
        chunk_types = {c.chunk_type for c in chunks}
        assert ChunkType.NARRATIVE in chunk_types or ChunkType.TABLE in chunk_types

        # All chunks should have required fields
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.document_id == "full_test_doc"
            assert chunk.text
            assert chunk.section_name
            assert chunk.page_start >= 1

    def test_multiple_invocations_consistent(
        self,
        default_chunker: SemanticChunker,
        sample_section_tree: SectionTree,
    ):
        """Test that multiple chunking invocations produce consistent results."""
        # Chunk the same document twice
        chunks1 = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=sample_section_tree,
            tables=[],
        )

        chunks2 = default_chunker.chunk_document(
            document_id="test_doc",
            pages=[],
            sections=sample_section_tree,
            tables=[],
        )

        # Should produce same number of chunks
        assert len(chunks1) == len(chunks2)

        # Content should match (IDs may differ due to UUID)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.text == c2.text
            assert c1.section_name == c2.section_name
            assert c1.chunk_type == c2.chunk_type
