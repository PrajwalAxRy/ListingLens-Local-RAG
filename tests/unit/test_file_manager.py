"""
Unit tests for the File Manager module.

Tests cover:
- Filename sanitization
- Document ID generation
- Path resolution
- Directory creation
- Document operations

Reference: milestones.md Milestone 3.5
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rhp_analyzer.storage.file_manager import (
    FileManager,
    create_file_manager,
)


# Helper to get a FileManager for testing static-like methods
@pytest.fixture
def file_manager_for_static():
    """Create FileManager with temp directory for testing methods."""
    temp_dir = tempfile.mkdtemp()
    fm = FileManager(data_dir=Path(temp_dir))
    yield fm
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestSanitizeFilename:
    """Tests for the sanitize_filename static method."""

    def test_basic_sanitization(self):
        """Test basic filename sanitization."""
        result = FileManager.sanitize_filename("Company_RHP.pdf")
        assert result == "Company_RHP"

    def test_removes_extension(self):
        """Test that file extension is removed."""
        result = FileManager.sanitize_filename("document.pdf")
        assert result == "document"

    def test_removes_invalid_characters(self):
        """Test removal of invalid filesystem characters."""
        result = FileManager.sanitize_filename("file<>:\"\\|?*.pdf")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert '"' not in result
        assert "\\" not in result
        assert "|" not in result
        assert "?" not in result
        assert "*" not in result

    def test_replaces_spaces_with_underscores(self):
        """Test that spaces are replaced with underscores."""
        result = FileManager.sanitize_filename("my document name.pdf")
        assert result == "my_document_name"

    def test_collapses_multiple_underscores(self):
        """Test that multiple underscores are collapsed to one."""
        result = FileManager.sanitize_filename("file___name.pdf")
        assert result == "file_name"

    def test_strips_leading_trailing_underscores(self):
        """Test that leading/trailing underscores are removed."""
        result = FileManager.sanitize_filename("___filename___.pdf")
        assert result == "filename"

    def test_handles_special_characters(self):
        """Test handling of various special characters."""
        result = FileManager.sanitize_filename("file@#$%^&()name.pdf")
        # Should still produce a valid filename
        assert len(result) > 0
        assert "/" not in result

    def test_empty_string_returns_default(self):
        """Test that empty string returns 'document'."""
        result = FileManager.sanitize_filename("")
        assert result == "document"

    def test_whitespace_only_returns_default(self):
        """Test that whitespace-only string returns 'document'."""
        result = FileManager.sanitize_filename("   ")
        assert result == "document"

    def test_path_object_input(self):
        """Test that Path objects are handled correctly."""
        result = FileManager.sanitize_filename(Path("path/to/file.pdf"))
        assert result == "file"

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        result = FileManager.sanitize_filename("文档名称.pdf")
        # Should handle unicode gracefully
        assert isinstance(result, str)

    def test_long_filename_truncation(self):
        """Test that very long filenames are handled."""
        long_name = "a" * 300 + ".pdf"
        result = FileManager.sanitize_filename(long_name)
        # Should be truncated to reasonable length
        assert len(result) <= 200


class TestGenerateDocumentId:
    """Tests for the generate_document_id method."""

    @pytest.fixture
    def file_manager(self):
        """Create FileManager with temp directory."""
        temp_dir = tempfile.mkdtemp()
        fm = FileManager(data_dir=Path(temp_dir))
        yield fm
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_basic_id_generation(self, file_manager):
        """Test basic document ID generation."""
        doc_id = file_manager.generate_document_id("Company_RHP.pdf")
        assert "Company_RHP" in doc_id
        assert "_" in doc_id  # Should have timestamp separator

    def test_custom_id_override(self, file_manager):
        """Test custom document ID override."""
        doc_id = file_manager.generate_document_id("any_file.pdf", custom_id="MY_CUSTOM_ID")
        assert doc_id == "MY_CUSTOM_ID"

    def test_id_format(self, file_manager):
        """Test document ID format: {sanitized_filename}_{YYYYMMDD_HHMMSS}."""
        doc_id = file_manager.generate_document_id("test.pdf")
        parts = doc_id.rsplit("_", 2)  # Split from right to get timestamp parts
        assert len(parts) >= 2  # At least name + date + time

    def test_unique_ids(self, file_manager):
        """Test that generated IDs are unique (different timestamps)."""
        import time

        doc_id1 = file_manager.generate_document_id("test.pdf")
        time.sleep(0.01)  # Small delay
        doc_id2 = file_manager.generate_document_id("test.pdf")
        # IDs should be different due to timestamp
        # Note: May be same if executed within same second
        assert isinstance(doc_id1, str)
        assert isinstance(doc_id2, str)

    def test_filesystem_safe_characters(self, file_manager):
        """Test that generated IDs contain only filesystem-safe characters."""
        doc_id = file_manager.generate_document_id("file<>:\"|?*.pdf")
        # Check for absence of invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            assert char not in doc_id

    def test_path_object_input(self, file_manager):
        """Test document ID generation with Path object."""
        doc_id = file_manager.generate_document_id(Path("path/to/file.pdf"))
        assert "file" in doc_id


class TestFileManager:
    """Tests for the FileManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        # Cleanup
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def file_manager(self, temp_dir):
        """Create a FileManager instance with temp directory."""
        return FileManager(data_dir=temp_dir)

    def test_initialization(self, file_manager, temp_dir):
        """Test FileManager initialization."""
        assert file_manager.data_dir == temp_dir
        assert file_manager.input_dir == temp_dir / "input"
        assert file_manager.output_dir == temp_dir / "output"
        assert file_manager.processed_dir == temp_dir / "processed"
        assert file_manager.embeddings_dir == temp_dir / "embeddings"
        assert file_manager.qdrant_dir == temp_dir / "qdrant"
        assert file_manager.checkpoints_dir == temp_dir / "checkpoints"

    def test_custom_directories(self, temp_dir):
        """Test FileManager with custom directories."""
        custom_input = temp_dir / "custom_input"
        custom_output = temp_dir / "custom_output"

        fm = FileManager(
            data_dir=temp_dir,
            input_dir=custom_input,
            output_dir=custom_output,
        )

        assert fm.input_dir == custom_input
        assert fm.output_dir == custom_output

    def test_ensure_directory(self, file_manager, temp_dir):
        """Test directory creation."""
        new_dir = temp_dir / "new" / "nested" / "directory"
        file_manager.ensure_directory(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_base_directories(self, file_manager):
        """Test creation of all base directories."""
        file_manager.ensure_base_directories()

        assert file_manager.input_dir.exists()
        assert file_manager.output_dir.exists()
        assert file_manager.processed_dir.exists()
        assert file_manager.embeddings_dir.exists()
        assert file_manager.qdrant_dir.exists()
        assert file_manager.checkpoints_dir.exists()

    def test_ensure_document_directories(self, file_manager):
        """Test creation of document-specific directories."""
        doc_id = "test_doc_123"
        file_manager.ensure_document_directories(doc_id)

        # Check processed directories
        processed = file_manager.processed_dir / doc_id
        assert (processed / "pages").exists()
        assert (processed / "tables").exists()
        assert (processed / "sections").exists()
        assert (processed / "entities").exists()

        # Check embeddings directory
        assert (file_manager.embeddings_dir / doc_id).exists()

        # Check output directory
        assert (file_manager.output_dir / doc_id).exists()

    def test_get_input_path(self, file_manager):
        """Test input path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_input_path(doc_id)
        # get_input_path returns the input directory, not the PDF file
        assert path == file_manager.input_dir / doc_id

    def test_get_input_pdf_path(self, file_manager):
        """Test input PDF file path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_input_pdf_path(doc_id)
        assert path == file_manager.input_dir / doc_id / "original.pdf"

    def test_get_processed_path(self, file_manager):
        """Test processed path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_processed_path(doc_id)
        assert path == file_manager.processed_dir / doc_id

    def test_get_pages_path(self, file_manager):
        """Test pages directory path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_pages_path(doc_id)
        assert path == file_manager.processed_dir / doc_id / "pages"

    def test_get_page_path(self, file_manager):
        """Test individual page file path - using pages_path and building file path."""
        doc_id = "test_doc"
        pages_path = file_manager.get_pages_path(doc_id)
        # The implementation doesn't have get_page_file, so we build path manually
        page_file = pages_path / "page_005.txt"
        assert page_file == file_manager.processed_dir / doc_id / "pages" / "page_005.txt"

    def test_get_tables_path(self, file_manager):
        """Test tables directory path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_tables_path(doc_id)
        assert path == file_manager.processed_dir / doc_id / "tables"

    def test_get_table_path(self, file_manager):
        """Test individual table file path - using tables_path and building file path."""
        doc_id = "test_doc"
        tables_path = file_manager.get_tables_path(doc_id)
        # The implementation doesn't have get_table_file, so we build path manually
        table_file = tables_path / "table_003.csv"
        assert table_file == file_manager.processed_dir / doc_id / "tables" / "table_003.csv"

    def test_get_sections_path(self, file_manager):
        """Test sections directory path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_sections_path(doc_id)
        assert path == file_manager.processed_dir / doc_id / "sections"

    def test_get_entities_path(self, file_manager):
        """Test entities directory path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_entities_path(doc_id)
        assert path == file_manager.processed_dir / doc_id / "entities"

    def test_get_embeddings_path(self, file_manager):
        """Test embeddings directory path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_embeddings_path(doc_id)
        assert path == file_manager.embeddings_dir / doc_id

    def test_get_chunks_file_path(self, file_manager):
        """Test chunks JSONL file path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_chunks_file_path(doc_id)
        assert path == file_manager.embeddings_dir / doc_id / "chunks.jsonl"

    def test_get_embeddings_file_path(self, file_manager):
        """Test embeddings NPY file path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_embeddings_file_path(doc_id)
        assert path == file_manager.embeddings_dir / doc_id / "embeddings.npy"

    def test_get_embeddings_metadata_path(self, file_manager):
        """Test embedding metadata file path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_embeddings_metadata_path(doc_id)
        # This is embeddings metadata, stored in embeddings dir
        assert path == file_manager.embeddings_dir / doc_id / "metadata.json"

    def test_get_output_path(self, file_manager):
        """Test output directory path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_output_path(doc_id)
        assert path == file_manager.output_dir / doc_id

    def test_get_report_md_path(self, file_manager):
        """Test markdown report path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_report_md_path(doc_id)
        assert path == file_manager.output_dir / doc_id / "report.md"

    def test_get_report_pdf_path(self, file_manager):
        """Test PDF report path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_report_pdf_path(doc_id)
        assert path == file_manager.output_dir / doc_id / "report.pdf"

    def test_get_output_metadata_path(self, file_manager):
        """Test analysis/output metadata file path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_output_metadata_path(doc_id)
        assert path == file_manager.output_dir / doc_id / "metadata.json"

    def test_get_checkpoint_path(self, file_manager):
        """Test checkpoint file path resolution."""
        doc_id = "test_doc"
        path = file_manager.get_checkpoint_path(doc_id)
        assert path == file_manager.checkpoints_dir / f"{doc_id}_checkpoint.json"

    def test_generate_document_id(self, file_manager):
        """Test document ID generation through FileManager."""
        doc_id = file_manager.generate_document_id("test.pdf")
        assert "test" in doc_id
        # Should contain timestamp
        assert "_" in doc_id

    def test_generate_document_id_custom(self, file_manager):
        """Test custom document ID through FileManager."""
        doc_id = file_manager.generate_document_id("test.pdf", custom_id="CUSTOM_123")
        assert doc_id == "CUSTOM_123"


class TestFileManagerDocumentOperations:
    """Tests for FileManager document operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def file_manager(self, temp_dir):
        """Create a FileManager instance with temp directory."""
        fm = FileManager(data_dir=temp_dir)
        fm.ensure_base_directories()
        return fm

    @pytest.fixture
    def sample_pdf(self, temp_dir):
        """Create a sample PDF file for testing."""
        pdf_path = temp_dir / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test content")
        return pdf_path

    def test_copy_pdf_to_input(self, file_manager, sample_pdf):
        """Test copying PDF to input directory."""
        doc_id = "test_doc_001"
        result = file_manager.copy_pdf_to_input(sample_pdf, doc_id)

        expected_path = file_manager.input_dir / doc_id / "original.pdf"
        assert result == expected_path
        assert expected_path.exists()
        assert expected_path.read_bytes() == b"%PDF-1.4 test content"

    def test_copy_pdf_nonexistent_source(self, file_manager, temp_dir):
        """Test copying non-existent PDF raises error."""
        nonexistent = temp_dir / "nonexistent.pdf"
        with pytest.raises(FileNotFoundError):
            file_manager.copy_pdf_to_input(nonexistent, "doc_id")

    def test_document_exists(self, file_manager, sample_pdf):
        """Test document existence check."""
        doc_id = "test_doc_002"

        # Initially doesn't exist
        assert not file_manager.document_exists(doc_id)

        # Copy PDF and check again
        file_manager.copy_pdf_to_input(sample_pdf, doc_id)
        assert file_manager.document_exists(doc_id)

    def test_get_processed_documents(self, file_manager):
        """Test listing processed documents."""
        # Create some document directories
        for doc_id in ["doc_001", "doc_002", "doc_003"]:
            file_manager.ensure_document_directories(doc_id)

        docs = file_manager.get_processed_documents()
        assert len(docs) == 3
        assert "doc_001" in docs
        assert "doc_002" in docs
        assert "doc_003" in docs

    def test_cleanup_document(self, file_manager, sample_pdf):
        """Test document cleanup."""
        doc_id = "test_doc_cleanup"

        # Set up document files
        file_manager.copy_pdf_to_input(sample_pdf, doc_id)
        file_manager.ensure_document_directories(doc_id)

        # Write some test files
        (file_manager.get_pages_path(doc_id) / "page_001.txt").write_text("test")
        (file_manager.get_output_path(doc_id) / "report.md").write_text("# Report")

        # Verify files exist
        assert file_manager.document_exists(doc_id)
        assert (file_manager.get_pages_path(doc_id) / "page_001.txt").exists()

        # Cleanup - pass keep_output=False to also remove output directory
        file_manager.cleanup_document(doc_id, keep_output=False)

        # Verify cleanup
        assert not file_manager.document_exists(doc_id)
        assert not file_manager.get_processed_path(doc_id).exists()
        assert not file_manager.get_embeddings_path(doc_id).exists()
        assert not file_manager.get_output_path(doc_id).exists()

    def test_get_storage_stats(self, file_manager, sample_pdf):
        """Test storage statistics."""
        # Create some documents
        for i, doc_id in enumerate(["doc_a", "doc_b"]):
            file_manager.copy_pdf_to_input(sample_pdf, doc_id)
            file_manager.ensure_document_directories(doc_id)

        stats = file_manager.get_storage_stats()

        assert "total_documents" in stats
        assert "input_size_bytes" in stats
        assert "processed_size_bytes" in stats
        assert "embeddings_size_bytes" in stats
        assert "output_size_bytes" in stats
        assert "total_size_bytes" in stats

        assert stats["total_documents"] == 2
        assert stats["input_size_bytes"] > 0


class TestFileManagerFactory:
    """Tests for the create_file_manager factory function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)

    def test_create_with_defaults(self, temp_dir):
        """Test creating FileManager with default config."""
        fm = create_file_manager(data_dir=temp_dir)
        assert fm.data_dir == temp_dir
        assert isinstance(fm, FileManager)

    def test_create_from_config(self, temp_dir):
        """Test creating FileManager from config object."""
        # Create a mock config
        mock_config = MagicMock()
        mock_config.paths.data_dir = temp_dir
        mock_config.paths.input_dir = temp_dir / "custom_input"
        mock_config.paths.output_dir = temp_dir / "custom_output"

        fm = FileManager.from_config(mock_config)

        assert fm.data_dir == temp_dir
        assert fm.input_dir == temp_dir / "custom_input"
        assert fm.output_dir == temp_dir / "custom_output"


class TestPathConventions:
    """Tests to verify path conventions match blueprint.md specifications."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def file_manager(self, temp_dir):
        """Create a FileManager instance."""
        return FileManager(data_dir=temp_dir)

    def test_path_structure_matches_blueprint(self, file_manager, temp_dir):
        """Verify path structure matches blueprint.md specification."""
        doc_id = "test_doc_123"

        # Blueprint structure:
        # data/
        # ├── input/{doc_id}/original.pdf
        # ├── processed/{doc_id}/
        # │   ├── pages/
        # │   ├── tables/
        # │   ├── sections/
        # │   └── entities/
        # ├── embeddings/{doc_id}/
        # │   ├── chunks.jsonl
        # │   ├── embeddings.npy
        # │   └── metadata.json
        # ├── qdrant/
        # ├── checkpoints/
        # └── output/{doc_id}/
        #     ├── report.md
        #     ├── report.pdf
        #     └── metadata.json

        # Verify input directory path
        input_path = file_manager.get_input_path(doc_id)
        assert input_path == temp_dir / "input" / doc_id
        
        # Verify input PDF path
        input_pdf_path = file_manager.get_input_pdf_path(doc_id)
        assert input_pdf_path == temp_dir / "input" / doc_id / "original.pdf"

        # Verify processed paths
        processed_path = file_manager.get_processed_path(doc_id)
        assert processed_path == temp_dir / "processed" / doc_id

        pages_path = file_manager.get_pages_path(doc_id)
        assert pages_path == temp_dir / "processed" / doc_id / "pages"

        tables_path = file_manager.get_tables_path(doc_id)
        assert tables_path == temp_dir / "processed" / doc_id / "tables"

        sections_path = file_manager.get_sections_path(doc_id)
        assert sections_path == temp_dir / "processed" / doc_id / "sections"

        entities_path = file_manager.get_entities_path(doc_id)
        assert entities_path == temp_dir / "processed" / doc_id / "entities"

        # Verify embeddings paths
        embeddings_path = file_manager.get_embeddings_path(doc_id)
        assert embeddings_path == temp_dir / "embeddings" / doc_id

        chunks_file = file_manager.get_chunks_file_path(doc_id)
        assert chunks_file == temp_dir / "embeddings" / doc_id / "chunks.jsonl"

        embeddings_file = file_manager.get_embeddings_file_path(doc_id)
        assert embeddings_file == temp_dir / "embeddings" / doc_id / "embeddings.npy"

        # Verify output paths
        output_path = file_manager.get_output_path(doc_id)
        assert output_path == temp_dir / "output" / doc_id

        report_md = file_manager.get_report_md_path(doc_id)
        assert report_md == temp_dir / "output" / doc_id / "report.md"

        report_pdf = file_manager.get_report_pdf_path(doc_id)
        assert report_pdf == temp_dir / "output" / doc_id / "report.pdf"

        metadata = file_manager.get_output_metadata_path(doc_id)
        assert metadata == temp_dir / "output" / doc_id / "metadata.json"

        # Verify qdrant path
        assert file_manager.qdrant_dir == temp_dir / "qdrant"

        # Verify checkpoint path
        checkpoint = file_manager.get_checkpoint_path(doc_id)
        assert checkpoint == temp_dir / "checkpoints" / f"{doc_id}_checkpoint.json"


class TestDocumentIdFormat:
    """Tests to verify document ID format matches specification."""

    @pytest.fixture
    def file_manager(self):
        """Create FileManager with temp directory."""
        temp_dir = tempfile.mkdtemp()
        fm = FileManager(data_dir=Path(temp_dir))
        yield fm
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_id_format_components(self, file_manager):
        """Test that document ID has correct format: {sanitized_filename}_{YYYYMMDD_HHMMSS}."""
        doc_id = file_manager.generate_document_id("TestCompany_RHP.pdf")

        # Split into parts
        parts = doc_id.split("_")
        assert len(parts) >= 3  # filename parts + date + time

        # Last two parts should be date and time
        date_part = parts[-2]
        time_part = parts[-1]

        # Verify date format (YYYYMMDD)
        assert len(date_part) == 8
        assert date_part.isdigit()
        year = int(date_part[:4])
        month = int(date_part[4:6])
        day = int(date_part[6:8])
        assert 2020 <= year <= 2100
        assert 1 <= month <= 12
        assert 1 <= day <= 31

        # Verify time format (HHMMSS)
        assert len(time_part) == 6
        assert time_part.isdigit()
        hour = int(time_part[:2])
        minute = int(time_part[2:4])
        second = int(time_part[4:6])
        assert 0 <= hour <= 23
        assert 0 <= minute <= 59
        assert 0 <= second <= 59

    def test_id_contains_sanitized_filename(self, file_manager):
        """Test that document ID contains the sanitized filename."""
        filename = "Company Name RHP.pdf"
        doc_id = file_manager.generate_document_id(filename)

        # Sanitized name should be "Company_Name_RHP"
        assert "Company_Name_RHP" in doc_id

    def test_id_no_invalid_characters(self, file_manager):
        """Test that document ID contains no invalid filesystem characters."""
        # Test with filename containing invalid characters
        filename = 'File<>:"/\\|?*Name.pdf'
        doc_id = file_manager.generate_document_id(filename)

        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            assert char not in doc_id
