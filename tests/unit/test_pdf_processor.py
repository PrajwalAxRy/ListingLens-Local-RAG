"""
Unit tests for PDF Processor Module.

Tests cover PDF validation, text extraction, page analysis,
and scanned page detection functionality.
"""

import contextlib
import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rhp_analyzer.ingestion.pdf_processor import (
    PageInfo,
    PDFMetadata,
    PDFProcessingError,
    PDFProcessor,
    PDFValidationError,
)

# ============================================================================
# Helper Functions for Mock Creation
# ============================================================================


def create_mock_page(
    page_num: int, text: str | None = None, has_images: bool = False, width: float = 612, height: float = 792
):
    """Create a properly configured mock fitz page."""
    if text is None:
        text = f"Page {page_num + 1} content. This is test text for the page."

    mock_page = MagicMock()
    mock_page.number = page_num

    # Configure get_text to return proper values based on argument
    def get_text_side_effect(option="text"):
        if option == "dict":
            return {
                "blocks": [
                    {
                        "type": 0,
                        "bbox": (72, 72, 540, 720),
                        "lines": [{"spans": [{"font": "Arial", "size": 12.0, "text": text[:50]}]}],
                    }
                ]
            }
        return text

    mock_page.get_text = MagicMock(side_effect=get_text_side_effect)

    # Configure rect
    mock_rect = MagicMock()
    mock_rect.width = width
    mock_rect.height = height
    mock_page.rect = mock_rect

    # Configure images
    mock_page.get_images = MagicMock(return_value=[(1, 0, 0, 0, 0, "image")] if has_images else [])

    return mock_page


def create_mock_document(
    num_pages: int = 5, is_encrypted: bool = False, metadata: dict | None = None, pages: list | None = None
):
    """Create a properly configured mock fitz document."""
    mock_doc = MagicMock()
    mock_doc.page_count = num_pages
    mock_doc.is_encrypted = is_encrypted
    mock_doc.metadata = metadata or {
        "title": "Test RHP Document",
        "author": "Test Author",
        "subject": "IPO Prospectus",
        "creator": "Test Creator",
    }

    # Create mock pages if not provided
    if pages is None:
        pages = [create_mock_page(i) for i in range(num_pages)]

    # Configure iteration
    mock_doc.__len__ = MagicMock(return_value=num_pages)
    mock_doc.__iter__ = MagicMock(return_value=iter(pages))
    mock_doc.load_page = MagicMock(side_effect=lambda n: pages[n] if 0 <= n < len(pages) else None)
    mock_doc.__getitem__ = MagicMock(side_effect=lambda n: pages[n] if 0 <= n < len(pages) else None)

    # Configure context manager
    mock_doc.__enter__ = MagicMock(return_value=mock_doc)
    mock_doc.__exit__ = MagicMock(return_value=False)
    mock_doc.close = MagicMock()

    return mock_doc


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_invalid_pdf() -> Generator[str, None, None]:
    """Create a temporary invalid PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"Not a valid PDF content")
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_empty_file() -> Generator[str, None, None]:
    """Create a temporary empty file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_valid_pdf_path() -> Generator[str, None, None]:
    """Create a temporary file path that will be mocked as valid PDF."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 test content")
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================================================
# PDF Validation Tests
# ============================================================================


class TestPDFValidation:
    """Test PDF validation functionality."""

    def test_nonexistent_file_raises_error(self):
        """Test that non-existent file raises PDFValidationError."""
        with pytest.raises(PDFValidationError) as exc_info:
            PDFProcessor("/nonexistent/path/to/file.pdf")

        assert "does not exist" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()

    def test_non_pdf_extension_raises_error(self):
        """Test that non-PDF file raises PDFValidationError."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Not a PDF")
            temp_path = f.name

        try:
            with pytest.raises(PDFValidationError) as exc_info:
                PDFProcessor(temp_path)
            assert "pdf" in str(exc_info.value).lower()
        finally:
            os.unlink(temp_path)

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_encrypted_pdf_detection(self, mock_fitz_open, temp_valid_pdf_path):
        """Test that encrypted PDFs are detected."""
        mock_doc = create_mock_document(is_encrypted=True)
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)
        result = processor.validate()

        assert result.is_encrypted is True

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_valid_pdf_validation(self, mock_fitz_open, temp_valid_pdf_path):
        """Test validation of a valid PDF."""
        mock_doc = create_mock_document(num_pages=10)
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)
        result = processor.validate()

        assert result.page_count == 10
        assert result.is_encrypted is False


# ============================================================================
# Page Extraction Tests
# ============================================================================


class TestPageExtraction:
    """Test page extraction functionality."""

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_extract_all_pages(self, mock_fitz_open, temp_valid_pdf_path):
        """Test extracting all pages from PDF."""
        mock_doc = create_mock_document(num_pages=3)
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)
        pages = processor.extract_all_pages()

        assert len(pages) == 3
        assert all(isinstance(p, PageInfo) for p in pages)

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_extract_single_page(self, mock_fitz_open, temp_valid_pdf_path):
        """Test extracting a single page."""
        mock_doc = create_mock_document(num_pages=5)
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)
        pages = processor.extract_page_range(2, 3)  # Gets page 2 (0-indexed), exclusive end

        assert len(pages) == 1

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_extract_page_range(self, mock_fitz_open, temp_valid_pdf_path):
        """Test extracting a range of pages."""
        mock_doc = create_mock_document(num_pages=10)
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)
        pages = processor.extract_page_range(3, 7)  # 0-indexed, exclusive end

        assert len(pages) == 4  # Pages 3, 4, 5, 6 (0-indexed)

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_page_text_extraction(self, mock_fitz_open, temp_valid_pdf_path):
        """Test that page text is extracted correctly."""
        pages = [create_mock_page(0, text="This is specific test content.")]
        mock_doc = create_mock_document(num_pages=1, pages=pages)
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)
        extracted_pages = processor.extract_all_pages()

        assert len(extracted_pages) == 1
        assert "test content" in extracted_pages[0].text.lower()

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_invalid_page_range(self, mock_fitz_open, temp_valid_pdf_path):
        """Test invalid page range returns empty list."""
        mock_doc = create_mock_document(num_pages=5)
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)
        # Range beyond document length returns empty list
        pages = processor.extract_page_range(10, 15)
        assert len(pages) == 0


# ============================================================================
# Page Analysis Tests
# ============================================================================


class TestPageAnalysis:
    """Test page analysis functionality."""

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_page_dimensions(self, mock_fitz_open, temp_valid_pdf_path):
        """Test page dimensions are captured."""
        pages = [create_mock_page(0, width=612, height=792)]
        mock_doc = create_mock_document(num_pages=1, pages=pages)
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)
        extracted_pages = processor.extract_all_pages()

        assert len(extracted_pages) == 1
        assert extracted_pages[0].page_width == 612
        assert extracted_pages[0].page_height == 792

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_page_with_images(self, mock_fitz_open, temp_valid_pdf_path):
        """Test detection of pages with images."""
        pages = [create_mock_page(0, has_images=True)]
        mock_doc = create_mock_document(num_pages=1, pages=pages)
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)
        extracted_pages = processor.extract_all_pages()

        assert len(extracted_pages) == 1
        assert extracted_pages[0].has_images is True

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_word_and_char_count(self, mock_fitz_open, temp_valid_pdf_path):
        """Test word and character counting."""
        text = "This is a test with exactly nine words here."
        pages = [create_mock_page(0, text=text)]
        mock_doc = create_mock_document(num_pages=1, pages=pages)
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)
        extracted_pages = processor.extract_all_pages()

        assert len(extracted_pages) == 1
        assert extracted_pages[0].word_count == 9
        assert extracted_pages[0].char_count == len(text)


# ============================================================================
# Scanned Page Detection Tests
# ============================================================================


class TestScannedPageDetection:
    """Test scanned page detection functionality."""

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_detect_no_scanned_pages(self, mock_fitz_open, temp_valid_pdf_path):
        """Test detection when no pages are scanned."""
        pages = [
            create_mock_page(0, text="Normal text content here " * 50),
            create_mock_page(1, text="More normal text content " * 50),
        ]
        mock_doc = create_mock_document(num_pages=2, pages=pages)
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)
        scanned = processor.detect_scanned_pages()

        assert len(scanned) == 0

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_detect_scanned_pages_with_images_low_text(self, mock_fitz_open, temp_valid_pdf_path):
        """Test detection of scanned pages (images + low text)."""
        pages = [
            create_mock_page(0, text="", has_images=True),  # Scanned: no text + has images
            create_mock_page(1, text="Normal text content " * 100, has_images=False),  # Not scanned
        ]
        mock_doc = create_mock_document(num_pages=2, pages=pages)
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)
        scanned = processor.detect_scanned_pages()

        assert len(scanned) > 0  # At least one page should be detected as scanned


# ============================================================================
# Utility Method Tests
# ============================================================================


class TestUtilityMethods:
    """Test utility methods of PDFProcessor."""

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_get_all_text(self, mock_fitz_open, temp_valid_pdf_path):
        """Test getting all text from document."""
        pages = [
            create_mock_page(0, text="First page text."),
            create_mock_page(1, text="Second page text."),
        ]
        mock_doc = create_mock_document(num_pages=2, pages=pages)
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)
        all_text = processor.get_all_text()

        assert "First page" in all_text
        assert "Second page" in all_text

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_get_processing_summary(self, mock_fitz_open, temp_valid_pdf_path):
        """Test processing summary generation."""
        mock_doc = create_mock_document(num_pages=5)
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)
        processor.extract_all_pages()
        summary = processor.get_processing_summary()

        assert isinstance(summary, dict)
        assert "total_pages" in summary


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling in PDF processor."""

    @patch("rhp_analyzer.ingestion.pdf_processor.fitz.open")
    def test_corrupted_page_handling(self, mock_fitz_open, temp_valid_pdf_path):
        """Test handling of corrupted page data."""
        mock_page = create_mock_page(0)
        mock_page.get_text = MagicMock(side_effect=Exception("Page corrupted"))
        mock_doc = create_mock_document(num_pages=1, pages=[mock_page])
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor(temp_valid_pdf_path)

        # Should either raise error or handle gracefully
        with contextlib.suppress(PDFProcessingError):
            processor.extract_all_pages()
            # If it handles gracefully, pages may be empty or have empty text


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestPageInfoDataclass:
    """Test PageInfo dataclass."""

    def test_page_info_creation(self):
        """Test PageInfo creation with required fields."""
        page_info = PageInfo(
            page_num=0,
            text="Sample text",
            word_count=2,
            char_count=11,
            page_width=612.0,
            page_height=792.0,
            has_images=False,
            image_count=0,
            is_scanned=False,
            fonts=[],
            page_type="text",
        )

        assert page_info.page_num == 0
        assert page_info.text == "Sample text"
        assert page_info.word_count == 2
        assert page_info.is_scanned is False


class TestPDFMetadataDataclass:
    """Test PDFMetadata dataclass."""

    def test_pdf_metadata_creation(self):
        """Test PDFMetadata creation."""
        metadata = PDFMetadata(
            filename="test.pdf",
            file_size_bytes=5000000,
            page_count=100,
            is_encrypted=False,
            title="Test RHP",
            author="Company",
            subject="IPO",
            creator="PDF Creator",
            creation_date="2024-01-01",
            modification_date="2024-01-02",
        )

        assert metadata.page_count == 100
        assert metadata.is_encrypted is False
        assert metadata.title == "Test RHP"


# ============================================================================
# Integration Test (if real PDF available)
# ============================================================================


class TestIntegrationWithRealPDF:
    """Integration tests that run if a real PDF is available."""

    @pytest.fixture
    def sample_pdf_path(self) -> str:
        """Get path to a sample PDF for testing."""
        test_input_dir = Path(__file__).parent.parent.parent / "test" / "input"

        if test_input_dir.exists():
            pdfs = list(test_input_dir.glob("*.pdf"))
            if pdfs:
                return str(pdfs[0])

        return None

    def test_real_pdf_processing(self, sample_pdf_path):
        """Test processing a real PDF if available."""
        if sample_pdf_path is None:
            pytest.skip("No test PDF available")

        processor = PDFProcessor(sample_pdf_path)
        metadata = processor.validate()

        assert metadata.page_count > 0
        assert isinstance(metadata.file_size_bytes, int)

    def test_real_pdf_extraction(self, sample_pdf_path):
        """Test extracting pages from real PDF."""
        if sample_pdf_path is None:
            pytest.skip("No test PDF available")

        processor = PDFProcessor(sample_pdf_path)
        pages = processor.extract_all_pages()

        assert len(pages) > 0
        assert all(isinstance(p, PageInfo) for p in pages)
