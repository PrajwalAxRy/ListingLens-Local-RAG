"""
Unit tests for OCR Processor Module.

Tests cover OCR configuration, scanned page detection,
text extraction, and error handling.
"""

import os
import tempfile
from collections.abc import Generator
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

# Import the modules under test
from rhp_analyzer.ingestion.ocr_processor import (
    OCRConfig,
    OCRError,
    OCRProcessingError,
    OCRProcessor,
    OCRResult,
    TesseractNotFoundError,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ocr_config() -> OCRConfig:
    """Create a default OCR configuration."""
    return OCRConfig()


@pytest.fixture
def custom_ocr_config() -> OCRConfig:
    """Create a custom OCR configuration."""
    return OCRConfig(languages="eng", dpi=200, psm=3, oem=1, timeout=60, preserve_interword_spaces=False)


@pytest.fixture
def mock_image():
    """Create a mock PIL Image."""
    mock_img = MagicMock()
    mock_img.size = (612, 792)
    mock_img.mode = "RGB"
    return mock_img


@pytest.fixture
def temp_pdf_file() -> Generator[str, None, None]:
    """Create a temporary PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 fake pdf content for testing")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_image_file() -> Generator[str, None, None]:
    """Create a temporary image file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        # Write minimal PNG header
        f.write(b"\x89PNG\r\n\x1a\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_processor():
    """Create an OCRProcessor with mocked availability."""
    with patch.object(OCRProcessor, "_find_tesseract", return_value="/usr/bin/tesseract"), patch.object(
        OCRProcessor, "_check_availability", return_value=True
    ):
        processor = OCRProcessor()
        processor.is_available = True
        return processor


# ============================================================================
# OCRConfig Tests
# ============================================================================


class TestOCRConfig:
    """Test OCR configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OCRConfig()

        assert config.languages == "eng+hin"
        assert config.dpi == 300
        assert config.psm == 6
        assert config.oem == 3
        assert config.timeout == 120
        assert config.preserve_interword_spaces is True

    def test_custom_config(self, custom_ocr_config):
        """Test custom configuration values."""
        assert custom_ocr_config.languages == "eng"
        assert custom_ocr_config.dpi == 200
        assert custom_ocr_config.psm == 3
        assert custom_ocr_config.oem == 1
        assert custom_ocr_config.timeout == 60
        assert custom_ocr_config.preserve_interword_spaces is False

    def test_tesseract_config_string(self):
        """Test Tesseract config string generation."""
        config = OCRConfig(psm=4, oem=2, preserve_interword_spaces=True)
        config_str = config.tesseract_config

        assert "--psm 4" in config_str
        assert "--oem 2" in config_str
        assert "preserve_interword_spaces=1" in config_str

    def test_tesseract_config_no_spaces(self):
        """Test Tesseract config string without preserve_interword_spaces."""
        config = OCRConfig(psm=6, oem=3, preserve_interword_spaces=False)
        config_str = config.tesseract_config

        assert "preserve_interword_spaces" not in config_str


# ============================================================================
# OCRResult Tests
# ============================================================================


class TestOCRResult:
    """Test OCRResult dataclass."""

    def test_result_creation(self):
        """Test OCRResult creation."""
        result = OCRResult(
            page_num=1,
            text="Sample extracted text",
            confidence=95.5,
            language="eng",
            processing_time=2.5,
            word_count=3,
            char_count=21,
            warnings=[],
        )

        assert result.page_num == 1
        assert result.text == "Sample extracted text"
        assert result.confidence == 95.5
        assert result.language == "eng"
        assert result.processing_time == 2.5
        assert result.word_count == 3
        assert result.char_count == 21
        assert result.warnings == []

    def test_result_with_warnings(self):
        """Test OCRResult with warnings."""
        result = OCRResult(
            page_num=5,
            text="Low quality text",
            confidence=35.0,
            language="eng+hin",
            processing_time=5.0,
            word_count=3,
            char_count=16,
            warnings=["Low confidence: 35.0%", "Possible OCR artifacts"],
        )

        assert len(result.warnings) == 2
        assert "Low confidence" in result.warnings[0]

    def test_result_to_dict(self):
        """Test OCRResult can be converted to dict."""
        result = OCRResult(
            page_num=1,
            text="Text",
            confidence=90.0,
            language="eng",
            processing_time=1.0,
            word_count=1,
            char_count=4,
            warnings=[],
        )

        result_dict = asdict(result)
        assert isinstance(result_dict, dict)
        assert result_dict["page_num"] == 1
        assert result_dict["text"] == "Text"


# ============================================================================
# OCRProcessor Initialization Tests
# ============================================================================


class TestOCRProcessorInit:
    """Test OCRProcessor initialization."""

    def test_default_initialization(self, mock_processor):
        """Test processor with default config."""
        assert mock_processor.config.languages == "eng+hin"
        assert mock_processor.config.dpi == 300
        assert mock_processor.config.psm == 6
        assert mock_processor.config.oem == 3
        assert mock_processor.config.timeout == 120
        assert mock_processor.config.preserve_interword_spaces is True

    def test_custom_config_initialization(self):
        """Test processor with custom config."""
        config = OCRConfig(languages="eng", dpi=200)

        with patch.object(OCRProcessor, "_find_tesseract", return_value="/usr/bin/tesseract"), patch.object(
            OCRProcessor, "_check_availability", return_value=True
        ):
            processor = OCRProcessor(config)

            assert processor.config.languages == "eng"
            assert processor.config.dpi == 200

    def test_unavailable_when_no_tesseract(self):
        """Test processor not available when Tesseract not found."""
        with patch.object(OCRProcessor, "_find_tesseract", return_value=None):
            processor = OCRProcessor()
            assert processor.is_available is False


# ============================================================================
# Scanned Page Detection Tests
# ============================================================================


class TestScannedPageDetection:
    """Test scanned page detection functionality."""

    def test_is_scanned_page_with_image_no_text(self, mock_processor):
        """Test detection when page has image but no text."""
        # Page with image and no text = likely scanned
        result = mock_processor.is_scanned_page(page_text="", image_count=1)

        assert result is True

    def test_is_scanned_page_with_normal_text(self, mock_processor):
        """Test detection with normal text page."""
        # Page with substantial text (over char_threshold of 100)
        result = mock_processor.is_scanned_page(page_text="This is normal text content " * 100, image_count=0)

        assert result is False

    def test_is_scanned_page_with_sparse_text(self, mock_processor):
        """Test detection with sparse text (likely OCR artifacts)."""
        # Page with very little text and images (below char_threshold of 100)
        result = mock_processor.is_scanned_page(
            page_text="AB",  # Very sparse - likely OCR artifact
            image_count=1,
        )

        assert result is True

    def test_is_scanned_page_with_custom_threshold(self, mock_processor):
        """Test detection with custom character threshold."""
        # 50 characters with threshold of 40 should be considered digital
        result = mock_processor.is_scanned_page(page_text="A" * 50, image_count=1, char_threshold=40)

        assert result is False

    def test_is_scanned_empty_page_no_images(self, mock_processor):
        """Test blank page detection."""
        # No text and no images - could be blank or fully scanned
        result = mock_processor.is_scanned_page(page_text="", image_count=0)

        assert result is True

    def test_is_scanned_text_only(self, mock_processor):
        """Test digital text page."""
        result = mock_processor.is_scanned_page(
            page_text="Normal text content with many characters " * 5, image_count=0
        )

        assert result is False


# ============================================================================
# OCR Text Extraction Tests
# ============================================================================


class TestOCRTextExtraction:
    """Test OCR text extraction functionality."""

    @patch("rhp_analyzer.ingestion.ocr_processor.pytesseract")
    def test_extract_text_from_image(self, mock_pytesseract, mock_processor, mock_image):
        """Test text extraction from image."""
        # Configure the mocked pytesseract
        mock_pytesseract.image_to_string.return_value = "Extracted text from image"
        mock_pytesseract.image_to_data.return_value = {"conf": [95.0, 90.0, 88.0]}

        text, confidence = mock_processor.extract_text_from_image(mock_image)

        assert text == "Extracted text from image"
        assert confidence > 0  # Average of confidences

    @patch("rhp_analyzer.ingestion.ocr_processor.pytesseract")
    def test_extract_text_empty_result(self, mock_pytesseract, mock_processor, mock_image):
        """Test handling of empty OCR result."""
        mock_pytesseract.image_to_string.return_value = ""
        mock_pytesseract.image_to_data.return_value = {"conf": []}

        text, confidence = mock_processor.extract_text_from_image(mock_image)

        assert text == ""
        assert confidence == 0.0

    @patch("rhp_analyzer.ingestion.ocr_processor.pytesseract")
    def test_extract_text_with_language_override(self, mock_pytesseract, mock_processor, mock_image):
        """Test extraction with language override."""
        mock_pytesseract.image_to_string.return_value = "Test"
        mock_pytesseract.image_to_data.return_value = {"conf": [90.0]}

        text, confidence = mock_processor.extract_text_from_image(mock_image, language="hin")

        assert text == "Test"
        # Verify pytesseract was called with correct language
        mock_pytesseract.image_to_data.assert_called()

    def test_extract_text_not_available(self, mock_image):
        """Test extraction when OCR not available."""
        with patch.object(OCRProcessor, "_find_tesseract", return_value=None):
            processor = OCRProcessor()
            processor.is_available = False

            with pytest.raises(OCRError):
                processor.extract_text_from_image(mock_image)

    @patch("rhp_analyzer.ingestion.ocr_processor.pytesseract")
    def test_extract_text_tesseract_error(self, mock_pytesseract, mock_processor, mock_image):
        """Test handling of Tesseract error."""
        # Need to set TesseractError to a real exception class for except clause to work
        mock_pytesseract.TesseractError = Exception
        mock_pytesseract.image_to_data.side_effect = Exception("Tesseract failed")

        with pytest.raises(OCRProcessingError):
            mock_processor.extract_text_from_image(mock_image)


# ============================================================================
# Batch OCR Tests
# ============================================================================


class TestBatchOCR:
    """Test batch OCR processing functionality."""

    @patch("rhp_analyzer.ingestion.ocr_processor.pytesseract")
    def test_batch_ocr_from_images(self, mock_pytesseract, mock_processor):
        """Test batch OCR from multiple images."""
        mock_pytesseract.image_to_string.return_value = "Extracted text"
        mock_pytesseract.image_to_data.return_value = {"conf": [90.0]}

        # Create mock images
        images = [MagicMock() for _ in range(3)]
        for img in images:
            img.size = (612, 792)
            img.mode = "RGB"

        results = mock_processor.batch_ocr_from_images(images)

        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)  # (text, confidence)

    @patch("rhp_analyzer.ingestion.ocr_processor.pytesseract")
    def test_batch_ocr_with_failures(self, mock_pytesseract, mock_processor):
        """Test batch OCR handles failures gracefully."""
        # Need to set TesseractError to a real exception class
        mock_pytesseract.TesseractError = Exception
        mock_pytesseract.image_to_data.return_value = {"conf": [90.0]}

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # Raise the TesseractError which gets caught and re-raised as OCRProcessingError
                raise Exception("OCR failed")
            return "Extracted text"

        mock_pytesseract.image_to_string.side_effect = side_effect

        images = [MagicMock() for _ in range(3)]

        results = mock_processor.batch_ocr_from_images(images)

        assert len(results) == 3
        # Second result should be empty due to failure
        assert results[1] == ("", 0.0)

    def test_batch_ocr_not_available(self):
        """Test batch OCR when not available."""
        with patch.object(OCRProcessor, "_find_tesseract", return_value=None):
            processor = OCRProcessor()
            processor.is_available = False

            with pytest.raises(OCRError):
                processor.batch_ocr_from_images([MagicMock()])


# ============================================================================
# Installation Verification Tests
# ============================================================================


class TestInstallationVerification:
    """Test installation verification functionality."""

    def test_verify_installation(self, mock_processor):
        """Test installation verification."""
        status = mock_processor.verify_installation()

        assert isinstance(status, dict)
        assert "pytesseract_installed" in status
        assert "pdf2image_installed" in status
        assert "tesseract_found" in status

    def test_get_status_report(self, mock_processor):
        """Test status report generation."""
        report = mock_processor.get_status_report()

        assert isinstance(report, str)
        assert "OCR Installation Status" in report

    @patch("rhp_analyzer.ingestion.ocr_processor.pytesseract")
    def test_get_available_languages(self, mock_pytesseract, mock_processor):
        """Test getting available languages."""
        mock_pytesseract.get_languages.return_value = ["eng", "hin", "fra"]

        languages = mock_processor.get_available_languages()

        assert "eng" in languages
        assert "hin" in languages


# ============================================================================
# Exception Tests
# ============================================================================


class TestExceptions:
    """Test exception classes."""

    def test_ocr_error(self):
        """Test OCRError exception."""
        error = OCRError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_tesseract_not_found_error(self):
        """Test TesseractNotFoundError exception."""
        error = TesseractNotFoundError("Tesseract not found")
        assert str(error) == "Tesseract not found"
        assert isinstance(error, OCRError)

    def test_ocr_processing_error(self):
        """Test OCRProcessingError exception."""
        error = OCRProcessingError("Processing failed")
        assert str(error) == "Processing failed"
        assert isinstance(error, OCRError)


# ============================================================================
# Tesseract Path Finding Tests
# ============================================================================


class TestTesseractPathFinding:
    """Test Tesseract path finding functionality."""

    def test_find_tesseract_in_path(self):
        """Test finding Tesseract in PATH."""
        with patch("shutil.which", return_value="/usr/bin/tesseract"):
            processor = OCRProcessor.__new__(OCRProcessor)
            processor.config = OCRConfig()

            path = processor._find_tesseract()
            assert path == "/usr/bin/tesseract"

    def test_find_tesseract_from_env(self):
        """Test finding Tesseract from environment variable."""
        with patch("shutil.which", return_value=None), patch("os.path.isfile", return_value=False), patch.dict(
            os.environ, {"TESSERACT_CMD": "/custom/tesseract"}
        ):
            # Second isfile check for TESSERACT_CMD should return True
            def isfile_side_effect(path):
                return path == "/custom/tesseract"

            with patch("os.path.isfile", side_effect=isfile_side_effect):
                processor = OCRProcessor.__new__(OCRProcessor)
                processor.config = OCRConfig()

                path = processor._find_tesseract()
                assert path == "/custom/tesseract"

    def test_find_tesseract_not_found(self):
        """Test when Tesseract is not found."""
        with patch("shutil.which", return_value=None), patch.dict(os.environ, {}, clear=True), patch(
            "os.path.isfile", return_value=False
        ):
            processor = OCRProcessor.__new__(OCRProcessor)
            processor.config = OCRConfig()

            path = processor._find_tesseract()
            assert path is None


# ============================================================================
# Poppler Path Finding Tests
# ============================================================================


class TestPopplerPathFinding:
    """Test Poppler path finding functionality."""

    def test_find_poppler_in_path(self):
        """Test when pdftoppm is in PATH."""
        with patch("shutil.which", return_value="/usr/bin/pdftoppm"):
            processor = OCRProcessor.__new__(OCRProcessor)
            processor.config = OCRConfig()

            path = processor._find_poppler()
            assert path is None  # Returns None when in PATH

    def test_find_poppler_from_env(self):
        """Test finding Poppler from environment variable."""
        with patch("shutil.which", return_value=None), patch.dict(
            os.environ, {"POPPLER_PATH": "/custom/poppler/bin"}
        ), patch("os.path.isdir", return_value=True):
            processor = OCRProcessor.__new__(OCRProcessor)
            processor.config = OCRConfig()

            path = processor._find_poppler()
            assert path == "/custom/poppler/bin"

    def test_find_poppler_not_found(self):
        """Test when Poppler is not found."""
        with patch("shutil.which", return_value=None), patch.dict(os.environ, {}, clear=True), patch(
            "os.path.isdir", return_value=False
        ):
            processor = OCRProcessor.__new__(OCRProcessor)
            processor.config = OCRConfig()

            path = processor._find_poppler()
            assert path is None
