"""
OCR Processor Module for RHP Analyzer.

This module provides OCR capabilities for scanned/image-based PDF pages
using Tesseract OCR. Supports English and Hindi text extraction.
"""

import os
import shutil
from dataclasses import dataclass, field

from loguru import logger

# Lazy imports for optional dependencies
try:
    import pytesseract

    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed. OCR functionality will be limited.")

try:
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not installed. OCR functionality will be limited.")


class OCRError(Exception):
    """Base exception for OCR-related errors."""

    pass


class TesseractNotFoundError(OCRError):
    """Raised when Tesseract OCR is not found on the system."""

    pass


class OCRProcessingError(OCRError):
    """Raised when OCR processing fails."""

    pass


@dataclass
class OCRResult:
    """Result of OCR processing for a single page."""

    page_num: int
    text: str
    confidence: float  # 0.0 to 100.0
    language: str
    processing_time: float  # seconds
    word_count: int
    char_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""

    languages: str = "eng+hin"  # English + Hindi for Indian RHPs
    dpi: int = 300  # Higher DPI = better quality but slower
    psm: int = 6  # Page segmentation mode: Assume uniform block of text
    oem: int = 3  # OCR Engine mode: Default, based on what is available
    timeout: int = 120  # Timeout per page in seconds
    preserve_interword_spaces: bool = True

    @property
    def tesseract_config(self) -> str:
        """Generate Tesseract config string."""
        config_parts = [
            f"--psm {self.psm}",
            f"--oem {self.oem}",
        ]
        if self.preserve_interword_spaces:
            config_parts.append("-c preserve_interword_spaces=1")
        return " ".join(config_parts)


class OCRProcessor:
    """
    OCR processor for scanned/image-based PDF pages.

    Uses Tesseract OCR with support for English and Hindi text,
    which is common in Indian IPO RHP documents.

    Attributes:
        config: OCR configuration settings
        is_available: Whether OCR is available on this system
    """

    def __init__(self, config: OCRConfig | None = None):
        """
        Initialize OCR processor.

        Args:
            config: Optional OCR configuration. Uses defaults if not provided.
        """
        self.config = config or OCRConfig()
        self._tesseract_path: str | None = None
        self._poppler_path: str | None = None
        self.is_available = self._check_availability()

        if self.is_available:
            logger.info(f"OCR processor initialized with languages: {self.config.languages}")
        else:
            logger.warning("OCR processor initialized but OCR is not available")

    def _check_availability(self) -> bool:
        """Check if OCR dependencies are available."""
        if not PYTESSERACT_AVAILABLE:
            logger.error("pytesseract is not installed")
            return False

        if not PDF2IMAGE_AVAILABLE:
            logger.error("pdf2image is not installed")
            return False

        # Try to find Tesseract
        tesseract_path = self._find_tesseract()
        if tesseract_path is None:
            logger.error("Tesseract OCR is not installed or not in PATH")
            return False

        self._tesseract_path = tesseract_path
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Check available languages
        try:
            available_langs = pytesseract.get_languages()
            logger.debug(f"Available Tesseract languages: {available_langs}")

            # Check if required languages are installed
            required_langs = self.config.languages.split("+")
            missing_langs = [lang for lang in required_langs if lang not in available_langs]

            if missing_langs:
                logger.warning(f"Missing Tesseract language packs: {missing_langs}")
                # Don't fail, just warn - can still process with available languages
        except Exception as e:
            logger.warning(f"Could not check Tesseract languages: {e}")

        return True

    def _find_tesseract(self) -> str | None:
        """
        Find Tesseract executable on the system.

        Returns:
            Path to Tesseract executable, or None if not found.
        """
        # Check common Windows installation paths
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
        ]

        # First, check if it's in PATH
        tesseract_in_path = shutil.which("tesseract")
        if tesseract_in_path:
            logger.debug(f"Found Tesseract in PATH: {tesseract_in_path}")
            return tesseract_in_path

        # Check common installation paths on Windows
        for path in common_paths:
            if os.path.isfile(path):
                logger.debug(f"Found Tesseract at: {path}")
                return path

        # Check environment variable
        tesseract_env = os.environ.get("TESSERACT_CMD")
        if tesseract_env and os.path.isfile(tesseract_env):
            logger.debug(f"Found Tesseract from TESSERACT_CMD: {tesseract_env}")
            return tesseract_env

        return None

    def _find_poppler(self) -> str | None:
        """
        Find Poppler binaries (needed for pdf2image on Windows).

        Returns:
            Path to Poppler bin directory, or None if not found.
        """
        # Check common Windows installation paths
        common_paths = [
            r"C:\Program Files\poppler\Library\bin",
            r"C:\Program Files\poppler\bin",
            r"C:\poppler\bin",
            r"C:\Program Files (x86)\poppler\bin",
        ]

        # Check if pdftoppm is in PATH (Linux/Mac typically)
        if shutil.which("pdftoppm"):
            return None  # No path needed, it's in PATH

        # Check common paths on Windows
        for path in common_paths:
            pdftoppm_path = os.path.join(path, "pdftoppm.exe")
            if os.path.isfile(pdftoppm_path):
                logger.debug(f"Found Poppler at: {path}")
                return path

        # Check environment variable
        poppler_env = os.environ.get("POPPLER_PATH")
        if poppler_env and os.path.isdir(poppler_env):
            logger.debug(f"Found Poppler from POPPLER_PATH: {poppler_env}")
            return poppler_env

        return None

    def is_scanned_page(self, page_text: str, image_count: int, char_threshold: int = 100) -> bool:
        """
        Determine if a page is likely scanned (image-based).

        Args:
            page_text: Text extracted from the page using normal methods
            image_count: Number of images on the page
            char_threshold: Minimum characters to consider page as having text

        Returns:
            True if page is likely scanned and needs OCR
        """
        text_length = len(page_text.strip())

        # Page has images but very little text = likely scanned
        if image_count > 0 and text_length < char_threshold:
            return True

        # Page with no text and no images could be blank or fully scanned
        if text_length == 0 and image_count == 0:
            return True  # Will need OCR to verify if blank

        return False

    def extract_text_from_image(self, image, language: str | None = None) -> tuple[str, float]:
        """
        Extract text from a PIL Image using OCR.

        Args:
            image: PIL Image object
            language: Override language setting for this extraction

        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        if not self.is_available:
            raise OCRError("OCR is not available. Install Tesseract and required dependencies.")

        lang = language or self.config.languages

        try:
            # Get OCR data with confidence
            ocr_data = pytesseract.image_to_data(
                image,
                lang=lang,
                config=self.config.tesseract_config,
                output_type=pytesseract.Output.DICT,
                timeout=self.config.timeout,
            )

            # Calculate average confidence (excluding -1 which means no text)
            confidences = [c for c in ocr_data["conf"] if c != -1]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # Extract text
            text = pytesseract.image_to_string(
                image, lang=lang, config=self.config.tesseract_config, timeout=self.config.timeout
            )

            return text.strip(), avg_confidence

        except pytesseract.TesseractError as e:
            logger.error(f"Tesseract error: {e}")
            raise OCRProcessingError(f"Tesseract processing failed: {e}") from e

    def process_pdf_page(self, pdf_path: str, page_num: int, language: str | None = None) -> OCRResult:
        """
        Process a single PDF page with OCR.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            language: Override language setting

        Returns:
            OCRResult with extracted text and metadata
        """
        import time

        start_time = time.time()

        if not self.is_available:
            raise OCRError("OCR is not available")

        lang = language or self.config.languages
        warnings: list[str] = []

        try:
            # Find poppler path for pdf2image
            poppler_path = self._find_poppler()

            # Convert PDF page to image
            logger.debug(f"Converting PDF page {page_num} to image...")

            convert_kwargs = {
                "pdf_path": pdf_path,
                "first_page": page_num,
                "last_page": page_num,
                "dpi": self.config.dpi,
            }

            if poppler_path:
                convert_kwargs["poppler_path"] = poppler_path

            images = convert_from_path(**convert_kwargs)

            if not images:
                raise OCRProcessingError(f"Failed to convert page {page_num} to image")

            page_image = images[0]

            # Perform OCR
            logger.debug(f"Running OCR on page {page_num}...")
            text, confidence = self.extract_text_from_image(page_image, lang)

            # Calculate metrics
            processing_time = time.time() - start_time
            word_count = len(text.split()) if text else 0
            char_count = len(text) if text else 0

            # Add warnings for low confidence
            if confidence < 50:
                warnings.append(f"Low OCR confidence: {confidence:.1f}%")

            if word_count < 10:
                warnings.append("Very few words extracted - page may be blank or image-only")

            return OCRResult(
                page_num=page_num,
                text=text,
                confidence=confidence,
                language=lang,
                processing_time=processing_time,
                word_count=word_count,
                char_count=char_count,
                warnings=warnings,
            )

        except PDFInfoNotInstalledError as e:
            raise OCRError(
                "Poppler is not installed. On Windows, install Poppler and add to PATH or set POPPLER_PATH."
            ) from e
        except PDFPageCountError as e:
            raise OCRProcessingError(f"Could not read PDF page count: {e}") from e
        except Exception as e:
            logger.error(f"OCR processing error on page {page_num}: {e}")
            raise OCRProcessingError(f"Failed to OCR page {page_num}: {e}") from e

    def process_pdf_pages(
        self, pdf_path: str, page_numbers: list[int], language: str | None = None
    ) -> dict[int, OCRResult]:
        """
        Process multiple PDF pages with OCR.

        Args:
            pdf_path: Path to PDF file
            page_numbers: List of page numbers to process (1-indexed)
            language: Override language setting

        Returns:
            Dictionary mapping page numbers to OCRResult objects
        """
        if not self.is_available:
            raise OCRError("OCR is not available")

        results: dict[int, OCRResult] = {}

        logger.info(f"Processing {len(page_numbers)} pages with OCR...")

        for i, page_num in enumerate(page_numbers):
            logger.debug(f"Processing page {page_num} ({i+1}/{len(page_numbers)})")

            try:
                result = self.process_pdf_page(pdf_path, page_num, language)
                results[page_num] = result

                logger.debug(
                    f"Page {page_num}: {result.word_count} words, "
                    f"{result.confidence:.1f}% confidence, "
                    f"{result.processing_time:.2f}s"
                )

            except OCRProcessingError as e:
                logger.warning(f"Failed to OCR page {page_num}: {e}")
                # Create result with error
                results[page_num] = OCRResult(
                    page_num=page_num,
                    text="",
                    confidence=0.0,
                    language=language or self.config.languages,
                    processing_time=0.0,
                    word_count=0,
                    char_count=0,
                    warnings=[f"OCR failed: {e}"],
                )

        logger.info(f"OCR completed for {len(results)} pages")
        return results

    def process_scanned_pdf(self, pdf_path: str, scanned_page_indices: list[int] | None = None) -> dict[int, OCRResult]:
        """
        Process a PDF, applying OCR to scanned pages.

        Args:
            pdf_path: Path to PDF file
            scanned_page_indices: List of page numbers to OCR.
                                 If None, will attempt to detect scanned pages.

        Returns:
            Dictionary mapping page numbers to OCRResult objects
        """
        if scanned_page_indices is None:
            # Import here to avoid circular dependency
            from rhp_analyzer.ingestion.pdf_processor import PDFProcessor

            processor = PDFProcessor(pdf_path)
            scanned_page_indices = processor.detect_scanned_pages()

        if not scanned_page_indices:
            logger.info("No scanned pages detected")
            return {}

        logger.info(f"Found {len(scanned_page_indices)} scanned pages to OCR")
        return self.process_pdf_pages(pdf_path, scanned_page_indices)

    def batch_ocr_from_images(self, images: list, language: str | None = None) -> list[tuple[str, float]]:
        """
        Perform OCR on a batch of images.

        Args:
            images: List of PIL Image objects
            language: Override language setting

        Returns:
            List of (text, confidence) tuples
        """
        if not self.is_available:
            raise OCRError("OCR is not available")

        results = []
        for i, image in enumerate(images):
            logger.debug(f"OCR batch: processing image {i+1}/{len(images)}")
            try:
                text, confidence = self.extract_text_from_image(image, language)
                results.append((text, confidence))
            except OCRProcessingError as e:
                logger.warning(f"Failed to OCR image {i+1}: {e}")
                results.append(("", 0.0))

        return results

    def get_available_languages(self) -> list[str]:
        """
        Get list of available Tesseract languages.

        Returns:
            List of language codes available for OCR
        """
        if not self.is_available:
            return []

        try:
            return pytesseract.get_languages()
        except Exception as e:
            logger.error(f"Could not get languages: {e}")
            return []

    def verify_installation(self) -> dict[str, bool]:
        """
        Verify OCR installation and dependencies.

        Returns:
            Dictionary with status of each component
        """
        status = {
            "pytesseract_installed": PYTESSERACT_AVAILABLE,
            "pdf2image_installed": PDF2IMAGE_AVAILABLE,
            "tesseract_found": self._tesseract_path is not None,
            "poppler_found": self._find_poppler() is not None or shutil.which("pdftoppm") is not None,
            "english_lang": False,
            "hindi_lang": False,
        }

        if self.is_available:
            try:
                langs = pytesseract.get_languages()
                status["english_lang"] = "eng" in langs
                status["hindi_lang"] = "hin" in langs
            except Exception:
                pass

        return status

    def get_status_report(self) -> str:
        """
        Get a human-readable status report of OCR installation.

        Returns:
            Status report string
        """
        status = self.verify_installation()

        lines = ["OCR Installation Status:"]
        lines.append(f"  pytesseract installed: {'✓' if status['pytesseract_installed'] else '✗'}")
        lines.append(f"  pdf2image installed: {'✓' if status['pdf2image_installed'] else '✗'}")
        lines.append(f"  Tesseract found: {'✓' if status['tesseract_found'] else '✗'}")
        lines.append(f"  Poppler found: {'✓' if status['poppler_found'] else '✗'}")
        lines.append(f"  English language pack: {'✓' if status['english_lang'] else '✗'}")
        lines.append(f"  Hindi language pack: {'✓' if status['hindi_lang'] else '✗'}")

        if self._tesseract_path:
            lines.append(f"  Tesseract path: {self._tesseract_path}")

        poppler_path = self._find_poppler()
        if poppler_path:
            lines.append(f"  Poppler path: {poppler_path}")

        overall = "Available" if self.is_available else "Not Available"
        lines.append(f"\nOverall Status: {overall}")

        return "\n".join(lines)


# Convenience function for quick OCR
def ocr_pdf_page(pdf_path: str, page_num: int, languages: str = "eng") -> str:
    """
    Convenience function to OCR a single PDF page.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        languages: Tesseract language codes (e.g., "eng", "eng+hin")

    Returns:
        Extracted text from the page
    """
    config = OCRConfig(languages=languages)
    processor = OCRProcessor(config)

    if not processor.is_available:
        raise OCRError("OCR is not available on this system")

    result = processor.process_pdf_page(pdf_path, page_num)
    return result.text
