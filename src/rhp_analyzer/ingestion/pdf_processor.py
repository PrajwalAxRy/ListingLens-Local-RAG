"""
PDF Processing Module for RHP Analyzer.

This module handles PDF parsing with multiple strategies, extracting text,
metadata, and structural information from RHP documents.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import fitz  # PyMuPDF
from loguru import logger


class PDFProcessingError(Exception):
    """Exception raised for PDF processing failures."""

    pass


class PDFValidationError(PDFProcessingError):
    """Exception raised for PDF validation failures."""

    pass


@dataclass
class PageInfo:
    """Represents a single PDF page with extracted content and metadata."""

    page_num: int
    text: str
    char_count: int
    word_count: int
    has_images: bool
    image_count: int
    is_scanned: bool
    fonts: list[str] = field(default_factory=list)
    font_sizes: list[float] = field(default_factory=list)
    page_width: float = 0.0
    page_height: float = 0.0
    # Page type classification
    page_type: str = "text"  # 'cover', 'toc', 'text', 'table', 'blank', 'appendix'
    has_tables: bool = False
    # Layout info
    is_multi_column: bool = False
    header_text: str | None = None
    footer_text: str | None = None


@dataclass
class PDFMetadata:
    """Metadata extracted from PDF document."""

    filename: str
    file_size_bytes: int
    page_count: int
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None
    is_encrypted: bool = False
    is_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)
    scanned_page_count: int = 0
    processing_warnings: list[str] = field(default_factory=list)


class PDFProcessor:
    """
    Handles PDF parsing with multiple strategies.

    Primary extraction uses PyMuPDF (fitz) for fast, reliable text extraction.
    Includes page analysis, validation, and scanned page detection.
    """

    # Thresholds for scanned page detection
    MIN_TEXT_CHARS_FOR_DIGITAL = 100  # Minimum chars to consider page as digital
    MIN_TEXT_TO_IMAGE_RATIO = 0.1  # If text/image ratio below this, likely scanned

    # Page type detection patterns
    COVER_PAGE_PATTERNS: ClassVar[list[str]] = [
        r"(?i)red\s*herring\s*prospectus",
        r"(?i)draft\s*red\s*herring\s*prospectus",
        r"(?i)prospectus",
        r"(?i)initial\s*public\s*offer",
        r"(?i)book\s*building\s*issue",
    ]

    TOC_PATTERNS: ClassVar[list[str]] = [
        r"(?i)table\s*of\s*contents",
        r"(?i)contents",
        r"(?i)index",
        r"(?i)^contents$",
    ]

    APPENDIX_PATTERNS: ClassVar[list[str]] = [
        r"(?i)^appendix",
        r"(?i)^annexure",
        r"(?i)^schedule",
    ]

    def __init__(self, pdf_path: str | Path):
        """
        Initialize PDF processor with path to PDF file.

        Args:
            pdf_path: Path to the PDF file to process.

        Raises:
            PDFValidationError: If the PDF file doesn't exist or is invalid.
        """
        self.pdf_path = Path(pdf_path)
        self.metadata: PDFMetadata | None = None
        self.pages: list[PageInfo] = []
        self._doc: fitz.Document | None = None

        # Validate file exists
        if not self.pdf_path.exists():
            raise PDFValidationError(f"PDF file not found: {self.pdf_path}")

        if not self.pdf_path.is_file():
            raise PDFValidationError(f"Path is not a file: {self.pdf_path}")

        if self.pdf_path.suffix.lower() != ".pdf":
            raise PDFValidationError(f"File is not a PDF: {self.pdf_path}")

    def validate(self) -> PDFMetadata:
        """
        Validate PDF integrity and extract metadata.

        Returns:
            PDFMetadata object with validation results.

        Raises:
            PDFValidationError: If PDF cannot be opened or is corrupted.
        """
        logger.debug(f"Validating PDF: {self.pdf_path}")

        validation_errors: list[str] = []
        processing_warnings: list[str] = []

        try:
            # Get file size
            file_size = self.pdf_path.stat().st_size

            # Try to open the PDF
            with fitz.open(str(self.pdf_path)) as doc:
                # Check if encrypted
                is_encrypted = doc.is_encrypted
                if is_encrypted and not doc.authenticate(""):
                    validation_errors.append("PDF is password protected and cannot be decrypted")
                    logger.warning(f"PDF is encrypted: {self.pdf_path}")

                # Get page count
                page_count = len(doc)
                if page_count == 0:
                    validation_errors.append("PDF has no pages")

                # Get metadata
                pdf_metadata = doc.metadata

                # Check for scanned pages (quick sample check)
                scanned_count = 0
                sample_pages = min(10, page_count)  # Check first 10 pages
                for i in range(sample_pages):
                    page = doc[i]
                    text = page.get_text()
                    images = page.get_images()

                    if len(text.strip()) < self.MIN_TEXT_CHARS_FOR_DIGITAL and len(images) > 0:
                        scanned_count += 1

                # Estimate total scanned pages
                if sample_pages > 0:
                    scanned_ratio = scanned_count / sample_pages
                    estimated_scanned = int(scanned_ratio * page_count)
                    if scanned_ratio > 0.5:
                        processing_warnings.append(
                            f"Document appears to be mostly scanned (~{estimated_scanned} of {page_count} pages). "
                            "OCR may be required for accurate text extraction."
                        )
                else:
                    estimated_scanned = 0

            self.metadata = PDFMetadata(
                filename=self.pdf_path.name,
                file_size_bytes=file_size,
                page_count=page_count,
                title=pdf_metadata.get("title"),
                author=pdf_metadata.get("author"),
                subject=pdf_metadata.get("subject"),
                creator=pdf_metadata.get("creator"),
                producer=pdf_metadata.get("producer"),
                creation_date=pdf_metadata.get("creationDate"),
                modification_date=pdf_metadata.get("modDate"),
                is_encrypted=is_encrypted,
                is_valid=len(validation_errors) == 0,
                validation_errors=validation_errors,
                scanned_page_count=estimated_scanned,
                processing_warnings=processing_warnings,
            )

            logger.info(f"PDF validated: {page_count} pages, {file_size} bytes")

            return self.metadata

        except fitz.FileDataError as e:
            raise PDFValidationError(f"Corrupted or invalid PDF file: {e}") from e
        except Exception as e:
            raise PDFValidationError(f"Failed to validate PDF: {e}") from e

    def extract_all_pages(self) -> list[PageInfo]:
        """
        Extract text and metadata from all pages.

        Returns:
            List of PageInfo objects for each page.

        Raises:
            PDFProcessingError: If extraction fails.
        """
        logger.info(f"Extracting text from all pages: {self.pdf_path}")

        try:
            with fitz.open(str(self.pdf_path)) as self._doc:
                self.pages = []

                total_pages = len(self._doc)

                for page_num in range(total_pages):
                    page_info = self._extract_page(page_num)
                    self.pages.append(page_info)

                    if (page_num + 1) % 50 == 0:
                        logger.debug(f"Processed {page_num + 1}/{total_pages} pages")

                logger.info(f"Extracted {len(self.pages)} pages successfully")

            return self.pages

        except Exception as e:
            raise PDFProcessingError(f"Failed to extract pages: {e}") from e

    def extract_page_range(self, start: int, end: int) -> list[PageInfo]:
        """
        Extract text from a specific range of pages.

        Args:
            start: Starting page number (0-indexed).
            end: Ending page number (exclusive).

        Returns:
            List of PageInfo objects for the specified range.
        """
        logger.debug(f"Extracting pages {start} to {end}")

        try:
            with fitz.open(str(self.pdf_path)) as self._doc:
                pages = []

                for page_num in range(start, min(end, len(self._doc))):
                    page_info = self._extract_page(page_num)
                    pages.append(page_info)

                return pages

        except Exception as e:
            raise PDFProcessingError(f"Failed to extract page range: {e}") from e

    def _extract_page(self, page_num: int) -> PageInfo:
        """
        Extract content and metadata from a single page.

        Args:
            page_num: Page number (0-indexed).

        Returns:
            PageInfo object with extracted content.
        """
        if self._doc is None:
            raise PDFProcessingError("Document not opened")

        page = self._doc[page_num]

        # Get page dimensions
        rect = page.rect
        width = rect.width
        height = rect.height

        # Extract text
        text = page.get_text("text")

        # Get images
        images = page.get_images(full=True)
        image_count = len(images)
        has_images = image_count > 0

        # Detect if page is scanned
        is_scanned = self._detect_scanned_page(text, images, page)

        # Extract font information
        fonts, font_sizes = self._extract_font_info(page)

        # Detect layout
        is_multi_column = self._detect_multi_column(page)

        # Extract headers and footers
        header_text, footer_text = self._extract_header_footer(page, text)

        # Detect page type
        page_type = self._classify_page_type(page_num, text, has_images)

        # Detect tables (basic heuristic)
        has_tables = self._detect_tables(page)

        # Calculate word count
        words = text.split()
        word_count = len(words)

        return PageInfo(
            page_num=page_num + 1,  # 1-indexed for user display
            text=text,
            char_count=len(text),
            word_count=word_count,
            has_images=has_images,
            image_count=image_count,
            is_scanned=is_scanned,
            fonts=fonts,
            font_sizes=font_sizes,
            page_width=width,
            page_height=height,
            page_type=page_type,
            has_tables=has_tables,
            is_multi_column=is_multi_column,
            header_text=header_text,
            footer_text=footer_text,
        )

    def _detect_scanned_page(self, text: str, images: list, page: fitz.Page) -> bool:
        """
        Detect if a page is scanned (image-based) rather than digital text.

        Args:
            text: Extracted text from page.
            images: List of images on the page.
            page: The fitz Page object.

        Returns:
            True if page appears to be scanned.
        """
        text_len = len(text.strip())

        # If very little text and has images, likely scanned
        if text_len < self.MIN_TEXT_CHARS_FOR_DIGITAL and len(images) > 0:
            return True

        # Check if single large image covers most of page
        if len(images) == 1:
            try:
                img_info = images[0]
                img_rect = page.get_image_bbox(img_info)
                if img_rect:
                    page_area = page.rect.width * page.rect.height
                    img_area = img_rect.width * img_rect.height
                    # If image covers > 80% of page and little text, it's scanned
                    if img_area / page_area > 0.8 and text_len < 200:
                        return True
            except Exception:
                pass

        return False

    def _extract_font_info(self, page: fitz.Page) -> tuple[list[str], list[float]]:
        """
        Extract font names and sizes from a page.

        Args:
            page: The fitz Page object.

        Returns:
            Tuple of (font names list, font sizes list).
        """
        fonts = set()
        font_sizes = set()

        try:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            font_name = span.get("font", "")
                            font_size = span.get("size", 0)
                            if font_name:
                                fonts.add(font_name)
                            if font_size > 0:
                                font_sizes.add(round(font_size, 1))
        except Exception:
            pass

        return sorted(fonts), sorted(font_sizes, reverse=True)

    def _detect_multi_column(self, page: fitz.Page) -> bool:
        """
        Detect if page has multi-column layout.

        Args:
            page: The fitz Page object.

        Returns:
            True if page appears to have multiple columns.
        """
        try:
            blocks = page.get_text("blocks")
            if len(blocks) < 2:
                return False

            # Get x-coordinates of block starts
            x_coords = [block[0] for block in blocks if block[4].strip()]  # x0 of non-empty blocks

            if len(x_coords) < 4:
                return False

            # Check if there are two distinct groups of x-coordinates
            x_coords_sorted = sorted(set(x_coords))
            page_width = page.rect.width

            # If blocks start at significantly different x positions, might be multi-column
            if len(x_coords_sorted) >= 2:
                # Check if there's a gap in the middle of the page
                mid_page = page_width / 2
                left_blocks = [x for x in x_coords_sorted if x < mid_page * 0.4]
                right_blocks = [x for x in x_coords_sorted if x > mid_page * 0.6]

                if len(left_blocks) > 1 and len(right_blocks) > 1:
                    return True
        except Exception:
            pass

        return False

    def _extract_header_footer(self, page: fitz.Page, full_text: str) -> tuple[str | None, str | None]:
        """
        Extract header and footer text from a page.

        Args:
            page: The fitz Page object.
            full_text: Full text of the page.

        Returns:
            Tuple of (header text, footer text).
        """
        header = None
        footer = None

        try:
            page_height = page.rect.height
            header_zone = page_height * 0.08  # Top 8%
            footer_zone = page_height * 0.92  # Bottom 8%

            blocks = page.get_text("blocks")

            for block in blocks:
                if len(block) >= 5:
                    y0, y1 = block[1], block[3]
                    text = block[4].strip() if isinstance(block[4], str) else ""

                    # Header detection
                    if y1 < header_zone and text and len(text) < 200:  # Headers are usually short
                        header = text

                    # Footer detection
                    if y0 > footer_zone and text and len(text) < 200:  # Footers are usually short
                        footer = text
        except Exception:
            pass

        return header, footer

    def _classify_page_type(self, page_num: int, text: str, has_images: bool) -> str:
        """
        Classify the type of page based on content.

        Args:
            page_num: Page number (0-indexed).
            text: Text content of the page.
            has_images: Whether page has images.

        Returns:
            Page type string: 'cover', 'toc', 'text', 'table', 'blank', 'appendix'.
        """
        text_stripped = text.strip()

        # Blank page detection
        if len(text_stripped) < 50:
            return "blank"

        # Cover page detection (typically first few pages)
        if page_num < 5:
            for pattern in self.COVER_PAGE_PATTERNS:
                if re.search(pattern, text_stripped):
                    return "cover"

        # Table of contents detection
        for pattern in self.TOC_PATTERNS:
            if re.search(pattern, text_stripped[:500]):  # Check beginning of page
                # Additional check: ToC pages often have many page numbers
                page_number_count = len(re.findall(r"\b\d{1,3}\b", text_stripped))
                line_count = len(text_stripped.split("\n"))
                if page_number_count > 10 or page_number_count / max(line_count, 1) > 0.3:
                    return "toc"

        # Appendix detection
        for pattern in self.APPENDIX_PATTERNS:
            if re.search(pattern, text_stripped[:200]):
                return "appendix"

        # Default to text
        return "text"

    def _detect_tables(self, page: fitz.Page) -> bool:
        """
        Basic heuristic to detect if page contains tables.

        Args:
            page: The fitz Page object.

        Returns:
            True if page likely contains tables.
        """
        try:
            # Get text blocks
            blocks = page.get_text("blocks")

            # Heuristic: Tables often have many short lines with numbers
            total_blocks = len(blocks)
            numeric_blocks = 0
            short_line_blocks = 0

            for block in blocks:
                if len(block) >= 5 and isinstance(block[4], str):
                    text = block[4].strip()
                    lines = text.split("\n")

                    # Count numeric content
                    if re.search(r"[\d,]+\.\d{2}|\d{1,3}(?:,\d{3})+", text):
                        numeric_blocks += 1

                    # Count blocks with short lines
                    avg_line_len = sum(len(l_val) for l_val in lines) / max(len(lines), 1)
                    if avg_line_len < 30 and len(lines) > 2:
                        short_line_blocks += 1

            # If significant portion has table-like characteristics
            if total_blocks > 0:
                table_ratio = (numeric_blocks + short_line_blocks) / total_blocks
                return table_ratio > 0.3

        except Exception:
            pass

        return False

    def detect_scanned_pages(self) -> list[int]:
        """
        Identify all scanned pages in the document.

        Returns:
            List of page numbers (1-indexed) that are scanned.
        """
        if not self.pages:
            self.extract_all_pages()

        scanned_pages = [page.page_num for page in self.pages if page.is_scanned]

        logger.info(f"Detected {len(scanned_pages)} scanned pages")
        return scanned_pages

    def get_text_by_page(self, page_num: int) -> str:
        """
        Get text content for a specific page.

        Args:
            page_num: Page number (1-indexed).

        Returns:
            Text content of the page.
        """
        if not self.pages:
            self.extract_all_pages()

        # Convert to 0-indexed
        idx = page_num - 1
        if 0 <= idx < len(self.pages):
            return self.pages[idx].text

        raise PDFProcessingError(f"Invalid page number: {page_num}")

    def get_all_text(self) -> str:
        """
        Get concatenated text from all pages.

        Returns:
            All text content joined by page breaks.
        """
        if not self.pages:
            self.extract_all_pages()

        return "\n\n".join(page.text for page in self.pages)

    def get_page_count(self) -> int:
        """
        Get total number of pages in the document.

        Returns:
            Number of pages.
        """
        if self.metadata:
            return self.metadata.page_count

        with fitz.open(str(self.pdf_path)) as doc:
            return len(doc)

    def get_pages_by_type(self, page_type: str) -> list[PageInfo]:
        """
        Get all pages of a specific type.

        Args:
            page_type: Type to filter by ('cover', 'toc', 'text', 'table', 'blank', 'appendix').

        Returns:
            List of PageInfo objects matching the type.
        """
        if not self.pages:
            self.extract_all_pages()

        return [page for page in self.pages if page.page_type == page_type]

    def get_processing_summary(self) -> dict:
        """
        Get a summary of processing results.

        Returns:
            Dictionary with processing statistics.
        """
        if not self.pages:
            self.extract_all_pages()

        return {
            "total_pages": len(self.pages),
            "scanned_pages": len([p for p in self.pages if p.is_scanned]),
            "blank_pages": len([p for p in self.pages if p.page_type == "blank"]),
            "pages_with_tables": len([p for p in self.pages if p.has_tables]),
            "pages_with_images": len([p for p in self.pages if p.has_images]),
            "total_word_count": sum(p.word_count for p in self.pages),
            "total_char_count": sum(p.char_count for p in self.pages),
            "page_types": {
                "cover": len([p for p in self.pages if p.page_type == "cover"]),
                "toc": len([p for p in self.pages if p.page_type == "toc"]),
                "text": len([p for p in self.pages if p.page_type == "text"]),
                "appendix": len([p for p in self.pages if p.page_type == "appendix"]),
                "blank": len([p for p in self.pages if p.page_type == "blank"]),
            },
        }
