"""
Ingestion module for RHP Analyzer.

This module provides PDF processing and OCR capabilities for
Red Herring Prospectus document analysis.
"""

from rhp_analyzer.ingestion.ocr_processor import (
    OCRConfig,
    OCRError,
    OCRProcessingError,
    OCRProcessor,
    OCRResult,
    TesseractNotFoundError,
    ocr_pdf_page,
)
from rhp_analyzer.ingestion.pdf_processor import (
    PageInfo,
    PDFMetadata,
    PDFProcessingError,
    PDFProcessor,
    PDFValidationError,
)

__all__ = [
    # PDF Processor
    "PDFProcessor",
    "PDFProcessingError",
    "PDFValidationError",
    "PageInfo",
    "PDFMetadata",
    # OCR Processor
    "OCRProcessor",
    "OCRConfig",
    "OCRResult",
    "OCRError",
    "TesseractNotFoundError",
    "OCRProcessingError",
    "ocr_pdf_page",
]
