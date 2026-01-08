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
from rhp_analyzer.ingestion.section_mapper import (
    STANDARD_RHP_SECTIONS,
    Section,
    SectionMapper,
    SectionTree,
    SectionType,
    normalize_section_name,
)
from rhp_analyzer.ingestion.table_extractor import (
    FinancialData,
    FinancialTableParser,
    Table,
    TableClassifier,
    TableExtractor,
    TableType,
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
    # Table Extractor
    "TableExtractor",
    "TableClassifier",
    "FinancialTableParser",
    "Table",
    "TableType",
    "FinancialData",
    # Section Mapper
    "SectionMapper",
    "Section",
    "SectionTree",
    "SectionType",
    "STANDARD_RHP_SECTIONS",
    "normalize_section_name",
]
