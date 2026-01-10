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
from rhp_analyzer.ingestion.pipeline import (
    IngestionPipeline,
    PipelineCheckpoint,
    PipelineResult,
    PipelineStage,
)
from rhp_analyzer.ingestion.promoter_extractor import (
    PromoterDossier,
    PromoterExtractor,
)
from rhp_analyzer.ingestion.pre_ipo_analyzer import (
    PreIPOInvestor,
    PreIPOInvestorAnalyzer,
)
from rhp_analyzer.ingestion.order_book_analyzer import (
    OrderBookAnalysis,
    OrderBookAnalyzer,
)
from rhp_analyzer.ingestion.objects_tracker import (
    ObjectsOfIssueAnalysis,
    ObjectsOfIssueTracker,
)
from rhp_analyzer.ingestion.contingent_liability_analyzer import (
    ContingentLiabilityAnalysis,
    ContingentLiabilityCategorizer,
)
from rhp_analyzer.ingestion.stub_analyzer import (
    StubPeriodAnalysis,
    StubPeriodAnalyzer,
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
    # Ingestion Pipeline
    "IngestionPipeline",
    "PipelineCheckpoint",
    "PipelineResult",
    "PipelineStage",
    # Promoter & Investor analyzers
    "PromoterExtractor",
    "PromoterDossier",
    "PreIPOInvestor",
    "PreIPOInvestorAnalyzer",
    # Specialized analyzers
    "OrderBookAnalysis",
    "OrderBookAnalyzer",
    "ObjectsOfIssueTracker",
    "ObjectsOfIssueAnalysis",
    "ContingentLiabilityCategorizer",
    "ContingentLiabilityAnalysis",
    "StubPeriodAnalyzer",
    "StubPeriodAnalysis",
]
