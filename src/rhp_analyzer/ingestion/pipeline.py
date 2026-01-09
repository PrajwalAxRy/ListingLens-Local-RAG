"""
Ingestion Pipeline Orchestrator for RHP Analyzer.

This module orchestrates the complete document ingestion workflow:
PDF → Text → Tables → Sections → Entities

It provides checkpoint/resume capability for fault tolerance and
emits progress events at each stage.
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from rhp_analyzer.ingestion.entity_extractor import EntityExtractor, Entity
from rhp_analyzer.ingestion.pdf_processor import PDFProcessor, PDFProcessingError, PageInfo
from rhp_analyzer.ingestion.section_mapper import SectionMapper, SectionTree
from rhp_analyzer.ingestion.table_extractor import TableExtractor, Table
from rhp_analyzer.utils.progress import ProgressTracker


class PipelineStage(Enum):
    """Enumeration of pipeline stages."""
    INITIALIZED = "initialized"
    PDF_VALIDATED = "pdf_validated"
    PAGES_EXTRACTED = "pages_extracted"
    TABLES_EXTRACTED = "tables_extracted"
    SECTIONS_MAPPED = "sections_mapped"
    ENTITIES_EXTRACTED = "entities_extracted"
    COMPLETED = "completed"


@dataclass
class PipelineCheckpoint:
    """
    Represents a checkpoint in the pipeline execution.
    
    Used for saving and resuming pipeline state.
    """
    document_id: str
    pdf_path: str
    current_stage: PipelineStage
    completed_stages: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    stage_durations: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to dictionary for JSON serialization."""
        data = asdict(self)
        data["current_stage"] = self.current_stage.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineCheckpoint":
        """Create checkpoint from dictionary."""
        data["current_stage"] = PipelineStage(data["current_stage"])
        return cls(**data)


@dataclass
class PipelineResult:
    """
    Contains all outputs from the ingestion pipeline.
    
    This is the complete result of processing an RHP document.
    """
    document_id: str
    pdf_path: str
    success: bool
    
    # Extracted data
    pages: list[PageInfo] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    sections: SectionTree | None = None
    entities: list[Entity] = field(default_factory=list)
    
    # Metadata
    total_pages: int = 0
    total_tables: int = 0
    total_sections: int = 0
    total_entities: int = 0
    
    # Timing
    started_at: str = ""
    completed_at: str = ""
    total_duration_seconds: float = 0.0
    stage_durations: dict[str, float] = field(default_factory=dict)
    
    # Errors and warnings
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "document_id": self.document_id,
            "pdf_path": self.pdf_path,
            "success": self.success,
            "total_pages": self.total_pages,
            "total_tables": self.total_tables,
            "total_sections": self.total_sections,
            "total_entities": self.total_entities,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_seconds": self.total_duration_seconds,
            "stage_durations": self.stage_durations,
            "errors": self.errors,
            "warnings": self.warnings,
            # Note: pages, tables, sections, entities are not serialized 
            # to avoid circular references and large payloads
        }


class IngestionPipeline:
    """
    Orchestrates the complete RHP document ingestion process.
    
    Pipeline Stages:
    1. Validate PDF - Check integrity and metadata
    2. Extract Pages - Extract text from all pages
    3. Extract Tables - Detect and extract tables
    4. Map Sections - Build document hierarchy
    5. Extract Entities - Extract named entities
    
    Provides checkpoint/resume capability for fault tolerance.
    """
    
    # Number of phases for progress tracking
    TOTAL_PHASES = 5
    
    def __init__(
        self,
        checkpoint_dir: Path | str | None = None,
        enable_checkpoints: bool = True,
        show_progress: bool = True,
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            checkpoint_dir: Directory for checkpoint files. If None, uses data/checkpoints/
            enable_checkpoints: Whether to save checkpoints after each stage
            show_progress: Whether to show progress bars
        """
        self.enable_checkpoints = enable_checkpoints
        self.show_progress = show_progress
        
        # Set up checkpoint directory
        if checkpoint_dir is None:
            # Default to project data directory
            self.checkpoint_dir = Path("data/checkpoints")
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        
        if self.enable_checkpoints:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.pdf_processor: PDFProcessor | None = None
        self.table_extractor = TableExtractor()
        self.section_mapper = SectionMapper()
        self.entity_extractor = EntityExtractor()
        
        # Progress tracker
        self.progress: ProgressTracker | None = None
        
        # Current state
        self._checkpoint: PipelineCheckpoint | None = None
        self._current_document_id: str = ""
        self._stage_start_time: datetime | None = None
        
        # Cached data between stages
        self._pages: list[PageInfo] = []
        self._tables: list[Table] = []
        self._sections: SectionTree | None = None
        self._entities: list[Entity] = []
        
        logger.debug("IngestionPipeline initialized")
    
    def _generate_document_id(self, pdf_path: Path) -> str:
        """Generate a unique document ID from the PDF path and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = pdf_path.stem.replace(" ", "_")[:50]
        return f"{safe_name}_{timestamp}"
    
    def _get_checkpoint_path(self, document_id: str) -> Path:
        """Get the checkpoint file path for a document."""
        return self.checkpoint_dir / f"{document_id}_checkpoint.json"
    
    def _save_checkpoint(self) -> None:
        """Save current checkpoint to disk."""
        if not self.enable_checkpoints or self._checkpoint is None:
            return
        
        checkpoint_path = self._get_checkpoint_path(self._checkpoint.document_id)
        
        try:
            self._checkpoint.last_updated = datetime.now().isoformat()
            
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(self._checkpoint.to_dict(), f, indent=2)
            
            logger.debug(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, document_id: str) -> PipelineCheckpoint | None:
        """Load a checkpoint from disk if it exists."""
        if not self.enable_checkpoints:
            return None
        
        checkpoint_path = self._get_checkpoint_path(document_id)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            checkpoint = PipelineCheckpoint.from_dict(data)
            logger.info(f"Loaded checkpoint for {document_id}, stage: {checkpoint.current_stage.value}")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def _clear_checkpoint(self, document_id: str) -> None:
        """Clear the checkpoint file on successful completion."""
        if not self.enable_checkpoints:
            return
        
        checkpoint_path = self._get_checkpoint_path(document_id)
        
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.debug(f"Checkpoint cleared: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to clear checkpoint: {e}")
    
    def _start_stage(self, stage: PipelineStage, phase_name: str, total_steps: int | None = None) -> None:
        """Start a new pipeline stage with timing and progress."""
        self._stage_start_time = datetime.now()
        
        if self._checkpoint:
            self._checkpoint.current_stage = stage
            self._save_checkpoint()
        
        if self.progress:
            self.progress.start_phase(phase_name, total_steps)
        
        logger.info(f"Starting stage: {phase_name}")
    
    def _complete_stage(self, stage: PipelineStage) -> None:
        """Complete the current stage with timing."""
        if self._stage_start_time and self._checkpoint:
            duration = (datetime.now() - self._stage_start_time).total_seconds()
            self._checkpoint.stage_durations[stage.value] = duration
            self._checkpoint.completed_stages.append(stage.value)
            self._save_checkpoint()
        
        if self.progress:
            self.progress.complete_phase()
        
        logger.info(f"Completed stage: {stage.value}")
    
    def _should_skip_stage(self, stage: PipelineStage) -> bool:
        """Check if a stage should be skipped based on checkpoint."""
        if self._checkpoint is None:
            return False
        return stage.value in self._checkpoint.completed_stages
    
    def run(
        self,
        pdf_path: str | Path,
        document_id: str | None = None,
        resume: bool = True,
    ) -> PipelineResult:
        """
        Execute the complete ingestion pipeline.
        
        Args:
            pdf_path: Path to the RHP PDF document
            document_id: Optional document ID. If None, auto-generated.
            resume: Whether to resume from checkpoint if available
        
        Returns:
            PipelineResult with all extracted data
        
        Raises:
            PDFProcessingError: If PDF validation or processing fails
        """
        pdf_path = Path(pdf_path)
        started_at = datetime.now()
        
        # Generate or use provided document ID
        if document_id is None:
            document_id = self._generate_document_id(pdf_path)
        
        self._current_document_id = document_id
        
        logger.info(f"Starting ingestion pipeline for: {pdf_path}")
        logger.info(f"Document ID: {document_id}")
        
        # Try to resume from checkpoint
        if resume and self.enable_checkpoints:
            existing_checkpoint = self._load_checkpoint(document_id)
            if existing_checkpoint:
                self._checkpoint = existing_checkpoint
                logger.info(f"Resuming from stage: {self._checkpoint.current_stage.value}")
        
        # Create new checkpoint if not resuming
        if self._checkpoint is None:
            self._checkpoint = PipelineCheckpoint(
                document_id=document_id,
                pdf_path=str(pdf_path),
                current_stage=PipelineStage.INITIALIZED,
            )
        
        # Initialize progress tracker
        if self.show_progress:
            self.progress = ProgressTracker(total_phases=self.TOTAL_PHASES)
            self.progress.start(f"Processing: {pdf_path.name}")
        
        # Initialize result
        result = PipelineResult(
            document_id=document_id,
            pdf_path=str(pdf_path),
            success=False,
            started_at=started_at.isoformat(),
        )
        
        try:
            # Stage 1: Validate PDF
            if not self._should_skip_stage(PipelineStage.PDF_VALIDATED):
                self._run_pdf_validation(pdf_path, result)
            else:
                logger.info("Skipping PDF validation (already completed)")
                # Need to reinitialize PDF processor for subsequent stages
                self.pdf_processor = PDFProcessor(pdf_path)
                self.pdf_processor.validate()
            
            # Stage 2: Extract Pages
            if not self._should_skip_stage(PipelineStage.PAGES_EXTRACTED):
                self._run_page_extraction(result)
            else:
                logger.info("Skipping page extraction (already completed)")
                # Load cached pages if available (would need persistent storage)
            
            # Stage 3: Extract Tables
            if not self._should_skip_stage(PipelineStage.TABLES_EXTRACTED):
                self._run_table_extraction(pdf_path, result)
            else:
                logger.info("Skipping table extraction (already completed)")
            
            # Stage 4: Map Sections
            if not self._should_skip_stage(PipelineStage.SECTIONS_MAPPED):
                self._run_section_mapping(result)
            else:
                logger.info("Skipping section mapping (already completed)")
            
            # Stage 5: Extract Entities
            if not self._should_skip_stage(PipelineStage.ENTITIES_EXTRACTED):
                self._run_entity_extraction(result)
            else:
                logger.info("Skipping entity extraction (already completed)")
            
            # Mark as complete
            self._checkpoint.current_stage = PipelineStage.COMPLETED
            self._checkpoint.completed_stages.append(PipelineStage.COMPLETED.value)
            
            result.success = True
            
            # Clear checkpoint on success
            self._clear_checkpoint(document_id)
            
        except PDFProcessingError as e:
            logger.error(f"PDF processing error: {e}")
            result.errors.append(str(e))
            self._checkpoint.errors.append(str(e))
            self._save_checkpoint()
            
        except Exception as e:
            logger.exception(f"Unexpected error in pipeline: {e}")
            result.errors.append(f"Unexpected error: {e}")
            self._checkpoint.errors.append(str(e))
            self._save_checkpoint()
        
        finally:
            # Finalize result
            completed_at = datetime.now()
            result.completed_at = completed_at.isoformat()
            result.total_duration_seconds = (completed_at - started_at).total_seconds()
            
            if self._checkpoint:
                result.stage_durations = self._checkpoint.stage_durations.copy()
                result.warnings = self._checkpoint.warnings.copy()
            
            # Stop progress tracker
            if self.progress:
                self.progress.stop()
            
            # Log summary
            self._log_summary(result)
        
        return result
    
    def _run_pdf_validation(self, pdf_path: Path, result: PipelineResult) -> None:
        """Stage 1: Validate the PDF document."""
        self._start_stage(PipelineStage.PDF_VALIDATED, "PDF Validation")
        
        self.pdf_processor = PDFProcessor(pdf_path)
        metadata = self.pdf_processor.validate()
        
        result.total_pages = metadata.page_count
        
        if metadata.is_encrypted:
            raise PDFProcessingError(f"PDF is encrypted: {pdf_path}")
        
        # Check if document has significant scanned pages
        if metadata.scanned_page_count > metadata.page_count * 0.5:
            self._checkpoint.warnings.append("Document appears to be scanned - OCR may be required")
            result.warnings.append("Document appears to be scanned - OCR may be required")
        
        self._complete_stage(PipelineStage.PDF_VALIDATED)
    
    def _run_page_extraction(self, result: PipelineResult) -> None:
        """Stage 2: Extract text from all pages."""
        if self.pdf_processor is None:
            raise PDFProcessingError("PDF processor not initialized")
        
        self._start_stage(
            PipelineStage.PAGES_EXTRACTED,
            "Page Extraction",
            total_steps=result.total_pages,
        )
        
        self._pages = self.pdf_processor.extract_all_pages()
        result.pages = self._pages
        result.total_pages = len(self._pages)
        
        # Update progress for each page
        if self.progress:
            self.progress.update(advance=len(self._pages))
        
        self._complete_stage(PipelineStage.PAGES_EXTRACTED)
    
    def _run_table_extraction(self, pdf_path: Path, result: PipelineResult) -> None:
        """Stage 3: Extract tables from the document."""
        self._start_stage(
            PipelineStage.TABLES_EXTRACTED,
            "Table Extraction",
            total_steps=result.total_pages,
        )
        
        self._tables = self.table_extractor.extract_tables(
            str(pdf_path),
            classify=True,
        )
        result.tables = self._tables
        result.total_tables = len(self._tables)
        
        if self.progress:
            self.progress.update(advance=result.total_pages)
        
        self._complete_stage(PipelineStage.TABLES_EXTRACTED)
    
    def _run_section_mapping(self, result: PipelineResult) -> None:
        """Stage 4: Build document section hierarchy."""
        self._start_stage(
            PipelineStage.SECTIONS_MAPPED,
            "Section Mapping",
        )
        
        self._sections = self.section_mapper.build_hierarchy(self._pages)
        result.sections = self._sections
        result.total_sections = len(self._sections.all_sections) if self._sections else 0
        
        self._complete_stage(PipelineStage.SECTIONS_MAPPED)

    def _run_entity_extraction(self, result: PipelineResult) -> None:
        """Stage 5: Extract named entities from all pages."""
        self._start_stage(
            PipelineStage.ENTITIES_EXTRACTED,
            "Entity Extraction",
            total_steps=len(self._pages),
        )
        
        all_entities: list[Entity] = []
        
        for page in self._pages:
            if hasattr(page, 'text') and page.text:
                page_entities = self.entity_extractor.extract_entities(
                    page.text,
                    page_num=page.page_num,
                )
                all_entities.extend(page_entities)
            
            if self.progress:
                self.progress.update(advance=1)
        
        # Deduplicate entities
        self._entities = self._deduplicate_entities(all_entities)
        result.entities = self._entities
        result.total_entities = len(self._entities)
        
        self._complete_stage(PipelineStage.ENTITIES_EXTRACTED)
    
    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """
        Deduplicate entities by normalized text and type.
        
        Args:
            entities: List of entities to deduplicate
        
        Returns:
            Deduplicated list of entities
        """
        seen = set()
        unique = []
        
        for entity in entities:
            # Create a key based on normalized text and type
            key = (
                entity.normalized_text.lower() if entity.normalized_text else entity.text.lower(),
                entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
            )
            
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        
        return unique
    
    def _log_summary(self, result: PipelineResult) -> None:
        """Log a summary of the pipeline execution."""
        status = "SUCCESS" if result.success else "FAILED"
        
        logger.info("=" * 60)
        logger.info(f"Pipeline {status} for: {result.pdf_path}")
        logger.info(f"Document ID: {result.document_id}")
        logger.info(f"Total Duration: {result.total_duration_seconds:.2f}s")
        logger.info("-" * 40)
        logger.info(f"Pages Extracted: {result.total_pages}")
        logger.info(f"Tables Extracted: {result.total_tables}")
        logger.info(f"Sections Mapped: {result.total_sections}")
        logger.info(f"Entities Extracted: {result.total_entities}")
        
        if result.stage_durations:
            logger.info("-" * 40)
            logger.info("Stage Durations:")
            for stage, duration in result.stage_durations.items():
                logger.info(f"  {stage}: {duration:.2f}s")
        
        if result.warnings:
            logger.info("-" * 40)
            logger.warning(f"Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                logger.warning(f"  - {warning}")
        
        if result.errors:
            logger.info("-" * 40)
            logger.error(f"Errors ({len(result.errors)}):")
            for error in result.errors:
                logger.error(f"  - {error}")
        
        logger.info("=" * 60)
    
    def get_checkpoint_info(self, document_id: str) -> PipelineCheckpoint | None:
        """
        Get information about an existing checkpoint.
        
        Args:
            document_id: The document ID to check
        
        Returns:
            PipelineCheckpoint if exists, None otherwise
        """
        return self._load_checkpoint(document_id)
    
    def clear_checkpoint(self, document_id: str) -> bool:
        """
        Manually clear a checkpoint.
        
        Args:
            document_id: The document ID to clear
        
        Returns:
            True if checkpoint was cleared, False if not found
        """
        checkpoint_path = self._get_checkpoint_path(document_id)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Checkpoint cleared: {document_id}")
            return True
        return False
    
    def list_checkpoints(self) -> list[str]:
        """
        List all available checkpoints.
        
        Returns:
            List of document IDs with checkpoints
        """
        if not self.checkpoint_dir.exists():
            return []
        
        checkpoints = []
        for path in self.checkpoint_dir.glob("*_checkpoint.json"):
            doc_id = path.stem.replace("_checkpoint", "")
            checkpoints.append(doc_id)
        
        return checkpoints
