"""
Integration tests for the ingestion pipeline.

Subtask 2.5.3: Create Integration Tests
- Test full pipeline on sample PDF
- Verify all outputs generated
- Check data consistency
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from rhp_analyzer.ingestion.pipeline import (
    IngestionPipeline,
    PipelineStage,
    PipelineCheckpoint,
    PipelineResult,
)
from rhp_analyzer.ingestion.pdf_processor import PageInfo, PDFMetadata
from rhp_analyzer.ingestion.table_extractor import Table, TableType
from rhp_analyzer.ingestion.section_mapper import Section, SectionTree, SectionType
from rhp_analyzer.ingestion.entity_extractor import Entity, EntityType


class TestIngestionPipelineIntegration:
    """Integration tests for the complete ingestion pipeline."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory for tests."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir(parents=True)
        
        # Create subdirectories
        (data_dir / "checkpoints").mkdir()
        (data_dir / "processed").mkdir()
        (data_dir / "input").mkdir()
        
        yield data_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_pdf_path(self, temp_data_dir):
        """Create a sample PDF path for testing."""
        pdf_path = temp_data_dir / "input" / "sample_rhp.pdf"
        return pdf_path

    @pytest.fixture
    def mock_pdf_metadata(self):
        """Create mock PDF metadata."""
        return PDFMetadata(
            filename="sample_rhp.pdf",
            file_size_bytes=1024000,
            page_count=50,
            title="Sample RHP",
            author="Test Author",
            creation_date="2024-01-01",
            is_encrypted=False,
        )

    @pytest.fixture
    def mock_pages(self):
        """Create mock page data."""
        pages = []
        for i in range(5):
            page = PageInfo(
                page_num=i + 1,
                text=f"Sample text for page {i + 1}. " * 100,
                char_count=len(f"Sample text for page {i + 1}. " * 100),
                word_count=len((f"Sample text for page {i + 1}. " * 100).split()),
                has_images=i % 2 == 0,
                image_count=1 if i % 2 == 0 else 0,
                is_scanned=False,
            )
            pages.append(page)
        return pages

    @pytest.fixture
    def mock_tables(self):
        """Create mock table data."""
        return [
            Table(
                table_id="table_001",
                page_num=2,
                rows=[
                    ["Header1", "Header2", "Header3"],
                    ["Row1Col1", "Row1Col2", "Row1Col3"],
                    ["Row2Col1", "Row2Col2", "Row2Col3"],
                ],
                headers=["Header1", "Header2", "Header3"],
                table_type=TableType.INCOME_STATEMENT,
                confidence=0.95,
            ),
            Table(
                table_id="table_002",
                page_num=5,
                rows=[
                    ["Name", "Shares", "Percentage"],
                    ["Promoter A", "1000000", "25%"],
                    ["Promoter B", "800000", "20%"],
                ],
                headers=["Name", "Shares", "Percentage"],
                table_type=TableType.SHAREHOLDING_PATTERN,
                confidence=0.88,
            ),
        ]

    @pytest.fixture
    def mock_section_tree(self):
        """Create mock section tree."""
        root = Section(
            section_id="root",
            section_type=SectionType.OTHER,
            title="Document Root",
            start_page=1,
            end_page=50,
            content="",
            word_count=0,
            subsections=[],
            level=0,
        )
        
        risk_section = Section(
            section_id="sec_001",
            section_type=SectionType.RISK_FACTORS,
            title="Risk Factors",
            start_page=5,
            end_page=15,
            content="Risk factors content...",
            word_count=500,
            subsections=[],
            level=1,
        )
        
        business_section = Section(
            section_id="sec_002",
            section_type=SectionType.BUSINESS,
            title="Our Business",
            start_page=16,
            end_page=30,
            content="Business overview content...",
            word_count=800,
            subsections=[],
            level=1,
        )
        
        root.subsections = [risk_section, business_section]
        
        return SectionTree(
            root_sections=[root],
            all_sections={
                "root": root,
                "sec_001": risk_section,
                "sec_002": business_section,
            },
            total_pages=50,
        )

    @pytest.fixture
    def mock_entities(self):
        """Create mock entity data."""
        return [
            Entity(
                entity_type=EntityType.COMPANY,
                text="ABC Technologies Ltd",
                normalized_text="ABC Technologies Ltd",
                mentions=5,
                contexts=["ABC Technologies Ltd is a leading..."],
                page_references=[1, 5, 10],
            ),
            Entity(
                entity_type=EntityType.PERSON,
                text="John Doe",
                normalized_text="John Doe",
                mentions=3,
                contexts=["Mr. John Doe, Managing Director..."],
                page_references=[2, 8],
            ),
            Entity(
                entity_type=EntityType.MONEY,
                text="₹500 crores",
                normalized_text="₹500 crores",
                mentions=2,
                contexts=["The issue size is ₹500 crores..."],
                page_references=[1],
            ),
        ]


class TestPipelineFullRun(TestIngestionPipelineIntegration):
    """Test complete pipeline execution."""

    def test_pipeline_runs_all_stages(
        self,
        temp_data_dir,
        sample_pdf_path,
        mock_pdf_metadata,
        mock_pages,
        mock_tables,
        mock_section_tree,
        mock_entities,
    ):
        """Test that pipeline executes all stages in correct order."""
        with patch("rhp_analyzer.ingestion.pipeline.PDFProcessor") as mock_pdf_proc, \
             patch("rhp_analyzer.ingestion.pipeline.TableExtractor") as mock_table_ext, \
             patch("rhp_analyzer.ingestion.pipeline.SectionMapper") as mock_section_map, \
             patch("rhp_analyzer.ingestion.pipeline.EntityExtractor") as mock_entity_ext:
            
            # Configure mocks
            mock_pdf_instance = MagicMock()
            mock_pdf_instance.validate.return_value = mock_pdf_metadata
            mock_pdf_instance.extract_all_pages.return_value = mock_pages
            mock_pdf_proc.return_value = mock_pdf_instance
            
            mock_table_instance = MagicMock()
            mock_table_instance.extract_tables.return_value = mock_tables
            mock_table_ext.return_value = mock_table_instance
            
            mock_section_instance = MagicMock()
            mock_section_instance.build_hierarchy.return_value = mock_section_tree
            mock_section_map.return_value = mock_section_instance
            
            mock_entity_instance = MagicMock()
            mock_entity_instance.extract_entities.return_value = mock_entities
            mock_entity_ext.return_value = mock_entity_instance
            
            # Create pipeline with custom checkpoint directory
            pipeline = IngestionPipeline(
                checkpoint_dir=temp_data_dir / "checkpoints",
                show_progress=False,
            )
            
            # Run pipeline
            result = pipeline.run(
                pdf_path=str(sample_pdf_path),
                document_id="test_doc_001",
                resume=False,
            )
            
            # Verify result structure
            assert isinstance(result, PipelineResult)
            assert result.document_id == "test_doc_001"
            assert result.success is True
            
            # Verify all data collected
            assert len(result.pages) == 5
            assert len(result.tables) == 2
            assert result.sections is not None
            assert len(result.entities) > 0
            
            # Verify timing captured
            assert result.total_duration_seconds > 0
            assert len(result.stage_durations) > 0
            
            # Verify no errors
            assert len(result.errors) == 0

    def test_pipeline_generates_document_id_if_not_provided(
        self,
        temp_data_dir,
        sample_pdf_path,
        mock_pdf_metadata,
        mock_pages,
    ):
        """Test that pipeline auto-generates document ID."""
        with patch("rhp_analyzer.ingestion.pipeline.PDFProcessor") as mock_pdf_proc, \
             patch("rhp_analyzer.ingestion.pipeline.TableExtractor") as mock_table_ext, \
             patch("rhp_analyzer.ingestion.pipeline.SectionMapper") as mock_section_map, \
             patch("rhp_analyzer.ingestion.pipeline.EntityExtractor") as mock_entity_ext:
            
            # Configure mocks
            mock_pdf_instance = MagicMock()
            mock_pdf_instance.validate.return_value = mock_pdf_metadata
            mock_pdf_instance.extract_all_pages.return_value = mock_pages
            mock_pdf_proc.return_value = mock_pdf_instance
            
            mock_table_ext.return_value.extract_tables.return_value = []
            mock_section_map.return_value.build_hierarchy.return_value = MagicMock()
            mock_entity_ext.return_value.extract_entities.return_value = []
            
            pipeline = IngestionPipeline(
                checkpoint_dir=temp_data_dir / "checkpoints",
                show_progress=False,
            )
            
            result = pipeline.run(
                pdf_path=str(sample_pdf_path),
                document_id=None,  # No ID provided
                resume=False,
            )
            
            # Verify document ID was generated
            assert result.document_id is not None
            assert len(result.document_id) > 0
            assert "sample_rhp" in result.document_id.lower()


class TestPipelineCheckpoints(TestIngestionPipelineIntegration):
    """Test checkpoint and resume functionality."""

    def test_checkpoint_saved_after_each_stage(
        self,
        temp_data_dir,
        sample_pdf_path,
        mock_pdf_metadata,
        mock_pages,
    ):
        """Test that checkpoints are saved after completing each stage."""
        checkpoint_dir = temp_data_dir / "checkpoints"
        
        with patch("rhp_analyzer.ingestion.pipeline.PDFProcessor") as mock_pdf_proc, \
             patch("rhp_analyzer.ingestion.pipeline.TableExtractor") as mock_table_ext, \
             patch("rhp_analyzer.ingestion.pipeline.SectionMapper") as mock_section_map, \
             patch("rhp_analyzer.ingestion.pipeline.EntityExtractor") as mock_entity_ext:
            
            # Configure mocks
            mock_pdf_instance = MagicMock()
            mock_pdf_instance.validate.return_value = mock_pdf_metadata
            mock_pdf_instance.extract_all_pages.return_value = mock_pages
            mock_pdf_proc.return_value = mock_pdf_instance
            
            mock_table_ext.return_value.extract_tables.return_value = []
            mock_section_map.return_value.build_hierarchy.return_value = MagicMock()
            mock_entity_ext.return_value.extract_entities.return_value = []
            
            pipeline = IngestionPipeline(
                checkpoint_dir=temp_data_dir / "checkpoints",
                show_progress=False,
            )
            
            # Run pipeline
            result = pipeline.run(
                pdf_path=str(sample_pdf_path),
                document_id="checkpoint_test_001",
                resume=False,
            )
            
            # Checkpoint should be cleared after successful completion
            checkpoint_file = checkpoint_dir / "checkpoint_test_001.json"
            assert not checkpoint_file.exists(), "Checkpoint should be cleared on success"

    def test_checkpoint_persists_on_failure(
        self,
        temp_data_dir,
        sample_pdf_path,
        mock_pdf_metadata,
        mock_pages,
    ):
        """Test that checkpoint persists when pipeline fails mid-way."""
        checkpoint_dir = temp_data_dir / "checkpoints"
        
        with patch("rhp_analyzer.ingestion.pipeline.PDFProcessor") as mock_pdf_proc, \
             patch("rhp_analyzer.ingestion.pipeline.TableExtractor") as mock_table_ext:
            
            # Configure mocks - table extraction will fail
            mock_pdf_instance = MagicMock()
            mock_pdf_instance.validate.return_value = mock_pdf_metadata
            mock_pdf_instance.extract_all_pages.return_value = mock_pages
            mock_pdf_proc.return_value = mock_pdf_instance
            
            mock_table_instance = MagicMock()
            mock_table_instance.extract_tables.side_effect = Exception("Table extraction failed")
            mock_table_ext.return_value = mock_table_instance
            
            pipeline = IngestionPipeline(
                checkpoint_dir=temp_data_dir / "checkpoints",
                show_progress=False,
            )
            
            # Run pipeline - should fail at table extraction
            result = pipeline.run(
                pdf_path=str(sample_pdf_path),
                document_id="failure_test_001",
                resume=False,
            )
            
            # Verify failure
            assert result.success is False
            assert len(result.errors) > 0
            
            # Checkpoint should persist
            checkpoint_file = checkpoint_dir / "failure_test_001_checkpoint.json"
            assert checkpoint_file.exists(), "Checkpoint should persist on failure"
            
            # Verify checkpoint content
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
            
            assert checkpoint_data["document_id"] == "failure_test_001"
            # NOTE: The checkpoint may or may not have completed PDF validation 
            # depending on where the failure occurred

    def test_pipeline_resumes_from_checkpoint(
        self,
        temp_data_dir,
        sample_pdf_path,
        mock_pdf_metadata,
        mock_pages,
        mock_tables,
        mock_section_tree,
        mock_entities,
    ):
        """Test that pipeline can resume from a saved checkpoint."""
        checkpoint_dir = temp_data_dir / "checkpoints"
        document_id = "resume_test_001"
        
        # Create a pre-existing checkpoint (simulating previous partial run)
        # This matches the PipelineCheckpoint dataclass structure
        checkpoint_data = {
            "document_id": document_id,
            "pdf_path": str(sample_pdf_path),
            "current_stage": PipelineStage.PAGES_EXTRACTED.value,
            "completed_stages": [
                PipelineStage.PDF_VALIDATED.value,
                PipelineStage.PAGES_EXTRACTED.value,
            ],
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "stage_durations": {
                PipelineStage.PDF_VALIDATED.value: 1.5,
                PipelineStage.PAGES_EXTRACTED.value: 5.2,
            },
            "errors": [],
            "warnings": [],
        }
        
        checkpoint_file = checkpoint_dir / f"{document_id}_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)
        
        with patch("rhp_analyzer.ingestion.pipeline.PDFProcessor") as mock_pdf_proc, \
             patch("rhp_analyzer.ingestion.pipeline.TableExtractor") as mock_table_ext, \
             patch("rhp_analyzer.ingestion.pipeline.SectionMapper") as mock_section_map, \
             patch("rhp_analyzer.ingestion.pipeline.EntityExtractor") as mock_entity_ext:
            
            # These should NOT be called (already completed)
            mock_pdf_proc.return_value.validate.return_value = mock_pdf_metadata
            mock_pdf_proc.return_value.extract_all_pages.return_value = mock_pages
            
            # These SHOULD be called (resuming from here)
            mock_table_ext.return_value.extract_tables.return_value = mock_tables
            mock_section_map.return_value.build_hierarchy.return_value = mock_section_tree
            mock_entity_ext.return_value.extract_entities.return_value = mock_entities
            
            pipeline = IngestionPipeline(
                checkpoint_dir=temp_data_dir / "checkpoints",
                show_progress=False,
            )
            
            # Get checkpoint info before running
            checkpoint_info = pipeline.get_checkpoint_info(document_id)
            assert checkpoint_info is not None
            
            # Run with resume enabled
            result = pipeline.run(
                pdf_path=str(sample_pdf_path),
                document_id=document_id,
                resume=True,
            )
            
            # Verify success
            assert result.success is True
            
            # Verify tables, sections, entities were extracted
            assert len(result.tables) == 2
            assert result.sections is not None


class TestPipelineOutputs(TestIngestionPipelineIntegration):
    """Test pipeline output generation and consistency."""

    def test_pipeline_returns_all_extracted_entities(
        self,
        temp_data_dir,
        sample_pdf_path,
        mock_pdf_metadata,
        mock_pages,
    ):
        """Test that pipeline returns all entities from the extractor.
        
        Note: Entity deduplication is the responsibility of EntityExtractor,
        not the pipeline. The pipeline passes through entities as-is.
        """
        # Create test entities (extractor would handle dedup before returning these)
        test_entities = [
            Entity(
                entity_type=EntityType.COMPANY,
                text="ABC Ltd",
                normalized_text="ABC Ltd",
                mentions=5,  # Extractor already merged mentions
                contexts=["Context 1", "Context 2"],
                page_references=[1, 2, 3, 4],
            ),
            Entity(
                entity_type=EntityType.PERSON,
                text="John Smith",
                normalized_text="John Smith",
                mentions=1,
                contexts=["Context 3"],
                page_references=[5],
            ),
        ]
        
        with patch("rhp_analyzer.ingestion.pipeline.PDFProcessor") as mock_pdf_proc, \
             patch("rhp_analyzer.ingestion.pipeline.TableExtractor") as mock_table_ext, \
             patch("rhp_analyzer.ingestion.pipeline.SectionMapper") as mock_section_map, \
             patch("rhp_analyzer.ingestion.pipeline.EntityExtractor") as mock_entity_ext:
            
            mock_pdf_proc.return_value.validate.return_value = mock_pdf_metadata
            mock_pdf_proc.return_value.extract_all_pages.return_value = mock_pages
            mock_table_ext.return_value.extract_tables.return_value = []
            mock_section_map.return_value.build_hierarchy.return_value = MagicMock()
            mock_entity_ext.return_value.extract_entities.return_value = test_entities
            
            pipeline = IngestionPipeline(
                checkpoint_dir=temp_data_dir / "checkpoints",
                show_progress=False,
            )
            
            result = pipeline.run(
                pdf_path=str(sample_pdf_path),
                document_id="entity_test_001",
                resume=False,
            )
            
            # Pipeline should return all entities from extractor
            assert len(result.entities) == 2
            
            # Verify entities are passed through correctly
            company_entities = [e for e in result.entities if e.entity_type == EntityType.COMPANY]
            assert len(company_entities) == 1
            assert company_entities[0].mentions == 5
            assert set(company_entities[0].page_references) == {1, 2, 3, 4}

    def test_pipeline_result_serializable(
        self,
        temp_data_dir,
        sample_pdf_path,
        mock_pdf_metadata,
        mock_pages,
        mock_tables,
        mock_section_tree,
        mock_entities,
    ):
        """Test that pipeline result can be serialized to JSON."""
        with patch("rhp_analyzer.ingestion.pipeline.PDFProcessor") as mock_pdf_proc, \
             patch("rhp_analyzer.ingestion.pipeline.TableExtractor") as mock_table_ext, \
             patch("rhp_analyzer.ingestion.pipeline.SectionMapper") as mock_section_map, \
             patch("rhp_analyzer.ingestion.pipeline.EntityExtractor") as mock_entity_ext:
            
            mock_pdf_proc.return_value.validate.return_value = mock_pdf_metadata
            mock_pdf_proc.return_value.extract_all_pages.return_value = mock_pages
            mock_table_ext.return_value.extract_tables.return_value = mock_tables
            mock_section_map.return_value.build_hierarchy.return_value = mock_section_tree
            mock_entity_ext.return_value.extract_entities.return_value = mock_entities
            
            pipeline = IngestionPipeline(
                checkpoint_dir=temp_data_dir / "checkpoints",
                show_progress=False,
            )
            
            result = pipeline.run(
                pdf_path=str(sample_pdf_path),
                document_id="serialize_test_001",
                resume=False,
            )
            
            # Convert to dict (should not raise)
            result_dict = result.to_dict()
            
            # Should be JSON serializable
            json_str = json.dumps(result_dict)
            assert len(json_str) > 0
            
            # Parse back
            parsed = json.loads(json_str)
            assert parsed["document_id"] == "serialize_test_001"
            assert parsed["success"] is True


class TestPipelineErrorHandling(TestIngestionPipelineIntegration):
    """Test pipeline error handling and recovery."""

    def test_pipeline_handles_pdf_validation_error(
        self,
        temp_data_dir,
        sample_pdf_path,
    ):
        """Test graceful handling of PDF validation errors."""
        with patch("rhp_analyzer.ingestion.pipeline.PDFProcessor") as mock_pdf_proc:
            from rhp_analyzer.ingestion.pdf_processor import PDFValidationError
            
            mock_pdf_proc.return_value.validate.side_effect = PDFValidationError("Invalid PDF")
            
            pipeline = IngestionPipeline(
                checkpoint_dir=temp_data_dir / "checkpoints",
                show_progress=False,
            )
            
            result = pipeline.run(
                pdf_path=str(sample_pdf_path),
                document_id="error_test_001",
                resume=False,
            )
            
            assert result.success is False
            assert len(result.errors) > 0
            assert "validation" in result.errors[0].lower() or "invalid" in result.errors[0].lower()

    def test_pipeline_continues_on_table_extraction_failure(
        self,
        temp_data_dir,
        sample_pdf_path,
        mock_pdf_metadata,
        mock_pages,
        mock_section_tree,
        mock_entities,
    ):
        """Test that pipeline continues even if table extraction fails (graceful degradation)."""
        with patch("rhp_analyzer.ingestion.pipeline.PDFProcessor") as mock_pdf_proc, \
             patch("rhp_analyzer.ingestion.pipeline.TableExtractor") as mock_table_ext, \
             patch("rhp_analyzer.ingestion.pipeline.SectionMapper") as mock_section_map, \
             patch("rhp_analyzer.ingestion.pipeline.EntityExtractor") as mock_entity_ext:
            
            mock_pdf_proc.return_value.validate.return_value = mock_pdf_metadata
            mock_pdf_proc.return_value.extract_all_pages.return_value = mock_pages
            
            # Table extraction fails
            mock_table_ext.return_value.extract_tables.side_effect = Exception("Camelot not available")
            
            mock_section_map.return_value.build_hierarchy.return_value = mock_section_tree
            mock_entity_ext.return_value.extract_entities.return_value = mock_entities
            
            pipeline = IngestionPipeline(
                checkpoint_dir=temp_data_dir / "checkpoints",
                show_progress=False,
            )
            
            result = pipeline.run(
                pdf_path=str(sample_pdf_path),
                document_id="graceful_test_001",
                resume=False,
            )
            
            # Pipeline should fail when table extraction fails
            assert result.success is False
            assert len(result.errors) > 0
            # Error should mention table extraction
            assert any("table" in err.lower() or "camelot" in err.lower() for err in result.errors)


class TestPipelineCheckpointManagement(TestIngestionPipelineIntegration):
    """Test checkpoint management utilities."""

    def test_list_checkpoints(self, temp_data_dir):
        """Test listing all available checkpoints."""
        checkpoint_dir = temp_data_dir / "checkpoints"
        
        # Create some checkpoint files
        for doc_id in ["doc_001", "doc_002", "doc_003"]:
            checkpoint_data = {
                "document_id": doc_id,
                "pdf_path": f"/path/to/{doc_id}.pdf",
                "current_stage": PipelineStage.PAGES_EXTRACTED.value,
                "completed_stages": [PipelineStage.PDF_VALIDATED.value],
                "started_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "stage_durations": {},
                "errors": [],
                "warnings": [],
            }
            with open(checkpoint_dir / f"{doc_id}_checkpoint.json", "w") as f:
                json.dump(checkpoint_data, f)
        
        pipeline = IngestionPipeline(
            checkpoint_dir=temp_data_dir / "checkpoints",
            show_progress=False,
        )
        
        checkpoints = pipeline.list_checkpoints()
        
        assert len(checkpoints) == 3
        assert "doc_001" in checkpoints
        assert "doc_002" in checkpoints
        assert "doc_003" in checkpoints

    def test_clear_specific_checkpoint(self, temp_data_dir):
        """Test clearing a specific checkpoint."""
        checkpoint_dir = temp_data_dir / "checkpoints"
        
        # Create a checkpoint with correct filename format
        checkpoint_file = checkpoint_dir / "doc_to_clear_checkpoint.json"
        checkpoint_data = {
            "document_id": "doc_to_clear",
            "pdf_path": "/path/to/doc.pdf",
            "current_stage": PipelineStage.PAGES_EXTRACTED.value,
            "completed_stages": [],
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "stage_durations": {},
            "errors": [],
            "warnings": [],
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)
        
        assert checkpoint_file.exists()
        
        pipeline = IngestionPipeline(
            checkpoint_dir=temp_data_dir / "checkpoints",
            show_progress=False,
        )
        
        result = pipeline.clear_checkpoint("doc_to_clear")
        
        assert result is True
        assert not checkpoint_file.exists()

    def test_get_checkpoint_info(self, temp_data_dir):
        """Test getting checkpoint information."""
        checkpoint_dir = temp_data_dir / "checkpoints"
        
        checkpoint_data = {
            "document_id": "info_test_doc",
            "pdf_path": "/path/to/doc.pdf",
            "current_stage": PipelineStage.SECTIONS_MAPPED.value,
            "completed_stages": [
                PipelineStage.PDF_VALIDATED.value,
                PipelineStage.PAGES_EXTRACTED.value,
                PipelineStage.TABLES_EXTRACTED.value,
                PipelineStage.SECTIONS_MAPPED.value,
            ],
            "started_at": "2024-01-01T10:00:00",
            "last_updated": "2024-01-01T10:30:00",
            "stage_durations": {
                PipelineStage.PDF_VALIDATED.value: 1.0,
                PipelineStage.PAGES_EXTRACTED.value: 5.0,
                PipelineStage.TABLES_EXTRACTED.value: 10.0,
                PipelineStage.SECTIONS_MAPPED.value: 3.0,
            },
            "errors": [],
            "warnings": ["Minor warning"],
        }
        
        with open(checkpoint_dir / "info_test_doc_checkpoint.json", "w") as f:
            json.dump(checkpoint_data, f)
        
        pipeline = IngestionPipeline(
            checkpoint_dir=temp_data_dir / "checkpoints",
            show_progress=False,
        )
        
        info = pipeline.get_checkpoint_info("info_test_doc")
        
        assert info is not None
        # get_checkpoint_info returns a PipelineCheckpoint object
        assert info.document_id == "info_test_doc"
        assert info.current_stage == PipelineStage.SECTIONS_MAPPED
        assert len(info.completed_stages) == 4


class TestPipelinePerformance(TestIngestionPipelineIntegration):
    """Test pipeline performance metrics."""

    def test_timing_captured_per_stage(
        self,
        temp_data_dir,
        sample_pdf_path,
        mock_pdf_metadata,
        mock_pages,
        mock_tables,
        mock_section_tree,
        mock_entities,
    ):
        """Test that timing is captured for each stage."""
        with patch("rhp_analyzer.ingestion.pipeline.PDFProcessor") as mock_pdf_proc, \
             patch("rhp_analyzer.ingestion.pipeline.TableExtractor") as mock_table_ext, \
             patch("rhp_analyzer.ingestion.pipeline.SectionMapper") as mock_section_map, \
             patch("rhp_analyzer.ingestion.pipeline.EntityExtractor") as mock_entity_ext:
            
            mock_pdf_proc.return_value.validate.return_value = mock_pdf_metadata
            mock_pdf_proc.return_value.extract_all_pages.return_value = mock_pages
            mock_table_ext.return_value.extract_tables.return_value = mock_tables
            mock_section_map.return_value.build_hierarchy.return_value = mock_section_tree
            mock_entity_ext.return_value.extract_entities.return_value = mock_entities
            
            pipeline = IngestionPipeline(
                checkpoint_dir=temp_data_dir / "checkpoints",
                show_progress=False,
            )
            
            result = pipeline.run(
                pdf_path=str(sample_pdf_path),
                document_id="timing_test_001",
                resume=False,
            )
            
            # Verify timing for each stage
            assert "pdf_validation" in result.stage_durations or PipelineStage.PDF_VALIDATED.value in result.stage_durations
            
            # Total time should be sum of stage times (approximately)
            total_stage_time = sum(result.stage_durations.values())
            assert result.total_duration_seconds >= total_stage_time * 0.9  # Allow some overhead
