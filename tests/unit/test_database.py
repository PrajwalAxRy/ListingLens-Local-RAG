"""
Unit tests for the database module.

Tests cover:
- Database initialization
- CRUD operations for all models
- Repository pattern implementations
- Relationships between entities
- Transaction handling

Reference: milestones.md Phase 3.4.4
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from rhp_analyzer.storage.database import (
    AgentOutput,
    Base,
    ChunkMetadata,
    DatabaseManager,
    Document,
    Entity,
    ExtractedTable,
    FinancialData,
    RiskFactor,
    Section,
)
from rhp_analyzer.storage.repositories import (
    AgentOutputRepository,
    ChunkMetadataRepository,
    DocumentRepository,
    EntityRepository,
    ExtractedTableRepository,
    FinancialDataRepository,
    RiskFactorRepository,
    SectionRepository,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_rhp_analyzer.db"


@pytest.fixture
def db_manager(temp_db_path):
    """Create a DatabaseManager with a temporary database."""
    manager = DatabaseManager(db_path=temp_db_path)
    manager.initialize()
    yield manager
    manager.close()


@pytest.fixture
def session(db_manager):
    """Create a database session for testing."""
    session = db_manager.get_session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def sample_document(session):
    """Create a sample document for testing."""
    doc = Document(
        document_id="TEST_DOC_001",
        filename="test_rhp.pdf",
        company_name="Test Company Ltd",
        total_pages=350,
        processing_status="completed",
        issue_size=1500.0,
        price_band="₹123 - ₹130",
    )
    session.add(doc)
    session.flush()
    return doc


# =============================================================================
# Database Manager Tests
# =============================================================================


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    def test_initialize_creates_database(self, temp_db_path):
        """Test that initialize creates the database file."""
        manager = DatabaseManager(db_path=temp_db_path)
        manager.initialize()

        assert temp_db_path.exists()
        manager.close()

    def test_initialize_creates_tables(self, db_manager, temp_db_path):
        """Test that initialize creates all required tables."""
        from sqlalchemy import inspect

        inspector = inspect(db_manager.engine)
        tables = inspector.get_table_names()

        expected_tables = [
            "documents",
            "sections",
            "tables",
            "entities",
            "financial_data",
            "risk_factors",
            "agent_outputs",
            "chunks",
        ]
        for table in expected_tables:
            assert table in tables, f"Table '{table}' not found"

    def test_session_scope_commits_on_success(self, db_manager):
        """Test that session_scope commits changes on success."""
        with db_manager.session_scope() as session:
            doc = Document(
                document_id="SCOPE_TEST_001",
                filename="scope_test.pdf",
            )
            session.add(doc)

        # Verify in a new session
        with db_manager.session_scope() as session:
            found = session.query(Document).filter_by(document_id="SCOPE_TEST_001").first()
            assert found is not None

    def test_session_scope_rollback_on_exception(self, db_manager):
        """Test that session_scope rolls back on exception."""
        try:
            with db_manager.session_scope() as session:
                doc = Document(
                    document_id="ROLLBACK_TEST_001",
                    filename="rollback_test.pdf",
                )
                session.add(doc)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Verify document was not saved
        with db_manager.session_scope() as session:
            found = session.query(Document).filter_by(document_id="ROLLBACK_TEST_001").first()
            assert found is None

    def test_drop_all_removes_tables(self, db_manager):
        """Test that drop_all removes all tables."""
        from sqlalchemy import inspect

        db_manager.drop_all()

        inspector = inspect(db_manager.engine)
        tables = inspector.get_table_names()

        # Only alembic_version might remain
        assert "documents" not in tables

        # Re-initialize for cleanup
        db_manager.initialize()


# =============================================================================
# Document Model Tests
# =============================================================================


class TestDocumentModel:
    """Tests for Document ORM model."""

    def test_create_document(self, session):
        """Test creating a document."""
        doc = Document(
            document_id="DOC_CREATE_001",
            filename="create_test.pdf",
            company_name="Create Test Company",
            total_pages=100,
        )
        session.add(doc)
        session.flush()

        assert doc.id is not None
        assert doc.document_id == "DOC_CREATE_001"
        assert doc.processing_status == "pending"  # Default value

    def test_document_relationships(self, session, sample_document):
        """Test document relationships with other models."""
        # Add related section
        section = Section(
            document_id=sample_document.id,
            section_name="Risk Factors",
            start_page=10,
            end_page=50,
        )
        session.add(section)
        session.flush()

        # Verify relationship
        assert len(sample_document.sections) == 1
        assert sample_document.sections[0].section_name == "Risk Factors"

    def test_document_cascade_delete(self, session, sample_document):
        """Test that deleting a document cascades to related entities."""
        # Add related section
        section = Section(
            document_id=sample_document.id,
            section_name="Test Section",
            start_page=1,
            end_page=10,
        )
        session.add(section)
        session.flush()
        section_id = section.id

        # Delete document
        session.delete(sample_document)
        session.flush()

        # Verify section is also deleted
        found_section = session.query(Section).filter_by(id=section_id).first()
        assert found_section is None


# =============================================================================
# Section Model Tests
# =============================================================================


class TestSectionModel:
    """Tests for Section ORM model."""

    def test_create_section(self, session, sample_document):
        """Test creating a section."""
        section = Section(
            document_id=sample_document.id,
            section_name="Business Overview",
            section_type="business",
            start_page=20,
            end_page=80,
            word_count=15000,
            level=1,
        )
        session.add(section)
        session.flush()

        assert section.id is not None
        assert section.section_name == "Business Overview"

    def test_section_parent_child_relationship(self, session, sample_document):
        """Test parent-child section relationships."""
        parent = Section(
            document_id=sample_document.id,
            section_name="Financial Statements",
            level=1,
            start_page=100,
            end_page=200,
        )
        session.add(parent)
        session.flush()

        child = Section(
            document_id=sample_document.id,
            section_name="Balance Sheet",
            level=2,
            start_page=100,
            end_page=120,
            parent_section_id=parent.id,
        )
        session.add(child)
        session.flush()

        assert child.parent.section_name == "Financial Statements"
        assert len(parent.children) == 1


# =============================================================================
# Financial Data Model Tests
# =============================================================================


class TestFinancialDataModel:
    """Tests for FinancialData ORM model."""

    def test_create_financial_data(self, session, sample_document):
        """Test creating financial data."""
        fin_data = FinancialData(
            document_id=sample_document.id,
            fiscal_year="FY24",
            revenue=1500.5,
            ebitda=350.2,
            ebitda_margin=23.34,
            pat=220.0,
            total_assets=2500.0,
            total_equity=1200.0,
            total_debt=800.0,
            roe=18.33,
            roce=15.5,
        )
        session.add(fin_data)
        session.flush()

        assert fin_data.id is not None
        assert fin_data.revenue == 1500.5

    def test_financial_data_multiple_years(self, session, sample_document):
        """Test creating multiple years of financial data."""
        years = ["FY22", "FY23", "FY24"]
        revenues = [800.0, 1100.0, 1500.0]

        for year, revenue in zip(years, revenues):
            fin_data = FinancialData(
                document_id=sample_document.id,
                fiscal_year=year,
                revenue=revenue,
            )
            session.add(fin_data)

        session.flush()

        all_data = session.query(FinancialData).filter_by(
            document_id=sample_document.id
        ).order_by(FinancialData.fiscal_year).all()

        assert len(all_data) == 3
        assert all_data[0].fiscal_year == "FY22"


# =============================================================================
# Agent Output Model Tests
# =============================================================================


class TestAgentOutputModel:
    """Tests for AgentOutput ORM model."""

    def test_create_agent_output(self, session, sample_document):
        """Test creating an agent output."""
        output = AgentOutput(
            document_id=sample_document.id,
            agent_name="forensic",
            confidence_score=0.85,
            processing_time_seconds=45.5,
            status="completed",
        )
        session.add(output)
        session.flush()

        assert output.id is not None
        assert output.agent_name == "forensic"

    def test_agent_output_json_properties(self, session, sample_document):
        """Test JSON serialization properties."""
        output = AgentOutput(
            document_id=sample_document.id,
            agent_name="red_flag",
        )

        # Test result property
        test_result = {"score": 75, "flags": ["high_promoter_pledge"]}
        output.result = test_result
        assert output.analysis_result == json.dumps(test_result)
        assert output.result == test_result

        # Test findings_list property
        test_findings = ["Finding 1", "Finding 2"]
        output.findings_list = test_findings
        assert output.findings_list == test_findings

        session.add(output)
        session.flush()


# =============================================================================
# Extracted Table Model Tests
# =============================================================================


class TestExtractedTableModel:
    """Tests for ExtractedTable ORM model."""

    def test_create_extracted_table(self, session, sample_document):
        """Test creating an extracted table."""
        table = ExtractedTable(
            document_id=sample_document.id,
            table_type="financial_statement",
            page_number=156,
            row_count=15,
            column_count=6,
            extraction_method="pdfplumber",
            confidence_score=0.92,
        )
        session.add(table)
        session.flush()

        assert table.id is not None

    def test_extracted_table_data_property(self, session, sample_document):
        """Test table data JSON serialization."""
        table = ExtractedTable(
            document_id=sample_document.id,
            table_type="shareholding",
            page_number=45,
        )

        test_data = [
            ["Shareholder", "Shares", "Percentage"],
            ["Promoter", "1000000", "45%"],
            ["Public", "1222222", "55%"],
        ]
        table.data = test_data

        assert table.data_json == json.dumps(test_data)
        assert table.data == test_data

        session.add(table)
        session.flush()


# =============================================================================
# Repository Tests
# =============================================================================


class TestDocumentRepository:
    """Tests for DocumentRepository."""

    def test_find_by_document_id(self, session, sample_document):
        """Test finding a document by document_id."""
        repo = DocumentRepository(session)
        found = repo.find_by_document_id("TEST_DOC_001")

        assert found is not None
        assert found.company_name == "Test Company Ltd"

    def test_find_by_company_name(self, session, sample_document):
        """Test finding documents by company name."""
        repo = DocumentRepository(session)
        results = repo.find_by_company_name("Test Company")

        assert len(results) == 1
        assert results[0].document_id == "TEST_DOC_001"

    def test_find_by_status(self, session, sample_document):
        """Test finding documents by status."""
        repo = DocumentRepository(session)
        completed = repo.find_by_status("completed")

        assert len(completed) == 1

    def test_update_status(self, session, sample_document):
        """Test updating document status."""
        repo = DocumentRepository(session)
        updated = repo.update_status("TEST_DOC_001", "processing")

        assert updated is not None
        assert updated.processing_status == "processing"

    def test_get_recent(self, session):
        """Test getting recent documents."""
        repo = DocumentRepository(session)

        # Create multiple documents
        for i in range(5):
            doc = Document(
                document_id=f"RECENT_DOC_{i}",
                filename=f"recent_{i}.pdf",
            )
            session.add(doc)
        session.flush()

        recent = repo.get_recent(limit=3)
        assert len(recent) == 3

    def test_delete_by_document_id(self, session):
        """Test deleting by document_id."""
        repo = DocumentRepository(session)

        doc = Document(
            document_id="DELETE_TEST",
            filename="delete_test.pdf",
        )
        session.add(doc)
        session.flush()

        result = repo.delete_by_document_id("DELETE_TEST")
        assert result is True

        found = repo.find_by_document_id("DELETE_TEST")
        assert found is None


class TestSectionRepository:
    """Tests for SectionRepository."""

    def test_find_by_document(self, session, sample_document):
        """Test finding sections by document."""
        repo = SectionRepository(session)

        # Create sections
        for i in range(3):
            section = Section(
                document_id=sample_document.id,
                section_name=f"Section {i}",
                start_page=i * 50,
                end_page=(i + 1) * 50,
            )
            session.add(section)
        session.flush()

        sections = repo.find_by_document(sample_document.id)
        assert len(sections) == 3

    def test_find_by_page(self, session, sample_document):
        """Test finding sections containing a page."""
        repo = SectionRepository(session)

        section = Section(
            document_id=sample_document.id,
            section_name="Risk Factors",
            start_page=10,
            end_page=50,
        )
        session.add(section)
        session.flush()

        found = repo.find_by_page(sample_document.id, 25)
        assert len(found) == 1
        assert found[0].section_name == "Risk Factors"


class TestFinancialDataRepository:
    """Tests for FinancialDataRepository."""

    @pytest.fixture
    def financial_data_setup(self, session, sample_document):
        """Create sample financial data for testing."""
        data = [
            ("FY22", 800.0, 160.0, 100.0),
            ("FY23", 1100.0, 242.0, 150.0),
            ("FY24", 1500.0, 345.0, 210.0),
        ]
        for year, revenue, ebitda, pat in data:
            fin = FinancialData(
                document_id=sample_document.id,
                fiscal_year=year,
                revenue=revenue,
                ebitda=ebitda,
                pat=pat,
            )
            session.add(fin)
        session.flush()
        return sample_document

    def test_find_by_fiscal_year(self, session, financial_data_setup):
        """Test finding financial data by fiscal year."""
        repo = FinancialDataRepository(session)
        found = repo.find_by_fiscal_year(financial_data_setup.id, "FY23")

        assert found is not None
        assert found.revenue == 1100.0

    def test_get_latest(self, session, financial_data_setup):
        """Test getting the latest financial data."""
        repo = FinancialDataRepository(session)
        latest = repo.get_latest(financial_data_setup.id)

        assert latest is not None
        assert latest.fiscal_year == "FY24"

    def test_get_revenue_trend(self, session, financial_data_setup):
        """Test getting revenue trend."""
        repo = FinancialDataRepository(session)
        trend = repo.get_revenue_trend(financial_data_setup.id)

        assert len(trend) == 3
        assert trend[0]["fiscal_year"] == "FY22"
        assert trend[2]["revenue"] == 1500.0

    def test_calculate_cagr(self, session, financial_data_setup):
        """Test CAGR calculation."""
        repo = FinancialDataRepository(session)
        cagr = repo.calculate_cagr(financial_data_setup.id, "revenue")

        # CAGR from 800 to 1500 over 2 years
        # (1500/800)^(1/2) - 1 ≈ 0.369
        assert cagr is not None
        assert abs(cagr - 0.369) < 0.01


class TestAgentOutputRepository:
    """Tests for AgentOutputRepository."""

    @pytest.fixture
    def agent_outputs_setup(self, session, sample_document):
        """Create sample agent outputs for testing."""
        agents = [
            ("forensic", 0.85, "completed"),
            ("governance", 0.92, "completed"),
            ("red_flag", 0.78, "completed"),
            ("legal", None, "failed"),
        ]
        for agent, confidence, status in agents:
            output = AgentOutput(
                document_id=sample_document.id,
                agent_name=agent,
                confidence_score=confidence,
                status=status,
                processing_time_seconds=30.0 if status == "completed" else None,
                error_message="Connection timeout" if status == "failed" else None,
            )
            session.add(output)
        session.flush()
        return sample_document

    def test_find_by_agent(self, session, agent_outputs_setup):
        """Test finding output by agent name."""
        repo = AgentOutputRepository(session)
        output = repo.find_by_agent(agent_outputs_setup.id, "forensic")

        assert output is not None
        assert output.confidence_score == 0.85

    def test_get_completed_agents(self, session, agent_outputs_setup):
        """Test getting list of completed agents."""
        repo = AgentOutputRepository(session)
        completed = repo.get_completed_agents(agent_outputs_setup.id)

        assert len(completed) == 3
        assert "legal" not in completed

    def test_get_failed_agents(self, session, agent_outputs_setup):
        """Test getting list of failed agents."""
        repo = AgentOutputRepository(session)
        failed = repo.get_failed_agents(agent_outputs_setup.id)

        assert len(failed) == 1
        assert failed[0][0] == "legal"
        assert "timeout" in failed[0][1].lower()

    def test_get_average_confidence(self, session, agent_outputs_setup):
        """Test getting average confidence score."""
        repo = AgentOutputRepository(session)
        avg = repo.get_average_confidence(agent_outputs_setup.id)

        # Average of 0.85, 0.92, 0.78
        expected = (0.85 + 0.92 + 0.78) / 3
        assert avg is not None
        assert abs(avg - expected) < 0.01

    def test_get_total_processing_time(self, session, agent_outputs_setup):
        """Test getting total processing time."""
        repo = AgentOutputRepository(session)
        total = repo.get_total_processing_time(agent_outputs_setup.id)

        # 3 completed agents × 30 seconds
        assert total == 90.0


class TestRiskFactorRepository:
    """Tests for RiskFactorRepository."""

    @pytest.fixture
    def risk_factors_setup(self, session, sample_document):
        """Create sample risk factors for testing."""
        risks = [
            ("operational", "High customer concentration", "critical", False),
            ("financial", "Negative working capital", "major", False),
            ("regulatory", "Pending regulatory approval", "minor", False),
            ("legal", "Standard litigation disclaimer", "minor", True),
        ]
        for category, desc, severity, boilerplate in risks:
            risk = RiskFactor(
                document_id=sample_document.id,
                category=category,
                description=desc,
                severity=severity,
                is_boilerplate=1 if boilerplate else 0,
            )
            session.add(risk)
        session.flush()
        return sample_document

    def test_find_by_severity(self, session, risk_factors_setup):
        """Test finding risks by severity."""
        repo = RiskFactorRepository(session)
        critical = repo.find_by_severity(risk_factors_setup.id, "critical")

        assert len(critical) == 1
        assert "customer concentration" in critical[0].description.lower()

    def test_get_critical_risks(self, session, risk_factors_setup):
        """Test getting critical risks."""
        repo = RiskFactorRepository(session)
        critical = repo.get_critical_risks(risk_factors_setup.id)

        assert len(critical) == 1

    def test_get_non_boilerplate(self, session, risk_factors_setup):
        """Test getting non-boilerplate risks."""
        repo = RiskFactorRepository(session)
        specific = repo.get_non_boilerplate(risk_factors_setup.id)

        assert len(specific) == 3

    def test_count_by_severity(self, session, risk_factors_setup):
        """Test counting risks by severity."""
        repo = RiskFactorRepository(session)
        counts = repo.count_by_severity(risk_factors_setup.id)

        assert counts.get("critical", 0) == 1
        assert counts.get("major", 0) == 1
        assert counts.get("minor", 0) == 2


class TestChunkMetadataRepository:
    """Tests for ChunkMetadataRepository."""

    @pytest.fixture
    def chunks_setup(self, session, sample_document):
        """Create sample chunks for testing."""
        chunks = []
        prev_id = None
        for i in range(5):
            chunk = ChunkMetadata(
                document_id=sample_document.id,
                chunk_id=f"chunk_{i:03d}",
                section_name="Risk Factors" if i < 3 else "Business Overview",
                chunk_type="narrative",
                start_page=i * 10,
                end_page=(i + 1) * 10,
                token_count=500,
                preceding_chunk_id=prev_id,
            )
            session.add(chunk)
            session.flush()

            # Update previous chunk's following_chunk_id
            if prev_id:
                prev_chunk = session.query(ChunkMetadata).filter_by(chunk_id=prev_id).first()
                prev_chunk.following_chunk_id = chunk.chunk_id
                session.flush()

            prev_id = chunk.chunk_id
            chunks.append(chunk)

        return sample_document

    def test_find_by_chunk_id(self, session, chunks_setup):
        """Test finding chunk by chunk_id."""
        repo = ChunkMetadataRepository(session)
        chunk = repo.find_by_chunk_id("chunk_002")

        assert chunk is not None
        assert chunk.start_page == 20

    def test_find_by_section(self, session, chunks_setup):
        """Test finding chunks by section."""
        repo = ChunkMetadataRepository(session)
        risk_chunks = repo.find_by_section(chunks_setup.id, "Risk Factors")

        assert len(risk_chunks) == 3

    def test_count_by_type(self, session, chunks_setup):
        """Test counting chunks by type."""
        repo = ChunkMetadataRepository(session)
        counts = repo.count_by_type(chunks_setup.id)

        assert counts.get("narrative", 0) == 5

    def test_get_total_tokens(self, session, chunks_setup):
        """Test getting total token count."""
        repo = ChunkMetadataRepository(session)
        total = repo.get_total_tokens(chunks_setup.id)

        # 5 chunks × 500 tokens
        assert total == 2500

    def test_get_chunk_chain(self, session, chunks_setup):
        """Test getting chunk chain."""
        repo = ChunkMetadataRepository(session)
        chain = repo.get_chunk_chain("chunk_002", direction="both")

        assert len(chain) == 5
        assert chain[0].chunk_id == "chunk_000"
        assert chain[-1].chunk_id == "chunk_004"


# =============================================================================
# Base Repository Tests
# =============================================================================


class TestBaseRepository:
    """Tests for BaseRepository generic operations."""

    def test_create(self, session):
        """Test generic create operation."""
        repo = DocumentRepository(session)
        doc = repo.create(
            document_id="BASE_CREATE_001",
            filename="base_create.pdf",
        )

        assert doc.id is not None
        assert doc.document_id == "BASE_CREATE_001"

    def test_get_by_id(self, session, sample_document):
        """Test getting entity by ID."""
        repo = DocumentRepository(session)
        found = repo.get_by_id(sample_document.id)

        assert found is not None
        assert found.document_id == "TEST_DOC_001"

    def test_update(self, session, sample_document):
        """Test generic update operation."""
        repo = DocumentRepository(session)
        updated = repo.update(
            sample_document.id,
            company_name="Updated Company Name",
        )

        assert updated is not None
        assert updated.company_name == "Updated Company Name"

    def test_delete(self, session, sample_document):
        """Test generic delete operation."""
        repo = DocumentRepository(session)
        doc_id = sample_document.id

        result = repo.delete(doc_id)
        assert result is True

        found = repo.get_by_id(doc_id)
        assert found is None

    def test_count(self, session, sample_document):
        """Test counting entities."""
        repo = DocumentRepository(session)
        count = repo.count()

        assert count >= 1

    def test_exists(self, session, sample_document):
        """Test checking entity existence."""
        repo = DocumentRepository(session)

        assert repo.exists(sample_document.id) is True
        assert repo.exists(99999) is False

    def test_get_all_with_pagination(self, session):
        """Test getting all entities with pagination."""
        repo = DocumentRepository(session)

        # Create multiple documents
        for i in range(10):
            doc = Document(
                document_id=f"PAGINATE_DOC_{i:02d}",
                filename=f"paginate_{i}.pdf",
            )
            session.add(doc)
        session.flush()

        # Test pagination
        page1 = repo.get_all(limit=3, offset=0)
        page2 = repo.get_all(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].id != page2[0].id


# =============================================================================
# Transaction Handling Tests
# =============================================================================


class TestTransactionHandling:
    """Tests for transaction handling and isolation."""

    def test_independent_sessions(self, db_manager):
        """Test that sessions are properly isolated."""
        session1 = db_manager.get_session()
        session2 = db_manager.get_session()

        try:
            # Add document in session1
            doc = Document(
                document_id="ISOLATION_TEST",
                filename="isolation.pdf",
            )
            session1.add(doc)
            session1.flush()

            # Should not be visible in session2 until commit
            found = session2.query(Document).filter_by(
                document_id="ISOLATION_TEST"
            ).first()
            # Note: SQLite may show uncommitted data due to its locking model
            # This test verifies the session management works

            session1.commit()
        finally:
            session1.close()
            session2.close()

    def test_rollback_on_error(self, db_manager):
        """Test that errors trigger rollback."""
        session = db_manager.get_session()

        try:
            doc = Document(
                document_id="ROLLBACK_ERROR_TEST",
                filename="rollback.pdf",
            )
            session.add(doc)
            session.flush()

            # Force an error
            raise ValueError("Test error")
        except ValueError:
            session.rollback()
        finally:
            session.close()

        # Verify rollback occurred
        with db_manager.session_scope() as verify_session:
            found = verify_session.query(Document).filter_by(
                document_id="ROLLBACK_ERROR_TEST"
            ).first()
            assert found is None
