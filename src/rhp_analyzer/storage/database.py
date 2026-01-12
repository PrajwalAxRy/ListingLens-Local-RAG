"""
SQLite database module using SQLAlchemy ORM.

This module provides:
- SQLAlchemy ORM models for all entities
- Database initialization and session management
- Base repository pattern for data access

Reference: blueprint.md Section 3.2, milestones.md Phase 3.4
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, TypeVar

from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    relationship,
    sessionmaker,
)


def utc_now():
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


# Enable foreign key constraints for SQLite
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key constraints for SQLite connections."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class."""

    pass


# =============================================================================
# ORM Models
# =============================================================================


class Document(Base):
    """
    RHP document metadata.

    Stores core information about each analyzed RHP document including
    company details, issue information, and processing status.
    """

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(255), unique=True, nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    company_name = Column(String(255), nullable=True)
    upload_date = Column(DateTime, default=utc_now)
    total_pages = Column(Integer, nullable=True)
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    issue_size = Column(Float, nullable=True)  # In crores
    price_band = Column(String(100), nullable=True)  # e.g., "₹123 - ₹130"
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Relationships
    sections = relationship("Section", back_populates="document", cascade="all, delete-orphan")
    tables = relationship("ExtractedTable", back_populates="document", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="document", cascade="all, delete-orphan")
    financial_data = relationship("FinancialData", back_populates="document", cascade="all, delete-orphan")
    risk_factors = relationship("RiskFactor", back_populates="document", cascade="all, delete-orphan")
    agent_outputs = relationship("AgentOutput", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("ChunkMetadata", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, document_id='{self.document_id}', company='{self.company_name}')>"


class Section(Base):
    """
    Extracted RHP sections.

    Stores information about identified sections within the RHP document
    including their location, type, and word count.
    """

    __tablename__ = "sections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    section_name = Column(String(255), nullable=False)
    section_type = Column(String(100), nullable=True)  # e.g., 'risk_factors', 'business', 'financials'
    start_page = Column(Integer, nullable=True)
    end_page = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    level = Column(Integer, default=1)  # Hierarchy level (1=top, 2=sub, etc.)
    parent_section_id = Column(Integer, ForeignKey("sections.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    document = relationship("Document", back_populates="sections")
    parent = relationship("Section", remote_side=[id], backref="children")

    def __repr__(self) -> str:
        return f"<Section(id={self.id}, name='{self.section_name}', pages={self.start_page}-{self.end_page})>"


class ExtractedTable(Base):
    """
    Extracted table data from RHP.

    Stores structured table data with metadata about source location
    and table type classification.
    """

    __tablename__ = "tables"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    table_type = Column(String(100), nullable=True)  # e.g., 'financial_statement', 'shareholding'
    page_number = Column(Integer, nullable=True)
    section_name = Column(String(255), nullable=True)
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    data_json = Column(Text, nullable=True)  # JSON serialized table data
    caption = Column(Text, nullable=True)
    extraction_method = Column(String(50), nullable=True)  # pdfplumber, camelot, etc.
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    document = relationship("Document", back_populates="tables")

    @property
    def data(self) -> list[list[Any]] | None:
        """Deserialize table data from JSON."""
        if self.data_json:
            return json.loads(self.data_json)
        return None

    @data.setter
    def data(self, value: list[list[Any]]) -> None:
        """Serialize table data to JSON."""
        self.data_json = json.dumps(value) if value else None

    def __repr__(self) -> str:
        return f"<ExtractedTable(id={self.id}, type='{self.table_type}', page={self.page_number})>"


class Entity(Base):
    """
    Extracted named entities from RHP.

    Stores identified entities (companies, people, locations) with
    their type, mention frequency, and context.
    """

    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    entity_type = Column(String(50), nullable=False)  # 'company', 'person', 'location', 'amount'
    entity_name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=True)  # Cleaned/standardized name
    mentions = Column(Integer, default=1)
    context = Column(Text, nullable=True)  # Sample context where entity appears
    first_page = Column(Integer, nullable=True)
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    document = relationship("Document", back_populates="entities")

    def __repr__(self) -> str:
        return f"<Entity(id={self.id}, type='{self.entity_type}', name='{self.entity_name}')>"


class FinancialData(Base):
    """
    Parsed financial data from RHP.

    Stores key financial metrics for each fiscal year including
    revenue, profitability, and balance sheet items.
    """

    __tablename__ = "financial_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    fiscal_year = Column(String(20), nullable=False)  # e.g., 'FY24', 'FY2023-24'
    period_type = Column(String(20), default="annual")  # 'annual', 'quarterly', 'half-yearly'

    # Income statement
    revenue = Column(Float, nullable=True)  # In crores
    revenue_growth = Column(Float, nullable=True)  # YoY growth percentage
    ebitda = Column(Float, nullable=True)
    ebitda_margin = Column(Float, nullable=True)  # Percentage
    pat = Column(Float, nullable=True)  # Profit After Tax
    pat_margin = Column(Float, nullable=True)  # Percentage
    eps = Column(Float, nullable=True)  # Earnings per share

    # Balance sheet
    total_assets = Column(Float, nullable=True)
    total_equity = Column(Float, nullable=True)
    total_debt = Column(Float, nullable=True)
    net_worth = Column(Float, nullable=True)
    working_capital = Column(Float, nullable=True)

    # Ratios
    roe = Column(Float, nullable=True)  # Return on Equity
    roce = Column(Float, nullable=True)  # Return on Capital Employed
    debt_equity_ratio = Column(Float, nullable=True)
    current_ratio = Column(Float, nullable=True)
    interest_coverage = Column(Float, nullable=True)

    # Cash flow
    cfo = Column(Float, nullable=True)  # Cash from Operations
    cfi = Column(Float, nullable=True)  # Cash from Investing
    cff = Column(Float, nullable=True)  # Cash from Financing

    # Working capital metrics
    receivable_days = Column(Float, nullable=True)
    inventory_days = Column(Float, nullable=True)
    payable_days = Column(Float, nullable=True)
    cash_conversion_cycle = Column(Float, nullable=True)

    source_page = Column(Integer, nullable=True)
    is_restated = Column(Integer, default=0)  # Boolean as integer for SQLite
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    document = relationship("Document", back_populates="financial_data")

    def __repr__(self) -> str:
        return f"<FinancialData(id={self.id}, year='{self.fiscal_year}', revenue={self.revenue})>"


class RiskFactor(Base):
    """
    Extracted risk factors from RHP.

    Stores identified risk factors with categorization,
    severity assessment, and source reference.
    """

    __tablename__ = "risk_factors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    category = Column(String(100), nullable=False)  # 'operational', 'financial', 'regulatory', 'legal'
    sub_category = Column(String(100), nullable=True)
    title = Column(String(500), nullable=True)
    description = Column(Text, nullable=False)
    severity = Column(String(20), default="medium")  # 'critical', 'major', 'minor'
    is_boilerplate = Column(Integer, default=0)  # Boolean: 1 if generic/boilerplate risk
    page_reference = Column(Integer, nullable=True)
    quantified_impact = Column(Text, nullable=True)  # Any numerical impact mentioned
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    document = relationship("Document", back_populates="risk_factors")

    def __repr__(self) -> str:
        return f"<RiskFactor(id={self.id}, category='{self.category}', severity='{self.severity}')>"


class AgentOutput(Base):
    """
    Agent analysis results.

    Stores the output from each analysis agent including
    findings, confidence scores, and structured results.
    """

    __tablename__ = "agent_outputs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    agent_name = Column(String(100), nullable=False)  # e.g., 'forensic', 'governance', 'red_flag'
    agent_version = Column(String(20), nullable=True)
    analysis_result = Column(Text, nullable=True)  # JSON blob with structured output
    key_findings = Column(Text, nullable=True)  # JSON array of key findings
    concerns = Column(Text, nullable=True)  # JSON array of concerns
    confidence_score = Column(Float, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)
    token_count = Column(Integer, nullable=True)
    iteration = Column(Integer, default=1)  # For multi-revision agents
    status = Column(String(50), default="completed")  # 'pending', 'completed', 'failed'
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    document = relationship("Document", back_populates="agent_outputs")

    @property
    def result(self) -> dict[str, Any] | None:
        """Deserialize analysis result from JSON."""
        if self.analysis_result:
            return json.loads(self.analysis_result)
        return None

    @result.setter
    def result(self, value: dict[str, Any]) -> None:
        """Serialize analysis result to JSON."""
        self.analysis_result = json.dumps(value) if value else None

    @property
    def findings_list(self) -> list[str]:
        """Deserialize key findings from JSON."""
        if self.key_findings:
            return json.loads(self.key_findings)
        return []

    @findings_list.setter
    def findings_list(self, value: list[str]) -> None:
        """Serialize key findings to JSON."""
        self.key_findings = json.dumps(value) if value else None

    @property
    def concerns_list(self) -> list[str]:
        """Deserialize concerns from JSON."""
        if self.concerns:
            return json.loads(self.concerns)
        return []

    @concerns_list.setter
    def concerns_list(self, value: list[str]) -> None:
        """Serialize concerns to JSON."""
        self.concerns = json.dumps(value) if value else None

    def __repr__(self) -> str:
        return f"<AgentOutput(id={self.id}, agent='{self.agent_name}', confidence={self.confidence_score})>"


class ChunkMetadata(Base):
    """
    Chunk metadata for vector store reference.

    Stores metadata about text chunks stored in the vector database,
    enabling cross-referencing between structured and vector data.
    Note: Actual chunk content and embeddings are in Qdrant.
    """

    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(String(255), unique=True, nullable=False, index=True)  # UUID for vector store
    section_name = Column(String(255), nullable=True)
    chunk_type = Column(String(50), default="narrative")  # 'narrative', 'table', 'mixed'
    start_page = Column(Integer, nullable=True)
    end_page = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    char_count = Column(Integer, nullable=True)
    preceding_chunk_id = Column(String(255), nullable=True)
    following_chunk_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<ChunkMetadata(id={self.id}, chunk_id='{self.chunk_id}', section='{self.section_name}')>"


# =============================================================================
# Database Manager
# =============================================================================


class DatabaseManager:
    """
    Manages SQLite database connections and sessions.

    Provides session management, database initialization,
    and connection pooling for the RHP Analyzer.

    Example:
        db = DatabaseManager(db_path="./data/rhp_analyzer.db")
        db.initialize()

        with db.get_session() as session:
            doc = Document(document_id="doc_123", filename="example.pdf")
            session.add(doc)
            session.commit()
    """

    def __init__(self, db_path: Path | str = "./data/rhp_analyzer.db"):
        """
        Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._engine = None
        self._session_factory = None

    @property
    def engine(self) -> Engine:
        """Get or create the SQLAlchemy engine."""
        if self._engine is None:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create SQLite engine
            db_url = f"sqlite:///{self.db_path}"
            self._engine = create_engine(
                db_url,
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True,
                connect_args={"check_same_thread": False},
            )
            logger.debug(f"Created database engine: {db_url}")

        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        return self._session_factory

    def initialize(self) -> None:
        """
        Initialize the database by creating all tables.

        This is safe to call multiple times - it only creates tables
        that don't already exist.
        """
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized at {self.db_path}")

    def get_session(self) -> Session:
        """
        Create a new database session.

        Returns:
            A new SQLAlchemy session.

        Note:
            Caller is responsible for closing the session.
            Prefer using session_scope() context manager instead.
        """
        return self.session_factory()

    def session_scope(self):
        """
        Provide a transactional scope around a series of operations.

        Usage:
            with db.session_scope() as session:
                session.add(...)
                # Automatically committed on exit
                # Rolled back on exception

        Yields:
            SQLAlchemy session.
        """
        from contextlib import contextmanager

        @contextmanager
        def _session_scope():
            session = self.get_session()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        return _session_scope()

    def drop_all(self) -> None:
        """
        Drop all tables in the database.

        WARNING: This will delete all data! Use with caution.
        """
        Base.metadata.drop_all(self.engine)
        logger.warning(f"All tables dropped from {self.db_path}")

    def close(self) -> None:
        """Close the database engine and cleanup resources."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.debug("Database engine closed")


# =============================================================================
# Repository Base Class
# =============================================================================

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """
    Generic base repository providing common CRUD operations.

    Subclasses should specify the model class and can add
    custom query methods as needed.

    Example:
        class DocumentRepository(BaseRepository[Document]):
            model = Document

            def find_by_company(self, company_name: str) -> list[Document]:
                return self.session.query(self.model).filter(
                    self.model.company_name.ilike(f"%{company_name}%")
                ).all()
    """

    model: type[ModelType]

    def __init__(self, session: Session):
        """
        Initialize the repository with a database session.

        Args:
            session: SQLAlchemy session for database operations.
        """
        self.session = session

    def create(self, **kwargs) -> ModelType:
        """
        Create a new entity.

        Args:
            **kwargs: Entity attributes.

        Returns:
            The created entity instance.
        """
        instance = self.model(**kwargs)
        self.session.add(instance)
        self.session.flush()  # Get the ID without committing
        return instance

    def get_by_id(self, entity_id: int) -> ModelType | None:
        """
        Get an entity by its primary key ID.

        Args:
            entity_id: The primary key ID.

        Returns:
            The entity if found, None otherwise.
        """
        return self.session.query(self.model).filter(self.model.id == entity_id).first()

    def get_all(self, limit: int | None = None, offset: int = 0) -> list[ModelType]:
        """
        Get all entities with optional pagination.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of entities.
        """
        query = self.session.query(self.model).offset(offset)
        if limit is not None:
            query = query.limit(limit)
        return query.all()

    def update(self, entity_id: int, **kwargs) -> ModelType | None:
        """
        Update an entity by ID.

        Args:
            entity_id: The primary key ID.
            **kwargs: Attributes to update.

        Returns:
            The updated entity if found, None otherwise.
        """
        instance = self.get_by_id(entity_id)
        if instance:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            self.session.flush()
        return instance

    def delete(self, entity_id: int) -> bool:
        """
        Delete an entity by ID.

        Args:
            entity_id: The primary key ID.

        Returns:
            True if deleted, False if not found.
        """
        instance = self.get_by_id(entity_id)
        if instance:
            self.session.delete(instance)
            self.session.flush()
            return True
        return False

    def count(self) -> int:
        """
        Count total number of entities.

        Returns:
            Total count of entities.
        """
        return self.session.query(self.model).count()

    def exists(self, entity_id: int) -> bool:
        """
        Check if an entity exists by ID.

        Args:
            entity_id: The primary key ID.

        Returns:
            True if exists, False otherwise.
        """
        return self.session.query(self.model).filter(self.model.id == entity_id).count() > 0
