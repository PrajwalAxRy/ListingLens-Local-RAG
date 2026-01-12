"""
Repository pattern implementations for database entities.

This module provides specific repository classes for each entity type,
implementing common access patterns and custom query methods.

Reference: milestones.md Phase 3.4.2
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session, joinedload

from rhp_analyzer.storage.database import (
    AgentOutput,
    BaseRepository,
    ChunkMetadata,
    Document,
    Entity,
    ExtractedTable,
    FinancialData,
    RiskFactor,
    Section,
)


class DocumentRepository(BaseRepository[Document]):
    """
    Repository for Document entities.

    Provides document-specific query methods for RHP document management.
    """

    model = Document

    def find_by_document_id(self, document_id: str) -> Document | None:
        """
        Find a document by its unique document ID.

        Args:
            document_id: The unique document identifier (not the primary key).

        Returns:
            The document if found, None otherwise.
        """
        return (
            self.session.query(self.model)
            .filter(self.model.document_id == document_id)
            .first()
        )

    def find_by_company_name(self, company_name: str) -> list[Document]:
        """
        Find documents by company name (case-insensitive partial match).

        Args:
            company_name: Company name to search for.

        Returns:
            List of matching documents.
        """
        return (
            self.session.query(self.model)
            .filter(self.model.company_name.ilike(f"%{company_name}%"))
            .all()
        )

    def find_by_status(self, status: str) -> list[Document]:
        """
        Find documents by processing status.

        Args:
            status: Processing status ('pending', 'processing', 'completed', 'failed').

        Returns:
            List of documents with the given status.
        """
        return (
            self.session.query(self.model)
            .filter(self.model.processing_status == status)
            .all()
        )

    def get_recent(self, limit: int = 10) -> list[Document]:
        """
        Get most recently uploaded documents.

        Args:
            limit: Maximum number of documents to return.

        Returns:
            List of recent documents ordered by upload date.
        """
        return (
            self.session.query(self.model)
            .order_by(desc(self.model.upload_date))
            .limit(limit)
            .all()
        )

    def update_status(self, document_id: str, status: str) -> Document | None:
        """
        Update the processing status of a document.

        Args:
            document_id: The unique document identifier.
            status: New processing status.

        Returns:
            The updated document if found, None otherwise.
        """
        doc = self.find_by_document_id(document_id)
        if doc:
            doc.processing_status = status
            doc.updated_at = datetime.now(timezone.utc)
            self.session.flush()
        return doc

    def get_with_sections(self, document_id: str) -> Document | None:
        """
        Get a document with its sections eagerly loaded.

        Args:
            document_id: The unique document identifier.

        Returns:
            The document with sections loaded, or None if not found.
        """
        return (
            self.session.query(self.model)
            .options(joinedload(self.model.sections))
            .filter(self.model.document_id == document_id)
            .first()
        )

    def get_with_all_relations(self, document_id: str) -> Document | None:
        """
        Get a document with all related entities eagerly loaded.

        Args:
            document_id: The unique document identifier.

        Returns:
            The document with all relations loaded, or None if not found.
        """
        return (
            self.session.query(self.model)
            .options(
                joinedload(self.model.sections),
                joinedload(self.model.tables),
                joinedload(self.model.entities),
                joinedload(self.model.financial_data),
                joinedload(self.model.risk_factors),
                joinedload(self.model.agent_outputs),
            )
            .filter(self.model.document_id == document_id)
            .first()
        )

    def delete_by_document_id(self, document_id: str) -> bool:
        """
        Delete a document and all related data by document ID.

        Args:
            document_id: The unique document identifier.

        Returns:
            True if deleted, False if not found.
        """
        doc = self.find_by_document_id(document_id)
        if doc:
            self.session.delete(doc)
            self.session.flush()
            return True
        return False


class SectionRepository(BaseRepository[Section]):
    """
    Repository for Section entities.

    Provides section-specific query methods for document structure.
    """

    model = Section

    def find_by_document(self, document_id: int) -> list[Section]:
        """
        Find all sections for a document.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of sections ordered by start page.
        """
        return (
            self.session.query(self.model)
            .filter(self.model.document_id == document_id)
            .order_by(self.model.start_page)
            .all()
        )

    def find_by_name(self, document_id: int, section_name: str) -> Section | None:
        """
        Find a section by name within a document.

        Args:
            document_id: The document's primary key ID.
            section_name: Name of the section.

        Returns:
            The section if found, None otherwise.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.section_name.ilike(f"%{section_name}%"),
                )
            )
            .first()
        )

    def find_by_type(self, document_id: int, section_type: str) -> list[Section]:
        """
        Find sections by type within a document.

        Args:
            document_id: The document's primary key ID.
            section_type: Type of section (e.g., 'risk_factors', 'financials').

        Returns:
            List of matching sections.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.section_type == section_type,
                )
            )
            .all()
        )

    def find_by_page(self, document_id: int, page_number: int) -> list[Section]:
        """
        Find sections containing a specific page.

        Args:
            document_id: The document's primary key ID.
            page_number: Page number to search for.

        Returns:
            List of sections containing the page.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.start_page <= page_number,
                    self.model.end_page >= page_number,
                )
            )
            .all()
        )

    def get_top_level_sections(self, document_id: int) -> list[Section]:
        """
        Get top-level sections (level 1) for a document.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of top-level sections.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.level == 1,
                )
            )
            .order_by(self.model.start_page)
            .all()
        )


class FinancialDataRepository(BaseRepository[FinancialData]):
    """
    Repository for FinancialData entities.

    Provides financial data query methods for analysis.
    """

    model = FinancialData

    def find_by_document(self, document_id: int) -> list[FinancialData]:
        """
        Find all financial data for a document.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of financial data ordered by fiscal year.
        """
        return (
            self.session.query(self.model)
            .filter(self.model.document_id == document_id)
            .order_by(self.model.fiscal_year)
            .all()
        )

    def find_by_fiscal_year(
        self, document_id: int, fiscal_year: str
    ) -> FinancialData | None:
        """
        Find financial data for a specific fiscal year.

        Args:
            document_id: The document's primary key ID.
            fiscal_year: Fiscal year (e.g., 'FY24').

        Returns:
            The financial data if found, None otherwise.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.fiscal_year == fiscal_year,
                )
            )
            .first()
        )

    def get_latest(self, document_id: int) -> FinancialData | None:
        """
        Get the most recent financial data for a document.

        Args:
            document_id: The document's primary key ID.

        Returns:
            The most recent financial data or None.
        """
        return (
            self.session.query(self.model)
            .filter(self.model.document_id == document_id)
            .order_by(desc(self.model.fiscal_year))
            .first()
        )

    def get_revenue_trend(self, document_id: int) -> list[dict[str, Any]]:
        """
        Get revenue trend data for a document.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of dicts with fiscal_year and revenue.
        """
        results = (
            self.session.query(
                self.model.fiscal_year,
                self.model.revenue,
                self.model.revenue_growth,
            )
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.revenue.isnot(None),
                )
            )
            .order_by(self.model.fiscal_year)
            .all()
        )
        return [
            {"fiscal_year": r.fiscal_year, "revenue": r.revenue, "growth": r.revenue_growth}
            for r in results
        ]

    def get_profitability_metrics(self, document_id: int) -> list[dict[str, Any]]:
        """
        Get profitability metrics (EBITDA, PAT, margins) for a document.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of dicts with profitability metrics by year.
        """
        results = (
            self.session.query(
                self.model.fiscal_year,
                self.model.ebitda,
                self.model.ebitda_margin,
                self.model.pat,
                self.model.pat_margin,
                self.model.roe,
                self.model.roce,
            )
            .filter(self.model.document_id == document_id)
            .order_by(self.model.fiscal_year)
            .all()
        )
        return [
            {
                "fiscal_year": r.fiscal_year,
                "ebitda": r.ebitda,
                "ebitda_margin": r.ebitda_margin,
                "pat": r.pat,
                "pat_margin": r.pat_margin,
                "roe": r.roe,
                "roce": r.roce,
            }
            for r in results
        ]

    def calculate_cagr(
        self, document_id: int, metric: str = "revenue"
    ) -> float | None:
        """
        Calculate CAGR for a specific metric.

        Args:
            document_id: The document's primary key ID.
            metric: Metric to calculate CAGR for (e.g., 'revenue', 'pat').

        Returns:
            CAGR as a decimal (e.g., 0.15 for 15%), or None if insufficient data.
        """
        if not hasattr(self.model, metric):
            return None

        results = (
            self.session.query(
                self.model.fiscal_year,
                getattr(self.model, metric),
            )
            .filter(
                and_(
                    self.model.document_id == document_id,
                    getattr(self.model, metric).isnot(None),
                )
            )
            .order_by(self.model.fiscal_year)
            .all()
        )

        if len(results) < 2:
            return None

        start_value = results[0][1]
        end_value = results[-1][1]
        years = len(results) - 1

        if start_value <= 0 or end_value <= 0 or years <= 0:
            return None

        cagr = (end_value / start_value) ** (1 / years) - 1
        return cagr


class AgentOutputRepository(BaseRepository[AgentOutput]):
    """
    Repository for AgentOutput entities.

    Provides agent output query methods for analysis results.
    """

    model = AgentOutput

    def find_by_document(self, document_id: int) -> list[AgentOutput]:
        """
        Find all agent outputs for a document.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of agent outputs ordered by creation time.
        """
        return (
            self.session.query(self.model)
            .filter(self.model.document_id == document_id)
            .order_by(self.model.created_at)
            .all()
        )

    def find_by_agent(
        self, document_id: int, agent_name: str
    ) -> AgentOutput | None:
        """
        Find the latest output for a specific agent.

        Args:
            document_id: The document's primary key ID.
            agent_name: Name of the agent.

        Returns:
            The latest agent output if found, None otherwise.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.agent_name == agent_name,
                )
            )
            .order_by(desc(self.model.iteration), desc(self.model.created_at))
            .first()
        )

    def find_all_by_agent(
        self, document_id: int, agent_name: str
    ) -> list[AgentOutput]:
        """
        Find all outputs for a specific agent (all iterations).

        Args:
            document_id: The document's primary key ID.
            agent_name: Name of the agent.

        Returns:
            List of agent outputs for all iterations.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.agent_name == agent_name,
                )
            )
            .order_by(self.model.iteration)
            .all()
        )

    def get_completed_agents(self, document_id: int) -> list[str]:
        """
        Get list of agents that have completed analysis.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of agent names that have completed successfully.
        """
        results = (
            self.session.query(self.model.agent_name)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.status == "completed",
                )
            )
            .distinct()
            .all()
        )
        return [r.agent_name for r in results]

    def get_failed_agents(self, document_id: int) -> list[tuple[str, str]]:
        """
        Get list of agents that failed with their error messages.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of tuples (agent_name, error_message).
        """
        results = (
            self.session.query(self.model.agent_name, self.model.error_message)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.status == "failed",
                )
            )
            .all()
        )
        return [(r.agent_name, r.error_message) for r in results]

    def get_average_confidence(self, document_id: int) -> float | None:
        """
        Calculate average confidence score across all agents.

        Args:
            document_id: The document's primary key ID.

        Returns:
            Average confidence score or None if no data.
        """
        result = (
            self.session.query(func.avg(self.model.confidence_score))
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.status == "completed",
                    self.model.confidence_score.isnot(None),
                )
            )
            .scalar()
        )
        return float(result) if result else None

    def get_total_processing_time(self, document_id: int) -> float:
        """
        Get total processing time for all agents.

        Args:
            document_id: The document's primary key ID.

        Returns:
            Total processing time in seconds.
        """
        result = (
            self.session.query(func.sum(self.model.processing_time_seconds))
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.processing_time_seconds.isnot(None),
                )
            )
            .scalar()
        )
        return float(result) if result else 0.0


class EntityRepository(BaseRepository[Entity]):
    """
    Repository for Entity entities.

    Provides entity-specific query methods for named entity management.
    """

    model = Entity

    def find_by_document(self, document_id: int) -> list[Entity]:
        """
        Find all entities for a document.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of entities ordered by mention count.
        """
        return (
            self.session.query(self.model)
            .filter(self.model.document_id == document_id)
            .order_by(desc(self.model.mentions))
            .all()
        )

    def find_by_type(self, document_id: int, entity_type: str) -> list[Entity]:
        """
        Find entities of a specific type.

        Args:
            document_id: The document's primary key ID.
            entity_type: Type of entity (e.g., 'person', 'company').

        Returns:
            List of matching entities.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.entity_type == entity_type,
                )
            )
            .order_by(desc(self.model.mentions))
            .all()
        )

    def find_by_name(
        self, document_id: int, entity_name: str
    ) -> Entity | None:
        """
        Find an entity by name.

        Args:
            document_id: The document's primary key ID.
            entity_name: Name of the entity.

        Returns:
            The entity if found, None otherwise.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.entity_name.ilike(f"%{entity_name}%"),
                )
            )
            .first()
        )

    def get_top_entities(
        self, document_id: int, entity_type: str | None = None, limit: int = 10
    ) -> list[Entity]:
        """
        Get top entities by mention count.

        Args:
            document_id: The document's primary key ID.
            entity_type: Optional type filter.
            limit: Maximum number to return.

        Returns:
            List of top entities.
        """
        query = self.session.query(self.model).filter(
            self.model.document_id == document_id
        )
        if entity_type:
            query = query.filter(self.model.entity_type == entity_type)
        return query.order_by(desc(self.model.mentions)).limit(limit).all()


class RiskFactorRepository(BaseRepository[RiskFactor]):
    """
    Repository for RiskFactor entities.

    Provides risk factor query methods for risk analysis.
    """

    model = RiskFactor

    def find_by_document(self, document_id: int) -> list[RiskFactor]:
        """
        Find all risk factors for a document.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of risk factors ordered by severity.
        """
        severity_order = {"critical": 0, "major": 1, "minor": 2}
        results = (
            self.session.query(self.model)
            .filter(self.model.document_id == document_id)
            .all()
        )
        return sorted(results, key=lambda r: severity_order.get(r.severity, 3))

    def find_by_category(
        self, document_id: int, category: str
    ) -> list[RiskFactor]:
        """
        Find risk factors by category.

        Args:
            document_id: The document's primary key ID.
            category: Risk category (e.g., 'operational', 'financial').

        Returns:
            List of matching risk factors.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.category == category,
                )
            )
            .all()
        )

    def find_by_severity(
        self, document_id: int, severity: str
    ) -> list[RiskFactor]:
        """
        Find risk factors by severity.

        Args:
            document_id: The document's primary key ID.
            severity: Severity level ('critical', 'major', 'minor').

        Returns:
            List of matching risk factors.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.severity == severity,
                )
            )
            .all()
        )

    def get_critical_risks(self, document_id: int) -> list[RiskFactor]:
        """
        Get all critical severity risk factors.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of critical risk factors.
        """
        return self.find_by_severity(document_id, "critical")

    def get_non_boilerplate(self, document_id: int) -> list[RiskFactor]:
        """
        Get risk factors that are not boilerplate/generic.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of company-specific risk factors.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.is_boilerplate == 0,
                )
            )
            .all()
        )

    def count_by_severity(self, document_id: int) -> dict[str, int]:
        """
        Count risk factors by severity level.

        Args:
            document_id: The document's primary key ID.

        Returns:
            Dict mapping severity to count.
        """
        results = (
            self.session.query(
                self.model.severity, func.count(self.model.id)
            )
            .filter(self.model.document_id == document_id)
            .group_by(self.model.severity)
            .all()
        )
        return {severity: count for severity, count in results}


class ExtractedTableRepository(BaseRepository[ExtractedTable]):
    """
    Repository for ExtractedTable entities.

    Provides table-specific query methods for extracted tables.
    """

    model = ExtractedTable

    def find_by_document(self, document_id: int) -> list[ExtractedTable]:
        """
        Find all tables for a document.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of tables ordered by page number.
        """
        return (
            self.session.query(self.model)
            .filter(self.model.document_id == document_id)
            .order_by(self.model.page_number)
            .all()
        )

    def find_by_type(
        self, document_id: int, table_type: str
    ) -> list[ExtractedTable]:
        """
        Find tables of a specific type.

        Args:
            document_id: The document's primary key ID.
            table_type: Type of table (e.g., 'financial_statement').

        Returns:
            List of matching tables.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.table_type == table_type,
                )
            )
            .all()
        )

    def find_by_page(
        self, document_id: int, page_number: int
    ) -> list[ExtractedTable]:
        """
        Find tables on a specific page.

        Args:
            document_id: The document's primary key ID.
            page_number: Page number.

        Returns:
            List of tables on the page.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.page_number == page_number,
                )
            )
            .all()
        )

    def get_financial_tables(self, document_id: int) -> list[ExtractedTable]:
        """
        Get all financial statement tables.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of financial tables.
        """
        financial_types = [
            "financial_statement",
            "balance_sheet",
            "profit_loss",
            "cash_flow",
        ]
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.table_type.in_(financial_types),
                )
            )
            .all()
        )


class ChunkMetadataRepository(BaseRepository[ChunkMetadata]):
    """
    Repository for ChunkMetadata entities.

    Provides chunk metadata query methods for vector store integration.
    """

    model = ChunkMetadata

    def find_by_document(self, document_id: int) -> list[ChunkMetadata]:
        """
        Find all chunk metadata for a document.

        Args:
            document_id: The document's primary key ID.

        Returns:
            List of chunk metadata.
        """
        return (
            self.session.query(self.model)
            .filter(self.model.document_id == document_id)
            .all()
        )

    def find_by_chunk_id(self, chunk_id: str) -> ChunkMetadata | None:
        """
        Find chunk metadata by its unique chunk ID.

        Args:
            chunk_id: The unique chunk identifier.

        Returns:
            The chunk metadata if found, None otherwise.
        """
        return (
            self.session.query(self.model)
            .filter(self.model.chunk_id == chunk_id)
            .first()
        )

    def find_by_section(
        self, document_id: int, section_name: str
    ) -> list[ChunkMetadata]:
        """
        Find chunks by section name.

        Args:
            document_id: The document's primary key ID.
            section_name: Name of the section.

        Returns:
            List of chunks in the section.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.section_name.ilike(f"%{section_name}%"),
                )
            )
            .all()
        )

    def find_by_page_range(
        self, document_id: int, start_page: int, end_page: int
    ) -> list[ChunkMetadata]:
        """
        Find chunks within a page range.

        Args:
            document_id: The document's primary key ID.
            start_page: Start page (inclusive).
            end_page: End page (inclusive).

        Returns:
            List of chunks in the page range.
        """
        return (
            self.session.query(self.model)
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.start_page >= start_page,
                    self.model.end_page <= end_page,
                )
            )
            .all()
        )

    def get_chunk_chain(self, chunk_id: str, direction: str = "both") -> list[ChunkMetadata]:
        """
        Get the chain of chunks linked to this chunk.

        Args:
            chunk_id: The starting chunk ID.
            direction: 'preceding', 'following', or 'both'.

        Returns:
            List of linked chunks in order.
        """
        result = []
        current = self.find_by_chunk_id(chunk_id)

        if not current:
            return result

        result.append(current)

        # Get preceding chunks
        if direction in ("preceding", "both"):
            preceding = []
            prev_id = current.preceding_chunk_id
            while prev_id:
                prev_chunk = self.find_by_chunk_id(prev_id)
                if prev_chunk:
                    preceding.append(prev_chunk)
                    prev_id = prev_chunk.preceding_chunk_id
                else:
                    break
            result = list(reversed(preceding)) + result

        # Get following chunks
        if direction in ("following", "both"):
            next_id = current.following_chunk_id
            while next_id:
                next_chunk = self.find_by_chunk_id(next_id)
                if next_chunk:
                    result.append(next_chunk)
                    next_id = next_chunk.following_chunk_id
                else:
                    break

        return result

    def count_by_type(self, document_id: int) -> dict[str, int]:
        """
        Count chunks by type.

        Args:
            document_id: The document's primary key ID.

        Returns:
            Dict mapping chunk type to count.
        """
        results = (
            self.session.query(
                self.model.chunk_type, func.count(self.model.id)
            )
            .filter(self.model.document_id == document_id)
            .group_by(self.model.chunk_type)
            .all()
        )
        return {chunk_type: count for chunk_type, count in results}

    def get_total_tokens(self, document_id: int) -> int:
        """
        Get total token count for a document.

        Args:
            document_id: The document's primary key ID.

        Returns:
            Total token count.
        """
        result = (
            self.session.query(func.sum(self.model.token_count))
            .filter(
                and_(
                    self.model.document_id == document_id,
                    self.model.token_count.isnot(None),
                )
            )
            .scalar()
        )
        return int(result) if result else 0
