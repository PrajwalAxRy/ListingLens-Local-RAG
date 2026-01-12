"""
Storage and Indexing module for RHP Analyzer.

This module provides:
- Semantic chunking with section awareness
- Vector embedding generation with caching
- Vector database integration (Qdrant)
- SQL database for structured data (SQLite)
- File storage organization

Reference: blueprint.md Section 3.2, milestones.md Phase 3
"""

from rhp_analyzer.storage.chunker import (
    Chunk,
    ChunkingConfig,
    ChunkType,
    SemanticChunker,
    chunk_text,
)
from rhp_analyzer.storage.database import (
    AgentOutput,
    Base,
    BaseRepository,
    ChunkMetadata,
    DatabaseManager,
    Document,
    Entity,
    ExtractedTable,
    FinancialData,
    RiskFactor,
    Section,
)
from rhp_analyzer.storage.embeddings import (
    CacheMetadata,
    EmbeddingConfig,
    EmbeddingGenerator,
    compute_similarity,
    generate_embeddings,
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
from rhp_analyzer.storage.vector_store import (
    SearchResult,
    SearchUtility,
    VectorStore,
    VectorStoreConfig,
    create_vector_store,
)
from rhp_analyzer.storage.file_manager import (
    FileManager,
    create_file_manager,
)

__all__ = [
    # Chunking
    "Chunk",
    "ChunkType",
    "ChunkingConfig",
    "SemanticChunker",
    "chunk_text",
    # Database Models
    "Base",
    "Document",
    "Section",
    "ExtractedTable",
    "Entity",
    "FinancialData",
    "RiskFactor",
    "AgentOutput",
    "ChunkMetadata",
    # Database Manager
    "DatabaseManager",
    "BaseRepository",
    # Repositories
    "DocumentRepository",
    "SectionRepository",
    "ExtractedTableRepository",
    "EntityRepository",
    "FinancialDataRepository",
    "RiskFactorRepository",
    "AgentOutputRepository",
    "ChunkMetadataRepository",
    # Embeddings
    "CacheMetadata",
    "EmbeddingConfig",
    "EmbeddingGenerator",
    "compute_similarity",
    "generate_embeddings",
    # Vector Store
    "VectorStore",
    "VectorStoreConfig",
    "SearchResult",
    "SearchUtility",
    "create_vector_store",
    # File Manager
    "FileManager",
    "create_file_manager",
]
