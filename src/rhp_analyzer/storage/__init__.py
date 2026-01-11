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
from rhp_analyzer.storage.embeddings import (
    CacheMetadata,
    EmbeddingConfig,
    EmbeddingGenerator,
    compute_similarity,
    generate_embeddings,
)

__all__ = [
    # Chunking
    "Chunk",
    "ChunkType",
    "ChunkingConfig",
    "SemanticChunker",
    "chunk_text",
    # Embeddings
    "CacheMetadata",
    "EmbeddingConfig",
    "EmbeddingGenerator",
    "compute_similarity",
    "generate_embeddings",
]
