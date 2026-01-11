"""
Storage and Indexing module for RHP Analyzer.

This module provides:
- Semantic chunking with section awareness
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

__all__ = [
    "Chunk",
    "ChunkType",
    "ChunkingConfig",
    "SemanticChunker",
    "chunk_text",
]
