"""
Embedding Generation Module for RHP Analyzer.

This module provides vector embedding generation for document chunks using
sentence-transformers. It supports:
- Primary model: nomic-ai/nomic-embed-text-v1.5 (768 dimensions)
- Fallback model: all-MiniLM-L6-v2 (384 dimensions)
- Batch processing for efficiency
- Caching to disk (numpy format) to avoid recomputation
- Content-based cache invalidation

Configuration:
- Batch size: 32 chunks (configurable)
- Device: CPU (no GPU requirement)
- Embeddings normalized by default

Cache Structure:
    data/embeddings/{document_id}/
    ├── chunks.jsonl       # Chunk text and metadata
    ├── embeddings.npy     # Numpy array of embeddings
    └── metadata.json      # Cache metadata (content hash, model, timestamp)

Reference: milestones.md Milestone 3.2, blueprint.md Section 4.1
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from rhp_analyzer.storage.chunker import Chunk


# Default embedding models with their dimensions
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
FALLBACK_MODEL = "all-MiniLM-L6-v2"

MODEL_DIMENSIONS = {
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation.

    Attributes:
        model_name: Name of the sentence-transformers model
        device: Device for computation ('cpu' or 'cuda')
        batch_size: Number of texts to embed in one batch
        normalize_embeddings: Whether to L2-normalize embeddings
        cache_dir: Base directory for embedding cache
        show_progress: Whether to show progress bar during embedding
    """

    model_name: str = DEFAULT_MODEL
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("./data/embeddings"))
    show_progress: bool = True

    def __post_init__(self):
        """Ensure cache_dir is a Path object."""
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)


@dataclass
class CacheMetadata:
    """Metadata for embedding cache validation.

    Attributes:
        document_id: ID of the source document
        model_name: Embedding model used
        model_dimension: Dimension of embeddings
        chunk_count: Number of chunks embedded
        content_hash: Hash of chunk texts for invalidation
        created_at: Timestamp when cache was created
        embedding_time_seconds: Time taken to generate embeddings
    """

    document_id: str
    model_name: str
    model_dimension: int
    chunk_count: int
    content_hash: str
    created_at: str
    embedding_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "document_id": self.document_id,
            "model_name": self.model_name,
            "model_dimension": self.model_dimension,
            "chunk_count": self.chunk_count,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "embedding_time_seconds": self.embedding_time_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheMetadata":
        """Create from dictionary."""
        return cls(
            document_id=data["document_id"],
            model_name=data["model_name"],
            model_dimension=data["model_dimension"],
            chunk_count=data["chunk_count"],
            content_hash=data["content_hash"],
            created_at=data["created_at"],
            embedding_time_seconds=data.get("embedding_time_seconds", 0.0),
        )


class EmbeddingGenerator:
    """
    Generates vector embeddings for document chunks.

    This class handles:
    - Loading sentence-transformer models (with fallback)
    - Batch embedding generation for efficiency
    - Caching embeddings to disk
    - Cache validation and invalidation

    Example:
        >>> generator = EmbeddingGenerator()
        >>> chunks = [chunk1, chunk2, chunk3]
        >>> embeddings = generator.generate(chunks)
        >>> len(embeddings) == len(chunks)
        True
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        """Initialize the embedding generator.

        Args:
            config: Embedding configuration. Uses defaults if not provided.
        """
        self.config = config or EmbeddingConfig()
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None
        self._model_loaded = False

        logger.debug(
            f"EmbeddingGenerator initialized with model={self.config.model_name}, "
            f"device={self.config.device}, batch_size={self.config.batch_size}"
        )

    @property
    def model(self) -> "SentenceTransformer":
        """Lazy-load the sentence transformer model.

        Returns:
            Loaded SentenceTransformer model.

        Raises:
            RuntimeError: If model cannot be loaded.
        """
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension for the loaded model.

        Returns:
            Integer dimension of embeddings.
        """
        if self._dimension is None:
            # Get dimension from known models or by generating a test embedding
            self._dimension = MODEL_DIMENSIONS.get(self.config.model_name)
            if self._dimension is None:
                # Generate a test embedding to determine dimension
                test_embedding = self.model.encode("test", normalize_embeddings=False)
                self._dimension = len(test_embedding)
        return self._dimension

    def _load_model(self) -> "SentenceTransformer":
        """Load the sentence transformer model with fallback.

        Returns:
            Loaded SentenceTransformer model.

        Raises:
            RuntimeError: If neither primary nor fallback model can be loaded.
        """
        from sentence_transformers import SentenceTransformer

        model_name = self.config.model_name
        device = self.config.device

        try:
            logger.info(f"Loading embedding model: {model_name} on device: {device}")
            model = SentenceTransformer(model_name, device=device)
            self._dimension = MODEL_DIMENSIONS.get(model_name)
            self._model_loaded = True
            logger.info(f"Successfully loaded model: {model_name}")
            return model

        except Exception as e:
            logger.warning(f"Failed to load primary model {model_name}: {e}")

            if model_name != FALLBACK_MODEL:
                logger.info(f"Attempting fallback model: {FALLBACK_MODEL}")
                try:
                    model = SentenceTransformer(FALLBACK_MODEL, device=device)
                    self._dimension = MODEL_DIMENSIONS.get(FALLBACK_MODEL, 384)
                    self._model_loaded = True
                    self.config.model_name = FALLBACK_MODEL  # Update config to reflect actual model
                    logger.info(f"Successfully loaded fallback model: {FALLBACK_MODEL}")
                    return model

                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Failed to load both primary ({model_name}) and "
                        f"fallback ({FALLBACK_MODEL}) models. "
                        f"Primary error: {e}, Fallback error: {fallback_error}"
                    ) from fallback_error
            else:
                raise RuntimeError(f"Failed to load embedding model {model_name}: {e}") from e

    def generate(
        self,
        texts: list[str],
        document_id: str | None = None,
        use_cache: bool = True,
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            document_id: Optional document ID for caching.
            use_cache: Whether to use cached embeddings if available.

        Returns:
            List of embedding vectors (list of floats).
        """
        if not texts:
            logger.warning("Empty text list provided for embedding generation")
            return []

        # Check cache if document_id provided and caching enabled
        if use_cache and document_id:
            cached = self._load_from_cache(document_id, texts)
            if cached is not None:
                logger.info(f"Loaded {len(cached)} embeddings from cache for document {document_id}")
                return cached

        # Generate new embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={self.config.batch_size})")

        import time

        start_time = time.time()

        embeddings_array = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Generated {len(embeddings_array)} embeddings in {elapsed_time:.2f}s")

        # Convert to list of lists for consistency
        embeddings = embeddings_array.tolist()

        # Save to cache if document_id provided
        if document_id:
            self._save_to_cache(document_id, texts, embeddings_array, elapsed_time)

        return embeddings

    def generate_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector as list of floats.
        """
        if not text:
            raise ValueError("Cannot generate embedding for empty text")

        embedding = self.model.encode(
            text,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def generate_for_chunks(
        self,
        chunks: list["Chunk"],
        document_id: str | None = None,
        use_cache: bool = True,
    ) -> list[list[float]]:
        """Generate embeddings for a list of Chunk objects.

        This is a convenience method that extracts text from chunks.

        Args:
            chunks: List of Chunk objects.
            document_id: Optional document ID for caching.
            use_cache: Whether to use cached embeddings if available.

        Returns:
            List of embedding vectors.
        """
        texts = [chunk.text for chunk in chunks]
        return self.generate(texts, document_id=document_id, use_cache=use_cache)

    def _get_cache_dir(self, document_id: str) -> Path:
        """Get the cache directory for a specific document.

        Args:
            document_id: Document identifier.

        Returns:
            Path to the document's cache directory.
        """
        return self.config.cache_dir / document_id

    def _compute_content_hash(self, texts: list[str]) -> str:
        """Compute a hash of the text content for cache validation.

        Args:
            texts: List of text strings.

        Returns:
            SHA-256 hash of concatenated texts.
        """
        combined = "|||".join(texts)  # Use unique separator
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _load_from_cache(
        self,
        document_id: str,
        texts: list[str],
    ) -> list[list[float]] | None:
        """Load embeddings from cache if valid.

        Args:
            document_id: Document identifier.
            texts: List of texts (for content hash validation).

        Returns:
            List of embeddings if cache is valid, None otherwise.
        """
        cache_dir = self._get_cache_dir(document_id)
        metadata_path = cache_dir / "metadata.json"
        embeddings_path = cache_dir / "embeddings.npy"

        # Check if cache files exist
        if not metadata_path.exists() or not embeddings_path.exists():
            logger.debug(f"No cache found for document {document_id}")
            return None

        try:
            # Load and validate metadata
            with open(metadata_path, encoding="utf-8") as f:
                metadata_dict = json.load(f)
            metadata = CacheMetadata.from_dict(metadata_dict)

            # Validate content hash
            current_hash = self._compute_content_hash(texts)
            if metadata.content_hash != current_hash:
                logger.info(f"Cache invalidated for {document_id}: content has changed")
                return None

            # Validate model name
            if metadata.model_name != self.config.model_name:
                logger.info(
                    f"Cache invalidated for {document_id}: model changed "
                    f"({metadata.model_name} -> {self.config.model_name})"
                )
                return None

            # Validate chunk count
            if metadata.chunk_count != len(texts):
                logger.info(
                    f"Cache invalidated for {document_id}: chunk count changed "
                    f"({metadata.chunk_count} -> {len(texts)})"
                )
                return None

            # Load embeddings
            embeddings_array = np.load(embeddings_path)

            if len(embeddings_array) != len(texts):
                logger.warning(
                    f"Cache size mismatch for {document_id}: "
                    f"expected {len(texts)}, got {len(embeddings_array)}"
                )
                return None

            return embeddings_array.tolist()

        except Exception as e:
            logger.warning(f"Error loading cache for {document_id}: {e}")
            return None

    def _save_to_cache(
        self,
        document_id: str,
        texts: list[str],
        embeddings: np.ndarray,
        generation_time: float,
    ) -> None:
        """Save embeddings to cache.

        Args:
            document_id: Document identifier.
            texts: List of text strings.
            embeddings: Numpy array of embeddings.
            generation_time: Time taken to generate embeddings.
        """
        cache_dir = self._get_cache_dir(document_id)

        try:
            # Create cache directory
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Save embeddings as numpy array
            embeddings_path = cache_dir / "embeddings.npy"
            np.save(embeddings_path, embeddings)

            # Save chunk texts as JSONL
            chunks_path = cache_dir / "chunks.jsonl"
            with open(chunks_path, "w", encoding="utf-8") as f:
                for i, text in enumerate(texts):
                    chunk_data = {"index": i, "text": text[:500]}  # Truncate for storage
                    f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")

            # Save metadata
            metadata = CacheMetadata(
                document_id=document_id,
                model_name=self.config.model_name,
                model_dimension=self.dimension,
                chunk_count=len(texts),
                content_hash=self._compute_content_hash(texts),
                created_at=datetime.now().isoformat(),
                embedding_time_seconds=generation_time,
            )

            metadata_path = cache_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            logger.info(f"Saved {len(embeddings)} embeddings to cache for document {document_id}")

        except Exception as e:
            logger.error(f"Error saving cache for {document_id}: {e}")
            # Don't raise - caching is not critical

    def clear_cache(self, document_id: str) -> bool:
        """Clear cached embeddings for a document.

        Args:
            document_id: Document identifier.

        Returns:
            True if cache was cleared, False if no cache existed.
        """
        import shutil

        cache_dir = self._get_cache_dir(document_id)

        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                logger.info(f"Cleared cache for document {document_id}")
                return True
            except Exception as e:
                logger.error(f"Error clearing cache for {document_id}: {e}")
                return False
        return False

    def clear_all_cache(self) -> int:
        """Clear all cached embeddings.

        Returns:
            Number of document caches cleared.
        """
        import shutil

        if not self.config.cache_dir.exists():
            return 0

        cleared = 0
        for doc_dir in self.config.cache_dir.iterdir():
            if doc_dir.is_dir():
                try:
                    shutil.rmtree(doc_dir)
                    cleared += 1
                except Exception as e:
                    logger.error(f"Error clearing cache for {doc_dir.name}: {e}")

        logger.info(f"Cleared {cleared} document caches")
        return cleared

    def get_cache_info(self, document_id: str) -> CacheMetadata | None:
        """Get cache metadata for a document.

        Args:
            document_id: Document identifier.

        Returns:
            CacheMetadata if cache exists, None otherwise.
        """
        metadata_path = self._get_cache_dir(document_id) / "metadata.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, encoding="utf-8") as f:
                return CacheMetadata.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f"Error reading cache metadata for {document_id}: {e}")
            return None

    def is_cache_valid(self, document_id: str, texts: list[str]) -> bool:
        """Check if cache is valid for given texts.

        Args:
            document_id: Document identifier.
            texts: List of texts to validate against.

        Returns:
            True if cache exists and is valid for the given texts.
        """
        return self._load_from_cache(document_id, texts) is not None


def generate_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: str = "cpu",
    normalize: bool = True,
    show_progress: bool = True,
) -> list[list[float]]:
    """Convenience function to generate embeddings without class instantiation.

    Args:
        texts: List of text strings to embed.
        model_name: Sentence transformer model name.
        batch_size: Batch size for processing.
        device: Computation device ('cpu' or 'cuda').
        normalize: Whether to normalize embeddings.
        show_progress: Whether to show progress bar.

    Returns:
        List of embedding vectors.
    """
    config = EmbeddingConfig(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        normalize_embeddings=normalize,
        show_progress=show_progress,
    )
    generator = EmbeddingGenerator(config)
    return generator.generate(texts)


def compute_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector.
        embedding2: Second embedding vector.

    Returns:
        Cosine similarity score (0 to 1 for normalized embeddings).
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))
