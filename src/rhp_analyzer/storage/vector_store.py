"""
Vector Store Module for RHP Analyzer using Qdrant.

This module provides semantic search capabilities using Qdrant in embedded mode
(local file storage, no server needed). It handles:
- Collection creation and management per document
- Adding chunks with embeddings and rich metadata
- Semantic similarity search with metadata filtering
- CRUD operations for document lifecycle management

Configuration:
- Embedded mode: No Docker or server required
- Storage path: data/qdrant/
- Distance metric: Cosine similarity
- Vector size: 768 (matching nomic-embed-text-v1.5)

Reference: milestones.md Milestone 3.3, blueprint.md Section 3.2.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from rhp_analyzer.storage.chunker import Chunk

# Default configuration
DEFAULT_STORAGE_PATH = Path("./data/qdrant")
DEFAULT_VECTOR_SIZE = 768  # nomic-embed-text-v1.5
FALLBACK_VECTOR_SIZE = 384  # all-MiniLM-L6-v2


@dataclass
class VectorStoreConfig:
    """Configuration for the Qdrant vector store.

    Attributes:
        storage_path: Path to store Qdrant data (embedded mode)
        vector_size: Dimension of embedding vectors
        distance: Distance metric for similarity (COSINE, DOT, EUCLID)
        collection_prefix: Prefix for collection names
        on_disk_payload: Whether to store payloads on disk (saves RAM)
    """

    storage_path: Path = field(default_factory=lambda: DEFAULT_STORAGE_PATH)
    vector_size: int = DEFAULT_VECTOR_SIZE
    distance: str = "COSINE"
    collection_prefix: str = "rhp"
    on_disk_payload: bool = True

    def __post_init__(self):
        """Ensure storage_path is a Path object."""
        if isinstance(self.storage_path, str):
            self.storage_path = Path(self.storage_path)


@dataclass
class SearchResult:
    """Result from a vector similarity search.

    Attributes:
        chunk_id: Unique identifier of the matched chunk
        text: The text content of the chunk
        score: Similarity score (higher is more similar for cosine)
        section: Section name where chunk is located
        page_start: Starting page number
        page_end: Ending page number
        chunk_type: Type of chunk (narrative, table, mixed)
        metadata: Additional metadata from the chunk
    """

    chunk_id: str
    text: str
    score: float
    section: str
    page_start: int
    page_end: int
    chunk_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Readable representation."""
        preview = self.text[:100] + "..." if len(self.text) > 100 else self.text
        return (
            f"SearchResult(score={self.score:.4f}, section='{self.section}', "
            f"pages={self.page_start}-{self.page_end}, text='{preview}')"
        )


class VectorStore:
    """
    Qdrant-based vector store for semantic search.

    This class manages vector storage for RHP document chunks using Qdrant
    in embedded mode (local file-based storage).

    Features:
    - Per-document collections for isolation
    - Rich metadata filtering (section, page, chunk type)
    - Efficient batch operations
    - Persistent storage across runs

    Example:
        >>> store = VectorStore()
        >>> store.create_collection("doc_123")
        >>> store.add_chunks("doc_123", chunks, embeddings)
        >>> results = store.search("doc_123", query_embedding, top_k=5)
    """

    def __init__(self, config: VectorStoreConfig | None = None):
        """Initialize the vector store.

        Args:
            config: Vector store configuration. Uses defaults if not provided.
        """
        self.config = config or VectorStoreConfig()
        self._client = None
        self._initialized = False

        # Ensure storage directory exists
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"VectorStore initialized with path={self.config.storage_path}, "
            f"vector_size={self.config.vector_size}"
        )

    @property
    def client(self):
        """Lazy-load the Qdrant client.

        Returns:
            QdrantClient instance configured for embedded mode.

        Raises:
            RuntimeError: If qdrant-client is not installed.
        """
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        """Create and configure the Qdrant client.

        Returns:
            Configured QdrantClient instance.

        Raises:
            RuntimeError: If qdrant-client package is not available.
        """
        try:
            from qdrant_client import QdrantClient

            logger.info(f"Initializing Qdrant client at {self.config.storage_path}")
            client = QdrantClient(path=str(self.config.storage_path))
            self._initialized = True
            logger.info("Qdrant client initialized successfully")
            return client

        except ImportError as e:
            raise RuntimeError(
                "qdrant-client package is required for vector storage. "
                "Install with: pip install qdrant-client"
            ) from e

    def _get_collection_name(self, document_id: str) -> str:
        """Generate a collection name for a document.

        Args:
            document_id: Document identifier.

        Returns:
            Collection name with prefix.
        """
        # Sanitize document_id for collection name (alphanumeric and underscore only)
        safe_id = "".join(c if c.isalnum() or c == "_" else "_" for c in document_id)
        return f"{self.config.collection_prefix}_{safe_id}"

    def _get_distance_metric(self):
        """Get the Qdrant distance metric enum.

        Returns:
            Qdrant Distance enum value.
        """
        from qdrant_client.models import Distance

        distance_map = {
            "COSINE": Distance.COSINE,
            "DOT": Distance.DOT,
            "EUCLID": Distance.EUCLID,
        }
        return distance_map.get(self.config.distance.upper(), Distance.COSINE)

    def collection_exists(self, document_id: str) -> bool:
        """Check if a collection exists for a document.

        Args:
            document_id: Document identifier.

        Returns:
            True if collection exists, False otherwise.
        """
        collection_name = self._get_collection_name(document_id)
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            logger.warning(f"Error checking collection existence: {e}")
            return False

    def create_collection(
        self,
        document_id: str,
        vector_size: int | None = None,
        recreate: bool = False,
    ) -> bool:
        """Create a collection for a document.

        Args:
            document_id: Document identifier.
            vector_size: Optional override for vector dimension.
            recreate: If True, delete existing collection first.

        Returns:
            True if collection was created, False if already exists.

        Raises:
            RuntimeError: If collection creation fails.
        """
        from qdrant_client.models import VectorParams

        collection_name = self._get_collection_name(document_id)
        actual_vector_size = vector_size or self.config.vector_size

        # Check if collection already exists
        if self.collection_exists(document_id):
            if recreate:
                logger.info(f"Recreating collection: {collection_name}")
                self.delete_collection(document_id)
            else:
                logger.debug(f"Collection already exists: {collection_name}")
                return False

        try:
            logger.info(
                f"Creating collection: {collection_name} with vector_size={actual_vector_size}"
            )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=actual_vector_size,
                    distance=self._get_distance_metric(),
                    on_disk=self.config.on_disk_payload,
                ),
            )
            logger.info(f"Successfully created collection: {collection_name}")
            return True

        except Exception as e:
            raise RuntimeError(
                f"Failed to create collection {collection_name}: {e}"
            ) from e

    def delete_collection(self, document_id: str) -> bool:
        """Delete a document's collection.

        Args:
            document_id: Document identifier.

        Returns:
            True if collection was deleted, False if it didn't exist.
        """
        collection_name = self._get_collection_name(document_id)

        if not self.collection_exists(document_id):
            logger.debug(f"Collection does not exist: {collection_name}")
            return False

        try:
            logger.info(f"Deleting collection: {collection_name}")
            self.client.delete_collection(collection_name)
            logger.info(f"Successfully deleted collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False

    def add_chunks(
        self,
        document_id: str,
        chunks: list["Chunk"],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> int:
        """Add chunks with embeddings to a document's collection.

        Args:
            document_id: Document identifier.
            chunks: List of Chunk objects to add.
            embeddings: List of embedding vectors (must match chunks length).
            batch_size: Number of points to upsert in one batch.

        Returns:
            Number of chunks successfully added.

        Raises:
            ValueError: If chunks and embeddings length don't match.
            RuntimeError: If collection doesn't exist or upsert fails.
        """
        from qdrant_client.models import PointStruct

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )

        if not chunks:
            logger.warning("No chunks to add")
            return 0

        collection_name = self._get_collection_name(document_id)

        # Ensure collection exists
        if not self.collection_exists(document_id):
            vector_size = len(embeddings[0]) if embeddings else self.config.vector_size
            self.create_collection(document_id, vector_size=vector_size)

        logger.info(f"Adding {len(chunks)} chunks to collection {collection_name}")

        # Build points with payload
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Build payload from chunk attributes
            payload = {
                "text": chunk.text,
                "section": chunk.section_name,
                "section_type": chunk.section_type,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chunk_type": chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else str(chunk.chunk_type),
                "document_id": chunk.document_id,
                "token_count": chunk.token_count,
                "char_count": chunk.char_count,
                "has_table": chunk.has_table,
            }

            # Add optional fields
            if chunk.table_id:
                payload["table_id"] = chunk.table_id
            if chunk.metadata:
                payload["metadata"] = chunk.metadata
            if chunk.preceding_chunk_id:
                payload["preceding_chunk_id"] = chunk.preceding_chunk_id
            if chunk.following_chunk_id:
                payload["following_chunk_id"] = chunk.following_chunk_id

            # Use chunk_id as the point ID (hash to integer for Qdrant)
            point_id = self._chunk_id_to_point_id(chunk.chunk_id)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert in batches
        total_added = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True,
                )
                total_added += len(batch)
                logger.debug(f"Upserted batch {i // batch_size + 1}: {len(batch)} points")

            except Exception as e:
                logger.error(f"Failed to upsert batch {i // batch_size + 1}: {e}")
                raise RuntimeError(f"Failed to add chunks to collection: {e}") from e

        logger.info(f"Successfully added {total_added} chunks to {collection_name}")
        return total_added

    def _chunk_id_to_point_id(self, chunk_id: str) -> int:
        """Convert string chunk_id to integer point_id for Qdrant.

        Uses a deterministic hash to ensure the same chunk_id always maps
        to the same point_id.

        Args:
            chunk_id: String chunk identifier.

        Returns:
            Integer point ID.
        """
        import hashlib

        # Use first 15 hex chars of SHA-256 (60 bits) to stay within safe integer range
        hash_hex = hashlib.sha256(chunk_id.encode()).hexdigest()[:15]
        return int(hash_hex, 16)

    def search(
        self,
        document_id: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks using vector similarity.

        Args:
            document_id: Document identifier.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters (section, page, chunk_type).
            score_threshold: Optional minimum similarity score threshold.

        Returns:
            List of SearchResult objects sorted by similarity (highest first).

        Raises:
            RuntimeError: If collection doesn't exist or search fails.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

        collection_name = self._get_collection_name(document_id)

        if not self.collection_exists(document_id):
            logger.warning(f"Collection does not exist: {collection_name}")
            return []

        # Build Qdrant filter from the filters dict
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if value is not None:
                    if isinstance(value, (list, tuple)):
                        # Handle range filters (e.g., page_start: [10, 50])
                        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
                            conditions.append(
                                FieldCondition(
                                    key=key,
                                    range=Range(gte=value[0], lte=value[1]),
                                )
                            )
                        else:
                            # Match any value in list
                            for v in value:
                                conditions.append(
                                    FieldCondition(key=key, match=MatchValue(value=v))
                                )
                    else:
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )

            if conditions:
                qdrant_filter = Filter(must=conditions)

        try:
            logger.debug(
                f"Searching {collection_name} with top_k={top_k}, "
                f"filters={filters}, threshold={score_threshold}"
            )

            # Use query_points (qdrant-client 1.x API)
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )

            # Convert to SearchResult objects
            search_results = []
            for hit in response.points:
                payload = hit.payload or {}
                search_results.append(
                    SearchResult(
                        chunk_id=str(hit.id),
                        text=payload.get("text", ""),
                        score=hit.score,
                        section=payload.get("section", "unknown"),
                        page_start=payload.get("page_start", 0),
                        page_end=payload.get("page_end", 0),
                        chunk_type=payload.get("chunk_type", "narrative"),
                        metadata={
                            "section_type": payload.get("section_type"),
                            "token_count": payload.get("token_count"),
                            "char_count": payload.get("char_count"),
                            "has_table": payload.get("has_table"),
                            "table_id": payload.get("table_id"),
                            "document_id": payload.get("document_id"),
                        },
                    )
                )

            logger.debug(f"Search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Search failed in {collection_name}: {e}")
            raise RuntimeError(f"Vector search failed: {e}") from e

    def get_by_section(
        self,
        document_id: str,
        section_name: str,
        limit: int = 100,
    ) -> list[SearchResult]:
        """Get all chunks from a specific section.

        This is a filter-based retrieval (not similarity search).

        Args:
            document_id: Document identifier.
            section_name: Name of the section to retrieve.
            limit: Maximum number of chunks to return.

        Returns:
            List of SearchResult objects from the specified section.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        collection_name = self._get_collection_name(document_id)

        if not self.collection_exists(document_id):
            logger.warning(f"Collection does not exist: {collection_name}")
            return []

        try:
            # Use scroll to get points by filter (no similarity search)
            section_filter = Filter(
                must=[
                    FieldCondition(key="section", match=MatchValue(value=section_name))
                ]
            )

            results, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=section_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            # Convert to SearchResult (score=1.0 since not a similarity search)
            search_results = []
            for point in results:
                payload = point.payload or {}
                search_results.append(
                    SearchResult(
                        chunk_id=str(point.id),
                        text=payload.get("text", ""),
                        score=1.0,  # Not a similarity search
                        section=payload.get("section", "unknown"),
                        page_start=payload.get("page_start", 0),
                        page_end=payload.get("page_end", 0),
                        chunk_type=payload.get("chunk_type", "narrative"),
                        metadata={
                            "section_type": payload.get("section_type"),
                            "token_count": payload.get("token_count"),
                            "char_count": payload.get("char_count"),
                            "has_table": payload.get("has_table"),
                            "table_id": payload.get("table_id"),
                            "document_id": payload.get("document_id"),
                        },
                    )
                )

            # Sort by page number
            search_results.sort(key=lambda r: (r.page_start, r.page_end))

            logger.debug(
                f"Retrieved {len(search_results)} chunks from section '{section_name}'"
            )
            return search_results

        except Exception as e:
            logger.error(f"Failed to get section {section_name}: {e}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """Delete all data for a document.

        This removes the document's entire collection from storage.

        Args:
            document_id: Document identifier.

        Returns:
            True if deletion succeeded, False otherwise.
        """
        return self.delete_collection(document_id)

    def get_collection_info(self, document_id: str) -> dict[str, Any] | None:
        """Get information about a document's collection.

        Args:
            document_id: Document identifier.

        Returns:
            Dictionary with collection info, or None if collection doesn't exist.
        """
        collection_name = self._get_collection_name(document_id)

        if not self.collection_exists(document_id):
            return None

        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vector_size": info.config.params.vectors.size,
                "distance": str(info.config.params.vectors.distance),
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": str(info.status),
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None

    def list_collections(self) -> list[str]:
        """List all collections in the vector store.

        Returns:
            List of collection names.
        """
        try:
            collections = self.client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def list_documents(self) -> list[str]:
        """List all document IDs that have collections in the store.

        Returns:
            List of document IDs (extracted from collection names).
        """
        collections = self.list_collections()
        prefix = f"{self.config.collection_prefix}_"
        document_ids = []
        for name in collections:
            if name.startswith(prefix):
                doc_id = name[len(prefix):]
                document_ids.append(doc_id)
        return document_ids

    def get_chunk_count(self, document_id: str) -> int:
        """Get the number of chunks in a document's collection.

        Args:
            document_id: Document identifier.

        Returns:
            Number of chunks (points) in the collection, or 0 if not exists.
        """
        info = self.get_collection_info(document_id)
        return info.get("points_count", 0) if info else 0


class SearchUtility:
    """
    High-level search utility with automatic embedding.

    This class provides convenient search methods that automatically
    embed text queries and format results with context.

    Example:
        >>> from rhp_analyzer.storage.embeddings import EmbeddingGenerator
        >>> generator = EmbeddingGenerator()
        >>> search = SearchUtility(vector_store, generator)
        >>> results = search.search_text("doc_123", "revenue growth analysis")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: "EmbeddingGenerator",
    ):
        """Initialize the search utility.

        Args:
            vector_store: VectorStore instance.
            embedding_generator: EmbeddingGenerator for query embedding.
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

    def search_text(
        self,
        document_id: str,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        include_context: bool = False,
    ) -> list[SearchResult]:
        """Search using a text query (auto-embedded).

        Args:
            document_id: Document identifier.
            query: Text query to search for.
            top_k: Maximum number of results.
            filters: Optional metadata filters.
            include_context: If True, include surrounding context in results.

        Returns:
            List of SearchResult objects.
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_single(query)

        # Perform vector search
        results = self.vector_store.search(
            document_id=document_id,
            query_vector=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        logger.info(f"Text search for '{query[:50]}...' returned {len(results)} results")
        return results

    def search_by_section(
        self,
        document_id: str,
        query: str,
        section_name: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search within a specific section.

        Args:
            document_id: Document identifier.
            query: Text query to search for.
            section_name: Section to search within.
            top_k: Maximum number of results.

        Returns:
            List of SearchResult objects from the specified section.
        """
        return self.search_text(
            document_id=document_id,
            query=query,
            top_k=top_k,
            filters={"section": section_name},
        )

    def search_by_page_range(
        self,
        document_id: str,
        query: str,
        start_page: int,
        end_page: int,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search within a specific page range.

        Args:
            document_id: Document identifier.
            query: Text query to search for.
            start_page: Starting page number.
            end_page: Ending page number.
            top_k: Maximum number of results.

        Returns:
            List of SearchResult objects from the specified page range.
        """
        return self.search_text(
            document_id=document_id,
            query=query,
            top_k=top_k,
            filters={"page_start": [start_page, end_page]},
        )

    def format_results_with_citations(
        self,
        results: list[SearchResult],
        max_text_length: int = 500,
    ) -> list[dict[str, Any]]:
        """Format search results with citation information.

        Args:
            results: List of SearchResult objects.
            max_text_length: Maximum text length to include.

        Returns:
            List of formatted result dictionaries with citations.
        """
        formatted = []
        for i, result in enumerate(results, 1):
            text = result.text
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."

            formatted.append({
                "rank": i,
                "score": round(result.score, 4),
                "text": text,
                "citation": f"[{result.section}, Pages {result.page_start}-{result.page_end}]",
                "section": result.section,
                "page_start": result.page_start,
                "page_end": result.page_end,
                "chunk_type": result.chunk_type,
            })

        return formatted


# Convenience function for quick access
def create_vector_store(
    storage_path: str | Path | None = None,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    collection_prefix: str = "rhp",
) -> VectorStore:
    """Create a VectorStore instance with common configuration.

    Args:
        storage_path: Path for Qdrant storage. Uses default if not provided.
        vector_size: Embedding dimension.
        collection_prefix: Prefix for collection names.

    Returns:
        Configured VectorStore instance.
    """
    config = VectorStoreConfig()
    if storage_path:
        config.storage_path = Path(storage_path)
    config.vector_size = vector_size
    config.collection_prefix = collection_prefix

    return VectorStore(config)
