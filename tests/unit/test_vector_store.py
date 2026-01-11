"""
Unit tests for Vector Store module.

Tests cover:
- VectorStoreConfig initialization
- Collection creation and deletion
- Adding chunks with embeddings
- Search with query vectors
- Filtering by section and page range
- Error handling

Reference: milestones.md Subtask 3.3.4
"""

import tempfile
import shutil
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from rhp_analyzer.storage.vector_store import (
    VectorStore,
    VectorStoreConfig,
    SearchResult,
    SearchUtility,
    create_vector_store,
)
from rhp_analyzer.storage.chunker import Chunk, ChunkType


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for Qdrant storage."""
    temp_dir = tempfile.mkdtemp(prefix="qdrant_test_")
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def config(temp_storage_dir):
    """Create a VectorStoreConfig with temporary storage."""
    return VectorStoreConfig(
        storage_path=temp_storage_dir,
        vector_size=384,  # Use smaller dimension for tests
        distance="cosine",
        collection_prefix="test",  # No trailing underscore, implementation adds separator
        on_disk_payload=False,
    )


@pytest.fixture
def vector_store(config):
    """Create a VectorStore instance for testing."""
    return VectorStore(config)


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            chunk_id="chunk_001",
            document_id="doc_test",
            text="This is the first test chunk about risk factors.",
            section_name="Risk Factors",
            section_type="risk_factors",
            page_start=1,
            page_end=2,
            chunk_type=ChunkType.NARRATIVE,
            has_table=False,
            table_id=None,
            metadata={"word_count": 10, "position": 0},
            embedding=None,
        ),
        Chunk(
            chunk_id="chunk_002",
            document_id="doc_test",
            text="Financial performance analysis and key metrics.",
            section_name="Financial Information",
            section_type="financial",
            page_start=10,
            page_end=12,
            chunk_type=ChunkType.TABLE,
            has_table=True,
            table_id="table_001",
            metadata={"word_count": 7, "position": 1},
            embedding=None,
        ),
        Chunk(
            chunk_id="chunk_003",
            document_id="doc_test",
            text="Business overview and company operations description.",
            section_name="Business Overview",
            section_type="business",
            page_start=20,
            page_end=25,
            chunk_type=ChunkType.NARRATIVE,
            has_table=False,
            table_id=None,
            metadata={"word_count": 7, "position": 2},
            embedding=None,
        ),
    ]


@pytest.fixture
def sample_embeddings() -> List[List[float]]:
    """Create sample embeddings (384 dimensions for tests)."""
    import random
    random.seed(42)  # Reproducible
    return [
        [random.uniform(-1, 1) for _ in range(384)]
        for _ in range(3)
    ]


@pytest.fixture
def query_embedding() -> List[float]:
    """Create a sample query embedding."""
    import random
    random.seed(123)
    return [random.uniform(-1, 1) for _ in range(384)]


# =============================================================================
# VectorStoreConfig Tests
# =============================================================================

class TestVectorStoreConfig:
    """Tests for VectorStoreConfig dataclass."""
    
    def test_default_config(self, temp_storage_dir):
        """Test default configuration values."""
        config = VectorStoreConfig(storage_path=temp_storage_dir)
        
        # storage_path is converted to Path object
        assert config.storage_path == Path(temp_storage_dir)
        assert config.vector_size == 768
        assert config.distance.upper() == "COSINE"
        assert config.collection_prefix == "rhp"
        assert config.on_disk_payload is True
    
    def test_custom_config(self, temp_storage_dir):
        """Test custom configuration values."""
        config = VectorStoreConfig(
            storage_path=temp_storage_dir,
            vector_size=1024,
            distance="euclid",
            collection_prefix="custom_",
            on_disk_payload=False,
        )
        
        assert config.vector_size == 1024
        assert config.distance == "euclid"
        assert config.collection_prefix == "custom_"
        assert config.on_disk_payload is False
    
    def test_config_with_path_object(self, temp_storage_dir):
        """Test configuration with Path object."""
        config = VectorStoreConfig(storage_path=Path(temp_storage_dir))
        assert config.storage_path == Path(temp_storage_dir)


# =============================================================================
# VectorStore Initialization Tests
# =============================================================================

class TestVectorStoreInitialization:
    """Tests for VectorStore initialization."""
    
    def test_vector_store_creation(self, config):
        """Test VectorStore can be created with config."""
        store = VectorStore(config)
        
        assert store.config == config
        assert store._client is None  # Lazy initialization
    
    def test_vector_store_creates_storage_dir(self, temp_storage_dir):
        """Test that storage directory is created if it doesn't exist."""
        non_existent_path = Path(temp_storage_dir) / "new_subdir" / "qdrant"
        config = VectorStoreConfig(storage_path=str(non_existent_path))
        store = VectorStore(config)
        
        # Access client to trigger initialization
        _ = store.client
        
        assert non_existent_path.exists()
    
    def test_client_lazy_initialization(self, vector_store):
        """Test that Qdrant client is lazily initialized."""
        # Before accessing client
        assert vector_store._client is None
        
        # Access client triggers initialization
        client = vector_store.client
        assert client is not None
        assert vector_store._client is not None


# =============================================================================
# Collection Management Tests
# =============================================================================

class TestCollectionManagement:
    """Tests for collection creation and deletion."""
    
    def test_create_collection(self, vector_store):
        """Test creating a new collection."""
        result = vector_store.create_collection("test_doc_001")
        
        assert result is True
        
        # Verify collection exists
        collections = vector_store.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        # Implementation adds underscore separator: {prefix}_{document_id}
        expected_name = f"{vector_store.config.collection_prefix}_test_doc_001"
        assert expected_name in collection_names
    
    def test_create_collection_custom_vector_size(self, vector_store):
        """Test creating collection with custom vector size."""
        result = vector_store.create_collection("test_doc_002", vector_size=256)
        
        assert result is True
        
        # Verify collection config
        # Implementation adds underscore separator: {prefix}_{document_id}
        collection_name = f"{vector_store.config.collection_prefix}_test_doc_002"
        collection_info = vector_store.client.get_collection(collection_name)
        assert collection_info.config.params.vectors.size == 256
    
    def test_create_collection_already_exists(self, vector_store):
        """Test creating collection that already exists (without recreate)."""
        vector_store.create_collection("test_doc_003")
        
        # Creating again should return False (collection exists)
        result = vector_store.create_collection("test_doc_003", recreate=False)
        assert result is False
    
    def test_create_collection_recreate(self, vector_store):
        """Test recreating an existing collection."""
        vector_store.create_collection("test_doc_004")
        
        # Recreate should return True
        result = vector_store.create_collection("test_doc_004", recreate=True)
        assert result is True
    
    def test_delete_document(self, vector_store):
        """Test deleting a document collection."""
        # Create first
        vector_store.create_collection("test_doc_005")
        
        # Delete
        result = vector_store.delete_document("test_doc_005")
        assert result is True
        
        # Verify deleted
        collections = vector_store.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        expected_name = f"{vector_store.config.collection_prefix}_test_doc_005"
        assert expected_name not in collection_names
    
    def test_delete_nonexistent_document(self, vector_store):
        """Test deleting a document that doesn't exist."""
        result = vector_store.delete_document("nonexistent_doc")
        assert result is False
    
    def test_collection_exists(self, vector_store):
        """Test checking if collection exists."""
        # Before creation
        assert vector_store.collection_exists("test_doc_006") is False
        
        # After creation
        vector_store.create_collection("test_doc_006")
        assert vector_store.collection_exists("test_doc_006") is True
    
    def test_list_documents(self, vector_store):
        """Test listing all document collections."""
        # Create multiple collections
        vector_store.create_collection("doc_a")
        vector_store.create_collection("doc_b")
        vector_store.create_collection("doc_c")
        
        documents = vector_store.list_documents()
        
        assert "doc_a" in documents
        assert "doc_b" in documents
        assert "doc_c" in documents


# =============================================================================
# Add Chunks Tests
# =============================================================================

class TestAddChunks:
    """Tests for adding chunks to collections."""
    
    def test_add_chunks_basic(self, vector_store, sample_chunks, sample_embeddings):
        """Test adding chunks with embeddings."""
        document_id = "add_test_001"
        
        result = vector_store.add_chunks(
            document_id=document_id,
            chunks=sample_chunks,
            embeddings=sample_embeddings,
        )
        
        # add_chunks returns the count of added chunks
        assert result == 3
        
        # Verify points were added
        collection_name = f"{vector_store.config.collection_prefix}_{document_id}"
        count = vector_store.client.count(collection_name)
        assert count.count == 3
    
    def test_add_chunks_mismatched_lengths(self, vector_store, sample_chunks):
        """Test error when chunks and embeddings have different lengths."""
        short_embeddings = [[0.1] * 384, [0.2] * 384]  # Only 2 embeddings
        
        with pytest.raises(ValueError, match="mismatch"):
            vector_store.add_chunks(
                document_id="add_test_002",
                chunks=sample_chunks,
                embeddings=short_embeddings,
            )
    
    def test_add_chunks_empty_list(self, vector_store):
        """Test adding empty chunk list."""
        result = vector_store.add_chunks(
            document_id="add_test_003",
            chunks=[],
            embeddings=[],
        )
        
        assert result == 0  # Returns 0 for empty list
    
    def test_add_chunks_creates_collection(self, vector_store, sample_chunks, sample_embeddings):
        """Test that add_chunks creates collection if it doesn't exist."""
        document_id = "add_test_004"
        
        # Collection shouldn't exist yet
        assert vector_store.collection_exists(document_id) is False
        
        vector_store.add_chunks(
            document_id=document_id,
            chunks=sample_chunks,
            embeddings=sample_embeddings,
        )
        
        # Now it should exist
        assert vector_store.collection_exists(document_id) is True
    
    def test_add_chunks_batch_processing(self, vector_store):
        """Test adding many chunks in batches."""
        import random
        random.seed(42)
        
        # Create 150 chunks (should use multiple batches with default batch_size=100)
        chunks = [
            Chunk(
                chunk_id=f"batch_chunk_{i:03d}",
                document_id="batch_test",
                text=f"Test chunk number {i}",
                section_name="Test Section",
                section_type="test",
                page_start=i,
                page_end=i + 1,
                chunk_type=ChunkType.NARRATIVE,
                has_table=False,
                table_id=None,
                metadata={"index": i},
                embedding=None,
            )
            for i in range(150)
        ]
        embeddings = [[random.uniform(-1, 1) for _ in range(384)] for _ in range(150)]
        
        result = vector_store.add_chunks(
            document_id="batch_test",
            chunks=chunks,
            embeddings=embeddings,
            batch_size=50,  # Use smaller batch for testing
        )
        
        # Returns count of added chunks
        assert result == 150
        
        # Verify all added
        collection_name = f"{vector_store.config.collection_prefix}_batch_test"
        count = vector_store.client.count(collection_name)
        assert count.count == 150


# =============================================================================
# Search Tests
# =============================================================================

class TestSearch:
    """Tests for vector search functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_search_data(self, vector_store, sample_chunks, sample_embeddings):
        """Set up test data before each search test."""
        self.document_id = "search_test_doc"
        vector_store.add_chunks(
            document_id=self.document_id,
            chunks=sample_chunks,
            embeddings=sample_embeddings,
        )
        self.vector_store = vector_store
        self.sample_embeddings = sample_embeddings
    
    def test_search_basic(self, query_embedding):
        """Test basic vector search."""
        results = self.vector_store.search(
            document_id=self.document_id,
            query_vector=query_embedding,
            top_k=3,
        )
        
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_search_returns_correct_fields(self, query_embedding):
        """Test that search results have all required fields."""
        results = self.vector_store.search(
            document_id=self.document_id,
            query_vector=query_embedding,
            top_k=1,
        )
        
        if len(results) > 0:
            result = results[0]
            assert result.chunk_id is not None
            assert result.text is not None
            assert result.score is not None
            assert result.section is not None
            assert result.page_start is not None
            assert result.page_end is not None
    
    def test_search_with_score_threshold(self, query_embedding):
        """Test search with minimum score threshold."""
        results = self.vector_store.search(
            document_id=self.document_id,
            query_vector=query_embedding,
            top_k=10,
            score_threshold=0.99,  # Very high threshold
        )
        
        # Should return fewer or no results with high threshold
        for r in results:
            assert r.score >= 0.99
    
    def test_search_with_section_filter(self):
        """Test search with section filter."""
        # Use first embedding to find first chunk
        results = self.vector_store.search(
            document_id=self.document_id,
            query_vector=self.sample_embeddings[0],
            top_k=10,
            filters={"section": "Risk Factors"},
        )
        
        for r in results:
            assert r.section == "Risk Factors"
    
    def test_search_with_page_filter(self):
        """Test search with page range filter."""
        results = self.vector_store.search(
            document_id=self.document_id,
            query_vector=self.sample_embeddings[1],
            top_k=10,
            filters={"page_start": 10, "page_end": 12},
        )
        
        # Should find the financial chunk
        if len(results) > 0:
            assert any(r.section == "Financial Information" for r in results)
    
    def test_search_with_chunk_type_filter(self):
        """Test search with chunk type filter."""
        results = self.vector_store.search(
            document_id=self.document_id,
            query_vector=self.sample_embeddings[1],
            top_k=10,
            filters={"chunk_type": "table"},
        )
        
        for r in results:
            assert r.chunk_type == "table"
    
    def test_search_nonexistent_collection(self, query_embedding):
        """Test searching in non-existent collection."""
        results = self.vector_store.search(
            document_id="nonexistent_doc",
            query_vector=query_embedding,
            top_k=5,
        )
        
        assert results == []
    
    def test_search_similarity_ordering(self):
        """Test that results are ordered by similarity score."""
        # Use one of the sample embeddings to guarantee a match
        results = self.vector_store.search(
            document_id=self.document_id,
            query_vector=self.sample_embeddings[0],
            top_k=3,
        )
        
        # Results should be in descending score order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# Get By Section Tests
# =============================================================================

class TestGetBySection:
    """Tests for section-based retrieval."""
    
    @pytest.fixture(autouse=True)
    def setup_section_data(self, vector_store, sample_chunks, sample_embeddings):
        """Set up test data before each section test."""
        self.document_id = "section_test_doc"
        vector_store.add_chunks(
            document_id=self.document_id,
            chunks=sample_chunks,
            embeddings=sample_embeddings,
        )
        self.vector_store = vector_store
    
    def test_get_by_section_basic(self):
        """Test getting chunks by section name."""
        results = self.vector_store.get_by_section(
            document_id=self.document_id,
            section_name="Risk Factors",
        )
        
        assert len(results) >= 1
        assert all(r.section == "Risk Factors" for r in results)
    
    def test_get_by_section_with_limit(self):
        """Test section retrieval with limit."""
        results = self.vector_store.get_by_section(
            document_id=self.document_id,
            section_name="Risk Factors",
            limit=1,
        )
        
        assert len(results) <= 1
    
    def test_get_by_section_no_match(self):
        """Test section retrieval when section doesn't exist."""
        results = self.vector_store.get_by_section(
            document_id=self.document_id,
            section_name="Nonexistent Section",
        )
        
        assert results == []
    
    def test_get_by_section_nonexistent_document(self):
        """Test section retrieval for non-existent document."""
        results = self.vector_store.get_by_section(
            document_id="nonexistent_doc",
            section_name="Risk Factors",
        )
        
        assert results == []


# =============================================================================
# Search Utility Tests
# =============================================================================

class TestSearchUtility:
    """Tests for SearchUtility helper class."""
    
    @pytest.fixture
    def mock_embedding_generator(self):
        """Create a mock embedding generator."""
        mock = MagicMock()
        mock.generate_single.return_value = [0.1] * 384
        return mock
    
    @pytest.fixture
    def search_utility(self, vector_store, mock_embedding_generator):
        """Create a SearchUtility with mock embedding generator."""
        return SearchUtility(
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
        )
    
    def test_search_text(self, search_utility, vector_store, sample_chunks, sample_embeddings):
        """Test text-based search."""
        document_id = "utility_test_doc"
        vector_store.add_chunks(
            document_id=document_id,
            chunks=sample_chunks,
            embeddings=sample_embeddings,
        )
        
        results = search_utility.search_text(
            document_id=document_id,
            query="risk factors analysis",
            top_k=3,
        )
        
        assert isinstance(results, list)
        # Embedding generator should have been called
        search_utility.embedding_generator.generate_single.assert_called_once()
    
    def test_search_by_section(self, search_utility, vector_store, sample_chunks, sample_embeddings):
        """Test section-filtered text search."""
        document_id = "utility_section_test"
        vector_store.add_chunks(
            document_id=document_id,
            chunks=sample_chunks,
            embeddings=sample_embeddings,
        )
        
        results = search_utility.search_by_section(
            document_id=document_id,
            query="financial metrics",
            section_name="Financial Information",
            top_k=5,
        )
        
        for r in results:
            assert r.section == "Financial Information"
    
    def test_search_by_page_range(self, search_utility, vector_store, sample_chunks, sample_embeddings):
        """Test page range filtered search."""
        document_id = "utility_page_test"
        vector_store.add_chunks(
            document_id=document_id,
            chunks=sample_chunks,
            embeddings=sample_embeddings,
        )
        
        results = search_utility.search_by_page_range(
            document_id=document_id,
            query="business operations",
            start_page=15,
            end_page=30,
            top_k=5,
        )
        
        # Should not error, results should be in page range if any
        for r in results:
            # The filter checks page_start >= start_page
            assert r.page_start >= 15 or r.page_end <= 30
    
    def test_format_results_with_citations(self, search_utility):
        """Test formatting search results with citations."""
        results = [
            SearchResult(
                chunk_id="test_001",
                text="Sample text content for testing.",
                score=0.95,
                section="Risk Factors",
                page_start=10,
                page_end=12,
                chunk_type="narrative",
                metadata={},
            ),
            SearchResult(
                chunk_id="test_002",
                text="Another sample text.",
                score=0.85,
                section="Business",
                page_start=20,
                page_end=20,
                chunk_type="narrative",
                metadata={},
            ),
        ]
        
        formatted = search_utility.format_results_with_citations(results)
        
        # format_results_with_citations returns a list of dicts
        assert isinstance(formatted, list)
        assert len(formatted) == 2
        assert formatted[0]["citation"] == "[Risk Factors, Pages 10-12]"
        assert "Sample text content" in formatted[0]["text"]
        assert formatted[1]["citation"] == "[Business, Pages 20-20]"


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunction:
    """Tests for create_vector_store convenience function."""
    
    def test_create_vector_store_default(self, temp_storage_dir):
        """Test creating vector store with defaults."""
        store = create_vector_store(storage_path=temp_storage_dir)
        
        assert isinstance(store, VectorStore)
        # storage_path is converted to Path
        assert store.config.storage_path == Path(temp_storage_dir)
        assert store.config.vector_size == 768
    
    def test_create_vector_store_custom(self, temp_storage_dir):
        """Test creating vector store with custom settings."""
        store = create_vector_store(
            storage_path=temp_storage_dir,
            vector_size=512,
            collection_prefix="custom",
        )
        
        assert store.config.vector_size == 512
        assert store.config.collection_prefix == "custom"


# =============================================================================
# Integration Tests
# =============================================================================

class TestVectorStoreIntegration:
    """Integration tests for full workflow."""
    
    def test_full_workflow(self, vector_store, sample_chunks, sample_embeddings, query_embedding):
        """Test complete add → search → delete workflow."""
        document_id = "integration_test_doc"
        
        # 1. Add chunks
        add_result = vector_store.add_chunks(
            document_id=document_id,
            chunks=sample_chunks,
            embeddings=sample_embeddings,
        )
        # add_chunks returns the count of added chunks
        assert add_result == 3
        
        # 2. Search
        search_results = vector_store.search(
            document_id=document_id,
            query_vector=query_embedding,
            top_k=3,
        )
        assert len(search_results) == 3
        
        # 3. Get by section
        section_results = vector_store.get_by_section(
            document_id=document_id,
            section_name="Business Overview",
        )
        assert len(section_results) >= 1
        
        # 4. Delete
        delete_result = vector_store.delete_document(document_id)
        assert delete_result is True
        
        # 5. Verify deleted
        assert vector_store.collection_exists(document_id) is False
    
    def test_multiple_documents(self, vector_store, sample_chunks, sample_embeddings):
        """Test working with multiple documents."""
        # Add to doc1
        chunks_1 = [c for c in sample_chunks]
        for c in chunks_1:
            c.document_id = "multi_doc_1"
        
        vector_store.add_chunks(
            document_id="multi_doc_1",
            chunks=chunks_1,
            embeddings=sample_embeddings,
        )
        
        # Add to doc2
        chunks_2 = [c for c in sample_chunks]
        for c in chunks_2:
            c.document_id = "multi_doc_2"
        
        vector_store.add_chunks(
            document_id="multi_doc_2",
            chunks=chunks_2,
            embeddings=sample_embeddings,
        )
        
        # Verify both exist
        docs = vector_store.list_documents()
        assert "multi_doc_1" in docs
        assert "multi_doc_2" in docs
        
        # Delete one
        vector_store.delete_document("multi_doc_1")
        
        # Verify only one remains
        docs = vector_store.list_documents()
        assert "multi_doc_1" not in docs
        assert "multi_doc_2" in docs


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling scenarios."""
    
    def test_invalid_vector_size_search(self, vector_store, sample_chunks, sample_embeddings, query_embedding):
        """Test error when query dimension doesn't match collection."""
        # First create a collection with 384-dim vectors
        document_id = "dim_test"
        vector_store.add_chunks(
            document_id=document_id,
            chunks=sample_chunks,
            embeddings=sample_embeddings,  # 384-dim
        )
        
        # Now try to search with wrong dimension (should raise error)
        wrong_dim_query = [0.1] * 256  # Wrong dimension
        
        with pytest.raises(Exception):
            vector_store.search(
                document_id=document_id,
                query_vector=wrong_dim_query,
                top_k=5,
            )
    
    def test_chunk_id_collision_handling(self, vector_store, sample_embeddings):
        """Test that duplicate chunk IDs are handled."""
        # Create chunks with same IDs
        chunks = [
            Chunk(
                chunk_id="same_id",  # Same ID
                document_id="collision_test",
                text="First chunk",
                section_name="Test",
                section_type="test",
                page_start=1,
                page_end=1,
                chunk_type=ChunkType.NARRATIVE,
                has_table=False,
                table_id=None,
                metadata={},
                embedding=None,
            ),
            Chunk(
                chunk_id="same_id",  # Same ID - will have same point_id
                document_id="collision_test",
                text="Second chunk",
                section_name="Test",
                section_type="test",
                page_start=2,
                page_end=2,
                chunk_type=ChunkType.NARRATIVE,
                has_table=False,
                table_id=None,
                metadata={},
                embedding=None,
            ),
        ]
        
        # This should upsert (update) rather than fail
        result = vector_store.add_chunks(
            document_id="collision_test",
            chunks=chunks,
            embeddings=sample_embeddings[:2],
        )
        
        # add_chunks returns count of chunks processed
        assert result == 2
        
        # Should only have 1 point (second overwrites first due to same chunk_id)
        collection_name = f"{vector_store.config.collection_prefix}_collision_test"
        count = vector_store.client.count(collection_name)
        assert count.count == 1


# =============================================================================
# Point ID Conversion Tests
# =============================================================================

class TestPointIdConversion:
    """Tests for chunk_id to point_id conversion."""
    
    def test_consistent_point_id(self, vector_store):
        """Test that same chunk_id always produces same point_id."""
        chunk_id = "test_chunk_abc123"
        
        point_id_1 = vector_store._chunk_id_to_point_id(chunk_id)
        point_id_2 = vector_store._chunk_id_to_point_id(chunk_id)
        
        assert point_id_1 == point_id_2
    
    def test_different_chunk_ids_different_point_ids(self, vector_store):
        """Test that different chunk_ids produce different point_ids."""
        point_id_1 = vector_store._chunk_id_to_point_id("chunk_001")
        point_id_2 = vector_store._chunk_id_to_point_id("chunk_002")
        
        assert point_id_1 != point_id_2
    
    def test_point_id_is_integer(self, vector_store):
        """Test that point_id is a positive integer."""
        point_id = vector_store._chunk_id_to_point_id("any_chunk_id")
        
        assert isinstance(point_id, int)
        assert point_id > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
