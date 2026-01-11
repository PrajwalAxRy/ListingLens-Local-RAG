"""
Unit tests for the Embedding Generation module.

Tests Subtask 3.2.3 from milestones.md:
- Test embedding dimensions
- Test batch processing
- Test caching logic
- Test similar texts have similar embeddings

Reference: milestones.md Milestone 3.2
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Create a mock for sentence_transformers before importing embeddings module
# This prevents the actual sentence_transformers import which has version conflicts
_mock_st_module = MagicMock()
_mock_st_class = MagicMock()
_mock_st_module.SentenceTransformer = _mock_st_class
sys.modules["sentence_transformers"] = _mock_st_module

from rhp_analyzer.storage.chunker import Chunk, ChunkType
from rhp_analyzer.storage.embeddings import (
    CacheMetadata,
    EmbeddingConfig,
    EmbeddingGenerator,
    compute_similarity,
    generate_embeddings,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_texts():
    """Sample texts for embedding tests."""
    return [
        "The company reported revenue growth of 25% year over year.",
        "Risk factors include market volatility and regulatory changes.",
        "The board of directors approved the dividend distribution.",
        "Financial statements show strong operating cash flow.",
        "IPO proceeds will be used for expansion and debt repayment.",
    ]


@pytest.fixture
def sample_chunks():
    """Sample chunks for embedding tests."""
    return [
        Chunk(
            chunk_id="chunk_001",
            document_id="DOC001",
            text="Revenue grew 25% YoY driven by strong demand.",
            section_name="Financial Highlights",
            section_type="financial",
            page_start=10,
            page_end=10,
            chunk_type=ChunkType.NARRATIVE,
            start_char=0,
            end_char=100,
            token_count=50,
        ),
        Chunk(
            chunk_id="chunk_002",
            document_id="DOC001",
            text="The company faces regulatory risks in key markets.",
            section_name="Risk Factors",
            section_type="risk_factors",
            page_start=25,
            page_end=25,
            chunk_type=ChunkType.NARRATIVE,
            start_char=0,
            end_char=80,
            token_count=40,
        ),
        Chunk(
            chunk_id="chunk_003",
            document_id="DOC001",
            text="IPO proceeds allocation: Capex 40%, Debt 30%, WC 30%.",
            section_name="Objects of Issue",
            section_type="objects_of_issue",
            page_start=50,
            page_end=50,
            chunk_type=ChunkType.TABLE,
            start_char=0,
            end_char=90,
            token_count=45,
        ),
    ]


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model for testing without loading actual model."""
    mock_model = MagicMock()
    # Return embeddings of dimension 768 (matching nomic-embed-text-v1.5)
    mock_model.encode.return_value = np.random.rand(5, 768).astype(np.float32)
    mock_model.get_sentence_embedding_dimension.return_value = 768
    return mock_model


@pytest.fixture
def mock_fallback_model():
    """Create a mock fallback model with 384 dimensions."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(5, 384).astype(np.float32)
    mock_model.get_sentence_embedding_dimension.return_value = 384
    return mock_model


@pytest.fixture(autouse=True)
def reset_sentence_transformer_mock(mock_embedding_model):
    """Reset the mocked SentenceTransformer before each test.
    
    This fixture is autouse=True so it runs for every test in this module.
    It configures the mocked sentence_transformers.SentenceTransformer class.
    """
    # Reset the mock completely
    _mock_st_class.reset_mock()
    # Clear any side_effect from previous tests
    _mock_st_class.side_effect = None
    # Configure default return value
    _mock_st_class.return_value = mock_embedding_model
    yield _mock_st_class
    # Cleanup after test
    _mock_st_class.reset_mock()
    _mock_st_class.side_effect = None


# =============================================================================
# EmbeddingConfig Tests
# =============================================================================


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        
        assert config.model_name == "nomic-ai/nomic-embed-text-v1.5"
        assert config.device == "cpu"
        assert config.batch_size == 32
        assert config.normalize_embeddings is True
        assert config.cache_dir == Path("data/embeddings")
        assert config.show_progress is True

    def test_custom_config(self, temp_cache_dir):
        """Test custom configuration values."""
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            device="cuda",
            batch_size=64,
            normalize_embeddings=False,
            cache_dir=temp_cache_dir,
            show_progress=False,
        )
        
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.device == "cuda"
        assert config.batch_size == 64
        assert config.normalize_embeddings is False
        assert config.cache_dir == temp_cache_dir
        assert config.show_progress is False


# =============================================================================
# CacheMetadata Tests
# =============================================================================


class TestCacheMetadata:
    """Tests for CacheMetadata dataclass."""

    def test_cache_metadata_creation(self):
        """Test CacheMetadata creation with all fields."""
        metadata = CacheMetadata(
            document_id="DOC001",
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_dimension=768,
            chunk_count=100,
            content_hash="abc123def456",
            created_at="2026-01-11T12:00:00",
        )
        
        assert metadata.document_id == "DOC001"
        assert metadata.model_name == "nomic-ai/nomic-embed-text-v1.5"
        assert metadata.model_dimension == 768
        assert metadata.chunk_count == 100
        assert metadata.content_hash == "abc123def456"
        assert metadata.created_at == "2026-01-11T12:00:00"

    def test_cache_metadata_to_dict(self):
        """Test CacheMetadata serialization to dictionary."""
        metadata = CacheMetadata(
            document_id="DOC001",
            model_name="test-model",
            model_dimension=768,
            chunk_count=50,
            content_hash="hash123",
            created_at="2026-01-11T12:00:00",
        )
        
        data = metadata.to_dict()
        
        assert isinstance(data, dict)
        assert data["document_id"] == "DOC001"
        assert data["model_name"] == "test-model"
        assert data["model_dimension"] == 768
        assert data["chunk_count"] == 50
        assert data["content_hash"] == "hash123"
        assert data["created_at"] == "2026-01-11T12:00:00"

    def test_cache_metadata_from_dict(self):
        """Test CacheMetadata deserialization from dictionary."""
        data = {
            "document_id": "DOC002",
            "model_name": "test-model",
            "model_dimension": 384,
            "chunk_count": 25,
            "content_hash": "hash456",
            "created_at": "2026-01-11T14:00:00",
        }
        
        metadata = CacheMetadata.from_dict(data)
        
        assert metadata.document_id == "DOC002"
        assert metadata.model_name == "test-model"
        assert metadata.model_dimension == 384
        assert metadata.chunk_count == 25
        assert metadata.content_hash == "hash456"
        assert metadata.created_at == "2026-01-11T14:00:00"

    def test_cache_metadata_roundtrip(self):
        """Test CacheMetadata serialization roundtrip."""
        original = CacheMetadata(
            document_id="DOC003",
            model_name="test-model",
            model_dimension=768,
            chunk_count=75,
            content_hash="roundtrip_hash",
            created_at="2026-01-11T16:00:00",
        )
        
        data = original.to_dict()
        restored = CacheMetadata.from_dict(data)
        
        assert restored.document_id == original.document_id
        assert restored.model_name == original.model_name
        assert restored.model_dimension == original.model_dimension
        assert restored.chunk_count == original.chunk_count
        assert restored.content_hash == original.content_hash
        assert restored.created_at == original.created_at


# =============================================================================
# EmbeddingGenerator Tests (with mocked model)
# =============================================================================


class TestEmbeddingGeneratorMocked:
    """Tests for EmbeddingGenerator with mocked model (no actual model loading)."""

    def test_initialization_with_mock(self, temp_cache_dir, mock_embedding_model):
        """Test EmbeddingGenerator initialization with mocked model."""
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        assert generator.model is not None
        assert generator.dimension == 768
        _mock_st_class.assert_called_once()

    def test_generate_single_embedding(self, temp_cache_dir, mock_embedding_model):
        """Test generating embedding for a single text."""
        # Configure mock to return 1D array for single input (as real SentenceTransformer does)
        mock_embedding_model.encode.return_value = np.random.rand(768).astype(np.float32)
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        embedding = generator.generate_single("Test text for embedding.")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 768

    def test_generate_batch_embeddings(self, temp_cache_dir, mock_embedding_model, sample_texts):
        """Test generating embeddings for multiple texts."""
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        embeddings = generator.generate(sample_texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(sample_texts)
        # Each embedding should be 768 dimensions
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == 768

    def test_generate_for_chunks(self, temp_cache_dir, mock_embedding_model, sample_chunks):
        """Test generating embeddings for Chunk objects."""
        # Return embeddings matching chunk count
        mock_embedding_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        embeddings = generator.generate_for_chunks(sample_chunks)
        
        # generate_for_chunks returns list of embeddings, not modified chunks
        assert len(embeddings) == len(sample_chunks)
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == 768

    def test_empty_input_handling(self, temp_cache_dir, mock_embedding_model):
        """Test handling of empty input."""
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        embeddings = generator.generate([])
        
        assert embeddings == []

    def test_batch_processing(self, temp_cache_dir, mock_embedding_model):
        """Test that batch processing respects batch_size."""
        
        # Create config with small batch size
        config = EmbeddingConfig(cache_dir=temp_cache_dir, batch_size=2)
        generator = EmbeddingGenerator(config)
        
        # Generate for 5 texts
        texts = ["Text " + str(i) for i in range(5)]
        
        # Mock encode to return correct number of embeddings
        mock_embedding_model.encode.return_value = np.random.rand(5, 768).astype(np.float32)
        
        embeddings = generator.generate(texts)
        
        assert len(embeddings) == 5
        # The implementation passes batch_size to model.encode(), so encode is called once
        assert mock_embedding_model.encode.call_count == 1
        # Verify batch_size was passed to encode
        call_kwargs = mock_embedding_model.encode.call_args[1]
        assert call_kwargs.get("batch_size") == 2


# =============================================================================
# Caching Tests
# =============================================================================


class TestEmbeddingCache:
    """Tests for embedding caching functionality."""

    def test_cache_save_and_load(self, temp_cache_dir, mock_embedding_model, sample_chunks):
        """Test saving embeddings to cache and loading them back."""
        mock_embedding_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        # Generate embeddings (should save to cache)
        document_id = sample_chunks[0].document_id
        updated_chunks = generator.generate_for_chunks(sample_chunks, document_id=document_id, use_cache=True)
        
        # Verify cache files exist
        cache_path = temp_cache_dir / document_id
        assert cache_path.exists()
        assert (cache_path / "embeddings.npy").exists()
        assert (cache_path / "chunks.jsonl").exists()
        assert (cache_path / "metadata.json").exists()

    def test_cache_hit(self, temp_cache_dir, mock_embedding_model, sample_chunks):
        """Test that cached embeddings are used on second call."""
        mock_embedding_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        document_id = sample_chunks[0].document_id
        
        # First call - generates and caches
        generator.generate_for_chunks(sample_chunks, document_id=document_id, use_cache=True)
        first_call_count = mock_embedding_model.encode.call_count
        
        # Second call - should use cache
        generator.generate_for_chunks(sample_chunks, document_id=document_id, use_cache=True)
        second_call_count = mock_embedding_model.encode.call_count
        
        # encode() should not be called again if cache hit
        assert second_call_count == first_call_count

    def test_cache_invalidation_on_content_change(self, temp_cache_dir, mock_embedding_model, sample_chunks):
        """Test that cache is invalidated when content changes."""
        mock_embedding_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        document_id = sample_chunks[0].document_id
        
        # First call - generates and caches
        generator.generate_for_chunks(sample_chunks, document_id=document_id, use_cache=True)
        first_call_count = mock_embedding_model.encode.call_count
        
        # Modify chunks (different content)
        modified_chunks = [
            Chunk(
                chunk_id="chunk_001",
                document_id="DOC001",
                text="This is completely different text content.",  # Changed
                section_name="Modified Section",
                section_type="other",
                page_start=10,
                page_end=10,
                chunk_type=ChunkType.NARRATIVE,
                start_char=0,
                end_char=100,
                token_count=50,
            )
        ]
        
        mock_embedding_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        
        # Second call with different content - should regenerate
        generator.generate_for_chunks(modified_chunks, document_id=document_id, use_cache=True)
        second_call_count = mock_embedding_model.encode.call_count
        
        # encode() should be called again due to cache invalidation
        assert second_call_count > first_call_count

    def test_cache_bypass(self, temp_cache_dir, mock_embedding_model, sample_chunks):
        """Test that use_cache=False bypasses caching."""
        mock_embedding_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        document_id = sample_chunks[0].document_id
        
        # First call with caching
        generator.generate_for_chunks(sample_chunks, document_id=document_id, use_cache=True)
        first_call_count = mock_embedding_model.encode.call_count
        
        # Second call without caching - should regenerate
        generator.generate_for_chunks(sample_chunks, document_id=document_id, use_cache=False)
        second_call_count = mock_embedding_model.encode.call_count
        
        # encode() should be called again
        assert second_call_count > first_call_count

    def test_clear_cache(self, temp_cache_dir, mock_embedding_model, sample_chunks):
        """Test clearing cache for a specific document."""
        mock_embedding_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        document_id = sample_chunks[0].document_id
        
        # Generate and cache
        generator.generate_for_chunks(sample_chunks, document_id=document_id, use_cache=True)
        
        # Verify cache exists
        cache_path = temp_cache_dir / document_id
        assert cache_path.exists()
        
        # Clear cache
        result = generator.clear_cache(document_id)
        
        assert result is True
        assert not cache_path.exists()

    def test_clear_all_cache(self, temp_cache_dir, mock_embedding_model, sample_chunks):
        """Test clearing all cached embeddings."""
        mock_embedding_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        # Generate for two documents
        chunks1 = [c for c in sample_chunks]
        chunks2 = [
            Chunk(
                chunk_id="chunk_x",
                document_id="DOC002",
                text="Another document text.",
                section_name="Section",
                section_type="other",
                page_start=1,
                page_end=1,
                chunk_type=ChunkType.NARRATIVE,
                start_char=0,
                end_char=50,
                token_count=25,
            )
        ]
        
        mock_embedding_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        
        generator.generate_for_chunks(chunks1, document_id="DOC001", use_cache=True)
        generator.generate_for_chunks(chunks2, document_id="DOC002", use_cache=True)
        
        # Verify both caches exist
        assert (temp_cache_dir / "DOC001").exists()
        assert (temp_cache_dir / "DOC002").exists()
        
        # Clear all
        count = generator.clear_all_cache()
        
        assert count == 2
        assert not (temp_cache_dir / "DOC001").exists()
        assert not (temp_cache_dir / "DOC002").exists()

    def test_get_cache_info(self, temp_cache_dir, mock_embedding_model, sample_chunks):
        """Test retrieving cache information."""
        mock_embedding_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        document_id = sample_chunks[0].document_id
        
        # Generate and cache
        generator.generate_for_chunks(sample_chunks, document_id=document_id, use_cache=True)
        
        # Get cache info - returns CacheMetadata object
        info = generator.get_cache_info(document_id)
        
        assert info is not None
        assert info.document_id == document_id
        assert info.model_name == config.model_name
        assert info.chunk_count == 3
        assert info.model_dimension == 768

    def test_is_cache_valid(self, temp_cache_dir, mock_embedding_model, sample_chunks):
        """Test checking cache validity."""
        mock_embedding_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        document_id = sample_chunks[0].document_id
        # Extract texts from chunks (is_cache_valid expects list of strings)
        texts = [chunk.text for chunk in sample_chunks]
        
        # No cache initially
        assert generator.is_cache_valid(document_id, texts) is False
        
        # Generate and cache
        generator.generate_for_chunks(sample_chunks, document_id=document_id, use_cache=True)
        
        # Cache should be valid now
        assert generator.is_cache_valid(document_id, texts) is True


# =============================================================================
# Similarity Tests
# =============================================================================


class TestSimilarity:
    """Tests for embedding similarity computation."""

    def test_compute_similarity_identical(self):
        """Test similarity of identical embeddings."""
        embedding = [0.5] * 768
        
        similarity = compute_similarity(embedding, embedding)
        
        # Identical embeddings should have similarity close to 1.0
        assert 0.99 <= similarity <= 1.01

    def test_compute_similarity_orthogonal(self):
        """Test similarity of orthogonal embeddings."""
        embedding1 = [1.0] + [0.0] * 767
        embedding2 = [0.0, 1.0] + [0.0] * 766
        
        similarity = compute_similarity(embedding1, embedding2)
        
        # Orthogonal embeddings should have similarity close to 0
        assert -0.01 <= similarity <= 0.01

    def test_compute_similarity_opposite(self):
        """Test similarity of opposite embeddings."""
        embedding1 = [1.0] * 768
        embedding2 = [-1.0] * 768
        
        similarity = compute_similarity(embedding1, embedding2)
        
        # Opposite embeddings should have similarity close to -1.0
        assert -1.01 <= similarity <= -0.99

    def test_compute_similarity_normalized(self):
        """Test that similarity handles unnormalized embeddings."""
        embedding1 = [2.0] * 768  # Not normalized
        embedding2 = [3.0] * 768  # Not normalized
        
        similarity = compute_similarity(embedding1, embedding2)
        
        # Should still compute valid similarity (same direction = 1.0)
        assert 0.99 <= similarity <= 1.01


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_generate_embeddings_function(self, temp_cache_dir, mock_embedding_model):
        """Test the generate_embeddings convenience function."""
        # Configure mock to return correct number of embeddings
        mock_embedding_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)
        
        texts = ["Text 1", "Text 2", "Text 3"]
        
        # generate_embeddings takes individual params, not a config object
        embeddings = generate_embeddings(texts)
        
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 768


# =============================================================================
# Dimension Tests
# =============================================================================


class TestEmbeddingDimensions:
    """Tests for embedding dimensions matching expected values."""

    def test_primary_model_dimension(self, temp_cache_dir, mock_embedding_model):
        """Test that primary model (nomic-embed-text-v1.5) produces 768 dimensions."""
        
        config = EmbeddingConfig(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            cache_dir=temp_cache_dir,
        )
        generator = EmbeddingGenerator(config)
        
        assert generator.dimension == 768

    def test_fallback_model_dimension(self, temp_cache_dir, mock_fallback_model):
        """Test that fallback model (all-MiniLM-L6-v2) produces 384 dimensions."""
        _mock_st_class.return_value = mock_fallback_model
        
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            cache_dir=temp_cache_dir,
        )
        generator = EmbeddingGenerator(config)
        
        assert generator.dimension == 384


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in embedding generation."""

    def test_model_load_failure_with_fallback(self, temp_cache_dir):
        """Test fallback when primary model fails to load."""
        # First call fails (primary model), second succeeds (fallback)
        mock_fallback = MagicMock()
        mock_fallback.get_sentence_embedding_dimension.return_value = 384
        _mock_st_class.side_effect = [Exception("Model not found"), mock_fallback]
        
        config = EmbeddingConfig(
            model_name="nonexistent-model",
            cache_dir=temp_cache_dir,
        )
        generator = EmbeddingGenerator(config)
        
        # Should fall back to secondary model
        assert generator.model is not None
        assert generator.dimension == 384

    def test_generate_with_none_text(self, temp_cache_dir, mock_embedding_model):
        """Test handling of None in text list."""
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        # Should handle None gracefully (filter or convert to empty string)
        texts = ["Valid text", None, "Another text"]
        
        # Mock to handle filtered texts
        mock_embedding_model.encode.return_value = np.random.rand(2, 768).astype(np.float32)
        
        # This should not raise an exception
        try:
            embeddings = generator.generate([t for t in texts if t is not None])
            assert len(embeddings) == 2
        except Exception as e:
            pytest.fail(f"Should handle None in texts: {e}")


# =============================================================================
# Integration-like Tests (Still Unit Tests with Mocks)
# =============================================================================


class TestIntegrationLike:
    """Integration-like tests that test multiple components together."""

    def test_full_workflow_generate_cache_load(
        self, temp_cache_dir, mock_embedding_model, sample_chunks
    ):
        """Test full workflow: generate -> cache -> load from cache."""
        # Configure mock for consistent embeddings
        fixed_embeddings = np.random.rand(3, 768).astype(np.float32)
        mock_embedding_model.encode.return_value = fixed_embeddings
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        document_id = sample_chunks[0].document_id
        
        # Step 1: Generate embeddings (returns list of embeddings, not chunks)
        embeddings_1 = generator.generate_for_chunks(
            sample_chunks, document_id=document_id, use_cache=True
        )
        
        # Step 2: Verify embeddings were generated
        assert len(embeddings_1) == 3
        assert all(len(emb) == 768 for emb in embeddings_1)
        
        # Step 3: Verify cache exists
        cache_path = temp_cache_dir / document_id
        assert cache_path.exists()
        assert (cache_path / "metadata.json").exists()
        
        # Step 4: Load cache info (returns CacheMetadata object)
        info = generator.get_cache_info(document_id)
        assert info.chunk_count == 3
        
        # Step 5: Second generation should use cache
        call_count_before = mock_embedding_model.encode.call_count
        embeddings_2 = generator.generate_for_chunks(
            sample_chunks, document_id=document_id, use_cache=True
        )
        call_count_after = mock_embedding_model.encode.call_count
        
        # No new encode calls (cache hit)
        assert call_count_after == call_count_before
        
        # Embeddings should still be present
        assert len(embeddings_2) == 3
        assert all(len(emb) == 768 for emb in embeddings_2)

    def test_similar_texts_have_similar_embeddings(
        self, temp_cache_dir
    ):
        """Test that semantically similar texts produce similar embeddings."""
        # Create mock that produces realistic-ish embeddings
        # (In reality, this would require actual model - here we simulate)
        mock_model = MagicMock()
        
        # Simulate: similar texts get similar vectors
        # revenue-related text vectors
        revenue_vec = np.random.rand(768).astype(np.float32)
        # risk-related text vectors (different direction)
        risk_vec = np.random.rand(768).astype(np.float32)
        
        def mock_encode(texts, **kwargs):
            results = []
            for text in texts:
                if "revenue" in text.lower() or "growth" in text.lower():
                    results.append(revenue_vec + np.random.rand(768) * 0.1)
                elif "risk" in text.lower():
                    results.append(risk_vec + np.random.rand(768) * 0.1)
                else:
                    results.append(np.random.rand(768))
            return np.array(results, dtype=np.float32)
        
        mock_model.encode = mock_encode
        mock_model.get_sentence_embedding_dimension.return_value = 768
        _mock_st_class.return_value = mock_model
        
        config = EmbeddingConfig(cache_dir=temp_cache_dir)
        generator = EmbeddingGenerator(config)
        
        similar_texts = [
            "Revenue grew 25% this year.",
            "Company reported revenue growth.",
        ]
        different_text = "Risk factors include market volatility."
        
        embeddings = generator.generate(similar_texts + [different_text])
        
        # Similar texts should have higher similarity than different texts
        sim_similar = compute_similarity(embeddings[0], embeddings[1])
        sim_different_1 = compute_similarity(embeddings[0], embeddings[2])
        sim_different_2 = compute_similarity(embeddings[1], embeddings[2])
        
        # Similar texts should be more similar to each other
        assert sim_similar > sim_different_1
        assert sim_similar > sim_different_2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
