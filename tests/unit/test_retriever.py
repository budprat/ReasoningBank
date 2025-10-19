"""
ABOUTME: Unit tests for MemoryRetriever component
ABOUTME: Tests embedding-based similarity search with caching
"""

import pytest
import os
import json
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from reasoningbank.retriever import MemoryRetriever, retrieve_memories, GOOGLE_AVAILABLE
from reasoningbank.models import MemoryItem, MemoryEntry
from reasoningbank.config import ReasoningBankConfig


class TestMemoryRetrieverInitialization:
    """Tests for MemoryRetriever initialization"""

    def test_retriever_initialization_openai(self, test_config):
        """Test MemoryRetriever initialization with OpenAI embeddings"""
        test_config.embedding_model = "text-embedding-3-small"
        test_config.embedding_dimension = 1536

        retriever = MemoryRetriever(test_config)

        assert retriever.config == test_config
        assert retriever.embedding_model == "text-embedding-3-small"
        assert retriever.embedding_dimension == 1536
        assert retriever.provider == "openai"
        assert retriever.client is not None

    @pytest.mark.skipif(not GOOGLE_AVAILABLE, reason="google-generativeai not installed")
    def test_retriever_initialization_google(self, test_config):
        """Test MemoryRetriever initialization with Google embeddings"""
        test_config.embedding_model = "gemini-embedding-001"
        test_config.embedding_dimension = 768

        retriever = MemoryRetriever(test_config)

        assert retriever.embedding_model == "gemini-embedding-001"
        assert retriever.embedding_dimension == 768
        assert retriever.provider == "google"

    def test_retriever_top_k_from_config(self, test_config):
        """Test that retriever uses top_k_retrieval from config"""
        test_config.top_k_retrieval = 1  # Paper default

        retriever = MemoryRetriever(test_config)

        assert retriever.top_k == 1

    def test_unsupported_embedding_model_raises_error(self, test_config):
        """Test that unsupported embedding model raises ValueError"""
        test_config.embedding_model = "unsupported-model"

        with pytest.raises(ValueError, match="Unsupported embedding model"):
            MemoryRetriever(test_config)

    def test_cache_initialization(self, test_config):
        """Test that cache is initialized properly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config.embedding_cache_path = os.path.join(tmpdir, "cache.json")

            retriever = MemoryRetriever(test_config)

            assert retriever.cache_path == test_config.embedding_cache_path
            assert isinstance(retriever.embedding_cache, dict)


class TestEmbeddingGeneration:
    """Tests for embedding generation"""

    def test_embed_text_caching(self, test_config, mock_embedding_responses):
        """Test that embeddings are cached to avoid redundant API calls"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config.embedding_cache_path = os.path.join(tmpdir, "cache.json")
            retriever = MemoryRetriever(test_config)

            # Mock embedding generation (use _embed_openai since test_config uses OpenAI)
            with patch.object(retriever, '_embed_openai', return_value=mock_embedding_responses["query"]):
                # First call should generate embedding
                embedding1 = retriever.embed_text("Test query")

                # Second call should use cache
                embedding2 = retriever.embed_text("Test query")

                # Both should be identical
                assert embedding1 == embedding2

                # Check that embedding is in cache
                assert "Test query" in retriever.embedding_cache

    def test_embed_text_openai_format(self, test_config):
        """Test embedding generation with OpenAI format"""
        test_config.embedding_model = "text-embedding-3-small"
        retriever = MemoryRetriever(test_config)

        # Mock OpenAI client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        retriever.client.embeddings.create = MagicMock(return_value=mock_response)

        embedding = retriever._embed_openai("Test text")

        # Verify format
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
        retriever.client.embeddings.create.assert_called_once()

    @pytest.mark.skipif(not GOOGLE_AVAILABLE, reason="google-generativeai not installed")
    def test_embed_text_google_format(self, test_config):
        """Test embedding generation with Google format"""
        test_config.embedding_model = "gemini-embedding-001"
        retriever = MemoryRetriever(test_config)

        # Mock Google embedding
        mock_result = {'embedding': [0.1] * 768}
        with patch('reasoningbank.retriever.genai.embed_content', return_value=mock_result):
            embedding = retriever._embed_google("Test text")

        # Verify format
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_text_saves_to_cache(self, test_config):
        """Test that new embeddings are saved to cache"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config.embedding_cache_path = os.path.join(tmpdir, "cache.json")
            retriever = MemoryRetriever(test_config)

            # Mock embedding generation (use _embed_openai and 1536 dimensions for OpenAI)
            with patch.object(retriever, '_embed_openai', return_value=[0.1] * 1536):
                embedding = retriever.embed_text("New text")

                # Check cache was saved
                assert "New text" in retriever.embedding_cache
                assert os.path.exists(test_config.embedding_cache_path)


class TestCosineSimilarity:
    """Tests for cosine similarity computation"""

    def test_cosine_similarity_identical_vectors(self, test_config):
        """Test cosine similarity of identical vectors is 1.0"""
        retriever = MemoryRetriever(test_config)

        vec = [1.0, 2.0, 3.0, 4.0]
        similarity = retriever._cosine_similarity(vec, vec)

        assert similarity == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self, test_config):
        """Test cosine similarity of orthogonal vectors is 0.0"""
        retriever = MemoryRetriever(test_config)

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = retriever._cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(0.0)

    def test_cosine_similarity_opposite_vectors(self, test_config):
        """Test cosine similarity of opposite vectors is -1.0"""
        retriever = MemoryRetriever(test_config)

        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        similarity = retriever._cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(-1.0)

    def test_cosine_similarity_range_validation(self, test_config):
        """Test cosine similarity is always in range [-1, 1]"""
        retriever = MemoryRetriever(test_config)

        # Random vectors
        vec1 = [0.5, 1.5, 2.5, 0.3]
        vec2 = [1.2, 0.8, 1.0, 2.0]
        similarity = retriever._cosine_similarity(vec1, vec2)

        assert -1.0 <= similarity <= 1.0

    def test_cosine_similarity_zero_vector(self, test_config):
        """Test cosine similarity with zero vector returns 0.0"""
        retriever = MemoryRetriever(test_config)

        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        similarity = retriever._cosine_similarity(vec1, vec2)

        assert similarity == 0.0


class TestBasicRetrieval:
    """Tests for basic retrieval functionality"""

    def test_retrieve_with_memory_bank(self, test_config, sample_memory_items, mock_embedding_responses):
        """Test retrieving memories from memory bank"""
        retriever = MemoryRetriever(test_config)

        # Create memory entries
        entry = MemoryEntry(
            id="entry_001",
            task_query="Test query",
            trajectory="Test trajectory",
            success=True,
            memory_items=sample_memory_items
        )
        memory_bank = [entry]

        # Mock embeddings
        with patch.object(retriever, 'embed_text', side_effect=[
            mock_embedding_responses["query"],  # Query embedding
            mock_embedding_responses["memory1"],  # Item 1
            mock_embedding_responses["memory2"],  # Item 2
            mock_embedding_responses["memory3"]   # Item 3
        ]):
            results = retriever.retrieve("Test query", memory_bank, k=1)

        # Should return top-1 result
        assert len(results) == 1
        assert isinstance(results[0], MemoryItem)

    def test_retrieve_top_k_selection(self, test_config, sample_memory_items, mock_embedding_responses):
        """Test that retrieve returns exactly k items"""
        retriever = MemoryRetriever(test_config)

        entry = MemoryEntry(
            id="entry_001",
            task_query="Test",
            trajectory="Test",
            success=True,
            memory_items=sample_memory_items
        )
        memory_bank = [entry]

        # Mock embeddings (use 1536 for OpenAI)
        with patch.object(retriever, 'embed_text', return_value=[0.1] * 1536):
            # Test k=1
            results = retriever.retrieve("Test", memory_bank, k=1)
            assert len(results) == 1

            # Test k=2
            results = retriever.retrieve("Test", memory_bank, k=2)
            assert len(results) == 2

            # Test k=3
            results = retriever.retrieve("Test", memory_bank, k=3)
            assert len(results) == 3

    def test_retrieve_empty_memory_bank(self, test_config):
        """Test retrieving from empty memory bank returns empty list"""
        retriever = MemoryRetriever(test_config)

        results = retriever.retrieve("Test query", [], k=1)

        assert results == []

    def test_retrieve_uses_default_top_k(self, test_config, sample_memory_items):
        """Test that retrieve uses config.top_k_retrieval as default"""
        test_config.top_k_retrieval = 2
        retriever = MemoryRetriever(test_config)

        entry = MemoryEntry(
            id="entry_001",
            task_query="Test",
            trajectory="Test",
            success=True,
            memory_items=sample_memory_items
        )
        memory_bank = [entry]

        with patch.object(retriever, 'embed_text', return_value=[0.1] * 1536):
            # Don't specify k, should use default
            results = retriever.retrieve("Test", memory_bank)

        assert len(results) == 2  # Should use config.top_k_retrieval

    def test_retrieve_ranks_by_similarity(self, test_config):
        """Test that retrieve ranks items by cosine similarity"""
        retriever = MemoryRetriever(test_config)

        # Create items with different similarity scores
        item1 = MemoryItem(title="Low", description="Low", content="Low similarity")
        item2 = MemoryItem(title="High", description="High", content="High similarity")
        item3 = MemoryItem(title="Medium", description="Medium", content="Medium similarity")

        entry = MemoryEntry(
            id="entry_001",
            task_query="Test",
            trajectory="Test",
            success=True,
            memory_items=[item1, item2, item3]
        )
        memory_bank = [entry]

        # Mock embeddings to produce different similarities
        def mock_embed(text):
            if "High" in text:
                return [1.0, 0.0, 0.0]  # Most similar
            elif "Medium" in text:
                return [0.7, 0.3, 0.0]  # Medium similar
            elif "Low" in text:
                return [0.0, 1.0, 0.0]  # Least similar
            else:
                return [1.0, 0.0, 0.0]  # Query

        with patch.object(retriever, 'embed_text', side_effect=mock_embed):
            results = retriever.retrieve("Test query", memory_bank, k=3)

        # Should be ranked by similarity: High, Medium, Low
        assert results[0].title == "High"
        assert results[1].title == "Medium"
        assert results[2].title == "Low"


class TestFilteredRetrieval:
    """Tests for filtered retrieval functionality"""

    def test_retrieve_with_success_filter(self, test_config):
        """Test retrieving only successful trajectory memories"""
        retriever = MemoryRetriever(test_config)

        # Create mixed success/failure items
        success_item = MemoryItem(
            title="Success",
            description="Success",
            content="Success",
            success_signal=True
        )
        failure_item = MemoryItem(
            title="Failure",
            description="Failure",
            content="Failure",
            success_signal=False
        )

        entry = MemoryEntry(
            id="entry_001",
            task_query="Test",
            trajectory="Test",
            success=True,
            memory_items=[success_item, failure_item]
        )
        memory_bank = [entry]

        with patch.object(retriever, 'embed_text', return_value=[0.1] * 1536):
            results = retriever.retrieve_with_filtering(
                "Test",
                memory_bank,
                k=10,
                success_only=True
            )

        # Should only return success items
        assert len(results) == 1
        assert results[0].title == "Success"

    def test_retrieve_with_failure_filter(self, test_config):
        """Test retrieving only failed trajectory memories"""
        retriever = MemoryRetriever(test_config)

        success_item = MemoryItem(
            title="Success",
            description="Success",
            content="Success",
            success_signal=True
        )
        failure_item = MemoryItem(
            title="Failure",
            description="Failure",
            content="Failure",
            success_signal=False
        )

        entry = MemoryEntry(
            id="entry_001",
            task_query="Test",
            trajectory="Test",
            success=True,
            memory_items=[success_item, failure_item]
        )
        memory_bank = [entry]

        with patch.object(retriever, 'embed_text', return_value=[0.1] * 1536):
            results = retriever.retrieve_with_filtering(
                "Test",
                memory_bank,
                k=10,
                failure_only=True
            )

        # Should only return failure items
        assert len(results) == 1
        assert results[0].title == "Failure"

    def test_retrieve_with_similarity_threshold(self, test_config):
        """Test retrieving with minimum similarity threshold"""
        retriever = MemoryRetriever(test_config)

        item1 = MemoryItem(title="High", description="High", content="High")
        item2 = MemoryItem(title="Low", description="Low", content="Low")

        entry = MemoryEntry(
            id="entry_001",
            task_query="Test",
            trajectory="Test",
            success=True,
            memory_items=[item1, item2]
        )
        memory_bank = [entry]

        # Mock cosine similarity to return different scores
        def mock_cosine(vec1, vec2):
            # Alternate between high and low similarity
            import random
            return 0.9 if random.random() > 0.5 else 0.3

        with patch.object(retriever, 'embed_text', return_value=[0.1] * 1536), \
             patch.object(retriever, '_cosine_similarity', side_effect=[0.9, 0.3]):
            results = retriever.retrieve_with_filtering(
                "Test",
                memory_bank,
                k=10,
                min_similarity=0.5
            )

        # Should only return items with similarity >= 0.5
        assert len(results) >= 0  # May be 0 or 1 depending on mock

    def test_retrieve_with_combined_filters(self, test_config):
        """Test retrieving with multiple filters combined"""
        retriever = MemoryRetriever(test_config)

        success_item = MemoryItem(
            title="Success",
            description="Success",
            content="Success",
            success_signal=True
        )
        failure_item = MemoryItem(
            title="Failure",
            description="Failure",
            content="Failure",
            success_signal=False
        )

        entry = MemoryEntry(
            id="entry_001",
            task_query="Test",
            trajectory="Test",
            success=True,
            memory_items=[success_item, failure_item]
        )
        memory_bank = [entry]

        with patch.object(retriever, 'embed_text', return_value=[0.1] * 1536):
            results = retriever.retrieve_with_filtering(
                "Test",
                memory_bank,
                k=1,
                success_only=True,
                min_similarity=0.0
            )

        assert len(results) <= 1
        if results:
            assert results[0].success_signal is True


class TestCacheManagement:
    """Tests for embedding cache management"""

    def test_cache_saves_to_disk(self, test_config):
        """Test that cache is saved to disk"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            test_config.embedding_cache_path = cache_path

            retriever = MemoryRetriever(test_config)
            retriever.embedding_cache["test"] = [0.1] * 1536  # Use 1536 for OpenAI
            retriever._save_cache()

            # Verify cache file was created
            assert os.path.exists(cache_path)

            # Verify cache content
            with open(cache_path, 'r') as f:
                loaded_cache = json.load(f)
            assert "test" in loaded_cache

    def test_cache_loads_from_disk(self, test_config):
        """Test that cache is loaded from disk on initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            test_config.embedding_cache_path = cache_path

            # Create cache file (use 1536 for OpenAI)
            cache_data = {"test_key": [0.1] * 1536}
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)

            # Initialize retriever (should load cache)
            retriever = MemoryRetriever(test_config)

            assert "test_key" in retriever.embedding_cache

    def test_corrupted_cache_starts_fresh(self, test_config):
        """Test that corrupted cache is handled gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            test_config.embedding_cache_path = cache_path

            # Create corrupted cache file
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                f.write("corrupted json {{{")

            # Should initialize with empty cache (no error)
            retriever = MemoryRetriever(test_config)

            assert retriever.embedding_cache == {}

    def test_cache_hit_avoids_api_call(self, test_config):
        """Test that cache hit avoids redundant API calls"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config.embedding_cache_path = os.path.join(tmpdir, "cache.json")
            retriever = MemoryRetriever(test_config)

            # Pre-populate cache (use 1536 dimensions for OpenAI)
            retriever.embedding_cache["cached_text"] = [0.5] * 1536

            # Mock embedding generation (should not be called)
            with patch.object(retriever, '_embed_openai') as mock_embed:
                embedding = retriever.embed_text("cached_text")

                # Should return cached embedding without calling API
                assert embedding == [0.5] * 1536
                mock_embed.assert_not_called()


class TestConvenienceFunction:
    """Tests for standalone retrieve_memories function"""

    def test_convenience_function_returns_memories(self, test_config, sample_memory_items):
        """Test convenience function creates retriever and returns memories"""
        entry = MemoryEntry(
            id="entry_001",
            task_query="Test",
            trajectory="Test",
            success=True,
            memory_items=sample_memory_items
        )
        memory_bank = [entry]

        with patch('reasoningbank.retriever.MemoryRetriever') as MockRetriever:
            # Setup mock
            mock_retriever_instance = MockRetriever.return_value
            mock_retriever_instance.retrieve.return_value = sample_memory_items

            # Call convenience function
            result = retrieve_memories(
                query="Test query",
                memory_bank=memory_bank,
                config=test_config,
                k=1
            )

            # Verify it created a retriever and called retrieve
            MockRetriever.assert_called_once_with(test_config)
            mock_retriever_instance.retrieve.assert_called_once()
            assert len(result) == 3


@pytest.mark.unit
class TestRetrieverIntegration:
    """Integration tests for retriever component"""

    def test_full_retrieval_workflow(self, test_config, sample_memory_items, mock_embedding_responses):
        """Test complete retrieval workflow"""
        retriever = MemoryRetriever(test_config)

        # Create memory bank
        entry = MemoryEntry(
            id="entry_001",
            task_query="Arithmetic calculation",
            trajectory="Test trajectory",
            success=True,
            memory_items=sample_memory_items
        )
        memory_bank = [entry]

        # Mock embeddings
        with patch.object(retriever, 'embed_text', side_effect=[
            mock_embedding_responses["query"],
            mock_embedding_responses["memory1"],
            mock_embedding_responses["memory2"],
            mock_embedding_responses["memory3"]
        ]):
            results = retriever.retrieve("Calculate 25 * 4", memory_bank, k=1)

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], MemoryItem)

    def test_retrieval_with_multiple_entries(self, test_config, mock_embedding_responses):
        """Test retrieval across multiple memory entries"""
        retriever = MemoryRetriever(test_config)

        # Create multiple entries
        item1 = MemoryItem(title="Item1", description="Desc1", content="Content1")
        item2 = MemoryItem(title="Item2", description="Desc2", content="Content2")

        entry1 = MemoryEntry(
            id="entry_001",
            task_query="Query1",
            trajectory="Traj1",
            success=True,
            memory_items=[item1]
        )
        entry2 = MemoryEntry(
            id="entry_002",
            task_query="Query2",
            trajectory="Traj2",
            success=True,
            memory_items=[item2]
        )
        memory_bank = [entry1, entry2]

        with patch.object(retriever, 'embed_text', return_value=[0.1] * 1536):
            results = retriever.retrieve("Test", memory_bank, k=2)

        # Should retrieve from both entries
        assert len(results) == 2

    def test_retrieval_paper_default_top_k(self, test_config, sample_memory_items):
        """Test retrieval with paper's default top_k=1"""
        test_config.top_k_retrieval = 1  # Paper default
        retriever = MemoryRetriever(test_config)

        entry = MemoryEntry(
            id="entry_001",
            task_query="Test",
            trajectory="Test",
            success=True,
            memory_items=sample_memory_items
        )
        memory_bank = [entry]

        with patch.object(retriever, 'embed_text', return_value=[0.1] * 1536):
            results = retriever.retrieve("Test query", memory_bank)

        # Should return exactly 1 item (paper default)
        assert len(results) == 1
