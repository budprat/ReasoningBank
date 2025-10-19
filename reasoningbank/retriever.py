"""
ABOUTME: Memory retrieval implementation using embedding-based similarity search
ABOUTME: Implements cosine similarity search with OpenAI and Google embeddings
"""

from typing import List, Optional, Dict, Any
import numpy as np
import json
import os
import openai

from .config import ReasoningBankConfig
from .models import MemoryEntry, MemoryItem

# Lazy import for google-generativeai to avoid import errors when not installed
try:
    from google import generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None


class MemoryRetriever:
    """
    Embedding-based retrieval for ReasoningBank memory system.

    Uses cosine similarity between query embeddings and memory embeddings
    to find relevant past experiences.

    Supports:
    - OpenAI embeddings (text-embedding-3-small, 1536 dim)
    - Google embeddings (gemini-embedding-001, 768 dim)
    """

    def __init__(self, config: ReasoningBankConfig):
        """
        Initialize the memory retriever with configuration.

        Args:
            config: ReasoningBankConfig with embedding settings
        """
        self.config = config
        self.embedding_model = config.embedding_model
        self.embedding_dimension = config.embedding_dimension
        self.top_k = config.top_k_retrieval

        # Initialize embedding client
        if "text-embedding" in self.embedding_model:
            # OpenAI embeddings
            api_key = config.openai_api_key or config.llm_api_key
            self.client = openai.OpenAI(api_key=api_key)
            self.provider = "openai"
        elif "gemini" in self.embedding_model or "embedding" in self.embedding_model:
            # Google embeddings
            if not GOOGLE_AVAILABLE:
                raise ImportError(
                    "google-generativeai is not installed. "
                    "Install it with: pip install google-generativeai>=0.3.0"
                )
            # Use google_api_key if available, otherwise fall back to llm_api_key
            api_key = config.google_api_key or config.llm_api_key
            if not api_key:
                raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable")
            genai.configure(api_key=api_key)
            self.provider = "google"
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")

        # Load embedding cache if available
        self.cache_path = config.embedding_cache_path
        self.embedding_cache: Dict[str, List[float]] = {}
        self._load_cache()

    def retrieve(
        self,
        query: str,
        memory_bank: List[MemoryEntry],
        k: Optional[int] = None
    ) -> List[MemoryItem]:
        """
        Retrieve top-k relevant memory items for a query.

        Args:
            query: Task query to find relevant memories for
            memory_bank: List of all memory entries
            k: Number of items to retrieve (default: config.top_k_retrieval)

        Returns:
            List[MemoryItem]: Top-k most relevant memory items
        """
        if k is None:
            k = self.top_k

        # Handle empty memory bank
        if not memory_bank:
            return []

        # Flatten memory items from all entries
        all_memory_items = []
        for entry in memory_bank:
            all_memory_items.extend(entry.memory_items)

        if not all_memory_items:
            return []

        # Generate query embedding
        query_embedding = self.embed_text(query)

        # Generate embeddings for all memory items
        memory_embeddings = []
        for item in all_memory_items:
            # Combine title, description, and content for embedding
            item_text = f"{item.title}\n{item.description}\n{item.content}"
            item_embedding = self.embed_text(item_text)
            memory_embeddings.append(item_embedding)

        # Compute cosine similarities
        similarities = []
        for i, mem_emb in enumerate(memory_embeddings):
            sim = self._cosine_similarity(query_embedding, mem_emb)
            similarities.append((sim, all_memory_items[i]))

        # Sort by similarity (descending) and return top-k
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k_items = [item for _, item in similarities[:k]]

        return top_k_items

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Uses caching to avoid redundant API calls.

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector
        """
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        # Generate embedding
        if self.provider == "openai":
            embedding = self._embed_openai(text)
        elif self.provider == "google":
            embedding = self._embed_google(text)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Cache the embedding
        self.embedding_cache[text] = embedding
        self._save_cache()

        return embedding

    def _embed_openai(self, text: str) -> List[float]:
        """
        Generate embedding using OpenAI API.

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector (1536 dimensions)
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def _embed_google(self, text: str) -> List[float]:
        """
        Generate embedding using Google Generative AI API.

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector (768 dimensions)
        """
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Cosine similarity in range [-1, 1]
        """
        # Convert to numpy arrays
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        # Compute cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    self.embedding_cache = json.load(f)
            except Exception:
                # If cache is corrupted, start fresh
                self.embedding_cache = {}

    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

            with open(self.cache_path, 'w') as f:
                json.dump(self.embedding_cache, f)
        except Exception as e:
            # Non-critical error, continue without caching
            pass

    def retrieve_with_filtering(
        self,
        query: str,
        memory_bank: List[MemoryEntry],
        k: Optional[int] = None,
        success_only: bool = False,
        failure_only: bool = False,
        min_similarity: float = 0.0
    ) -> List[MemoryItem]:
        """
        Retrieve memories with filtering options.

        Args:
            query: Task query
            memory_bank: List of memory entries
            k: Number of items to retrieve
            success_only: Only retrieve from successful trajectories
            failure_only: Only retrieve from failed trajectories
            min_similarity: Minimum similarity threshold

        Returns:
            List[MemoryItem]: Filtered and ranked memory items
        """
        if k is None:
            k = self.top_k

        # Flatten and filter memory items
        all_memory_items = []
        for entry in memory_bank:
            for item in entry.memory_items:
                # Apply filters
                if success_only and not item.success_signal:
                    continue
                if failure_only and item.success_signal is not False:
                    continue

                all_memory_items.append(item)

        if not all_memory_items:
            return []

        # Generate query embedding
        query_embedding = self.embed_text(query)

        # Generate embeddings and compute similarities
        similarities = []
        for item in all_memory_items:
            item_text = f"{item.title}\n{item.description}\n{item.content}"
            item_embedding = self.embed_text(item_text)
            sim = self._cosine_similarity(query_embedding, item_embedding)

            # Apply similarity threshold
            if sim >= min_similarity:
                similarities.append((sim, item))

        # Sort by similarity and return top-k
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k_items = [item for _, item in similarities[:k]]

        return top_k_items


def retrieve_memories(
    query: str,
    memory_bank: List[MemoryEntry],
    config: ReasoningBankConfig,
    k: Optional[int] = None
) -> List[MemoryItem]:
    """
    Convenience function for one-off memory retrieval.

    Args:
        query: Task query
        memory_bank: List of memory entries
        config: ReasoningBankConfig
        k: Number of items to retrieve

    Returns:
        List[MemoryItem]: Top-k relevant memory items
    """
    retriever = MemoryRetriever(config)
    return retriever.retrieve(query, memory_bank, k)
