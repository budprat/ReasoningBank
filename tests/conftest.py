"""
ABOUTME: Pytest configuration and shared fixtures for ReasoningBank tests
ABOUTME: Provides mock LLM responses, test configurations, and sample data
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import ReasoningBank components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from reasoningbank import ReasoningBankConfig
from reasoningbank.models import MemoryItem, MemoryEntry, TrajectoryResult


# ============= Configuration Fixtures =============

@pytest.fixture
def test_config():
    """
    Default test configuration with isolated test data paths.

    Uses temporary directories that are cleaned up after tests.
    Uses OpenAI embeddings to avoid Google dependency in tests.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ReasoningBankConfig(
            llm_provider="anthropic",
            llm_model="claude-3-5-sonnet-20241022",
            llm_api_key="test-api-key",
            openai_api_key="test-api-key",  # For OpenAI embeddings in tests
            embedding_model="text-embedding-3-small",  # Use OpenAI embeddings for tests
            embedding_dimension=1536,  # OpenAI embedding size
            agent_temperature=0.7,
            judge_temperature=0.0,
            extractor_temperature=1.0,
            selector_temperature=0.0,
            memory_bank_path=os.path.join(tmpdir, "memory_bank_test.json"),
            embedding_cache_path=os.path.join(tmpdir, "embeddings_test.json"),
            top_k_retrieval=1,
            max_memory_items_per_trajectory=3,
            max_memory_items_aggregated=5,
            max_steps_per_task=30,
            enable_logging=False  # Disable logging during tests
        )
        yield config


@pytest.fixture
def paper_config():
    """
    Configuration matching paper's experimental setup.

    Uses Gemini-2.5-flash as in paper's experiments.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ReasoningBankConfig(
            llm_provider="google",
            llm_model="gemini-2.5-flash",
            llm_api_key="test-api-key",
            agent_temperature=0.7,
            judge_temperature=0.0,
            extractor_temperature=1.0,
            top_k_retrieval=1,
            max_steps_per_task=30,
            memory_bank_path=os.path.join(tmpdir, "memory_bank_test.json"),
            embedding_cache_path=os.path.join(tmpdir, "embeddings_test.json"),
            enable_logging=False
        )
        yield config


# ============= Mock LLM Response Fixtures =============

@pytest.fixture
def mock_judge_responses():
    """
    Mock responses for TrajectoryJudge component.

    Returns responses for both success and failure cases.
    """
    return {
        "success": "Thoughts: The agent successfully completed the task with correct output.\nStatus: success",
        "failure": "Thoughts: The agent failed to complete the task correctly.\nStatus: failure",
        "ambiguous": "Status: success"  # Minimal response format
    }


@pytest.fixture
def mock_extractor_responses():
    """
    Mock responses for MemoryExtractor component.

    Returns properly formatted Markdown responses as per paper Figure 8.
    """
    return {
        "success": """# Memory Item 1
## Title Step-by-step arithmetic calculation
## Description Always break down complex calculations into smaller steps

## Content When solving multi-step arithmetic problems, first identify the operations and their order. Execute multiplication before addition following PEMDAS rules.

# Memory Item 2
## Title Verification of intermediate results
## Description Verify each calculation step before proceeding to the next

## Content Double-check intermediate results to catch errors early. This prevents compounding mistakes in multi-step calculations.""",

        "failure": """# Memory Item 1
## Title Check search query formatting
## Description Ensure search queries are properly formatted before execution

## Content Failed searches often result from incorrectly formatted queries. Verify query syntax and structure before attempting search operations.

# Memory Item 2
## Title Handle no-results scenarios
## Description Implement fallback strategies when searches return no results

## Content When initial search fails, try alternative search terms or broader queries. Don't immediately report failure without attempting recovery.""",

        "self_contrast": """# Memory Item 1
## Title Consistent verification approach
## Description Successful trajectories verified intermediate steps systematically

## Content Comparing successful and failed trajectories reveals that verification of intermediate results is critical. Failed attempts skipped validation steps that successful ones performed consistently.

# Memory Item 2
## Title Error recovery mechanisms
## Description Failed trajectories lacked proper error handling

## Content Successful agents implemented fallback strategies when encountering errors. Failed agents terminated immediately without attempting recovery or alternative approaches."""
    }


@pytest.fixture
def mock_embedding_responses():
    """
    Mock embedding vectors for retrieval testing.

    Returns 1536-dimensional vectors (OpenAI embedding size for tests).
    Provides sufficient embeddings for all test scenarios.
    """
    import numpy as np

    # Create deterministic embeddings for testing
    np.random.seed(42)

    return {
        "query": np.random.rand(1536).tolist(),
        "query1": np.random.rand(1536).tolist(),
        "query2": np.random.rand(1536).tolist(),
        "memory1": np.random.rand(1536).tolist(),
        "memory2": np.random.rand(1536).tolist(),
        "memory3": np.random.rand(1536).tolist(),  # Added for retriever tests
        "memory4": np.random.rand(1536).tolist(),  # Extra for future tests
        "memory5": np.random.rand(1536).tolist(),  # Extra for future tests
    }


# ============= Sample Data Fixtures =============

@pytest.fixture
def sample_successful_trajectory():
    """
    Sample successful trajectory for testing.

    Based on simple arithmetic task from paper examples.
    """
    return {
        "query": "What is 25 * 4 + 15?",
        "trajectory": """<think>I need to calculate 25 * 4 first according to order of operations</think>
<action>calculate 25 * 4</action>
<observation>Result: 100</observation>
<think>Now I need to add 15 to the result</think>
<action>calculate 100 + 15</action>
<observation>Result: 115</observation>
<think>The final answer is 115</think>
<action>answer 115</action>""",
        "final_state": "Calculation completed. Result displayed: 115",
        "model_output": "The answer is 115",
        "success": True,
        "steps_taken": 3
    }


@pytest.fixture
def sample_failed_trajectory():
    """
    Sample failed trajectory for testing.

    Represents a failed information retrieval task.
    """
    return {
        "query": "Find the author of the book '1984'",
        "trajectory": """<think>I'll search for the book 1984</think>
<action>search "1984"</action>
<observation>Error: Search query too broad, multiple results</observation>
<think>Let me try a direct search</think>
<action>search "1984 author"</action>
<observation>No results found</observation>
<think>Unable to find information</think>
<action>answer "Unable to find information"</action>""",
        "final_state": "Search failed, no results displayed",
        "model_output": "Unable to find the author information",
        "success": False,
        "steps_taken": 3
    }


@pytest.fixture
def sample_memory_items():
    """
    Sample pre-extracted memory items for testing retrieval.

    Contains both success and failure memory items.
    """
    return [
        MemoryItem(
            title="Order of Operations in Multi-Step Calculations",
            description="Always perform multiplication before addition in arithmetic expressions",
            content="When solving problems with multiple operations, follow PEMDAS/BODMAS order to ensure correct results. This prevents common arithmetic errors.",
            success_signal=True,
            source_task_id="task_001"
        ),
        MemoryItem(
            title="Search Query Refinement",
            description="Refine broad search queries to get more specific results",
            content="When search queries are too broad, add specific keywords or filters. For book searches, include 'author' or 'book' in the query to narrow results.",
            success_signal=False,
            source_task_id="task_002"
        ),
        MemoryItem(
            title="Intermediate Result Verification",
            description="Verify each calculation step before proceeding",
            content="Double-check intermediate results in multi-step calculations. This catches errors early and prevents compounding mistakes in subsequent steps.",
            success_signal=True,
            source_task_id="task_003"
        )
    ]


@pytest.fixture
def sample_memory_entries(sample_memory_items):
    """
    Sample memory bank entries for testing consolidation.

    Contains complete MemoryEntry objects with trajectories.
    """
    return [
        MemoryEntry(
            id="entry_001",
            task_query="What is 25 * 4 + 15?",
            trajectory="<think>Calculate 25 * 4 first</think><action>calculate</action>",
            success=True,
            memory_items=[sample_memory_items[0], sample_memory_items[2]],
            final_state="Result: 115",
            model_output="The answer is 115",
            steps_taken=3
        ),
        MemoryEntry(
            id="entry_002",
            task_query="Find the author of '1984'",
            trajectory="<think>Search for 1984</think><action>search</action>",
            success=False,
            memory_items=[sample_memory_items[1]],
            final_state="Search failed",
            model_output="Unable to find information",
            steps_taken=3
        )
    ]


# ============= Mock Environment Fixtures =============

@pytest.fixture
def mock_environment():
    """
    Mock environment for agent testing.

    Simulates agent interactions without real API calls.
    """
    from tests.fixtures.mock_environments import MockEnvironment
    return MockEnvironment()


# ============= Mock LLM Client Fixtures =============

@pytest.fixture
def mock_anthropic_client(mock_judge_responses, mock_extractor_responses):
    """
    Mock Anthropic client for testing without API calls.

    Returns predefined responses for judge and extractor.
    """
    with patch('anthropic.Anthropic') as mock_client:
        mock_instance = Mock()

        # Mock messages.create for judge and extractor
        def create_mock_response(model, max_tokens, temperature, messages):
            content = messages[0]["content"]

            # Determine response type based on prompt content
            if "evaluating the performance" in content:
                # Judge prompt
                response_text = mock_judge_responses["success"]
            elif "extract and summarize useful insights" in content:
                # Extractor prompt
                if "successfully accomplished" in content:
                    response_text = mock_extractor_responses["success"]
                else:
                    response_text = mock_extractor_responses["failure"]
            else:
                response_text = "Mock response"

            # Create mock response object
            mock_response = Mock()
            mock_content = Mock()
            mock_content.text = response_text
            mock_response.content = [mock_content]
            return mock_response

        mock_instance.messages.create = Mock(side_effect=create_mock_response)
        mock_client.return_value = mock_instance

        yield mock_client


@pytest.fixture
def mock_embedding_client(mock_embedding_responses):
    """
    Mock embedding client for testing retrieval without API calls.

    Returns predefined embedding vectors (OpenAI format for tests).
    """
    with patch('openai.OpenAI') as mock_openai:
        def embed_mock(model, input):
            # Return embedding based on content hash for consistency
            import hashlib
            content_hash = int(hashlib.md5(input.encode()).hexdigest()[:8], 16)

            import numpy as np
            np.random.seed(content_hash)
            embedding = np.random.rand(1536).tolist()

            # Mock OpenAI response structure
            mock_response = Mock()
            mock_data = Mock()
            mock_data.embedding = embedding
            mock_response.data = [mock_data]
            return mock_response

        mock_instance = Mock()
        mock_instance.embeddings.create = Mock(side_effect=embed_mock)
        mock_openai.return_value = mock_instance
        yield mock_openai


# ============= Pytest Hooks and Configuration =============

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for closed-loop system")
    config.addinivalue_line("markers", "matts: Test-time scaling tests")
    config.addinivalue_line("markers", "performance: Performance and benchmarking tests")
    config.addinivalue_line("markers", "slow: Slow tests that may take significant time")


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """
    Automatically cleanup test data after each test.

    Removes temporary files and directories created during tests.
    """
    yield

    # Cleanup logic runs after test
    test_data_dirs = [
        "./test_data",
        "./data_test",
        "./__pycache__"
    ]

    for dir_path in test_data_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)


# ============= Helper Functions =============

@pytest.fixture
def assert_memory_item_valid():
    """
    Helper function to validate MemoryItem structure.

    Ensures all required fields are present and properly formatted.
    """
    def _assert(item: MemoryItem):
        assert item.title is not None and len(item.title) > 0
        assert item.description is not None and len(item.description) > 0
        assert item.content is not None and len(item.content) > 0
        assert isinstance(item.success_signal, (bool, type(None)))

        # Title should be concise (< 100 chars)
        assert len(item.title) < 100, f"Title too long: {len(item.title)} chars"

        # Description should be one sentence
        assert len(item.description.split('.')) <= 2, "Description should be one sentence"

        # Content should be 1-5 sentences
        sentence_count = len([s for s in item.content.split('.') if s.strip()])
        assert 1 <= sentence_count <= 5, f"Content should be 1-5 sentences, got {sentence_count}"

    return _assert
