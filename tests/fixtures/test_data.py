"""
ABOUTME: Shared test utilities and helper functions
ABOUTME: Common assertions, validators, and test data generators
"""

from typing import List, Dict, Any, Optional
from reasoningbank.models import MemoryItem, MemoryEntry, TrajectoryResult
import re


# ============= Validation Functions =============

def validate_memory_item(item: MemoryItem) -> tuple[bool, Optional[str]]:
    """
    Validate MemoryItem conforms to paper specifications.

    Paper Reference: Section 3.2 - Memory Schema

    Args:
        item: MemoryItem to validate

    Returns:
        (is_valid, error_message) tuple
    """
    # Check required fields
    if not item.title or not item.description or not item.content:
        return False, "Missing required fields (title, description, or content)"

    # Title constraints
    if len(item.title) == 0:
        return False, "Title cannot be empty"
    if len(item.title) > 100:
        return False, f"Title too long ({len(item.title)} > 100 chars)"

    # Description constraints (should be one sentence)
    desc_sentences = [s.strip() for s in item.description.split('.') if s.strip()]
    if len(desc_sentences) > 2:
        return False, f"Description should be one sentence, got {len(desc_sentences)}"

    # Content constraints (should be 1-5 sentences)
    content_sentences = [s.strip() for s in item.content.split('.') if s.strip()]
    if len(content_sentences) < 1:
        return False, "Content must have at least one sentence"
    if len(content_sentences) > 5:
        return False, f"Content should be 1-5 sentences, got {len(content_sentences)}"

    # Success signal should be bool or None
    if item.success_signal not in [True, False, None]:
        return False, f"Invalid success_signal: {item.success_signal}"

    return True, None


def validate_markdown_format(text: str) -> tuple[bool, Optional[str]]:
    """
    Validate text follows paper's Markdown extraction format.

    Paper Reference: Figure 8 (Page 18) - Output Format

    Expected format:
    ```
    # Memory Item i
    ## Title <title>
    ## Description <description>
    ## Content <content>
    ```

    Args:
        text: Markdown text to validate

    Returns:
        (is_valid, error_message) tuple
    """
    # Check for memory item headers
    if not re.search(r'# Memory Item', text):
        return False, "Missing '# Memory Item' header"

    # Check for required sections
    if not re.search(r'## Title', text):
        return False, "Missing '## Title' section"
    if not re.search(r'## Description', text):
        return False, "Missing '## Description' section"
    if not re.search(r'## Content', text):
        return False, "Missing '## Content' section"

    return True, None


def validate_temperature_settings(config) -> tuple[bool, Optional[str]]:
    """
    Validate temperature settings match paper specifications.

    Paper Reference: Appendix A.2

    Args:
        config: ReasoningBankConfig object

    Returns:
        (is_valid, error_message) tuple
    """
    expected = {
        "agent_temperature": 0.7,
        "judge_temperature": 0.0,
        "extractor_temperature": 1.0,
        "selector_temperature": 0.0
    }

    for key, expected_val in expected.items():
        actual_val = getattr(config, key)
        if actual_val != expected_val:
            return False, f"{key}: expected {expected_val}, got {actual_val}"

    return True, None


def validate_extraction_limits(config) -> tuple[bool, Optional[str]]:
    """
    Validate extraction limits match paper specifications.

    Paper Reference: Appendix A.1

    Args:
        config: ReasoningBankConfig object

    Returns:
        (is_valid, error_message) tuple
    """
    if config.max_memory_items_per_trajectory != 3:
        return False, f"max_memory_items_per_trajectory should be 3, got {config.max_memory_items_per_trajectory}"

    if config.max_memory_items_aggregated != 5:
        return False, f"max_memory_items_aggregated should be 5, got {config.max_memory_items_aggregated}"

    return True, None


# ============= Comparison Functions =============

def compare_memory_items(item1: MemoryItem, item2: MemoryItem) -> bool:
    """
    Compare two memory items for equality.

    Args:
        item1, item2: MemoryItem objects to compare

    Returns:
        bool: True if items are equivalent
    """
    return (
        item1.title == item2.title and
        item1.description == item2.description and
        item1.content == item2.content and
        item1.success_signal == item2.success_signal
    )


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity (Jaccard similarity).

    Args:
        text1, text2: Texts to compare

    Returns:
        float: Similarity score between 0 and 1
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


# ============= Test Data Generators =============

def generate_test_trajectory(
    num_steps: int,
    success: bool = True,
    task_type: str = "arithmetic"
) -> Dict[str, Any]:
    """
    Generate synthetic trajectory for testing.

    Args:
        num_steps: Number of steps in trajectory
        success: Whether trajectory should be successful
        task_type: Type of task (arithmetic, search, navigation)

    Returns:
        Dict with trajectory data
    """
    steps = []
    for i in range(num_steps):
        steps.append(f"<think>Step {i+1} thinking</think>\n<action>Action {i+1}</action>\n<observation>Observation {i+1}</observation>")

    trajectory = "\n".join(steps)

    return {
        "query": f"Test {task_type} query",
        "trajectory": trajectory,
        "final_state": f"Final state after {num_steps} steps",
        "model_output": "Success output" if success else "Failure output",
        "success": success,
        "steps_taken": num_steps
    }


def generate_test_memory_items(count: int, success_signal: bool = True) -> List[MemoryItem]:
    """
    Generate test memory items.

    Args:
        count: Number of items to generate
        success_signal: Success/failure signal for items

    Returns:
        List of MemoryItem objects
    """
    items = []
    for i in range(count):
        item = MemoryItem(
            title=f"Test Memory Item {i+1}",
            description=f"Test description for item {i+1}",
            content=f"Test content for item {i+1}. This is a complete sentence explaining the insight.",
            success_signal=success_signal,
            source_task_id=f"test_task_{i+1}"
        )
        items.append(item)

    return items


# ============= Mock Response Generators =============

def generate_mock_judge_response(success: bool) -> str:
    """
    Generate mock judge response.

    Args:
        success: Whether to generate success or failure response

    Returns:
        str: Mock judge response in correct format
    """
    if success:
        return "Thoughts: The agent completed the task successfully with correct output.\nStatus: success"
    else:
        return "Thoughts: The agent failed to complete the task correctly.\nStatus: failure"


def generate_mock_extraction_response(
    num_items: int = 3,
    success_type: bool = True
) -> str:
    """
    Generate mock extraction response in Markdown format.

    Args:
        num_items: Number of memory items (max 3 per paper)
        success_type: True for success extraction, False for failure

    Returns:
        str: Mock extraction response in Markdown format
    """
    items = []
    prefix = "Success Strategy" if success_type else "Preventative Lesson"

    for i in range(min(num_items, 3)):
        item = f"""# Memory Item {i+1}
## Title {prefix} {i+1}
## Description Test description for {prefix.lower()} {i+1}
## Content Test content explaining the {prefix.lower()}. This provides actionable insights. The strategy is generalizable."""
        items.append(item)

    return "\n\n".join(items)


# ============= Assertion Helpers =============

class AssertionHelpers:
    """Collection of assertion helper methods for testing"""

    @staticmethod
    def assert_valid_memory_item(item: MemoryItem):
        """Assert memory item is valid"""
        is_valid, error = validate_memory_item(item)
        assert is_valid, f"Invalid memory item: {error}"

    @staticmethod
    def assert_valid_markdown(text: str):
        """Assert text is valid Markdown extraction format"""
        is_valid, error = validate_markdown_format(text)
        assert is_valid, f"Invalid Markdown format: {error}"

    @staticmethod
    def assert_temperature_correct(config):
        """Assert temperature settings match paper"""
        is_valid, error = validate_temperature_settings(config)
        assert is_valid, f"Invalid temperature settings: {error}"

    @staticmethod
    def assert_extraction_limits_correct(config):
        """Assert extraction limits match paper"""
        is_valid, error = validate_extraction_limits(config)
        assert is_valid, f"Invalid extraction limits: {error}"

    @staticmethod
    def assert_memory_count(items: List[MemoryItem], expected_max: int):
        """Assert memory item count within limits"""
        assert len(items) <= expected_max, f"Too many memory items: {len(items)} > {expected_max}"

    @staticmethod
    def assert_cosine_similarity_valid(similarity: float):
        """Assert cosine similarity in valid range"""
        assert -1.0 <= similarity <= 1.0, f"Invalid cosine similarity: {similarity}"
        assert isinstance(similarity, float), f"Similarity should be float, got {type(similarity)}"


# ============= Test Constants =============

# From paper specifications
PAPER_MAX_STEPS = 30
PAPER_MAX_ITEMS_PER_TRAJECTORY = 3
PAPER_MAX_ITEMS_AGGREGATED = 5
PAPER_TOP_K_RETRIEVAL = 1
PAPER_EMBEDDING_DIM_GEMINI = 768
PAPER_EMBEDDING_DIM_OPENAI = 1536

# Temperature settings from paper
PAPER_AGENT_TEMP = 0.7
PAPER_JUDGE_TEMP = 0.0
PAPER_EXTRACTOR_TEMP = 1.0
PAPER_SELECTOR_TEMP = 0.0

# MaTTS scaling factors from paper
MATTS_K_VALUES = [1, 3, 5, 7]

# Expected performance improvements from paper (relative %)
EXPECTED_IMPROVEMENTS = {
    "webarena": 17,  # 22.1% → 25.8%
    "mind2web": 15,  # 31.5% → 36.1%
    "swebench": 43   # 7.0% → 10.0%
}


# ============= Debugging Helpers =============

def print_memory_item(item: MemoryItem):
    """Pretty print memory item for debugging"""
    print("=" * 60)
    print(f"Title: {item.title}")
    print(f"Description: {item.description}")
    print(f"Content: {item.content}")
    print(f"Success Signal: {item.success_signal}")
    print(f"Source Task: {item.source_task_id}")
    print("=" * 60)


def print_trajectory_result(result: TrajectoryResult):
    """Pretty print trajectory result for debugging"""
    print("=" * 60)
    print(f"Query: {result.query}")
    print(f"Success: {result.success}")
    print(f"Steps: {result.steps_taken}")
    print(f"Output: {result.model_output}")
    if result.memory_items:
        print(f"Memory Items: {len(result.memory_items)}")
    print("=" * 60)
