"""
ABOUTME: Sample memory items and entries for testing ReasoningBank retrieval
ABOUTME: Pre-extracted memory items based on paper's memory schema
"""

from reasoningbank.models import MemoryItem, MemoryEntry
from typing import List
import time


# ============= Success Memory Items =============

MEMORY_ARITHMETIC_SUCCESS = MemoryItem(
    title="Order of Operations in Arithmetic",
    description="Follow PEMDAS/BODMAS rules for multi-operation calculations",
    content="When solving arithmetic problems with multiple operations, always apply order of operations: Parentheses, Exponents, Multiplication/Division (left to right), Addition/Subtraction (left to right). This prevents calculation errors.",
    success_signal=True,
    source_task_id="task_arithmetic_001",
    extraction_timestamp=time.time()
)

MEMORY_STEP_VERIFICATION = MemoryItem(
    title="Intermediate Step Verification",
    description="Verify each calculation step before proceeding to next operation",
    content="In multi-step calculations, verify intermediate results immediately after each operation. This catches errors early and prevents them from compounding in subsequent steps.",
    success_signal=True,
    source_task_id="task_arithmetic_002",
    extraction_timestamp=time.time()
)

MEMORY_SPECIFIC_SEARCH_TERMS = MemoryItem(
    title="Use Specific Search Keywords",
    description="Add specific keywords to search queries for better results",
    content="When searching for information, include specific keywords that narrow the scope. For book searches, add 'author' or 'book'; for products, add model numbers or categories.",
    success_signal=True,
    source_task_id="task_search_001",
    extraction_timestamp=time.time()
)

MEMORY_GOAL_ORIENTED_NAVIGATION = MemoryItem(
    title="Goal-Oriented Navigation Strategy",
    description="Keep target goal in mind when navigating through pages",
    content="During site navigation, maintain clear awareness of the destination goal. Verify each page transition moves closer to the objective before proceeding further.",
    success_signal=True,
    source_task_id="task_navigation_001",
    extraction_timestamp=time.time()
)

MEMORY_RESULT_RANKING = MemoryItem(
    title="Systematic Result Comparison",
    description="Compare all results systematically when finding optimal choice",
    content="When tasked with finding the best option (cheapest, fastest, highest-rated), collect all candidates first, then apply comparison criteria systematically rather than selecting prematurely.",
    success_signal=True,
    source_task_id="task_reasoning_001",
    extraction_timestamp=time.time()
)


# ============= Failure Memory Items (Preventative Lessons) =============

MEMORY_AVOID_OPERATION_CONFUSION = MemoryItem(
    title="Clarify Operation Requirements First",
    description="Identify required operations before executing calculations",
    content="Failed attempts often result from confusion about which operations to apply. Before calculating, explicitly identify each operation type and sequence to avoid mixing addition with multiplication incorrectly.",
    success_signal=False,
    source_task_id="task_arithmetic_003",
    extraction_timestamp=time.time()
)

MEMORY_AVOID_BROAD_SEARCHES = MemoryItem(
    title="Refine Overly Broad Search Queries",
    description="Narrow search scope when too many irrelevant results appear",
    content="Searches with generic terms often return overwhelming irrelevant results. When this occurs, add specific filters or context keywords rather than abandoning the search immediately.",
    success_signal=False,
    source_task_id="task_search_002",
    extraction_timestamp=time.time()
)

MEMORY_AVOID_RANDOM_CLICKING = MemoryItem(
    title="Plan Navigation Path Before Clicking",
    description="Avoid clicking random links without clear direction",
    content="Random navigation wastes time and often leads to dead ends. Before clicking any link or button, verify it logically leads toward the target destination.",
    success_signal=False,
    source_task_id="task_navigation_002",
    extraction_timestamp=time.time()
)

MEMORY_IMPLEMENT_ERROR_RECOVERY = MemoryItem(
    title="Implement Error Recovery Strategies",
    description="Don't immediately fail when encountering errors - attempt recovery",
    content="When operations fail or return errors, implement fallback strategies: try alternative approaches, use broader queries, or attempt related solutions before reporting failure.",
    success_signal=False,
    source_task_id="task_error_handling_001",
    extraction_timestamp=time.time()
)

MEMORY_COMPLETE_REQUIREMENTS = MemoryItem(
    title="Verify All Requirements Met",
    description="Check that all parts of the request are fulfilled before completing",
    content="Partial completion is a failure. Before finalizing, verify every aspect of the request is addressed. If asking for 'top 3' items, ensure exactly 3 are provided.",
    success_signal=False,
    source_task_id="task_completeness_001",
    extraction_timestamp=time.time()
)


# ============= Self-Contrast Extracted Memories =============

MEMORY_CONTRAST_SYSTEMATIC_APPROACH = MemoryItem(
    title="Systematic Approach Distinguishes Success from Failure",
    description="Successful trajectories follow structured methodical approaches",
    content="Comparing successful and failed attempts reveals that systematic step-by-step execution consistently leads to success, while ad-hoc reactive approaches typically fail. Structure and planning are critical success factors.",
    success_signal=None,  # Mixed - extracted from comparison
    source_task_id="matts_parallel_001",
    extraction_timestamp=time.time()
)

MEMORY_CONTRAST_VERIFICATION_IMPORTANCE = MemoryItem(
    title="Verification Differentiates Outcomes",
    description="Successful agents verify steps while failed agents skip validation",
    content="Analysis of multiple trajectories shows successful agents consistently verify intermediate results and page states, while failed agents proceed without validation. This verification habit prevents error propagation.",
    success_signal=None,
    source_task_id="matts_parallel_002",
    extraction_timestamp=time.time()
)


# ============= Collections =============

ALL_SUCCESS_MEMORIES: List[MemoryItem] = [
    MEMORY_ARITHMETIC_SUCCESS,
    MEMORY_STEP_VERIFICATION,
    MEMORY_SPECIFIC_SEARCH_TERMS,
    MEMORY_GOAL_ORIENTED_NAVIGATION,
    MEMORY_RESULT_RANKING
]

ALL_FAILURE_MEMORIES: List[MemoryItem] = [
    MEMORY_AVOID_OPERATION_CONFUSION,
    MEMORY_AVOID_BROAD_SEARCHES,
    MEMORY_AVOID_RANDOM_CLICKING,
    MEMORY_IMPLEMENT_ERROR_RECOVERY,
    MEMORY_COMPLETE_REQUIREMENTS
]

ALL_SELF_CONTRAST_MEMORIES: List[MemoryItem] = [
    MEMORY_CONTRAST_SYSTEMATIC_APPROACH,
    MEMORY_CONTRAST_VERIFICATION_IMPORTANCE
]

ALL_MEMORY_ITEMS: List[MemoryItem] = (
    ALL_SUCCESS_MEMORIES +
    ALL_FAILURE_MEMORIES +
    ALL_SELF_CONTRAST_MEMORIES
)


# ============= Memory Entries (with trajectories) =============

ENTRY_ARITHMETIC_SUCCESS = MemoryEntry(
    id="entry_arithmetic_success_001",
    task_query="What is 25 * 4 + 15?",
    trajectory="<think>Calculate 25*4 first (PEMDAS)</think><action>calculate 25*4</action><observation>100</observation><think>Now add 15</think><action>calculate 100+15</action><observation>115</observation>",
    success=True,
    memory_items=[MEMORY_ARITHMETIC_SUCCESS, MEMORY_STEP_VERIFICATION],
    final_state="Result: 115",
    model_output="The answer is 115",
    steps_taken=2,
    timestamp=time.time()
)

ENTRY_SEARCH_SUCCESS = MemoryEntry(
    id="entry_search_success_001",
    task_query="Who wrote 'To Kill a Mockingbird'?",
    trajectory="<think>Search with specific terms</think><action>search 'To Kill a Mockingbird author'</action><observation>Harper Lee</observation>",
    success=True,
    memory_items=[MEMORY_SPECIFIC_SEARCH_TERMS],
    final_state="Author found: Harper Lee",
    model_output="Harper Lee",
    steps_taken=1,
    timestamp=time.time()
)

ENTRY_ARITHMETIC_FAILURE = MemoryEntry(
    id="entry_arithmetic_failure_001",
    task_query="What is 25 * 4 + 15?",
    trajectory="<think>Just add all numbers</think><action>calculate 25+4+15</action><observation>44</observation>",
    success=False,
    memory_items=[MEMORY_AVOID_OPERATION_CONFUSION],
    final_state="Incorrect result: 44",
    model_output="44",
    steps_taken=1,
    timestamp=time.time()
)

ENTRY_SEARCH_FAILURE = MemoryEntry(
    id="entry_search_failure_001",
    task_query="Find author of '1984'",
    trajectory="<think>Search for 1984</think><action>search '1984'</action><observation>Too many results</observation>",
    success=False,
    memory_items=[MEMORY_AVOID_BROAD_SEARCHES],
    final_state="Search failed - too broad",
    model_output="Unable to find",
    steps_taken=1,
    timestamp=time.time()
)

ENTRY_NAVIGATION_SUCCESS = MemoryEntry(
    id="entry_navigation_success_001",
    task_query="Navigate to checkout page",
    trajectory="<think>Look for cart button</think><action>click 'View Cart'</action><observation>Cart page</observation><think>Now find checkout</think><action>click 'Checkout'</action><observation>Checkout page</observation>",
    success=True,
    memory_items=[MEMORY_GOAL_ORIENTED_NAVIGATION],
    final_state="Checkout page reached",
    model_output="At checkout",
    steps_taken=2,
    timestamp=time.time()
)


ALL_MEMORY_ENTRIES: List[MemoryEntry] = [
    ENTRY_ARITHMETIC_SUCCESS,
    ENTRY_SEARCH_SUCCESS,
    ENTRY_ARITHMETIC_FAILURE,
    ENTRY_SEARCH_FAILURE,
    ENTRY_NAVIGATION_SUCCESS
]


# ============= Memory Bank Statistics =============

def get_memory_bank_stats(entries: List[MemoryEntry]) -> dict:
    """
    Calculate statistics for a memory bank.

    Args:
        entries: List of MemoryEntry objects

    Returns:
        Dict with statistics (total_entries, success_count, failure_count, etc.)
    """
    total_entries = len(entries)
    success_count = sum(1 for e in entries if e.success)
    failure_count = total_entries - success_count

    total_memory_items = sum(len(e.memory_items) for e in entries)
    avg_items_per_entry = total_memory_items / total_entries if total_entries > 0 else 0

    return {
        "total_entries": total_entries,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_count / total_entries if total_entries > 0 else 0,
        "total_memory_items": total_memory_items,
        "avg_items_per_entry": avg_items_per_entry
    }


# ============= Helper Functions =============

def get_memories_by_type(memory_type: str) -> List[MemoryItem]:
    """
    Get memory items filtered by type.

    Args:
        memory_type: One of "success", "failure", "self_contrast", "all"

    Returns:
        List of MemoryItem objects
    """
    type_map = {
        "success": ALL_SUCCESS_MEMORIES,
        "failure": ALL_FAILURE_MEMORIES,
        "self_contrast": ALL_SELF_CONTRAST_MEMORIES,
        "all": ALL_MEMORY_ITEMS
    }

    if memory_type not in type_map:
        raise ValueError(f"Unknown memory type: {memory_type}. Choose from: success, failure, self_contrast, all")

    return type_map[memory_type]


def create_minimal_memory_bank() -> List[MemoryEntry]:
    """
    Create a minimal memory bank for basic testing (3 entries).

    Returns:
        List of 3 MemoryEntry objects
    """
    return [
        ENTRY_ARITHMETIC_SUCCESS,
        ENTRY_SEARCH_SUCCESS,
        ENTRY_ARITHMETIC_FAILURE
    ]


def create_full_memory_bank() -> List[MemoryEntry]:
    """
    Create a full memory bank for comprehensive testing (all entries).

    Returns:
        List of all MemoryEntry objects
    """
    return ALL_MEMORY_ENTRIES.copy()
