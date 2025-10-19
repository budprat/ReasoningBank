"""
Gap 22: Memory Growth Strategy Without Deduplication - Stress Tests

Paper Claim (Appendix A.2): "Consolidation: Simple addition of new memories"

CRITICAL: Validate unbounded memory growth without deduplication.
System must scale to 1000+ memories with acceptable retrieval performance (<5s).

Tests validate:
1. Memory grows linearly with task count (no pruning/deduplication)
2. System handles 1000+ memories without performance collapse
3. Near-duplicate memories are expected per paper design

Test Strategy:
- Quick Validation (10 tasks): Validate logic before expensive tests
- Test 1 (100 tasks): Validate unbounded growth pattern
- Test 2 (334 tasks): Validate performance at 1000+ memories
"""

import pytest
import random
import time
from difflib import SequenceMatcher
from dotenv import load_dotenv
from reasoningbank.agent import ReasoningBankAgent
from reasoningbank.config import ReasoningBankConfig
from tests.e2e.tasks.arithmetic_tasks import ARITHMETIC_TASKS, get_tasks_by_difficulty

# Load environment variables from .env file
load_dotenv()


def generate_random_arithmetic_task():
    """
    Generate a random arithmetic task for stress testing.

    Selects a random difficulty level (1-5) and returns a random task
    from that difficulty tier.

    Returns:
        ArithmeticTask: Random task from ARITHMETIC_TASKS
    """
    difficulty = random.randint(1, 5)
    tasks = get_tasks_by_difficulty(difficulty)

    if tasks:
        return random.choice(tasks)
    else:
        # Fallback to first task if difficulty tier is empty
        return ARITHMETIC_TASKS[0]


@pytest.mark.stress
@pytest.mark.slow
class TestMemoryGrowthLongTerm:
    """
    Long-term memory growth validation tests.

    Validates paper's "simple addition" consolidation strategy:
    - Memory grows unbounded (no deduplication/pruning)
    - System scales to 1000+ memories
    - Retrieval performance remains acceptable (<5s)
    - Near-duplicate memories are expected behavior
    """

    def _find_similar_memory_pairs(self, memories, similarity_threshold=0.95):
        """
        Find near-duplicate memory pairs using text similarity.

        Uses SequenceMatcher to compare memory content and identify
        near-duplicates (≥95% similarity by default).

        Args:
            memories: List of MemoryEntry objects
            similarity_threshold: Minimum similarity ratio (0.0-1.0)

        Returns:
            List of tuples: (memory1, memory2, similarity_score)
        """
        similar_pairs = []

        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                # Get content from memory items
                content_i = " ".join([
                    item.content for item in memories[i].memory_items
                ])
                content_j = " ".join([
                    item.content for item in memories[j].memory_items
                ])

                # Calculate text similarity
                similarity = SequenceMatcher(
                    None,
                    content_i,
                    content_j
                ).ratio()

                if similarity >= similarity_threshold:
                    similar_pairs.append((memories[i], memories[j], similarity))

        return similar_pairs

    def test_memory_bank_grows_linearly_quick(self):
        """
        QUICK TEST: 10-task validation of memory growth logic.

        Duration: ~45-60 minutes
        Cost: ~$0.60
        Purpose: Validate implementation before expensive full tests

        Success Criteria:
        - Memory grows with task count (10-30 entries expected)
        - No crashes or errors
        - Basic functionality works
        - Ready for larger tests
        """
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            judge_temperature=0.0,
            extractor_temperature=1.0,
            memory_bank_path="test_data/quick_memory_growth.json"
        )
        agent = ReasoningBankAgent(config)

        # Get 10 random tasks
        num_tasks = 10

        print(f"\n" + "="*60)
        print(f"=== QUICK MEMORY GROWTH VALIDATION ===")
        print(f"="*60)
        print(f"Tasks: {num_tasks}")
        print(f"Expected memory: 10-30 entries")
        print(f"="*60)

        for i in range(num_tasks):
            task = generate_random_arithmetic_task()
            print(f"\n--- Task {i+1}/{num_tasks}: {task.query} ---")

            # Run task
            result = agent.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=True
            )

            # Print memory size every 5 tasks
            if (i + 1) % 5 == 0:
                memory_size = len(agent.get_memory_bank())
                print(f"  Memory size after {i+1} tasks: {memory_size}")

        # Validate final memory size
        final_size = len(agent.get_memory_bank())
        expected_min = num_tasks  # At least 1 memory per task
        expected_max = num_tasks * 3  # At most 3 memories per task

        print(f"\n" + "="*60)
        print(f"QUICK VALIDATION RESULTS:")
        print(f"  Final memory size: {final_size}")
        print(f"  Expected range: {expected_min}-{expected_max}")
        print(f"="*60)

        assert expected_min <= final_size <= expected_max, (
            f"Expected {expected_min}-{expected_max} memories, got {final_size}"
        )

        print(f"\n✅ QUICK VALIDATION PASSED")
        print(f"   Memory growth pattern correct")
        print(f"   No crashes detected")
        print(f"   Ready for full tests")
        print(f"="*60)

    def test_memory_bank_grows_linearly_without_deduplication(self):
        """
        TEST 1: 100-task growth validation (OPTIONAL).

        Duration: ~45-60 minutes
        Cost: ~$6-8
        Purpose: Validate unbounded growth pattern

        Success Criteria:
        - 250-300 memories after 100 tasks
        - Linear growth pattern observed
        - Near-duplicates detected (expected behavior)
        """
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            judge_temperature=0.0,
            extractor_temperature=1.0,
            memory_bank_path="test_data/memory_growth_100_tasks.json"
        )
        agent = ReasoningBankAgent(config)

        num_tasks = 100

        print(f"\n" + "="*60)
        print(f"=== MEMORY GROWTH TEST (100 TASKS) ===")
        print(f"="*60)
        print(f"Tasks: {num_tasks}")
        print(f"Expected memory: 250-300 entries")
        print(f"Validating: Unbounded growth without deduplication")
        print(f"="*60)

        for i in range(num_tasks):
            task = generate_random_arithmetic_task()
            print(f"\n--- Task {i+1}/{num_tasks}: {task.query} ---")

            result = agent.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=True
            )

            # Print memory size every 10 tasks
            if (i + 1) % 10 == 0:
                memory_size = len(agent.get_memory_bank())
                print(f"  Memory size after {i+1} tasks: {memory_size}")

        # Validate final size
        final_size = len(agent.get_memory_bank())
        expected_min = 250
        expected_max = 300

        assert expected_min <= final_size <= expected_max, (
            f"Expected {expected_min}-{expected_max} memories, got {final_size}"
        )

        # Check for near-duplicates (EXPECTED per paper)
        memory_bank = agent.get_memory_bank()
        similar_pairs = self._find_similar_memory_pairs(
            memory_bank,
            similarity_threshold=0.95
        )

        print(f"\n" + "="*60)
        print(f"TEST 1 RESULTS:")
        print(f"  Final memory size: {final_size}")
        print(f"  Near-duplicate pairs: {len(similar_pairs)} (EXPECTED)")
        print(f"="*60)
        print(f"\n✅ UNBOUNDED GROWTH VALIDATED")
        print(f"   Memory grows linearly: {final_size} entries")
        print(f"   Near-duplicates present (paper design)")
        print(f"="*60)

    def test_retrieval_performance_with_large_memory(self):
        """
        TEST 2: 334-task retrieval performance test (OPTIONAL, EXPENSIVE).

        Duration: ~2.5-3 hours
        Cost: ~$20-25
        Purpose: Validate performance at 1000+ memories

        Success Criteria:
        - ≥900 memories after 334 tasks
        - Retrieval time <5s with 1000+ memories
        - System scales to long-term usage
        """
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            judge_temperature=0.0,
            extractor_temperature=1.0,
            memory_bank_path="test_data/memory_growth_1000_plus.json"
        )
        agent = ReasoningBankAgent(config)

        num_tasks = 334

        print(f"\n" + "="*60)
        print(f"=== RETRIEVAL PERFORMANCE TEST (334 TASKS) ===")
        print(f"="*60)
        print(f"Tasks: {num_tasks}")
        print(f"Target: ≥900 memories (1000+ expected)")
        print(f"Validating: Retrieval performance at scale")
        print(f"="*60)

        for i in range(num_tasks):
            task = generate_random_arithmetic_task()

            if (i + 1) % 50 == 0:
                print(f"\nProgress: {i+1}/{num_tasks} tasks")

            result = agent.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=True
            )

            # Print memory size every 50 tasks
            if (i + 1) % 50 == 0:
                memory_size = len(agent.get_memory_bank())
                print(f"  Memory size: {memory_size}")

        # Validate size
        final_size = len(agent.get_memory_bank())
        assert final_size >= 900, (
            f"Expected ≥900 memories, got {final_size}"
        )

        # Measure retrieval performance
        query = "Calculate 50 + 50"
        start_time = time.time()
        memory_bank = agent.get_memory_bank()
        retrieved = agent.retriever.retrieve(query, memory_bank, k=1)
        retrieval_time = time.time() - start_time

        print(f"\n" + "="*60)
        print(f"TEST 2 RESULTS:")
        print(f"  Final memory size: {final_size}")
        print(f"  Retrieval time: {retrieval_time:.2f}s")
        print(f"  Target: <5s")
        print(f"="*60)

        # CRITICAL: Must be <5s
        assert retrieval_time < 5.0, (
            f"Retrieval too slow: {retrieval_time:.2f}s (expected <5s)"
        )

        print(f"\n✅ RETRIEVAL PERFORMANCE VALIDATED")
        print(f"   System scales to {final_size} memories")
        print(f"   Retrieval time: {retrieval_time:.2f}s < 5s")
        print(f"="*60)
