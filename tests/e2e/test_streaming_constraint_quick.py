"""
Quick streaming constraint validation (5 tasks instead of 20).

Use this to verify implementation before running full expensive test.
"""

import pytest
from dotenv import load_dotenv
from reasoningbank.agent import ReasoningBankAgent
from reasoningbank.config import ReasoningBankConfig
from tests.e2e.tasks.arithmetic_tasks import ARITHMETIC_TASKS

# Load environment variables from .env file
load_dotenv()


@pytest.mark.e2e
@pytest.mark.quick
class TestStreamingConstraintQuick:
    """Quick validation of streaming constraint with 5 tasks"""

    def test_agent_cannot_access_future_tasks_quick(self):
        """
        QUICK TEST: Validate streaming constraint with 5 tasks.

        This is a reduced version for rapid validation before running
        the full 20-task test.
        """
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            judge_temperature=0.0,
            extractor_temperature=1.0,
            memory_bank_path="test_data/quick_streaming_constraint.json"
        )
        agent = ReasoningBankAgent(config)

        # Get 5 tasks for quick validation
        tasks = ARITHMETIC_TASKS[:5]

        print(f"\n" + "="*60)
        print(f"=== QUICK STREAMING CONSTRAINT VALIDATION ===")
        print(f"="*60)
        print(f"Tasks: {len(tasks)} (reduced from 20 for quick test)")
        print(f"="*60)

        for i, task in enumerate(tasks):
            print(f"\n--- Task {i+1}/{len(tasks)}: {task.query} ---")

            # Run current task
            result = agent.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=True
            )

            # Validate memory bank
            memory_bank = agent.get_memory_bank()
            tasks_processed = i + 1

            print(f"  Memory bank size: {len(memory_bank)}")
            print(f"  Expected max: {tasks_processed * 3}")

            # CHECK 1: No future task queries in memory
            future_task_queries = [t.query for t in tasks[i+1:]]

            memory_leak_detected = False
            for memory_entry in memory_bank:
                # Check if task query is from future
                if memory_entry.task_query in future_task_queries:
                    print(f"\n❌ MEMORY LEAK: Future task query in memory!")
                    print(f"  Future task: {memory_entry.task_query}")
                    memory_leak_detected = True
                    break

            assert not memory_leak_detected, "Memory leak detected!"

            # CHECK 2: Memory count within bounds
            max_expected = tasks_processed * 3
            assert len(memory_bank) <= max_expected, (
                f"Memory count {len(memory_bank)} > expected {max_expected}"
            )

            print(f"  ✓ No future leakage")
            print(f"  ✓ Memory count OK: {len(memory_bank)} ≤ {max_expected}")

        print(f"\n" + "="*60)
        print(f"✅ QUICK VALIDATION PASSED")
        print(f"="*60)
        print(f"   No future task leakage in {len(tasks)} tasks")
        print(f"   Memory growth pattern correct")
        print(f"   Ready for full 20-task test")
        print(f"="*60)

    def test_memory_grows_monotonically_quick(self):
        """Quick validation of monotonic memory growth with 5 tasks"""
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            memory_bank_path="test_data/quick_temporal_order.json"
        )
        agent = ReasoningBankAgent(config)

        tasks = ARITHMETIC_TASKS[:5]
        memory_sizes = []

        print(f"\n" + "="*60)
        print(f"=== QUICK TEMPORAL ORDERING VALIDATION ===")
        print(f"="*60)

        for i, task in enumerate(tasks):
            print(f"\nTask {i+1}/{len(tasks)}: {task.query}")

            result = agent.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=True
            )

            current_size = len(agent.get_memory_bank())
            memory_sizes.append(current_size)

            if i > 0:
                previous_size = memory_sizes[i-1]
                assert current_size >= previous_size, (
                    f"Memory decreased: {previous_size} → {current_size}"
                )
                growth = current_size - previous_size
                if growth > 0:
                    print(f"  Memory: {current_size} (+{growth})")
                else:
                    print(f"  Memory: {current_size} (no extraction)")
            else:
                print(f"  Memory: {current_size} (initial)")

        initial = memory_sizes[0]
        final = memory_sizes[-1]

        print(f"\n" + "="*60)
        print(f"Memory: {initial} → {final} (+{final - initial})")

        assert final > initial, "Memory must grow over tasks"

        print(f"\n✅ MONOTONIC GROWTH VALIDATED")
        print(f"="*60)
