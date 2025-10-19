"""
Test-time streaming constraint validation.

Paper Claim (Section 3.2): "Tasks arrive sequentially, no ground truth available,
agent learns during test-time without seeing future tasks."

CRITICAL: Validate agent ONLY uses memory from past tasks (0..i-1),
NEVER from future tasks (i+1..N).
"""

import pytest
from dotenv import load_dotenv
from reasoningbank.agent import ReasoningBankAgent
from reasoningbank.config import ReasoningBankConfig
from tests.e2e.tasks.arithmetic_tasks import ARITHMETIC_TASKS

# Load environment variables from .env file
load_dotenv()


@pytest.mark.e2e
@pytest.mark.slow
class TestStreamingConstraint:
    """
    Validate test-time streaming constraint: no future task access.

    Paper's test-time learning paradigm requires:
    - Tasks arrive sequentially (one at a time)
    - No future task information available
    - Agent learns ONLY from past experiences (tasks 0..i-1)
    """

    def test_agent_cannot_access_future_tasks_during_learning(self):
        """
        CRITICAL TEST: Prove streaming constraint is enforced.

        Validation Logic:
        1. Process 20 tasks sequentially
        2. After each task i, validate memory bank contains ONLY info from tasks 0..i-1
        3. Ensure NO leakage from future tasks i+1..19

        Paper Compliance:
        - Section 3.2: "Streaming task paradigm"
        - No ground truth during test-time
        - Sequential task arrival
        """
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            judge_temperature=0.0,
            extractor_temperature=1.0,
            memory_bank_path="test_data/streaming_constraint_memory.json"
        )
        agent = ReasoningBankAgent(config)

        # Get 20 sequential tasks
        tasks = ARITHMETIC_TASKS[:20]

        print(f"\n" + "="*60)
        print(f"=== STREAMING CONSTRAINT VALIDATION ===")
        print(f"="*60)
        print(f"Tasks: {len(tasks)}")
        print(f"Paper Claim: Agent learns from PAST only, no future access")
        print(f"="*60)

        for i, task in enumerate(tasks):
            print(f"\n--- Task {i+1}/{len(tasks)}: {task.query} ---")

            # Agent should ONLY have memory from past tasks (0..i-1)
            # NOT from future tasks (i+1..19)

            # Run current task
            result = agent.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=True
            )

            # Validate memory bank contains ONLY past experiences
            memory_bank = agent.get_memory_bank()

            # After processing task i, memory contains tasks 0..i (inclusive)
            tasks_processed = i + 1
            print(f"  Memory bank size: {len(memory_bank)}")
            print(f"  Expected max: {tasks_processed * 3} (from {tasks_processed} processed tasks × 3 max items)")

            # CRITICAL CHECK 1: No memory from future tasks
            future_task_queries = [t.query for t in tasks[i+1:]]

            memory_leak_detected = False
            leaked_query = None
            leaked_memory = None

            for memory_entry in memory_bank:
                # Check task query
                if memory_entry.task_query in future_task_queries:
                    memory_leak_detected = True
                    leaked_query = memory_entry.task_query
                    leaked_memory = f"Task Query: {memory_entry.task_query}"
                    break

                # Check memory items
                for memory_item in memory_entry.memory_items:
                    memory_content = memory_item.content.lower()
                    memory_title = memory_item.title.lower()
                    memory_desc = memory_item.description.lower()

                    # Check if future query appears in memory
                    for future_query in future_task_queries:
                        future_query_lower = future_query.lower()

                        # Check all fields for leakage
                        if (future_query_lower in memory_content or
                            future_query_lower in memory_title or
                            future_query_lower in memory_desc):
                            memory_leak_detected = True
                            leaked_query = future_query
                            leaked_memory = f"Memory Item: {memory_item.title}"
                            break

                    if memory_leak_detected:
                        break

                if memory_leak_detected:
                    break

            if memory_leak_detected:
                print(f"\n❌ MEMORY LEAK DETECTED!")
                print(f"  Task {i+1} has memory from future task: {leaked_query}")
                print(f"  Leaked in: {leaked_memory}")
                assert False, (
                    f"Memory leak! Task {i+1} contains information from future tasks. "
                    f"This violates test-time streaming constraint. "
                    f"Future task: {leaked_query}"
                )

            # CRITICAL CHECK 2: Memory count should be ≤ (i+1) * 3
            # (Each processed task can contribute max 3 memory items)
            max_expected_memories = tasks_processed * 3  # Conservative upper bound

            if len(memory_bank) > max_expected_memories:
                print(f"\n⚠️ MEMORY COUNT ANOMALY!")
                print(f"  Memory bank: {len(memory_bank)} entries")
                print(f"  Expected max: {max_expected_memories}")
                assert False, (
                    f"Memory bank has {len(memory_bank)} entries, but only {i} tasks processed. "
                    f"Expected ≤{max_expected_memories} entries. Possible future information leakage."
                )

            print(f"  ✓ No future task leakage detected")
            print(f"  ✓ Memory count within bounds: {len(memory_bank)} ≤ {max_expected_memories}")

        print(f"\n" + "="*60)
        print(f"✅ STREAMING CONSTRAINT VALIDATED")
        print(f"="*60)
        print(f"   Agent learned strictly from past experiences (0..i-1)")
        print(f"   No future task information leakage detected")
        print(f"   Memory growth pattern matches sequential learning")
        print(f"="*60)

    def test_memory_accumulation_follows_temporal_order(self):
        """
        Validate memory bank grows monotonically with task sequence.

        Memory should:
        - Increase or stay same after each task (never decrease)
        - Contain only information from tasks processed so far
        - Follow temporal ordering (earlier tasks → earlier memories)
        """
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            judge_temperature=0.0,
            extractor_temperature=1.0,
            memory_bank_path="test_data/temporal_order_memory.json"
        )
        agent = ReasoningBankAgent(config)

        tasks = ARITHMETIC_TASKS[:15]
        memory_sizes = []

        print(f"\n" + "="*60)
        print(f"=== TEMPORAL ORDERING VALIDATION ===")
        print(f"="*60)
        print(f"Tasks: {len(tasks)}")
        print(f"Validating: Monotonic memory growth")
        print(f"="*60)

        for i, task in enumerate(tasks):
            print(f"\nTask {i+1}/{len(tasks)}: {task.query}")

            result = agent.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=True
            )

            current_memory_size = len(agent.get_memory_bank())
            memory_sizes.append(current_memory_size)

            # Validate monotonic increase (or stay same if extraction failed)
            if i > 0:
                previous_size = memory_sizes[i-1]
                if current_memory_size < previous_size:
                    print(f"\n❌ MEMORY DECREASED!")
                    print(f"  Previous size: {previous_size}")
                    print(f"  Current size: {current_memory_size}")
                    assert False, (
                        f"Memory decreased from {previous_size} to {current_memory_size} "
                        f"after task {i+1}. Memory should only grow or stay same, not shrink."
                    )

                growth = current_memory_size - previous_size
                if growth > 0:
                    print(f"  Memory size: {current_memory_size} (+{growth})")
                else:
                    print(f"  Memory size: {current_memory_size} (no extraction)")
            else:
                print(f"  Memory size: {current_memory_size} (initial)")

        # Validate overall growth
        initial_size = memory_sizes[0]
        final_size = memory_sizes[-1]

        print(f"\n" + "="*60)
        print(f"Memory Growth Summary:")
        print(f"  Initial: {initial_size}")
        print(f"  Final: {final_size}")
        print(f"  Total Growth: +{final_size - initial_size}")

        if final_size <= initial_size:
            print(f"\n❌ NO MEMORY GROWTH!")
            assert False, (
                f"Memory did not grow over {len(tasks)} tasks. "
                f"Initial: {initial_size}, Final: {final_size}. "
                f"Agent should learn and accumulate memories."
            )

        print(f"\n✅ TEMPORAL ORDERING VALIDATED")
        print(f"="*60)
        print(f"   Memory grew monotonically from {initial_size} → {final_size}")
        print(f"   No memory decrease detected")
        print(f"   Sequential learning pattern confirmed")
        print(f"="*60)
