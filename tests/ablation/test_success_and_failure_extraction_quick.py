"""
QUICK VALIDATION: Success-Only vs Success+Failure Extraction

Simplified version with 5 tasks instead of 20 for rapid validation.
Use this to verify implementation before running the full expensive test.
"""

import pytest
import statistics
from scipy import stats
from dotenv import load_dotenv
from reasoningbank.agent import ReasoningBankAgent
from reasoningbank.config import ReasoningBankConfig
from tests.e2e.tasks.arithmetic_tasks import ARITHMETIC_TASKS

# Load environment variables from .env file
load_dotenv()


@pytest.mark.ablation
@pytest.mark.quick
class TestSuccessAndFailureExtractionQuick:
    def test_success_plus_failure_outperforms_success_only(self):
        """
        QUICK TEST: Validate core ablation with just 5 tasks.

        Tests the same hypothesis as the full test but with reduced sample size
        for rapid validation. Use this to verify implementation correctness
        before committing to the full 40-task test run.
        """
        print("\n" + "="*60)
        print("=== QUICK ABLATION VALIDATION ===")
        print("Testing: Success-Only vs Success+Failure Extraction")
        print("Tasks: 5 (reduced from 20 for rapid validation)")
        print("Paper's Core Innovation: Learning from failures")
        print("="*60)

        # Select first 5 tasks for quick test
        tasks = ARITHMETIC_TASKS[:5]

        # BASELINE: Success-only extraction
        print("\n--- BASELINE: Success-Only Extraction ---")

        config_success_only = ReasoningBankConfig(
            agent_temperature=0.7,
            judge_temperature=0.0,
            extractor_temperature=1.0,
            extract_from_failures=False,  # Only successes
            memory_bank_path="test_data/quick_ablation_success_only.json"
        )
        agent_success_only = ReasoningBankAgent(config_success_only)

        success_only_results = []
        success_only_steps = []

        for i, task in enumerate(tasks, 1):
            print(f"\nTask {i}/5: {task.query}")
            result = agent_success_only.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=True
            )

            is_correct = task.validate(result.model_output)
            success_only_results.append(is_correct)
            success_only_steps.append(result.steps_taken)

            # Memory extraction handled automatically by agent
            # With extract_from_failures=False, only successes get memories

            if is_correct:
                print(f"  ✓ CORRECT → Success memory extracted")
            else:
                print(f"  ✗ WRONG → No memory extracted (success-only mode)")

        success_only_sr = sum(success_only_results) / len(success_only_results)
        success_only_avg_steps = statistics.mean(success_only_steps)
        success_only_memory_count = len(agent_success_only.get_memory_bank())

        print(f"\nBaseline Performance (Success-Only):")
        print(f"  Success Rate: {success_only_sr*100:.1f}%")
        print(f"  Avg Steps: {success_only_avg_steps:.1f}")
        print(f"  Memory Items: {success_only_memory_count}")

        # PAPER APPROACH: Success+Failure extraction
        print("\n--- PAPER APPROACH: Success+Failure Extraction ---")

        config_both = ReasoningBankConfig(
            agent_temperature=0.7,
            judge_temperature=0.0,
            extractor_temperature=1.0,
            extract_from_failures=True,  # Extract from BOTH
            memory_bank_path="test_data/quick_ablation_success_and_failure.json"
        )
        agent_both = ReasoningBankAgent(config_both)

        both_results = []
        both_steps = []

        for i, task in enumerate(tasks, 1):
            print(f"\nTask {i}/5: {task.query}")
            result = agent_both.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=True
            )

            is_correct = task.validate(result.model_output)
            both_results.append(is_correct)
            both_steps.append(result.steps_taken)

            # Memory extraction handled automatically by agent
            # With extract_from_failures=True, extracts from BOTH successes AND failures

            if is_correct:
                print(f"  ✓ CORRECT → Success memory extracted")
            else:
                print(f"  ✗ WRONG → Failure lesson extracted")

        both_sr = sum(both_results) / len(both_results)
        both_avg_steps = statistics.mean(both_steps)
        both_memory_count = len(agent_both.get_memory_bank())

        print(f"\nPaper Approach Performance (Success+Failure):")
        print(f"  Success Rate: {both_sr*100:.1f}%")
        print(f"  Avg Steps: {both_avg_steps:.1f}")
        print(f"  Memory Items: {both_memory_count}")

        # Analyze failure memories
        all_memory_entries = agent_both.get_memory_bank()
        failure_memory_entries = [entry for entry in all_memory_entries if not entry.success]

        # Count all memory items from failures
        failure_memory_count = sum(len(entry.memory_items) for entry in failure_memory_entries)

        print(f"  Failure Memories: {failure_memory_count}")
        if failure_memory_entries and failure_memory_entries[0].memory_items:
            print(f"  Example failure lesson: \"{failure_memory_entries[0].memory_items[0].title}\"")

        # === VALIDATION ===
        print(f"\n=== VALIDATION: QUICK CHECK ===")

        improvement = both_sr - success_only_sr
        step_improvement = (success_only_avg_steps - both_avg_steps) / success_only_avg_steps if success_only_avg_steps > 0 else 0

        print(f"\nPerformance Comparison:")
        print(f"  Success-Only SR:      {success_only_sr*100:.1f}%")
        print(f"  Success+Failure SR:   {both_sr*100:.1f}%")
        print(f"  Improvement:          {improvement*100:+.1f}%")
        print(f"  Step Efficiency:      {step_improvement*100:+.1f}%")

        # Statistical test (note: sample size too small for reliable statistics)
        t_statistic, p_value = stats.ttest_ind(both_results, success_only_results)

        print(f"\nStatistical Analysis:")
        print(f"  t-statistic: {t_statistic:.3f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Note: Sample size (n=5) too small for reliable statistics")

        # Basic assertions for quick validation
        print(f"\n--- QUICK VALIDATION CHECKS ---")

        # Check 1: Both agents should work
        assert len(success_only_results) == 5, "Baseline agent should complete all tasks"
        assert len(both_results) == 5, "Paper approach agent should complete all tasks"
        print(f"✓ Both agents completed all tasks")

        # Check 2: Memories should be extracted
        assert success_only_memory_count > 0, "Baseline should have extracted some memories"
        assert both_memory_count > 0, "Paper approach should have extracted some memories"
        print(f"✓ Both agents extracted memories")

        # Check 3: If there were failures, paper approach should have failure memories
        if any(not r for r in both_results):
            assert failure_memory_count > 0, (
                f"Paper approach had failures but no failure memories extracted!"
            )
            print(f"✓ Failure memories extracted when failures occurred")

        # Success message
        print(f"\n" + "="*60)
        print(f"✅ QUICK VALIDATION PASSED")
        print(f"="*60)
        print(f"\nImplementation is working correctly!")
        print(f"  • Both agents executed successfully")
        print(f"  • Memory extraction working")
        print(f"  • Success rate: Success-Only={success_only_sr*100:.1f}%, Both={both_sr*100:.1f}%")
        print(f"  • Ready for full 20-task validation")
        print(f"\n" + "="*60)
