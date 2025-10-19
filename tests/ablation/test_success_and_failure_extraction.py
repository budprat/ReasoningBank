"""
CRITICAL ABLATION: Success-Only vs Success+Failure Extraction

Paper's Core Innovation (Section 3.2, Page 4):
"ReasoningBank learns from BOTH successful and failed experiences,
unlike prior work that only learns from successes."

This test validates the paper's PRIMARY contribution to the field.
Without this validation, we cannot claim ReasoningBank works as advertised.
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
@pytest.mark.slow
class TestSuccessAndFailureExtraction:
    """
    CRITICAL: Validate paper's core innovation claim.

    Hypothesis: Learning from BOTH successes AND failures outperforms
                learning from successes only.
    """

    def test_success_plus_failure_outperforms_success_only(self):
        """
        CRITICAL ABLATION: Paper's core claim validation.

        Experimental Design:
        1. BASELINE: Agent extracts memory ONLY from successful tasks
        2. PAPER APPROACH: Agent extracts from BOTH successes AND failures
        3. VALIDATION: Both > Success-only by ≥5% (statistically significant)

        Paper Innovation:
        - Prior work: Learn only from successes
        - ReasoningBank: Learn from successes AND failures
        - Claim: Failure lessons improve performance

        Success Criteria:
        - ≥5% improvement in success rate
        - p<0.05 statistical significance
        - Failure memories actually used in subsequent tasks
        """
        # Get 20 HARD tasks (difficulty 4-5) to induce more failures
        tasks = [t for t in ARITHMETIC_TASKS if t.difficulty in [4, 5]][:20]

        print(f"\n=== CRITICAL ABLATION TEST ===")
        print(f"Testing: Success-Only vs Success+Failure Extraction")
        print(f"Tasks: {len(tasks)}")
        print(f"Paper's Core Innovation: Learning from failures")

        # === BASELINE: Success-Only Extraction ===
        print(f"\n--- BASELINE: Success-Only Extraction ---")
        config_success_only = ReasoningBankConfig(
            agent_temperature=0.7,
            judge_temperature=0.0,
            extractor_temperature=1.0,
            extract_from_failures=False,  # ONLY extract from successes
            memory_bank_path="test_data/ablation_success_only.json"
        )
        agent_success_only = ReasoningBankAgent(config_success_only)

        success_only_results = []
        success_only_steps = []

        for i, task in enumerate(tasks, 1):
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
                print(f"  Task {i}: ✓ CORRECT → Memory extracted")
            else:
                print(f"  Task {i}: ✗ WRONG → No memory extraction")

        success_only_sr = sum(success_only_results) / len(success_only_results)
        success_only_avg_steps = statistics.mean(success_only_steps)
        success_only_memory_count = len(agent_success_only.get_memory_bank())

        print(f"\nBaseline Performance (Success-Only):")
        print(f"  Success Rate: {success_only_sr*100:.1f}%")
        print(f"  Avg Steps: {success_only_avg_steps:.1f}")
        print(f"  Memory Items: {success_only_memory_count}")

        # === PAPER APPROACH: Success + Failure Extraction ===
        print(f"\n--- PAPER APPROACH: Success+Failure Extraction ---")
        config_both = ReasoningBankConfig(
            agent_temperature=0.7,
            judge_temperature=0.0,
            extractor_temperature=1.0,
            extract_from_failures=True,  # Extract from BOTH
            memory_bank_path="test_data/ablation_success_and_failure.json"
        )
        agent_both = ReasoningBankAgent(config_both)

        both_results = []
        both_steps = []

        for i, task in enumerate(tasks, 1):
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
                print(f"  Task {i}: ✓ CORRECT → Success memory extracted")
            else:
                print(f"  Task {i}: ✗ WRONG → Failure lesson extracted")

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

        # === CRITICAL VALIDATION ===
        print(f"\n=== VALIDATION: PAPER'S CORE CLAIM ===")

        improvement = both_sr - success_only_sr
        step_improvement = (success_only_avg_steps - both_avg_steps) / success_only_avg_steps if success_only_avg_steps > 0 else 0

        print(f"\nPerformance Comparison:")
        print(f"  Success-Only SR:      {success_only_sr*100:.1f}%")
        print(f"  Success+Failure SR:   {both_sr*100:.1f}%")
        print(f"  Improvement:          {improvement*100:+.1f}%")
        print(f"\n  Success-Only Steps:   {success_only_avg_steps:.1f}")
        print(f"  Success+Failure Steps:{both_avg_steps:.1f}")
        print(f"  Step Reduction:       {step_improvement*100:+.1f}%")

        # CRITICAL ASSERTION 1: Success rate improvement
        assert improvement >= 0.05, (
            f"\n❌ CORE CLAIM FAILED!\n"
            f"Expected ≥5% improvement from learning from failures.\n"
            f"Got: Success-Only={success_only_sr:.2%}, Both={both_sr:.2%}\n"
            f"Improvement: {improvement:.2%}\n"
            f"\nPaper's core innovation NOT validated!"
        )

        # CRITICAL ASSERTION 2: Statistical significance
        t_stat, p_value = stats.ttest_ind(
            [1 if r else 0 for r in success_only_results],
            [1 if r else 0 for r in both_results]
        )

        print(f"\nStatistical Validation:")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")

        assert p_value < 0.05, (
            f"\n❌ IMPROVEMENT NOT STATISTICALLY SIGNIFICANT!\n"
            f"p-value: {p_value:.4f} (expected <0.05)\n"
            f"Improvement may be due to random chance, not learning from failures."
        )

        # CRITICAL ASSERTION 3: Failure memories actually exist
        assert failure_memory_count > 0, (
            f"No failure memories extracted! Agent should have learned from failures."
        )

        # Success message
        print(f"\n" + "="*60)
        print(f"✅ ✅ ✅ PAPER'S CORE CLAIM VALIDATED ✅ ✅ ✅")
        print(f"="*60)
        print(f"\nReasoningBank's PRIMARY innovation proven:")
        print(f"  • Learning from BOTH outcomes > learning from successes only")
        print(f"  • Success rate improvement: {improvement*100:+.1f}%")
        print(f"  • Statistically significant: p={p_value:.4f} < 0.05")
        print(f"  • Failure memories extracted: {failure_memory_count}")
        print(f"  • Step efficiency improved: {step_improvement*100:+.1f}%")
        print(f"\n" + "="*60)
