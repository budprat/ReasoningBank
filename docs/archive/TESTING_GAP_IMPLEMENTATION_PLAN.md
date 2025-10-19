# ReasoningBank Testing Gap Implementation Plan
**ABOUTME: Comprehensive plan to address all testing gaps identified in ultrathink analysis**
**ABOUTME: No shortcuts, no mocks for validation - only real end-to-end testing**

Generated: 2025-01-XX
Status: DRAFT - Ready for Implementation

---

## Executive Summary

This plan addresses **24 critical testing gaps** (20 original + 4 from cross-check) identified in the comprehensive ultrathink analysis of ReasoningBank test coverage. The core issue: we validated that components work (mechanism) but didn't prove agents self-evolve (outcome).

**CRITICAL UPDATE (from Cross-Check Analysis)**:
Systematic cross-check against ReasoningBank.pdf revealed **4 additional critical gaps** that must be addressed to fully validate the paper's claims:
- **Gap 21** (P0): Test-time streaming constraint validation
- **Gap 22** (P1): Memory growth without deduplication (1000+ memories)
- **Gap 23** (P2): Deterministic tie-breaking for reproducibility
- **Gap 24** (P0): **Success+Failure ablation - validates PAPER'S PRIMARY INNOVATION**

Most critical: **Gap 24 proves that learning from BOTH successes AND failures outperforms learning from successes only** - this is ReasoningBank's core contribution to the field.

**Key Principle**: NO SHORTCUTS
- No mock data for validation tests
- Real LLM calls for end-to-end tests
- Actual task complexity matching paper benchmarks
- Real memory accumulation and retrieval
- Genuine performance measurement

**Timeline**: 3 weeks + 2 days (17 working days)
**Effort**: ~160 hours implementation + validation (includes 4 new gaps from cross-check)

---

## Part 1: Critical Gaps (P0) - Week 1

### Gap 1: Self-Evolution Not Proven 🚨 CRITICAL

**Current State**: Tests mock that agents improve; never prove actual improvement
**Target**: Prove agents reduce steps by 20%+ and increase success rate by 15%+ over 20+ similar tasks

#### Implementation Tasks:

**1.1 Create Real Task Suite for Self-Evolution Testing**
```
File: tests/e2e/tasks/arithmetic_tasks.py
Purpose: 50 progressively complex arithmetic problems
- No mocking - real calculations required
- Difficulty progression: simple (10+5) → complex (fibonacci, factorials)
- Success validation: actual computation correctness
```

**Implementation Details**:
```python
# tests/e2e/tasks/arithmetic_tasks.py
"""Real arithmetic tasks for self-evolution testing"""

class ArithmeticTask:
    def __init__(self, query: str, expected_answer: int, difficulty: int):
        self.query = query
        self.expected_answer = expected_answer
        self.difficulty = difficulty  # 1-10 scale

    def validate(self, agent_answer: str) -> bool:
        """Validate agent's answer against expected result"""
        # Extract number from agent answer
        import re
        numbers = re.findall(r'-?\d+', agent_answer)
        if not numbers:
            return False
        return int(numbers[-1]) == self.expected_answer

# Progressive difficulty tasks
ARITHMETIC_TASKS = [
    # Level 1: Basic addition (5 tasks)
    ArithmeticTask("Calculate 10 + 5", 15, difficulty=1),
    ArithmeticTask("Calculate 23 + 17", 40, difficulty=1),
    ArithmeticTask("Calculate 45 + 38", 83, difficulty=1),
    ArithmeticTask("Calculate 67 + 89", 156, difficulty=1),
    ArithmeticTask("Calculate 102 + 98", 200, difficulty=1),

    # Level 2: Multiplication (5 tasks)
    ArithmeticTask("Calculate 12 * 8", 96, difficulty=2),
    ArithmeticTask("Calculate 15 * 7", 105, difficulty=2),
    ArithmeticTask("Calculate 23 * 4", 92, difficulty=2),
    ArithmeticTask("Calculate 31 * 6", 186, difficulty=2),
    ArithmeticTask("Calculate 42 * 9", 378, difficulty=2),

    # Level 3: Multi-step (10 tasks)
    ArithmeticTask("Calculate (10 + 5) * 3", 45, difficulty=3),
    ArithmeticTask("Calculate 100 - (20 * 3)", 40, difficulty=3),
    ArithmeticTask("Calculate (45 + 15) / 4", 15, difficulty=3),
    ArithmeticTask("Calculate 200 - (50 + 30) * 2", 40, difficulty=3),
    ArithmeticTask("Calculate (12 * 3) + (8 * 5)", 76, difficulty=3),
    ArithmeticTask("Calculate 1000 / (10 + 15 - 5)", 50, difficulty=3),
    ArithmeticTask("Calculate ((20 + 10) * 2) - 15", 45, difficulty=3),
    ArithmeticTask("Calculate (100 / 5) + (30 * 2)", 80, difficulty=3),
    ArithmeticTask("Calculate 500 - ((20 + 30) * 8)", 100, difficulty=3),
    ArithmeticTask("Calculate (15 * 4) - (10 * 3)", 30, difficulty=3),

    # Level 4: Complex expressions (10 tasks)
    ArithmeticTask("Calculate the 5th Fibonacci number", 5, difficulty=4),
    ArithmeticTask("Calculate the 7th Fibonacci number", 13, difficulty=4),
    ArithmeticTask("Calculate factorial of 5", 120, difficulty=4),
    ArithmeticTask("Calculate factorial of 6", 720, difficulty=4),
    ArithmeticTask("Calculate sum of numbers 1 to 10", 55, difficulty=4),
    ArithmeticTask("Calculate sum of even numbers 1 to 20", 110, difficulty=4),
    ArithmeticTask("Calculate 2^8", 256, difficulty=4),
    ArithmeticTask("Calculate 3^5", 243, difficulty=4),
    ArithmeticTask("Calculate greatest common divisor of 48 and 18", 6, difficulty=4),
    ArithmeticTask("Calculate least common multiple of 12 and 8", 24, difficulty=4),

    # Level 5: Word problems (20 tasks)
    ArithmeticTask("If 3 apples cost $6, how much do 7 apples cost?", 14, difficulty=5),
    ArithmeticTask("A car travels 60 miles in 1 hour. How far in 3.5 hours?", 210, difficulty=5),
    ArithmeticTask("If 5 workers complete a task in 8 hours, how long for 10 workers?", 4, difficulty=5),
    ArithmeticTask("A rectangle has width 12 and length 15. What's the area?", 180, difficulty=5),
    ArithmeticTask("Circle with radius 7. What's the area? (use π≈3.14)", 154, difficulty=5),
    # ... 15 more word problems
]

def get_tasks_by_difficulty(difficulty: int) -> List[ArithmeticTask]:
    """Get all tasks of specific difficulty level"""
    return [task for task in ARITHMETIC_TASKS if task.difficulty == difficulty]

def get_progressive_task_sequence() -> List[ArithmeticTask]:
    """Get tasks in progressive difficulty order for learning"""
    return sorted(ARITHMETIC_TASKS, key=lambda t: t.difficulty)
```

**1.2 Create Self-Evolution Validation Test**
```
File: tests/e2e/test_self_evolution.py
Purpose: Prove agents improve over time with real tasks
- Real LLM calls (no mocking)
- Real memory accumulation
- Real performance metrics
- Statistical significance validation
```

**Implementation Details**:
```python
# tests/e2e/test_self_evolution.py
"""
End-to-end tests proving agent self-evolution through closed-loop learning.

CRITICAL: NO MOCKS - Real LLM calls, real memory, real performance measurement.
Paper Claim (Abstract): "Agents improve performance over time through accumulated memory"
"""

import pytest
import statistics
from reasoningbank.agent import ReasoningBankAgent
from reasoningbank.config import ReasoningBankConfig
from tests.e2e.tasks.arithmetic_tasks import ARITHMETIC_TASKS, get_progressive_task_sequence

@pytest.mark.e2e
@pytest.mark.slow  # Indicates real LLM calls
class TestSelfEvolutionRealTasks:
    """
    Validate self-evolution with real tasks (no mocking).

    Paper Validation:
    - Abstract: "Agents improve performance over time"
    - Figure 3 (p7): Success rate improves from 30% → 50% over 200 tasks
    - Section 5: Efficiency improves (fewer steps) with experience
    """

    def test_agent_improves_arithmetic_performance_over_20_tasks(self):
        """
        CRITICAL TEST: Prove agent improves on arithmetic through real learning.

        Validation Criteria (from paper):
        1. Success rate increases by ≥15% (baseline → experienced)
        2. Steps taken decreases by ≥20% (baseline → experienced)
        3. Memory bank quality improves (measured by strategy sophistication)

        NO MOCKING: Real LLM calls, real memory, real calculations.
        """
        # Setup with real API credentials
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            judge_temperature=0.0,
            extractor_temperature=1.0,
            max_steps_per_task=30,
            top_k_retrieval=1,
            memory_bank_path="test_data/self_evolution_memory_bank.json"
        )
        agent = ReasoningBankAgent(config)

        # Get 20 arithmetic tasks (progressive difficulty)
        tasks = get_progressive_task_sequence()[:20]

        # Phase 1: Baseline (first 5 tasks, NO memory injection)
        print("\n=== PHASE 1: BASELINE (No Memory) ===")
        baseline_results = []
        baseline_steps = []
        baseline_successes = []

        for i, task in enumerate(tasks[:5], 1):
            print(f"\nBaseline Task {i}/{5}: {task.query}")
            result = agent.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=False  # NO MEMORY for baseline
            )

            # Validate actual correctness (no mocking!)
            is_correct = task.validate(result.model_output)
            baseline_successes.append(is_correct)
            baseline_steps.append(result.steps_taken)
            baseline_results.append({
                'task': task.query,
                'correct': is_correct,
                'steps': result.steps_taken,
                'model_output': result.model_output
            })

            print(f"  Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            print(f"  Steps: {result.steps_taken}")
            print(f"  Output: {result.model_output}")

        baseline_success_rate = sum(baseline_successes) / len(baseline_successes)
        baseline_avg_steps = statistics.mean(baseline_steps)

        print(f"\nBaseline Performance:")
        print(f"  Success Rate: {baseline_success_rate*100:.1f}%")
        print(f"  Avg Steps: {baseline_avg_steps:.1f}")

        # Phase 2: With Memory (tasks 6-20, WITH memory injection)
        print("\n=== PHASE 2: WITH ACCUMULATED MEMORY ===")
        experienced_results = []
        experienced_steps = []
        experienced_successes = []

        for i, task in enumerate(tasks[5:20], 6):
            print(f"\nExperienced Task {i}/{20}: {task.query}")
            result = agent.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=True  # USE MEMORY
            )

            # Validate actual correctness
            is_correct = task.validate(result.model_output)
            experienced_successes.append(is_correct)
            experienced_steps.append(result.steps_taken)
            experienced_results.append({
                'task': task.query,
                'correct': is_correct,
                'steps': result.steps_taken,
                'model_output': result.model_output,
                'memory_bank_size': len(agent.get_memory_bank())
            })

            print(f"  Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            print(f"  Steps: {result.steps_taken}")
            print(f"  Memory Bank Size: {len(agent.get_memory_bank())}")
            print(f"  Output: {result.model_output}")

        experienced_success_rate = sum(experienced_successes) / len(experienced_successes)
        experienced_avg_steps = statistics.mean(experienced_steps)

        print(f"\nExperienced Performance:")
        print(f"  Success Rate: {experienced_success_rate*100:.1f}%")
        print(f"  Avg Steps: {experienced_avg_steps:.1f}")

        # Phase 3: Statistical Validation (CRITICAL ASSERTIONS)
        print("\n=== PHASE 3: VALIDATION ===")

        # Calculate improvements
        success_improvement = experienced_success_rate - baseline_success_rate
        step_reduction_pct = (baseline_avg_steps - experienced_avg_steps) / baseline_avg_steps if baseline_avg_steps > 0 else 0

        print(f"\nImprovement Metrics:")
        print(f"  Success Rate Δ: {success_improvement*100:+.1f}% ({baseline_success_rate*100:.1f}% → {experienced_success_rate*100:.1f}%)")
        print(f"  Steps Reduction: {step_reduction_pct*100:+.1f}% ({baseline_avg_steps:.1f} → {experienced_avg_steps:.1f})")
        print(f"  Memory Bank: {len(agent.get_memory_bank())} entries")

        # CRITICAL ASSERTIONS (from paper claims)
        assert success_improvement >= 0.15, (
            f"Expected ≥15% success improvement, got {success_improvement*100:.1f}%. "
            f"Agent did NOT self-evolve as claimed in paper!"
        )

        assert step_reduction_pct >= 0.20, (
            f"Expected ≥20% step reduction, got {step_reduction_pct*100:.1f}%. "
            f"Agent did NOT become more efficient with experience!"
        )

        assert len(agent.get_memory_bank()) >= 15, (
            f"Expected memory accumulation, got only {len(agent.get_memory_bank())} entries"
        )

        # Validate experienced agent performs better than baseline
        # Using statistical significance test (t-test)
        from scipy import stats
        t_statistic, p_value = stats.ttest_ind(baseline_steps, experienced_steps)

        assert p_value < 0.05, (
            f"Step reduction not statistically significant (p={p_value:.3f}). "
            f"Improvement may be random chance!"
        )

        print(f"\n✅ SELF-EVOLUTION VALIDATED")
        print(f"   Agent improved significantly with experience (p<0.05)")
        print(f"   Success: +{success_improvement*100:.1f}% | Steps: -{step_reduction_pct*100:.1f}%")

    def test_agent_improves_across_50_tasks_progressive_learning(self):
        """
        Extended test: 50 tasks to match paper's learning curves (Figure 3).

        Paper shows: Success rate improves from 30% → 50% over 200 tasks
        Our test: Validate similar improvement trend over 50 tasks
        """
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            memory_bank_path="test_data/progressive_learning_bank.json"
        )
        agent = ReasoningBankAgent(config)

        tasks = get_progressive_task_sequence()[:50]

        # Track performance in windows
        window_size = 10
        window_success_rates = []
        window_avg_steps = []

        all_results = []

        for i, task in enumerate(tasks, 1):
            enable_memory = i > window_size  # Enable after first window

            result = agent.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=enable_memory
            )

            is_correct = task.validate(result.model_output)
            all_results.append({
                'task_num': i,
                'correct': is_correct,
                'steps': result.steps_taken,
                'memory_enabled': enable_memory,
                'memory_bank_size': len(agent.get_memory_bank())
            })

            # Calculate rolling window metrics
            if i >= window_size:
                window_results = all_results[i-window_size:i]
                window_success = sum(1 for r in window_results if r['correct']) / window_size
                window_steps = statistics.mean([r['steps'] for r in window_results])
                window_success_rates.append(window_success)
                window_avg_steps.append(window_steps)

                print(f"Task {i}/{50}: Success Window: {window_success*100:.1f}%, Avg Steps: {window_steps:.1f}, Bank: {len(agent.get_memory_bank())}")

        # Validate learning trend
        early_success = statistics.mean(window_success_rates[:3])  # Windows 10-30
        late_success = statistics.mean(window_success_rates[-3:])  # Windows 40-50

        early_steps = statistics.mean(window_avg_steps[:3])
        late_steps = statistics.mean(window_avg_steps[-3:])

        improvement = late_success - early_success
        efficiency_gain = (early_steps - late_steps) / early_steps

        print(f"\n=== Progressive Learning Results ===")
        print(f"Early Performance (tasks 10-30): {early_success*100:.1f}% success, {early_steps:.1f} steps")
        print(f"Late Performance (tasks 40-50): {late_success*100:.1f}% success, {late_steps:.1f} steps")
        print(f"Improvement: +{improvement*100:.1f}% success, -{efficiency_gain*100:.1f}% steps")

        # Assertions
        assert improvement >= 0.15, f"Expected ≥15% improvement over 50 tasks, got {improvement*100:.1f}%"
        assert efficiency_gain >= 0.15, f"Expected ≥15% efficiency gain, got {efficiency_gain*100:.1f}%"

        # Validate monotonic improvement trend (success rates should generally increase)
        # Use Spearman rank correlation
        from scipy.stats import spearmanr
        task_numbers = list(range(1, len(window_success_rates) + 1))
        correlation, p_value = spearmanr(task_numbers, window_success_rates)

        assert correlation > 0.3, f"Expected positive learning trend, got correlation={correlation:.2f}"
        assert p_value < 0.05, f"Learning trend not significant (p={p_value:.3f})"

        print(f"✅ Progressive learning validated: ρ={correlation:.2f}, p<0.05")
```

**1.3 Success Criteria**:
- ✅ 20-task test shows ≥15% success improvement
- ✅ 20-task test shows ≥20% step reduction
- ✅ 50-task test shows monotonic improvement trend
- ✅ Statistical significance (p<0.05) for all improvements
- ✅ No mocking—all improvements from real learning

---

### Gap 2: No Real Complex Multi-Step Reasoning Tests 🚨 CRITICAL

**Current State**: Tests use trivial tasks ("Calculate 25 * 4")
**Target**: Tests with 15-30 step reasoning matching paper benchmarks

#### Implementation Tasks:

**2.1 Create Complex Multi-Step Task Environments**
```
File: tests/e2e/environments/web_arena_simulator.py
Purpose: Simulate WebArena-style multi-step web navigation
```

**Implementation Details**:
```python
# tests/e2e/environments/web_arena_simulator.py
"""
WebArena-style environment simulator for complex multi-step reasoning.

Simulates real web interactions without actual browser:
- Shopping cart workflows (add item → apply coupon → checkout)
- GitLab issue workflows (create → assign → track → resolve)
- Multi-domain tasks (search product → read reviews → compare → purchase)

NO MOCKS for agent behavior - only environment state simulation.
"""

class WebArenaState:
    """Maintains environment state for web navigation tasks"""

    def __init__(self):
        self.current_page = "homepage"
        self.cart_items = []
        self.current_product = None
        self.coupon_applied = False
        self.gitlab_issues = {}
        self.search_results = []
        self.navigation_history = []

    def reset(self):
        """Reset environment to initial state"""
        self.__init__()

class WebArenaEnvironment:
    """
    WebArena-style environment for complex multi-step tasks.

    Based on Paper Section 4.1, Table 1:
    - Shopping tasks: 15-25 steps average
    - GitLab tasks: 10-20 steps average
    - Multi-domain: 20-30 steps average
    """

    def __init__(self):
        self.state = WebArenaState()
        self.step_count = 0

        # Product catalog
        self.products = {
            "laptop": {"name": "Gaming Laptop", "price": 999, "rating": 4.5},
            "mouse": {"name": "Wireless Mouse", "price": 29, "rating": 4.2},
            "keyboard": {"name": "Mechanical Keyboard", "price": 89, "rating": 4.7},
            "monitor": {"name": "4K Monitor", "price": 399, "rating": 4.6},
            "headphones": {"name": "Noise-Cancelling Headphones", "price": 199, "rating": 4.4}
        }

        # Available coupons
        self.coupons = {
            "SAVE10": 0.10,  # 10% off
            "SAVE20": 0.20,  # 20% off
            "FREESHIP": 0.0  # Free shipping
        }

    def execute_action(self, action: str) -> str:
        """
        Execute agent action and return observation.

        Supports actions:
        - navigate <page>: Navigate to page
        - search <query>: Search for products
        - select <product>: Select product
        - add_to_cart: Add current product to cart
        - apply_coupon <code>: Apply discount coupon
        - checkout: Complete purchase
        - view_cart: View cart contents
        - create_issue <title>: Create GitLab issue
        - assign_issue <id> <user>: Assign issue to user
        - close_issue <id>: Close issue
        """
        self.step_count += 1
        self.state.navigation_history.append(action)

        action = action.strip().lower()

        # Navigation actions
        if action.startswith("navigate"):
            page = action.split()[-1]
            self.state.current_page = page
            return f"Navigated to {page}. Current location: {page}"

        # Search actions
        elif action.startswith("search"):
            query = " ".join(action.split()[1:])
            self.state.search_results = [
                prod for name, prod in self.products.items()
                if query.lower() in name or query.lower() in prod["name"].lower()
            ]
            return f"Found {len(self.state.search_results)} results for '{query}': {[p['name'] for p in self.state.search_results]}"

        # Product selection
        elif action.startswith("select"):
            product_key = action.split()[-1]
            if product_key in self.products:
                self.state.current_product = self.products[product_key]
                return f"Selected: {self.state.current_product['name']} (${self.state.current_product['price']}, {self.state.current_product['rating']}★)"
            else:
                return f"Error: Product '{product_key}' not found"

        # Cart operations
        elif action == "add_to_cart":
            if self.state.current_product:
                self.state.cart_items.append(self.state.current_product)
                return f"Added {self.state.current_product['name']} to cart. Cart has {len(self.state.cart_items)} items."
            else:
                return "Error: No product selected"

        elif action == "view_cart":
            if not self.state.cart_items:
                return "Cart is empty"
            total = sum(item['price'] for item in self.state.cart_items)
            if self.state.coupon_applied:
                discount = total * 0.10  # Assuming SAVE10
                total -= discount
                return f"Cart: {[item['name'] for item in self.state.cart_items]}, Subtotal: ${total:.2f} (coupon applied)"
            return f"Cart: {[item['name'] for item in self.state.cart_items]}, Total: ${total:.2f}"

        elif action.startswith("apply_coupon"):
            code = action.split()[-1].upper()
            if code in self.coupons:
                self.state.coupon_applied = True
                return f"Coupon {code} applied successfully! {int(self.coupons[code]*100)}% discount"
            else:
                return f"Error: Invalid coupon code '{code}'"

        elif action == "checkout":
            if not self.state.cart_items:
                return "Error: Cannot checkout with empty cart"
            total = sum(item['price'] for item in self.state.cart_items)
            if self.state.coupon_applied:
                total *= 0.9  # 10% discount
            return f"CHECKOUT COMPLETE! Order total: ${total:.2f}. Thank you for your purchase!"

        # GitLab issue management
        elif action.startswith("create_issue"):
            title = " ".join(action.split()[1:])
            issue_id = len(self.state.gitlab_issues) + 1
            self.state.gitlab_issues[issue_id] = {
                "title": title,
                "status": "open",
                "assignee": None
            }
            return f"Created issue #{issue_id}: {title}"

        elif action.startswith("assign_issue"):
            parts = action.split()
            issue_id = int(parts[1])
            assignee = parts[2]
            if issue_id in self.state.gitlab_issues:
                self.state.gitlab_issues[issue_id]["assignee"] = assignee
                return f"Assigned issue #{issue_id} to {assignee}"
            return f"Error: Issue #{issue_id} not found"

        elif action.startswith("close_issue"):
            issue_id = int(action.split()[-1])
            if issue_id in self.state.gitlab_issues:
                self.state.gitlab_issues[issue_id]["status"] = "closed"
                return f"Closed issue #{issue_id}"
            return f"Error: Issue #{issue_id} not found"

        else:
            return f"Unknown action: {action}"

# Complex task definitions
COMPLEX_WEBARENA_TASKS = [
    {
        "name": "shopping_with_coupon",
        "query": "Find a gaming laptop, add it to cart, apply the SAVE10 coupon, and checkout",
        "expected_steps": 6,  # navigate → search → select → add_to_cart → apply_coupon → checkout
        "success_condition": lambda env: (
            "laptop" in [item.get("name", "").lower() for item in env.state.cart_items] and
            env.state.coupon_applied and
            "CHECKOUT COMPLETE" in env.state.navigation_history[-1] if env.state.navigation_history else False
        ),
        "difficulty": 3
    },
    {
        "name": "multi_item_shopping",
        "query": "Add a mouse and keyboard to cart, apply coupon SAVE20, then checkout",
        "expected_steps": 7,
        "success_condition": lambda env: (
            len(env.state.cart_items) >= 2 and
            env.state.coupon_applied and
            "CHECKOUT COMPLETE" in str(env.state.navigation_history)
        ),
        "difficulty": 3
    },
    {
        "name": "gitlab_issue_workflow",
        "query": "Create an issue titled 'Fix login bug', assign it to developer1, then close it",
        "expected_steps": 3,
        "success_condition": lambda env: (
            any(issue["title"] == "Fix login bug" and issue["status"] == "closed"
                for issue in env.state.gitlab_issues.values())
        ),
        "difficulty": 2
    },
    {
        "name": "product_comparison",
        "query": "Search for 'monitor', select the 4K Monitor, check its rating, if rating > 4.5 add to cart, otherwise search for 'headphones' and add those",
        "expected_steps": 8,
        "success_condition": lambda env: len(env.state.cart_items) > 0,
        "difficulty": 4
    }
]
```

**2.2 Create Complex Multi-Step Reasoning Test**
```
File: tests/e2e/test_complex_reasoning.py
Purpose: Validate agent handles 15-30 step reasoning tasks
```

**Implementation**:
```python
# tests/e2e/test_complex_reasoning.py
"""
Complex multi-step reasoning tests matching WebArena benchmark complexity.

Paper Section 4.1: WebArena tasks average 15-25 steps
Target: Validate agent completes 15-30 step reasoning chains
"""

import pytest
from reasoningbank.agent import ReasoningBankAgent
from reasoningbank.config import ReasoningBankConfig
from tests.e2e.environments.web_arena_simulator import (
    WebArenaEnvironment,
    COMPLEX_WEBARENA_TASKS
)

@pytest.mark.e2e
@pytest.mark.slow
class TestComplexMultiStepReasoning:
    """Validate agent handles complex 15-30 step reasoning tasks"""

    def test_agent_completes_shopping_workflow_with_coupon(self):
        """
        Complex 6-step shopping workflow.

        Steps:
        1. Navigate to products
        2. Search for laptop
        3. Select laptop
        4. Add to cart
        5. Apply SAVE10 coupon
        6. Checkout

        NO MOCKING: Real LLM decides each step
        """
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            max_steps_per_task=30
        )
        env = WebArenaEnvironment()
        agent = ReasoningBankAgent(config, environment=env.execute_action)

        task = COMPLEX_WEBARENA_TASKS[0]  # shopping_with_coupon

        result = agent.run(
            query=task["query"],
            max_steps=30,
            enable_memory_injection=False
        )

        # Validate success
        assert task["success_condition"](env), (
            f"Task failed: {task['name']}. "
            f"Cart: {env.state.cart_items}, "
            f"Coupon: {env.state.coupon_applied}, "
            f"History: {env.state.navigation_history}"
        )

        # Validate reasonable step count
        assert result.steps_taken <= task["expected_steps"] * 2, (
            f"Took too many steps: {result.steps_taken} (expected ≤{task['expected_steps']*2})"
        )

        print(f"✅ Completed {task['name']} in {result.steps_taken} steps")
        print(f"   Final cart: {[item['name'] for item in env.state.cart_items]}")
        print(f"   Coupon applied: {env.state.coupon_applied}")

    def test_agent_improves_on_complex_tasks_with_memory(self):
        """
        Prove memory helps on complex multi-step tasks.

        Run same complex task twice:
        1. First time: No memory (baseline)
        2. Second time: With memory from first attempt

        Validate: Second attempt uses fewer steps
        """
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            max_steps_per_task=30,
            memory_bank_path="test_data/complex_task_memory.json"
        )

        env1 = WebArenaEnvironment()
        agent = ReasoningBankAgent(config, environment=env1.execute_action)

        task = COMPLEX_WEBARENA_TASKS[0]

        # First attempt (no memory)
        result1 = agent.run(
            query=task["query"],
            max_steps=30,
            enable_memory_injection=False
        )

        # Second attempt (with memory)
        env2 = WebArenaEnvironment()
        agent.environment = env2.execute_action

        result2 = agent.run(
            query=task["query"],
            max_steps=30,
            enable_memory_injection=True
        )

        # Validate improvement
        assert result2.steps_taken < result1.steps_taken, (
            f"Expected step reduction with memory. "
            f"First: {result1.steps_taken}, Second: {result2.steps_taken}"
        )

        step_reduction = (result1.steps_taken - result2.steps_taken) / result1.steps_taken
        assert step_reduction >= 0.15, (
            f"Expected ≥15% step reduction, got {step_reduction*100:.1f}%"
        )

        print(f"✅ Memory reduced steps by {step_reduction*100:.1f}%")
        print(f"   Baseline: {result1.steps_taken} steps")
        print(f"   With memory: {result2.steps_taken} steps")

    @pytest.mark.parametrize("task", COMPLEX_WEBARENA_TASKS)
    def test_all_complex_tasks_completable(self, task):
        """
        Validate agent can complete all complex WebArena-style tasks.

        Tests shopping, GitLab, and multi-domain workflows.
        """
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            max_steps_per_task=30
        )
        env = WebArenaEnvironment()
        agent = ReasoningBankAgent(config, environment=env.execute_action)

        result = agent.run(
            query=task["query"],
            max_steps=30,
            enable_memory_injection=False
        )

        assert task["success_condition"](env), (
            f"Failed task: {task['name']}. "
            f"Steps: {result.steps_taken}, "
            f"Success expected but not achieved"
        )

        print(f"✅ {task['name']}: {result.steps_taken} steps (expected: {task['expected_steps']})")
```

**2.3 Success Criteria**:
- ✅ Agent completes 6-step shopping workflow
- ✅ Agent completes 8-step product comparison
- ✅ Memory reduces steps by ≥15% on complex tasks
- ✅ All 4 WebArena-style tasks completable
- ✅ Step counts within 2x expected (reasonable efficiency)

---

### Gap 3: MaTTS (Memory-aware Test-Time Scaling) Not Tested 🚨 CRITICAL

**Current State**: Zero tests for parallel or sequential scaling
**Target**: Validate parallel scaling (N trajectories) and sequential scaling (refinement loops)

#### Implementation Tasks:

**3.1 Implement MaTTS Parallel Scaling**
```
File: reasoningbank/matts.py
Purpose: Implement memory-aware test-time scaling algorithms
```

**Implementation Details**:
```python
# reasoningbank/matts.py
"""
ABOUTME: Memory-aware Test-Time Scaling (MaTTS) implementation
ABOUTME: Parallel and sequential scaling as described in Paper Section 4.2

Paper Section 4.2: MaTTS improves performance through:
1. Parallel Scaling: Generate N trajectories, select best via self-contrast
2. Sequential Scaling: Iterative refinement within single trajectory
"""

from typing import List, Tuple, Optional
from reasoningbank.models import TrajectoryResult
from reasoningbank.agent import ReasoningBankAgent
import logging

logger = logging.getLogger(__name__)

class ParallelScaling:
    """
    Parallel MaTTS: Generate N trajectories and select best.

    Paper Section 4.2, Figure 4 (left):
    - Generate N independent trajectories
    - Use judge to evaluate each
    - Select trajectory with highest confidence
    - Better memory → better self-contrast → higher success rate
    """

    def __init__(self, agent: ReasoningBankAgent):
        self.agent = agent

    def generate_n_trajectories(
        self,
        query: str,
        n: int = 3,
        max_steps: int = 30,
        enable_memory: bool = True
    ) -> List[TrajectoryResult]:
        """
        Generate N independent trajectories for same query.

        Args:
            query: Task query
            n: Number of trajectories to generate
            max_steps: Max steps per trajectory
            enable_memory: Whether to use memory injection

        Returns:
            List of N trajectory results
        """
        logger.info(f"Generating {n} parallel trajectories for: {query}")

        trajectories = []
        for i in range(n):
            logger.info(f"  Trajectory {i+1}/{n}...")
            result = self.agent.run(
                query=query,
                max_steps=max_steps,
                enable_memory_injection=enable_memory
            )
            trajectories.append(result)
            logger.info(f"    Steps: {result.steps_taken}, Success: {result.success}")

        return trajectories

    def select_best_trajectory(
        self,
        trajectories: List[TrajectoryResult]
    ) -> Tuple[TrajectoryResult, int]:
        """
        Select best trajectory using self-contrast.

        Paper: Use judge's confidence scores to select best

        Returns:
            (best_trajectory, best_index)
        """
        if not trajectories:
            raise ValueError("No trajectories to select from")

        # Prioritize: success > fewer steps > first in list
        successful = [t for t in trajectories if t.success]

        if successful:
            # Among successful, pick one with fewest steps
            best = min(successful, key=lambda t: t.steps_taken)
            best_idx = trajectories.index(best)
        else:
            # If all failed, pick first (could enhance with judge scoring)
            best = trajectories[0]
            best_idx = 0

        logger.info(f"Selected trajectory {best_idx}: {best.steps_taken} steps, success={best.success}")
        return best, best_idx

    def run_with_parallel_scaling(
        self,
        query: str,
        n: int = 3,
        max_steps: int = 30,
        enable_memory: bool = True
    ) -> TrajectoryResult:
        """
        Run query with parallel scaling (N trajectories).

        Returns best trajectory from N attempts.
        """
        trajectories = self.generate_n_trajectories(
            query=query,
            n=n,
            max_steps=max_steps,
            enable_memory=enable_memory
        )

        best, _ = self.select_best_trajectory(trajectories)
        return best

class SequentialScaling:
    """
    Sequential MaTTS: Iterative refinement within trajectory.

    Paper Section 4.2, Figure 4 (right):
    - Generate initial trajectory
    - If failed, refine using memory of failure
    - Iterate up to K times
    - Better memory → better refinement → higher success
    """

    def __init__(self, agent: ReasoningBankAgent):
        self.agent = agent

    def run_with_sequential_scaling(
        self,
        query: str,
        max_refinements: int = 3,
        max_steps: int = 30,
        enable_memory: bool = True
    ) -> TrajectoryResult:
        """
        Run query with sequential refinement.

        Args:
            query: Task query
            max_refinements: Max refinement iterations
            max_steps: Max steps per attempt
            enable_memory: Use memory injection

        Returns:
            Best trajectory after refinements
        """
        logger.info(f"Running with sequential scaling: max {max_refinements} refinements")

        # Initial attempt
        result = self.agent.run(
            query=query,
            max_steps=max_steps,
            enable_memory_injection=enable_memory
        )

        logger.info(f"Initial attempt: steps={result.steps_taken}, success={result.success}")

        # If successful on first try, return
        if result.success:
            logger.info("Success on first attempt, no refinement needed")
            return result

        # Iterative refinement
        for refinement_num in range(1, max_refinements + 1):
            logger.info(f"Refinement {refinement_num}/{max_refinements}...")

            # Memory now contains failure lesson from previous attempt
            refined_result = self.agent.run(
                query=query,
                max_steps=max_steps,
                enable_memory_injection=True  # Always use memory for refinement
            )

            logger.info(f"  Steps: {refined_result.steps_taken}, Success: {refined_result.success}")

            # If this refinement succeeded, return it
            if refined_result.success:
                logger.info(f"Success after {refinement_num} refinements")
                return refined_result

            # Otherwise continue refining
            result = refined_result

        # Return best attempt (even if failed)
        logger.info(f"Failed after {max_refinements} refinements")
        return result
```

**3.2 Create MaTTS Tests**
```
File: tests/e2e/test_matts.py
Purpose: Validate parallel and sequential scaling
```

**Implementation**:
```python
# tests/e2e/test_matts.py
"""
Memory-aware Test-Time Scaling (MaTTS) validation tests.

Paper Section 4.2, Figure 4:
- Parallel Scaling: N=1,3,5 trajectories → success rate improves
- Sequential Scaling: K=1,3,5 refinements → success rate improves
"""

import pytest
import statistics
from reasoningbank.agent import ReasoningBankAgent
from reasoningbank.config import ReasoningBankConfig
from reasoningbank.matts import ParallelScaling, SequentialScaling
from tests.e2e.tasks.arithmetic_tasks import ARITHMETIC_TASKS

@pytest.mark.e2e
@pytest.mark.slow
class TestMaTTSParallelScaling:
    """
    Validate parallel scaling improves success rate.

    Paper Figure 4 (left): N=1,3,5 trajectories
    Expected: Success rate increases with N
    """

    def test_parallel_scaling_n1_vs_n3_vs_n5(self):
        """
        Compare success rates: N=1 vs N=3 vs N=5 trajectories.

        Paper claim: More trajectories → better self-contrast → higher success
        """
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            max_steps_per_task=30,
            memory_bank_path="test_data/matts_parallel_memory.json"
        )
        agent = ReasoningBankAgent(config)
        parallel_scaler = ParallelScaling(agent)

        # Get 20 moderately difficult tasks
        tasks = [t for t in ARITHMETIC_TASKS if t.difficulty == 3][:20]

        # Test with N=1, N=3, N=5
        n_values = [1, 3, 5]
        results_by_n = {}

        for n in n_values:
            print(f"\n=== Testing with N={n} trajectories ===")
            successes = []

            for i, task in enumerate(tasks, 1):
                if n == 1:
                    # Baseline: single trajectory
                    result = agent.run(
                        query=task.query,
                        max_steps=30,
                        enable_memory_injection=True
                    )
                else:
                    # Parallel scaling
                    result = parallel_scaler.run_with_parallel_scaling(
                        query=task.query,
                        n=n,
                        max_steps=30,
                        enable_memory=True
                    )

                is_correct = task.validate(result.model_output)
                successes.append(is_correct)
                print(f"  Task {i}/{len(tasks)}: {'✓' if is_correct else '✗'} (N={n})")

            success_rate = sum(successes) / len(successes)
            results_by_n[n] = success_rate
            print(f"N={n} Success Rate: {success_rate*100:.1f}%")

        # Validate improvement with more trajectories
        print(f"\n=== MaTTS Parallel Scaling Results ===")
        print(f"N=1: {results_by_n[1]*100:.1f}%")
        print(f"N=3: {results_by_n[3]*100:.1f}%")
        print(f"N=5: {results_by_n[5]*100:.1f}%")

        # CRITICAL ASSERTIONS (from paper)
        assert results_by_n[3] > results_by_n[1], (
            f"N=3 should outperform N=1. Got N=1:{results_by_n[1]*100:.1f}%, N=3:{results_by_n[3]*100:.1f}%"
        )

        assert results_by_n[5] >= results_by_n[3], (
            f"N=5 should match or exceed N=3. Got N=3:{results_by_n[3]*100:.1f}%, N=5:{results_by_n[5]*100:.1f}%"
        )

        # Validate meaningful improvement (≥10% from N=1 to N=5)
        improvement = results_by_n[5] - results_by_n[1]
        assert improvement >= 0.10, (
            f"Expected ≥10% improvement from N=1 to N=5, got {improvement*100:.1f}%"
        )

        print(f"\n✅ Parallel scaling validated: +{improvement*100:.1f}% (N=1→N=5)")

@pytest.mark.e2e
@pytest.mark.slow
class TestMaTTSSequentialScaling:
    """
    Validate sequential refinement improves success rate.

    Paper Figure 4 (right): K=1,3,5 refinements
    Expected: Success rate increases with refinements
    """

    def test_sequential_scaling_k1_vs_k3_vs_k5(self):
        """
        Compare success rates: K=1 vs K=3 vs K=5 refinements.

        Paper claim: More refinements → better self-correction → higher success
        """
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            max_steps_per_task=30,
            memory_bank_path="test_data/matts_sequential_memory.json"
        )
        agent = ReasoningBankAgent(config)
        sequential_scaler = SequentialScaling(agent)

        # Get difficult tasks (more likely to need refinement)
        tasks = [t for t in ARITHMETIC_TASKS if t.difficulty >= 4][:20]

        k_values = [1, 3, 5]
        results_by_k = {}

        for k in k_values:
            print(f"\n=== Testing with K={k} refinements ===")
            successes = []
            refinement_counts = []

            for i, task in enumerate(tasks, 1):
                if k == 1:
                    # Baseline: no refinement
                    result = agent.run(
                        query=task.query,
                        max_steps=30,
                        enable_memory_injection=True
                    )
                    actual_refinements = 0
                else:
                    # Sequential scaling
                    result = sequential_scaler.run_with_sequential_scaling(
                        query=task.query,
                        max_refinements=k,
                        max_steps=30,
                        enable_memory=True
                    )
                    # Count how many refinements were actually used
                    actual_refinements = len([r for r in agent.get_memory_bank() if task.query in r.task_query]) - 1

                is_correct = task.validate(result.model_output)
                successes.append(is_correct)
                refinement_counts.append(actual_refinements)
                print(f"  Task {i}/{len(tasks)}: {'✓' if is_correct else '✗'} (K={k}, used {actual_refinements} refinements)")

            success_rate = sum(successes) / len(successes)
            avg_refinements = statistics.mean(refinement_counts)
            results_by_k[k] = {
                'success_rate': success_rate,
                'avg_refinements': avg_refinements
            }
            print(f"K={k} Success Rate: {success_rate*100:.1f}%, Avg Refinements Used: {avg_refinements:.1f}")

        # Validate improvement with more refinements
        print(f"\n=== MaTTS Sequential Scaling Results ===")
        for k in k_values:
            print(f"K={k}: {results_by_k[k]['success_rate']*100:.1f}% (avg {results_by_k[k]['avg_refinements']:.1f} refinements used)")

        # CRITICAL ASSERTIONS
        assert results_by_k[3]['success_rate'] > results_by_k[1]['success_rate'], (
            f"K=3 should outperform K=1. Got K=1:{results_by_k[1]['success_rate']*100:.1f}%, K=3:{results_by_k[3]['success_rate']*100:.1f}%"
        )

        assert results_by_k[5]['success_rate'] >= results_by_k[3]['success_rate'], (
            f"K=5 should match or exceed K=3"
        )

        # Validate meaningful improvement
        improvement = results_by_k[5]['success_rate'] - results_by_k[1]['success_rate']
        assert improvement >= 0.10, (
            f"Expected ≥10% improvement from K=1 to K=5, got {improvement*100:.1f}%"
        )

        print(f"\n✅ Sequential scaling validated: +{improvement*100:.1f}% (K=1→K=5)")
```

**3.3 Success Criteria**:
- ✅ Parallel scaling: N=5 outperforms N=1 by ≥10%
- ✅ Sequential scaling: K=5 outperforms K=1 by ≥10%
- ✅ Both scaling methods show monotonic improvement
- ✅ MaTTS implementation matches paper algorithms
- ✅ Real LLM calls, no mocking of scaling benefits

---

## Part 2: Additional Critical Gaps (from Cross-Check) - Week 1 Extended

### Gap 21: Test-Time Streaming Constraint Validation 🚨 CRITICAL (NEW)

**Current State**: Benchmark tests don't explicitly validate streaming constraint
**Target**: Prove agent never uses information from future tasks during test-time learning
**Paper Reference**: Section 3.2 (Page 4) - "Tasks arrive in streaming fashion, no future access"

**Why Critical**: ReasoningBank's core paradigm is test-time learning without future task knowledge. If agent accidentally leaks information from task N+1 while processing task N, results are invalid and paper claims unproven.

#### Implementation Tasks:

**21.1 Create Streaming Constraint Validation Test**
```
File: tests/e2e/test_streaming_constraint.py
Purpose: Validate no future task information leakage during learning
```

**Implementation Details**:
```python
# tests/e2e/test_streaming_constraint.py
"""
Test-time streaming constraint validation.

Paper Claim (Section 3.2): "Tasks arrive sequentially, no ground truth available,
agent learns during test-time without seeing future tasks."

CRITICAL: Validate agent ONLY uses memory from past tasks (0..i-1),
NEVER from future tasks (i+1..N).
"""

import pytest
from reasoningbank.agent import ReasoningBankAgent
from reasoningbank.config import ReasoningBankConfig
from tests.e2e.tasks.arithmetic_tasks import ARITHMETIC_TASKS

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

        print(f"\n=== STREAMING CONSTRAINT VALIDATION ===")
        print(f"Tasks: {len(tasks)}")

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
            memory_bank = agent.memory_manager.get_all_memories()

            print(f"  Memory bank size: {len(memory_bank)}")
            print(f"  Expected max: {i * 3} (from {i} past tasks × 3 max items)")

            # CRITICAL CHECK 1: No memory from future tasks
            future_task_queries = [t.query for t in tasks[i+1:]]

            memory_leak_detected = False
            for memory_item in memory_bank:
                memory_content = memory_item.content.lower()
                memory_title = memory_item.title.lower()
                memory_desc = memory_item.description.lower()

                # Check if memory contains information from future tasks
                for future_query in future_task_queries:
                    # Extract key numbers/operations from future query
                    future_query_lower = future_query.lower()

                    # Check if future query appears in memory
                    if future_query_lower in memory_content:
                        print(f"\n❌ MEMORY LEAK DETECTED!")
                        print(f"  Task {i} has memory from future task: {future_query}")
                        print(f"  Memory item: {memory_item.title}")
                        memory_leak_detected = True
                        break

                if memory_leak_detected:
                    break

            assert not memory_leak_detected, (
                f"Memory leak! Task {i} contains information from future tasks. "
                f"This violates test-time streaming constraint."
            )

            # CRITICAL CHECK 2: Memory count should be ≤ i * 3
            # (Each past task can contribute max 3 memory items)
            max_expected_memories = i * 3  # Conservative upper bound

            assert len(memory_bank) <= max_expected_memories, (
                f"Memory bank has {len(memory_bank)} items, but only {i} tasks processed. "
                f"Expected ≤{max_expected_memories} items. Possible future information leakage."
            )

            print(f"  ✓ No future task leakage detected")
            print(f"  ✓ Memory count within bounds: {len(memory_bank)} ≤ {max_expected_memories}")

        print(f"\n✅ STREAMING CONSTRAINT VALIDATED")
        print(f"   Agent learned strictly from past experiences (0..i-1)")
        print(f"   No future task information leakage detected")
        print(f"   Memory growth pattern matches sequential learning")

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
            memory_bank_path="test_data/temporal_order_memory.json"
        )
        agent = ReasoningBankAgent(config)

        tasks = ARITHMETIC_TASKS[:15]
        memory_sizes = []

        for i, task in enumerate(tasks):
            result = agent.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=True
            )

            current_memory_size = len(agent.memory_manager.get_all_memories())
            memory_sizes.append(current_memory_size)

            # Validate monotonic increase (or stay same if extraction failed)
            if i > 0:
                assert current_memory_size >= memory_sizes[i-1], (
                    f"Memory decreased from {memory_sizes[i-1]} to {current_memory_size} "
                    f"after task {i}. Memory should only grow, not shrink."
                )

            print(f"Task {i+1}: Memory size = {current_memory_size}")

        # Validate overall growth
        initial_size = memory_sizes[0]
        final_size = memory_sizes[-1]

        assert final_size > initial_size, (
            f"Memory did not grow over {len(tasks)} tasks. "
            f"Initial: {initial_size}, Final: {final_size}"
        )

        print(f"\n✅ Temporal ordering validated")
        print(f"   Memory grew from {initial_size} → {final_size} over {len(tasks)} tasks")
```

**21.2 Success Criteria**:
- ✅ No future task information leakage detected across 20 tasks
- ✅ Memory count matches historical task count (≤ i × 3)
- ✅ Memory bank follows temporal ordering (monotonic growth)
- ✅ Agent learns strictly from past experiences (0..i-1)
- ✅ Validates paper's test-time streaming paradigm

---

### Gap 22: Memory Growth Strategy Without Deduplication 🚨 HIGH PRIORITY (NEW)

**Current State**: Tests don't validate long-term memory growth behavior
**Target**: Validate memory grows unbounded (simple addition, no deduplication) and retrieval still works
**Paper Reference**: Section 3.2, Appendix A.2 - "Consolidation: Simple addition of new memories"

**Why Critical**: Paper uses "simple addition" with NO deduplication. After 1000 tasks = 3000 memories. Must validate:
- Retrieval performance doesn't degrade (<5s)
- Near-duplicate memories are expected (per paper design)
- System handles large memory banks

#### Implementation Tasks:

**22.1 Create Memory Growth Validation Tests**
```
File: tests/stress/test_memory_growth_long_term.py
Purpose: Validate unbounded memory growth and retrieval performance
```

**Implementation Details**:
```python
# tests/stress/test_memory_growth_long_term.py
"""
Long-term memory growth validation.

Paper Design (Appendix A.2): "Simple addition" consolidation strategy
- No deduplication mentioned
- No pruning strategy mentioned
- Memory grows unbounded with task count

CRITICAL: Validate system handles 1000+ memories without degradation.
"""

import pytest
import time
import random
from reasoningbank.agent import ReasoningBankAgent
from reasoningbank.config import ReasoningBankConfig
from tests.e2e.tasks.arithmetic_tasks import ARITHMETIC_TASKS, get_tasks_by_difficulty

def generate_random_arithmetic_task():
    """Generate random arithmetic task for stress testing"""
    difficulty = random.randint(1, 5)
    tasks = get_tasks_by_difficulty(difficulty)
    return random.choice(tasks) if tasks else ARITHMETIC_TASKS[0]

@pytest.mark.stress
@pytest.mark.slow
class TestMemoryGrowthLongTerm:
    """
    Validate memory bank grows linearly without deduplication.

    Paper's consolidation strategy: Simple addition (no deduplication, no pruning)
    Expected behavior: Memory grows unbounded, may contain near-duplicates
    """

    def test_memory_bank_grows_linearly_without_deduplication(self):
        """
        Paper uses simple addition with NO deduplication.
        Validate behavior matches: memory grows unbounded.

        Test Parameters:
        - 100 tasks
        - Expected: ~300 memories (100 tasks × 3 max items per task)
        - Allow range: 250-300 (some tasks may extract <3 items)
        """
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            judge_temperature=0.0,
            extractor_temperature=1.0,
            memory_bank_path="test_data/memory_growth_100tasks.json"
        )
        agent = ReasoningBankAgent(config)

        print(f"\n=== MEMORY GROWTH TEST: 100 TASKS ===")

        # Run 100 tasks with varying difficulty
        for i in range(100):
            task = generate_random_arithmetic_task()

            result = agent.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=True
            )

            if (i + 1) % 10 == 0:
                memory_size = len(agent.memory_manager.get_all_memories())
                print(f"  Tasks {i+1}/100: Memory size = {memory_size}")

        # Final memory bank size
        final_memory_bank = agent.memory_manager.get_all_memories()
        final_size = len(final_memory_bank)

        print(f"\nFinal Memory Size: {final_size}")
        print(f"Expected Range: 250-300 (100 tasks × ~2.5-3 items)")

        # ASSERTION: Expected ~300 memories
        assert 250 <= final_size <= 300, (
            f"Expected 250-300 memories after 100 tasks, got {final_size}. "
            f"Memory growth pattern doesn't match paper's simple addition strategy."
        )

        # Check for near-duplicate memories (expected per paper design)
        similar_pairs = self._find_similar_memory_pairs(
            final_memory_bank,
            similarity_threshold=0.95
        )

        print(f"\nNear-duplicate memory pairs found: {len(similar_pairs)}")
        print(f"  (This is EXPECTED behavior per paper - no deduplication)")

        # Document that duplicates exist (as per paper design)
        if similar_pairs:
            print(f"\nExample near-duplicate pair:")
            pair = similar_pairs[0]
            print(f"  Memory 1: {pair[0].title}")
            print(f"  Memory 2: {pair[1].title}")
            print(f"  Similarity: {pair[2]:.2%}")

        print(f"\n✅ Memory growth validated: {final_size} memories after 100 tasks")
        print(f"   Simple addition strategy confirmed (no deduplication)")

    def test_retrieval_performance_degrades_with_memory_growth(self):
        """
        Validate retrieval still works with large memory banks.

        Target: <5s retrieval time with 1000+ memories

        Paper doesn't specify performance requirements, but system
        must remain usable at scale.
        """
        config = ReasoningBankConfig(
            agent_temperature=0.7,
            memory_bank_path="test_data/memory_growth_1000memories.json"
        )
        agent = ReasoningBankAgent(config)

        print(f"\n=== RETRIEVAL PERFORMANCE TEST: 1000 MEMORIES ===")

        # Grow memory to 1000 items (334 tasks × 3 ≈ 1000)
        print(f"Growing memory bank to 1000+ items...")
        for i in range(334):
            task = generate_random_arithmetic_task()
            agent.run(
                query=task.query,
                max_steps=30,
                enable_memory_injection=True
            )

            if (i + 1) % 50 == 0:
                memory_size = len(agent.memory_manager.get_all_memories())
                print(f"  Tasks {i+1}/334: Memory size = {memory_size}")

        # Final memory bank size
        final_memory_bank = agent.memory_manager.get_all_memories()
        final_size = len(final_memory_bank)

        print(f"\nFinal Memory Size: {final_size}")

        assert final_size >= 900, (
            f"Expected ≥900 memories after 334 tasks, got {final_size}. "
            f"Memory growth insufficient for stress test."
        )

        # Measure retrieval performance with large memory bank
        query = "Calculate 50 + 50"

        print(f"\nMeasuring retrieval performance...")
        print(f"  Query: {query}")
        print(f"  Memory bank size: {final_size}")

        start_time = time.time()
        retrieved_memories = agent.memory_manager.retrieve(query, top_k=1)
        retrieval_time = time.time() - start_time

        print(f"  Retrieval time: {retrieval_time:.3f}s")
        print(f"  Retrieved: {len(retrieved_memories)} memories")

        # ASSERTION: Retrieval should complete in <5s even with 1000 memories
        assert retrieval_time < 5.0, (
            f"Retrieval too slow with {final_size} memories: {retrieval_time:.2f}s. "
            f"Expected <5s. System may not scale to long-term usage."
        )

        print(f"\n✅ Retrieval performance validated")
        print(f"   {final_size} memories: {retrieval_time:.3f}s retrieval time")
        print(f"   System scales to long-term usage")

    def _find_similar_memory_pairs(self, memories, similarity_threshold=0.95):
        """
        Find near-duplicate memory pairs using text similarity.

        Returns: List of (memory1, memory2, similarity_score) tuples
        """
        from difflib import SequenceMatcher

        similar_pairs = []

        for i in range(len(memories)):
            for j in range(i+1, len(memories)):
                # Compare content similarity
                similarity = SequenceMatcher(
                    None,
                    memories[i].content,
                    memories[j].content
                ).ratio()

                if similarity >= similarity_threshold:
                    similar_pairs.append((memories[i], memories[j], similarity))

        return similar_pairs
```

**22.2 Success Criteria**:
- ✅ Memory grows to 250-300 after 100 tasks (unbounded growth)
- ✅ Memory grows to 900+ after 334 tasks (linear growth)
- ✅ Near-duplicate memories exist (expected per paper)
- ✅ Retrieval time <5s with 1000+ memories
- ✅ System scales to long-term usage (no performance collapse)

---

### Gap 23: Tie-Breaking Strategy for Equal Similarity Scores 🔧 MEDIUM PRIORITY (NEW)

**Current State**: No tests for retrieval behavior when multiple memories have identical similarity scores
**Target**: Validate deterministic tie-breaking for reproducibility
**Paper Reference**: Appendix A.2 - "Top-k=1 retrieval with cosine similarity"

**Why Important**: Paper specifies top-k=1 but doesn't define tie-breaking. Non-deterministic behavior breaks reproducibility requirements.

#### Implementation Tasks:

**23.1 Create Tie-Breaking Validation Test**
```
File: tests/unit/test_retrieval_tie_breaking.py
Purpose: Validate deterministic tie-breaking for equal similarity scores
```

**Implementation Details**:
```python
# tests/unit/test_retrieval_tie_breaking.py
"""
Retrieval tie-breaking strategy validation.

Paper (Appendix A.2): "Top-k=1 retrieval using cosine similarity"
- Doesn't specify what happens when multiple memories have same score
- Behavior must be deterministic for reproducibility

CRITICAL: Validate consistent tie-breaking across multiple retrievals.
"""

import pytest
from reasoningbank.memory_manager import MemoryManager
from reasoningbank.models import MemoryItem

@pytest.mark.unit
class TestRetrievalTieBreaking:
    """Validate deterministic tie-breaking for equal similarity scores"""

    def test_retrieval_tie_breaking_strategy_is_deterministic(self):
        """
        When multiple memories have IDENTICAL similarity scores,
        validate retrieval returns same result consistently.

        Reproducibility requirement: Same query → same retrieval
        """
        manager = MemoryManager(embedding_model="gemini-embedding-001")

        # Create 3 memories with IDENTICAL content/embeddings
        # (will have same cosine similarity for same query)
        identical_content = "Always use search feature before applying filters to products"

        memory1 = MemoryItem(
            title="Strategy A - First Inserted",
            description="Search then filter approach",
            content=identical_content,
            task_query="Find products"
        )

        memory2 = MemoryItem(
            title="Strategy B - Second Inserted",
            description="Search then filter approach",
            content=identical_content,
            task_query="Find products"
        )

        memory3 = MemoryItem(
            title="Strategy C - Third Inserted",
            description="Search then filter approach",
            content=identical_content,
            task_query="Find products"
        )

        # Add to memory bank in specific order
        manager.add(memory1)
        manager.add(memory2)
        manager.add(memory3)

        # Retrieve with top-k=1 (should break tie deterministically)
        query = "How to find and filter products?"

        print(f"\n=== TIE-BREAKING TEST ===")
        print(f"Memories with identical content: 3")
        print(f"Insertion order: A → B → C")

        # Run retrieval 10 times - should return SAME memory every time
        results = []
        for i in range(10):
            retrieved = manager.retrieve(query, top_k=1)
            assert len(retrieved) == 1, "Should return exactly 1 memory"
            results.append(retrieved[0].title)

        # All results should be identical (deterministic)
        unique_results = set(results)

        print(f"Retrieval runs: 10")
        print(f"Unique results: {len(unique_results)}")
        print(f"Results: {unique_results}")

        assert len(unique_results) == 1, (
            f"Non-deterministic tie-breaking! Got different results across runs: {unique_results}. "
            f"Expected same result every time for reproducibility."
        )

        # Document which strategy tie-breaking uses
        selected_strategy = list(unique_results)[0]
        print(f"\nTie-breaking strategy: Always selects '{selected_strategy}'")

        # Validate it's one of the expected strategies
        assert selected_strategy in [memory1.title, memory2.title, memory3.title], (
            f"Unexpected tie-breaking result: {selected_strategy}"
        )

        print(f"\n✅ Tie-breaking is deterministic")
        print(f"   Same query → same retrieval result across 10 runs")
        print(f"   Reproducibility validated")

    def test_tie_breaking_respects_insertion_order(self):
        """
        Validate tie-breaking uses a consistent rule (e.g., first/last inserted).
        """
        manager = MemoryManager(embedding_model="gemini-embedding-001")

        # Add 5 memories with identical similarity
        for i in range(5):
            memory = MemoryItem(
                title=f"Memory_{i}",
                description="Test memory",
                content="Identical content for all memories",
                task_query="test"
            )
            manager.add(memory)

        # Retrieve with top-k=1
        query = "test query"
        retrieved = manager.retrieve(query, top_k=1)

        # Document which memory was selected
        selected_title = retrieved[0].title
        print(f"\nSelected from 5 identical memories: {selected_title}")

        # Validate it's either first or last (common tie-breaking strategies)
        assert selected_title in ["Memory_0", "Memory_4"], (
            f"Unexpected tie-breaking: {selected_title}. "
            f"Expected either first (Memory_0) or last (Memory_4) insertion order."
        )

        print(f"✅ Tie-breaking uses insertion order (deterministic)")
```

**23.2 Success Criteria**:
- ✅ Tie-breaking is deterministic (same result across 10 runs)
- ✅ Strategy is documented (insertion order, recency, or seeded random)
- ✅ Behavior matches paper's reproducibility claims
- ✅ No random tie-breaking that breaks reproducibility

---

### Gap 24: Ablation - Success-Only vs Success+Failure Extraction 🚨 CRITICAL (NEW)

**Current State**: No test validates paper's PRIMARY innovation claim
**Target**: Prove learning from BOTH outcomes outperforms learning from successes only
**Paper Reference**: Section 3.2 (Page 4) - **PAPER'S CORE INNOVATION**

**Why CRITICAL**: This is ReasoningBank's PRIMARY contribution:
> "ReasoningBank learns from BOTH successful and failed experiences, unlike prior work that only learns from successes"

**Must prove with evidence that adding failure extraction improves performance beyond success-only learning.**

#### Implementation Tasks:

**24.1 Create Success+Failure Ablation Test**
```
File: tests/ablation/test_success_and_failure_extraction.py
Purpose: Validate paper's core innovation claim
```

**Implementation Details**:
```python
# tests/ablation/test_success_and_failure_extraction.py
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
from reasoningbank.agent import ReasoningBankAgent
from reasoningbank.config import ReasoningBankConfig
from tests.e2e.tasks.arithmetic_tasks import ARITHMETIC_TASKS

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
        # Get 20 moderately challenging tasks
        tasks = [t for t in ARITHMETIC_TASKS if t.difficulty in [3, 4]][:20]

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

            # Memory ONLY extracted from successes
            if is_correct:
                agent_success_only.memory_manager.extract_and_add(result)
                print(f"  Task {i}: ✓ CORRECT → Memory extracted")
            else:
                print(f"  Task {i}: ✗ WRONG → No memory extraction")

        success_only_sr = sum(success_only_results) / len(success_only_results)
        success_only_avg_steps = statistics.mean(success_only_steps)
        success_only_memory_count = len(agent_success_only.memory_manager.get_all_memories())

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

            # Extract from BOTH successes AND failures
            agent_both.memory_manager.extract_and_add(result)

            if is_correct:
                print(f"  Task {i}: ✓ CORRECT → Success memory extracted")
            else:
                print(f"  Task {i}: ✗ WRONG → Failure lesson extracted")

        both_sr = sum(both_results) / len(both_results)
        both_avg_steps = statistics.mean(both_steps)
        both_memory_count = len(agent_both.memory_manager.get_all_memories())

        print(f"\nPaper Approach Performance (Success+Failure):")
        print(f"  Success Rate: {both_sr*100:.1f}%")
        print(f"  Avg Steps: {both_avg_steps:.1f}")
        print(f"  Memory Items: {both_memory_count}")

        # Analyze failure memories
        failure_memories = [
            m for m in agent_both.memory_manager.get_all_memories()
            if any(keyword in m.description.lower() for keyword in ["fail", "error", "wrong", "incorrect"])
        ]

        print(f"  Failure Memories: {len(failure_memories)}")
        if failure_memories:
            print(f"  Example failure lesson: \"{failure_memories[0].title}\"")

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
        assert len(failure_memories) > 0, (
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
        print(f"  • Failure memories extracted: {len(failure_memories)}")
        print(f"  • Step efficiency improved: {step_improvement*100:+.1f}%")
        print(f"\n" + "="*60)
```

**24.2 Success Criteria**:
- ✅ Success+Failure extraction outperforms Success-only by ≥5%
- ✅ Improvement is statistically significant (p<0.05)
- ✅ Failure memories are extracted and used
- ✅ Step efficiency also improves with failure lessons
- ✅ **VALIDATES PAPER'S PRIMARY INNOVATION CLAIM**

---

*[PLAN CONTINUES WITH GAPS 4-20 IN SAME DETAIL...]*

**Due to token limits, I'll provide the remaining gaps in summary form with full implementation details available on request:**

### Gap 4: Emergent Behaviors Not Validated
- Implement sophistication scoring for memory evolution
- Track title/description complexity over 50 tasks
- Validate procedural → adaptive → compositional progression
- **Files**: `tests/e2e/test_emergent_behaviors.py`, `reasoningbank/analysis/sophistication_scorer.py`

### Gap 5: No Stress Testing
- Test with 1000+ memories
- Validate retrieval performance <5s
- Test JSON corruption recovery
- Test concurrent agent execution
- **Files**: `tests/stress/test_scale.py`, `tests/stress/test_corruption.py`

### Gaps 6-9: Benchmark Suite
- WebArena shopping workflows (20 tasks)
- Mind2Web cross-domain transfer (20 tasks)
- SWE-Bench code debugging (10 tasks)
- Progressive learning validation (50 tasks)
- **Files**: `tests/benchmarks/` directory with suite

### Gaps 10-20: Edge Cases & Robustness
- Memory bank at scale (1000+ entries)
- Retrieval degradation tests
- Embedding quality validation
- Judge consistency tests
- Extractor failure handling
- Max steps boundary tests
- Environment timeout simulation
- Memory corruption recovery
- **Files**: `tests/edge_cases/` directory

---

## Implementation Timeline (3 Weeks + 2 Days)

### Week 1: Critical Gaps (P0)
**Days 1-2**: Self-Evolution Tests (Gap 1)
- Arithmetic task suite (50 tasks)
- 20-task self-evolution test
- 50-task progressive learning test

**Days 3-4**: Complex Reasoning Tests (Gap 2)
- WebArena simulator environment
- Multi-step workflow tests (6-8 steps)
- Memory improvement validation

**Day 5**: MaTTS Implementation (Gap 3)
- Parallel scaling (N=1,3,5)
- Sequential scaling (K=1,3,5)
- Validation tests

### Week 1 Extended: Additional Critical Gaps (P0)
**Day 6**: Streaming Constraint Validation (Gap 21 - NEW)
- Test-time streaming constraint tests
- Future task leakage detection
- Temporal ordering validation
- **Effort**: 8 hours

**Day 7**: Success+Failure Ablation (Gap 24 - NEW) ⚠️ **PAPER'S CORE CLAIM**
- Success-only baseline implementation
- Success+failure comparison
- Statistical validation (p<0.05)
- **Effort**: 16 hours
- **CRITICAL**: Validates paper's PRIMARY innovation

### Week 2: High Priority (P1)
**Days 8-9**: Emergent Behaviors (Gap 4)
- Sophistication scoring system
- Evolution tracking tests
- Strategy progression validation

**Days 10-11**: Benchmark Suite (Gap 5)
- WebArena workflows
- Mind2Web cross-domain
- SWE-Bench debugging scenarios

**Day 12**: Memory Growth Tests (Gap 22 - NEW)
- 100-task memory growth validation
- 1000-memory retrieval performance
- Deduplication behavior documentation
- **Effort**: 12 hours

### Week 3: Medium Priority (P2)
**Days 13-14**: Stress Tests (Gap 6)
- Concurrent execution
- Performance regression framework
- Additional robustness tests

**Days 15-16**: Edge Cases (Gaps 15-16)
- Boundary conditions
- Failure modes
- Timeout handling
- Quality validation

**Day 17**: Tie-Breaking Tests (Gap 23 - NEW)
- Deterministic retrieval validation
- Insertion order testing
- Reproducibility verification
- **Effort**: 4 hours

### Week 3 Final
**Day 18**: Final Validation
- Complete test suite run (240 tests total)
- Production readiness assessment
- Documentation updates
- CI/CD integration

---

## Success Metrics

### Mandatory (P0):
- ✅ Self-evolution: ≥15% success improvement, ≥20% step reduction (Gap 1)
- ✅ Complex tasks: All 4 WebArena-style tasks completable (Gap 2)
- ✅ MaTTS: Both scaling methods show ≥10% improvement (Gap 3)
- ✅ **Streaming constraint: No future task leakage detected (Gap 21 - NEW)**
- ✅ **Success+Failure ablation: ≥5% improvement, p<0.05 (Gap 24 - NEW) - PAPER'S CORE CLAIM**
- ✅ Statistical significance: p<0.05 for all improvements
- ✅ Zero mocking in validation tests

### High Priority (P1):
- ✅ Emergent behaviors: Strategy sophistication increases 30%+ (Gap 4)
- ✅ Benchmark suite: Match paper performance within 10% (Gap 5)
- ✅ 50-task learning: Monotonic improvement trend (ρ>0.3) (Gap 1)
- ✅ **Memory growth: 250-300 after 100 tasks, retrieval <5s with 1000+ (Gap 22 - NEW)**

### Medium Priority (P2):
- ✅ **Tie-breaking: Deterministic retrieval across 10 runs (Gap 23 - NEW)**
- ✅ Stress tests: Concurrent execution, no degradation (Gap 6)
- ✅ Robustness: 95%+ edge case coverage (Gaps 15-16)
- ✅ Performance: No regression on existing 228 tests

---

## Resource Requirements

**LLM API Costs** (estimated):
- Self-evolution tests: ~1000 LLM calls × $0.002 = $2 (Gap 1)
- Complex reasoning: ~500 calls × $0.002 = $1 (Gap 2)
- MaTTS tests: ~1500 calls × $0.002 = $3 (Gap 3)
- Benchmark suite: ~2000 calls × $0.002 = $4 (Gap 5)
- **Streaming constraint: ~500 calls × $0.002 = $1 (Gap 21 - NEW)**
- **Memory growth: ~800 calls × $0.002 = $2 (Gap 22 - NEW)**
- **Success+Failure ablation: ~1000 calls × $0.002 = $2 (Gap 24 - NEW)**
- Tie-breaking: Unit tests only = $0 (Gap 23 - NEW)
- **Total**: ~$15 for original plan + ~$5 for new gaps = **$20 total**
- **Conservative estimate with retries/failures**: **$25-30**

**Compute Resources**:
- CPU: Standard development machine
- Memory: 8GB+ RAM (16GB recommended for stress tests)
- Storage: 2GB for test data (increased from 1GB for larger memory banks)
- Network: Stable internet for API calls

**Development Time**:
- Original plan: 120 hours
- **New gaps: +40 hours (Gap 21: 8h, Gap 22: 12h, Gap 23: 4h, Gap 24: 16h)**
- **Total: 160 hours implementation + validation**
- 40 hours/week = 4 weeks (3 weeks + 2 days)
- Can parallelize with 2 developers → 2 weeks

---

## Risk Mitigation

**Risk**: Real LLM calls may be expensive
**Mitigation**: Use caching, run tests in batches, monitor costs

**Risk**: Tests may fail due to LLM variability
**Mitigation**: Run each test 3x, use statistical validation, set confidence thresholds

**Risk**: Complex tasks may be too hard for current agent
**Mitigation**: Progressive difficulty, start simple, iterate on agent improvements

**Risk**: Timeline may slip
**Mitigation**: Prioritize P0 gaps first, P1-P2 can extend if needed

---

## Deliverables

1. **Test Suite** (`tests/e2e/`, `tests/benchmarks/`, `tests/stress/`, `tests/ablation/`)
   - **Original**: 228 tests
   - **New**: +12 tests from 4 new gaps
   - **Total**: **240 comprehensive tests**
2. **MaTTS Implementation** (`reasoningbank/matts.py`)
3. **Task Environments** (`tests/e2e/environments/`, `tests/e2e/tasks/`)
4. **Analysis Tools** (`reasoningbank/analysis/`)
5. **New Test Files** (from cross-check):
   - `tests/e2e/test_streaming_constraint.py` (Gap 21)
   - `tests/stress/test_memory_growth_long_term.py` (Gap 22)
   - `tests/unit/test_retrieval_tie_breaking.py` (Gap 23)
   - `tests/ablation/test_success_and_failure_extraction.py` (Gap 24 - **CRITICAL**)
6. **Documentation** (README updates, test running guide, gap analysis report)
7. **CI/CD Integration** (GitHub Actions workflow with all 240 tests)
8. **Production Readiness Report** (evidence-based assessment with all gaps closed)

---

## Next Steps

1. ✅ Review and approve this plan
2. ✅ Set up test data directories (including new `/tests/ablation/` directory)
3. ✅ Configure API credentials for real LLM calls
4. ✅ Begin Week 1 Day 1: Arithmetic task suite implementation (Gap 1)
5. ✅ Week 1 Day 6-7: Implement NEW critical gaps (Gap 21, 24)
6. ✅ Week 2 Day 12: Memory growth tests (Gap 22)
7. ✅ Week 3 Day 17: Tie-breaking tests (Gap 23)
8. ✅ Daily progress tracking and plan adjustments

---

## Summary

**This plan comprehensively addresses all 24 gaps (20 original + 4 from cross-check) identified in the ultrathink analysis with ZERO shortcuts, ZERO mocks for validation, and REAL end-to-end testing that proves ReasoningBank's core self-evolution claims.**

### Updated Coverage (Post Cross-Check):
- **Original Gaps**: 20 gaps covering mechanism validation
- **New Gaps from Cross-Check**: 4 critical gaps covering outcome validation
  - **Gap 21** (P0): Test-time streaming constraint (no future task leakage)
  - **Gap 22** (P1): Memory growth without deduplication (1000+ memories)
  - **Gap 23** (P2): Deterministic tie-breaking for reproducibility
  - **Gap 24** (P0): **Success+Failure ablation - PAPER'S PRIMARY INNOVATION CLAIM**

### Total Implementation:
- **Test Count**: 228 → **240 tests** (+12 new tests)
- **Timeline**: 3 weeks → **3 weeks + 2 days** (+40 hours)
- **Cost**: $10-15 → **$25-30** (+$5 base, +10-15 buffer)
- **Priority**: 2 P0 gaps added (Gap 21, 24), 1 P1 (Gap 22), 1 P2 (Gap 23)

### Most Critical Addition:
**Gap 24 (Success+Failure Ablation)** validates ReasoningBank's PRIMARY innovation claim that learning from BOTH successful AND failed experiences outperforms learning from successes only. Without this test, we cannot claim the paper's core contribution is proven.

### Final Validation Status:
- ✅ **95% of paper claims covered** (from cross-check analysis)
- ✅ **100% of paper claims will be covered** after implementing 4 new gaps
- ✅ **Zero shortcuts, zero mocks** - all validation uses real LLM calls
- ✅ **Statistical rigor** - p<0.05 for all improvement claims
- ✅ **Paper's core innovation validated** - Success+Failure > Success-only
