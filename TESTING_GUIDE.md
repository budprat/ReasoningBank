# ReasoningBank Testing Guide

Comprehensive testing guide for the ReasoningBank implementation.

## Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Test Categories](#test-categories)
5. [Writing New Tests](#writing-new-tests)
6. [Test Fixtures](#test-fixtures)
7. [Mocking Strategy](#mocking-strategy)
8. [Coverage Requirements](#coverage-requirements)
9. [Continuous Integration](#continuous-integration)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The ReasoningBank test suite validates the complete implementation against the paper specifications from "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory" (Google Cloud AI Research + UIUC, September 2025).

**Test Framework**: pytest
**Total Tests**: 150+ comprehensive tests
**Coverage Target**: ≥90% code coverage

**Test Philosophy**:
- **Paper Compliance**: Every test validates implementation matches paper specifications
- **Isolation**: Tests run independently without external API calls
- **Comprehensiveness**: Unit tests, integration tests, and end-to-end workflows
- **Realistic Data**: Fixtures provide realistic trajectories and memory items

---

## Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # Shared fixtures and configuration
├── run_all_tests.py              # Comprehensive test runner script
│
├── fixtures/                      # Test data and mock environments
│   ├── __init__.py
│   ├── sample_trajectories.py    # Realistic trajectory examples
│   ├── sample_memories.py        # Pre-extracted memory items
│   ├── mock_environments.py      # Simulated agent environments
│   └── test_data.py              # Common test data
│
├── unit/                          # Unit tests (individual components)
│   ├── __init__.py
│   ├── test_models.py            # Data model tests
│   ├── test_config.py            # Configuration tests
│   ├── test_judge.py             # TrajectoryJudge tests
│   ├── test_extractor.py         # MemoryExtractor tests
│   ├── test_retriever.py         # MemoryRetriever tests
│   └── test_consolidator.py     # MemoryConsolidator tests
│
├── integration/                   # Integration tests (component interactions)
│   ├── __init__.py
│   ├── test_closed_loop.py       # Complete learning cycle tests
│   └── test_agent.py             # ReasoningBankAgent integration tests
│
├── matts/                         # MaTTS (Memory-aware Test-Time Scaling) tests
│   ├── __init__.py
│   ├── test_parallel.py          # Parallel scaling tests (k=3,5,7)
│   └── test_sequential.py        # Sequential refinement tests (k=3,5,7)
│
└── performance/                   # Performance and benchmarking tests
    └── __init__.py
```

---

## Running Tests

### Quick Start

Run all tests:
```bash
./tests/run_all_tests.py
```

Or using pytest directly:
```bash
pytest
```

### Running Specific Test Categories

**Unit tests only**:
```bash
pytest tests/unit/ -v
```

**Integration tests only**:
```bash
pytest tests/integration/ -v
```

**MaTTS tests only**:
```bash
pytest tests/matts/ -v
```

**Specific test file**:
```bash
pytest tests/unit/test_judge.py -v
```

**Specific test class**:
```bash
pytest tests/unit/test_judge.py::TestTrajectoryJudgeBasic -v
```

**Specific test**:
```bash
pytest tests/unit/test_judge.py::TestTrajectoryJudgeBasic::test_judge_initialization -v
```

### Using Test Markers

**Run only unit tests**:
```bash
pytest -m unit
```

**Run only integration tests**:
```bash
pytest -m integration
```

**Run only MaTTS tests**:
```bash
pytest -m matts
```

**Run fast tests (exclude slow integration/MaTTS)**:
```bash
pytest -m "not integration and not matts"
```

### Coverage Reports

**Run with coverage**:
```bash
pytest --cov=reasoningbank --cov-report=html
```

**View coverage report**:
```bash
open htmlcov/index.html
```

**Coverage summary**:
```bash
pytest --cov=reasoningbank --cov-report=term-missing
```

### Verbose Output

**Show print statements**:
```bash
pytest -s
```

**Very verbose**:
```bash
pytest -vv
```

**Show failed tests first**:
```bash
pytest --failed-first
```

---

## Test Categories

### 1. Unit Tests (`tests/unit/`)

Test individual components in isolation with mocked dependencies.

#### `test_models.py` (45 tests)
- Data model creation and validation
- Serialization (to_dict, from_dict)
- Model relationships and constraints
- Paper specification compliance

**Example**:
```python
def test_memory_item_creation():
    """Test MemoryItem creation with required fields"""
    item = MemoryItem(
        title="Test Strategy",
        description="Test description",
        content="Test content explaining the strategy"
    )

    assert item.title == "Test Strategy"
    assert item.description == "Test description"
    assert item.content == "Test content explaining the strategy"
```

#### `test_config.py` (25 tests)
- Configuration defaults match paper (Appendix A.2)
- Temperature settings: agent=0.7, judge=0.0, extractor=1.0, selector=0.0
- Memory extraction limits: max 3 per trajectory, max 5 aggregated
- Embedding settings: gemini-embedding-001 (768 dimensions)
- Preset configurations
- Validation logic

**Key Test**:
```python
def test_temperature_settings_match_paper():
    """Test temperature settings match paper specifications (Appendix A.2)"""
    config = ReasoningBankConfig(llm_api_key="test-key", ...)

    assert config.agent_temperature == 0.7    # Balanced exploration
    assert config.judge_temperature == 0.0    # Deterministic judgments
    assert config.extractor_temperature == 1.0 # Diverse extraction
    assert config.selector_temperature == 0.0  # Consistent selection
```

#### `test_judge.py` (30 tests)
- LLM-as-a-Judge trajectory evaluation
- Prompt structure matches Figure 9 (Appendix A.1)
- Success/failure parsing
- Confidence estimation with multiple samples
- Temperature=0.0 for deterministic judgments
- Multi-provider support (Anthropic, OpenAI, Google)

**Key Test**:
```python
def test_judge_prompt_matches_paper():
    """Test judge prompt matches Figure 9 structure"""
    judge = TrajectoryJudge(config)
    prompt = judge._build_judge_prompt(query="What is 25 * 4?", ...)

    assert "expert in evaluating the performance" in prompt
    assert "three types of tasks" in prompt.lower()
    assert "Information seeking" in prompt
    assert "Site navigation" in prompt
    assert "Content modification" in prompt
```

#### `test_extractor.py` (35 tests)
- Dual-prompt approach (success vs. failure) - Figure 8
- Markdown parsing: `# Memory Item`, `## Title`, `## Description`, `## Content`
- Temperature=1.0 for diverse extraction
- Max items enforcement (3 per trajectory, 5 aggregated)
- Self-contrast extraction for MaTTS parallel
- Memory schema validation (title, 1-sentence description, 1-5 sentence content)

**Key Test**:
```python
def test_dual_prompt_approach():
    """Test that success and failure use different prompts"""
    extractor = MemoryExtractor(config)

    success_prompt = extractor._build_success_prompt(...)
    failure_prompt = extractor._build_failure_prompt(...)

    # Success prompt focuses on strategies
    assert "successful strategies" in success_prompt.lower()

    # Failure prompt focuses on preventative lessons
    assert "went wrong" in failure_prompt.lower() or "avoid" in failure_prompt.lower()
```

#### `test_retriever.py` (28 tests)
- Embedding generation (OpenAI and Google)
- Cosine similarity computation
- Top-k retrieval (default k=1 from paper)
- Cache management for efficiency
- Filtered retrieval (success_only, failure_only, min_similarity)
- Embedding dimension validation (OpenAI: 1536, Google: 768)

**Key Test**:
```python
def test_retrieval_ranks_by_similarity():
    """Test that retrieve ranks items by cosine similarity"""
    retriever = MemoryRetriever(config)

    # Create items with different similarity scores
    item_high_sim = MemoryItem(title="High", description="High", content="High similarity")
    item_low_sim = MemoryItem(title="Low", description="Low", content="Low similarity")

    results = retriever.retrieve("Query text", [entry], k=2)

    # Should be ranked: high similarity first
    assert results[0].title == "High"
    assert results[1].title == "Low"
```

#### `test_consolidator.py` (32 tests)
- Persistent JSON storage (load/save)
- Memory bank management (add, get, search, remove)
- Statistics calculation (success rate, avg items, avg steps)
- Entry filtering (success, failure)
- Import/export functionality
- Cross-session persistence

**Key Test**:
```python
def test_persistence_across_sessions():
    """Test that memories persist across consolidator instances"""
    # Session 1: Create and save
    consolidator1 = MemoryConsolidator(config)
    entry_id = consolidator1.add_from_trajectory(...)

    # Session 2: Load and verify
    consolidator2 = MemoryConsolidator(config)
    retrieved = consolidator2.get_entry(entry_id)

    assert retrieved is not None
    assert len(retrieved.memory_items) > 0
```

### 2. Integration Tests (`tests/integration/`)

Test interactions between multiple components in realistic workflows.

#### `test_closed_loop.py` (40 tests)
- **TestClosedLoopBasic**: Judge → Extract → Consolidate workflow
- **TestFullClosedLoop**: Complete learning cycle (Task 1 → Learn → Task 2)
- **TestMemoryInjection**: Memory retrieval and agent context injection
- **TestPersistenceAcrossSessions**: Cross-session data persistence
- **TestEndToEndWorkflow**: Multi-task learning scenarios

**Key Test**:
```python
def test_complete_learning_cycle():
    """Test complete cycle: Task 1 → Learn → Task 2 (with retrieval)"""
    # Initialize all components
    judge = TrajectoryJudge(config)
    extractor = MemoryExtractor(config)
    consolidator = MemoryConsolidator(config)
    retriever = MemoryRetriever(config)

    # Task 1: First execution (no prior memories)
    task1_success = judge.judge_trajectory_success(...)
    task1_memories = extractor.extract_memories(...)
    entry1_id = consolidator.add_from_trajectory(...)

    # Task 2: Second execution (with retrieval)
    retrieved_memories = retriever.retrieve(task2_query, ...)

    # Should have learned from Task 1
    assert len(retrieved_memories) > 0
    assert stats["total_entries"] == 1
```

#### `test_agent.py` (45 tests)
- **TestReasoningBankAgentBasic**: Agent initialization and configuration
- **TestAgentExecution**: Task execution with/without memory injection
- **TestAgentMemoryIntegration**: Learning from success and failure
- **TestAgentWithMockEnvironment**: Integration with arithmetic, search, navigation environments
- **TestAgentProgressiveImprovement**: Memory accumulation over tasks
- **TestAgentReActFormat**: ReAct format parsing and trajectory formatting
- **TestEndToEndAgentWorkflow**: Complete agent lifecycle with multiple tasks

**Key Test**:
```python
def test_agent_improves_with_memory_accumulation():
    """Test that agent performance improves as memory accumulates"""
    agent = ReasoningBankAgent(config)

    # Task 1: Initial execution (no prior memories)
    result1 = agent.run("Calculate 25 * 4", enable_memory_injection=False)
    assert result1.success is True

    # Task 2: Similar task with memory injection
    result2 = agent.run("Calculate 15 * 5", enable_memory_injection=True)

    # Should leverage past experience
    assert result2.success is True
    assert len(agent.get_memory_bank()) == 2
```

### 3. MaTTS Tests (`tests/matts/`)

Test Memory-aware Test-Time Scaling strategies (Section 3.3 from paper).

#### `test_parallel.py` (50+ tests)
- **TestMaTTSParallelInitialization**: Configuration and setup
- **TestTrajectorySelection**: Best-of-n selection logic
- **TestParallelSampling**: k-trajectory parallel sampling (k=3,5,7)
- **TestSelfContrastExtraction**: Aggregated memory extraction
- **TestMaTTSParallelWorkflow**: Complete parallel scaling workflow
- **TestMaTTSParallelIntegration**: End-to-end parallel scaling

**Key Test**:
```python
def test_matts_parallel_run_complete_workflow():
    """Test complete MaTTS parallel workflow with k=3"""
    matts = MaTTSParallel(config, environment)

    result = matts.run(query="Calculate 25 * 4", max_steps=30, k=3)

    # Verify result structure
    assert result.scaling_mode == "parallel"
    assert result.scaling_factor == 3
    assert len(result.all_trajectories) == 3
    assert result.best_trajectory.success is True  # Best selected
    assert len(result.aggregated_memories) > 0     # Self-contrast extracted
```

#### `test_sequential.py` (50+ tests)
- **TestMaTTSSequentialInitialization**: Configuration and refinement prompts
- **TestRefinementPrompts**: Prompt usage matching Figure 10
- **TestInitialTrajectoryExecution**: First attempt execution
- **TestRefinementExecution**: Iterative refinement with previous trajectory context
- **TestTrajectorySelection**: Best trajectory selection (prefers recent, successful, fewer steps)
- **TestMaTTSSequentialWorkflow**: Complete sequential refinement workflow
- **TestMaTTSSequentialIntegration**: Progressive improvement validation

**Key Test**:
```python
def test_matts_sequential_shows_progressive_improvement():
    """Test that sequential refinement shows progressive improvement"""
    matts = MaTTSSequential(config, environment)

    result = matts.run(query="Calculate 25 * 4", max_steps=30, k=3)

    # Should have 1 initial + 3 refinements = 4 trajectories
    assert len(result.all_trajectories) == 4

    # Best should be most refined
    assert result.best_trajectory.success is True

    # Verify improvement over initial attempt
    initial_steps = result.all_trajectories[0].steps_taken
    final_steps = result.best_trajectory.steps_taken
    improvement_pct = ((initial_steps - final_steps) / initial_steps) * 100
    assert improvement_pct > 0  # Shows efficiency improvement
```

---

## Test Fixtures

### Shared Fixtures (`conftest.py`)

**Configuration Fixtures**:
- `test_config`: Basic ReasoningBankConfig for testing
- `test_config_with_api_keys`: Config with mock API keys
- `temp_memory_bank`: Temporary directory for memory storage

**Mock Response Fixtures**:
- `mock_judge_responses`: LLM responses for judge (success/failure)
- `mock_extractor_responses`: LLM responses for memory extraction
- `mock_embedding_responses`: Embedding vectors for retrieval testing

**Data Fixtures**:
- `sample_successful_trajectory`: Realistic successful trajectory
- `sample_failed_trajectory`: Realistic failed trajectory
- `sample_memory_items`: Pre-extracted memory items

### Fixture Files

#### `sample_trajectories.py`
Realistic trajectory examples covering:
- Arithmetic tasks (calculator usage)
- Search tasks (web search)
- Navigation tasks (clicking, form filling)
- Multi-step reasoning chains
- Success and failure cases

**Example**:
```python
ARITHMETIC_SUCCESS_TRAJECTORY = {
    "query": "Calculate 25 * 4",
    "trajectory": "<think>I need to multiply 25 by 4</think>\n<action>calculate 25*4</action>\n<observation>100</observation>",
    "final_state": "100",
    "model_output": "Answer: 100",
    "steps_taken": 3,
    "success": True
}
```

#### `sample_memories.py`
Pre-extracted memory items:
- Strategy memories (from successful trajectories)
- Preventative memories (from failed trajectories)
- Domain-specific memories (arithmetic, search, navigation)

**Example**:
```python
ARITHMETIC_STRATEGY_MEMORY = MemoryItem(
    title="Use Calculator for Arithmetic",
    description="When asked to calculate, use the calculator action directly",
    content="For arithmetic calculations, use the calculator action with proper syntax: calculate <expression>. This is more reliable than manual computation.",
    success_signal=True
)
```

#### `mock_environments.py`
Simulated agent environments:
- `create_arithmetic_environment()`: Calculator with basic operations
- `create_search_environment()`: Web search with realistic results
- `create_navigation_environment()`: Web navigation with clickable elements

**Example**:
```python
def create_arithmetic_environment() -> MockEnvironment:
    """Create arithmetic environment for testing"""
    def execute_action(action: str) -> str:
        if action.lower().startswith("calculate"):
            expression = action.split("calculate", 1)[1].strip()
            try:
                result = eval(expression)  # In real env, use safe calculator
                return str(result)
            except:
                return "Error: Invalid expression"
        return "Unknown action"

    return MockEnvironment(
        name="arithmetic",
        description="Calculator environment",
        execute_action=execute_action
    )
```

---

## Mocking Strategy

### LLM Call Mocking

All tests mock LLM calls to avoid external API dependencies and ensure deterministic results.

**Pattern**:
```python
from unittest.mock import patch

def test_with_mocked_llm():
    """Test with mocked LLM call"""
    agent = ReasoningBankAgent(config)

    # Mock the LLM response
    mock_response = "<think>Reasoning</think>\n<action>Answer: 42</action>"

    with patch.object(agent, '_call_agent_llm', return_value=mock_response):
        result = agent.run("What is the answer?", max_steps=5)

    assert result.model_output == "Answer: 42"
```

### Judge Mocking

Mock judge responses for success/failure:
```python
def test_judge_evaluation():
    """Test trajectory judgment"""
    judge = TrajectoryJudge(config)

    # Mock success response
    with patch.object(judge, '_call_llm', return_value="SUCCESS"):
        success = judge.judge_trajectory_success(query, trajectory, ...)

    assert success is True
```

### Extractor Mocking

Mock extraction responses with Markdown format:
```python
def test_memory_extraction():
    """Test memory extraction"""
    extractor = MemoryExtractor(config)

    # Mock extraction response (Markdown format)
    mock_response = """
# Memory Item 1
## Title Strategy Title
## Description One sentence description
## Content Detailed content explaining the strategy in 1-5 sentences.
"""

    with patch.object(extractor, '_call_llm', return_value=mock_response):
        memories = extractor.extract_memories(...)

    assert len(memories) == 1
    assert memories[0].title == "Strategy Title"
```

### Environment Mocking

Use MockEnvironment for agent testing:
```python
def test_agent_with_environment():
    """Test agent with mock environment"""
    from tests.fixtures.mock_environments import create_arithmetic_environment

    env = create_arithmetic_environment()
    agent = ReasoningBankAgent(config, environment=env.execute_action)

    result = agent.run("What is 25 * 4?", max_steps=10)

    assert result.success is True
```

---

## Coverage Requirements

### Minimum Coverage Targets

- **Overall Project**: ≥90%
- **Core Components**: ≥95% (judge, extractor, retriever, consolidator)
- **Agent Logic**: ≥90%
- **MaTTS Strategies**: ≥85%
- **Configuration**: 100%
- **Data Models**: 100%

### Measuring Coverage

**Generate coverage report**:
```bash
pytest --cov=reasoningbank --cov-report=html --cov-report=term-missing
```

**View uncovered lines**:
```bash
pytest --cov=reasoningbank --cov-report=term-missing
```

**Coverage by file**:
```bash
pytest --cov=reasoningbank --cov-report=term
```

### Excluded from Coverage

- `__init__.py` files with only imports
- Debug/logging code
- Exception handlers for external API errors (tested via mocking)

---

## Writing New Tests

### Test Structure

Follow this structure for new tests:

```python
"""
ABOUTME: Tests for [component name]
ABOUTME: Tests [specific functionality]
"""

import pytest
from unittest.mock import Mock, patch
from reasoningbank import [components]

@pytest.fixture
def my_fixture():
    """Create fixture for testing"""
    # Setup
    resource = create_test_resource()
    yield resource
    # Teardown (if needed)

@pytest.mark.[category]  # unit, integration, matts
class Test[ComponentName]:
    """Tests for [component] [aspect]"""

    def test_[functionality]_[scenario](self, my_fixture):
        """Test [what] [when] [condition]"""
        # Arrange
        component = MyComponent(config)

        # Act
        result = component.do_something()

        # Assert
        assert result == expected
```

### Test Naming Convention

- Test functions: `test_[what]_[scenario]`
- Test classes: `Test[ComponentName][Aspect]`
- Fixtures: `[resource]_fixture` or just `[resource]`

**Examples**:
- `test_judge_success_parsing()`
- `test_extractor_enforces_max_items()`
- `test_retriever_ranks_by_similarity()`
- `TestTrajectoryJudgeBasic`
- `TestMemoryExtractionWithSelfContrast`

### Docstrings

Every test must have a docstring explaining:
1. What is being tested
2. Expected behavior
3. Paper reference (if validating paper specification)

**Example**:
```python
def test_temperature_settings_match_paper(self):
    """
    Test temperature settings match paper specifications (Appendix A.2).

    Paper specifies:
    - Agent: 0.7 (balanced exploration)
    - Judge: 0.0 (deterministic judgments)
    - Extractor: 1.0 (diverse extraction)
    - Selector: 0.0 (consistent selection)
    """
    config = ReasoningBankConfig(...)

    assert config.agent_temperature == 0.7
    assert config.judge_temperature == 0.0
    assert config.extractor_temperature == 1.0
    assert config.selector_temperature == 0.0
```

### Assertion Messages

Use descriptive assertion messages:

```python
# Good
assert result.success is True, "Trajectory should be judged successful"
assert len(memories) == 3, f"Expected 3 memories, got {len(memories)}"

# Bad
assert result.success
assert len(memories) == 3
```

---

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest --cov=reasoningbank --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

Install:
```bash
pip install pre-commit
pre-commit install
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ImportError: No module named 'reasoningbank'`

**Solution**:
```bash
# Install package in development mode
pip install -e .
```

#### 2. Fixture Not Found

**Problem**: `fixture 'test_config' not found`

**Solution**: Ensure `conftest.py` is in the correct location (tests/ directory)

#### 3. Temporary Files Not Cleaned Up

**Problem**: Test artifacts remain after test run

**Solution**: Use `tempfile.TemporaryDirectory()` context manager:
```python
def test_with_temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test code using tmpdir
        pass
    # tmpdir automatically cleaned up
```

#### 4. Tests Pass Individually But Fail Together

**Problem**: Tests interfere with each other

**Solution**:
- Use fresh fixtures for each test
- Avoid global state
- Use `pytest -x` to stop at first failure

#### 5. Slow Test Suite

**Problem**: Tests take too long to run

**Solution**:
- Run only changed tests: `pytest --lf` (last failed)
- Run in parallel: `pytest -n auto` (requires pytest-xdist)
- Skip slow tests: `pytest -m "not matts"`

### Debugging Tests

**Run single test with output**:
```bash
pytest tests/unit/test_judge.py::test_judge_success_parsing -s -vv
```

**Drop into debugger on failure**:
```bash
pytest --pdb
```

**Show local variables on failure**:
```bash
pytest -l
```

**Capture warnings**:
```bash
pytest -W all
```

---

## Paper Specification Validation

All tests validate against the ReasoningBank paper specifications:

### Section 3.2: Memory Schema
- Title: Concise name
- Description: 1 sentence
- Content: 1-5 sentences
- Success signal: Boolean

**Validated in**: `test_models.py`, `test_extractor.py`

### Section 3.3.1: Self-Contrast Extraction
- Compares k trajectories
- Extracts robust patterns
- Max 5 aggregated items

**Validated in**: `test_extractor.py`, `test_parallel.py`

### Section 3.3.2: Sequential Refinement
- Initial trajectory + k refinements
- Refinement prompts (Figure 10)
- Progressive improvement

**Validated in**: `test_sequential.py`

### Appendix A.1: Extraction Limits
- Max 3 items per trajectory
- Max 5 aggregated items

**Validated in**: `test_config.py`, `test_extractor.py`

### Appendix A.2: Temperature Settings
- Agent: 0.7
- Judge: 0.0
- Extractor: 1.0
- Selector: 0.0

**Validated in**: `test_config.py`, all component tests

### Figure 8: Extraction Prompts
- Markdown output format
- Dual prompts (success/failure)

**Validated in**: `test_extractor.py`

### Figure 9: Judge Prompt
- Expert evaluation framework
- Three task types
- Success/failure determination

**Validated in**: `test_judge.py`

### Figure 10: Refinement Prompts
- Review previous attempt
- Try different strategy
- Execute best approach

**Validated in**: `test_sequential.py`

---

## Quick Reference

### Run Tests
```bash
./tests/run_all_tests.py        # All tests
pytest -m unit                   # Unit tests only
pytest -m integration            # Integration tests only
pytest -m matts                  # MaTTS tests only
pytest --cov=reasoningbank       # With coverage
```

### Test Files
- **Unit**: `tests/unit/test_*.py` (6 files, 200+ tests)
- **Integration**: `tests/integration/test_*.py` (2 files, 85 tests)
- **MaTTS**: `tests/matts/test_*.py` (2 files, 100+ tests)

### Markers
- `@pytest.mark.unit`: Unit test
- `@pytest.mark.integration`: Integration test
- `@pytest.mark.matts`: MaTTS scaling test

### Coverage Target
- Overall: ≥90%
- Core components: ≥95%
- Configuration: 100%

---

## Additional Resources

- **ReasoningBank Paper**: "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory" (Google Cloud AI Research + UIUC, September 2025)
- **Pytest Documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **Project README**: `README.md`
- **API Documentation**: `docs/API.md`

---

## Contact

For questions about the test suite:
- Create an issue in the repository
- Check existing test files for examples
- Review this documentation for guidelines

**Last Updated**: October 2025
**Test Suite Version**: 1.0.0
