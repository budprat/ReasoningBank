# 🚀 ReasoningBank Quick Start Guide

**Get ReasoningBank running in 5 minutes.**

## Prerequisites

- Python 3.9+
- API key for at least one LLM provider (Anthropic, Google, OpenAI)

---

## 1. Quick Installation

### Installation using venv + pip

```bash
# Clone the repository
git clone https://github.com/budprat/ReasoningBank.git
cd ReasoningBank

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "from reasoningbank import ReasoningBankAgent, get_config_for_claude; print('✓ ReasoningBank installed successfully')"
```

---

## 2. Configure API Keys

### Option A: Environment Variable (Quick Testing)

```bash
# Anthropic Claude (recommended)
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"

# Or Google Gemini
export GOOGLE_API_KEY="your-google-key-here"

# Or OpenAI
export OPENAI_API_KEY="sk-your-openai-key-here"
```

### Option B: .env File (Persistent)

Create a `.env` file in the project root:

```bash
# Choose ONE or more providers:

# Anthropic Claude (recommended)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Gemini
GOOGLE_API_KEY=your_google_api_key_here

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 3. Basic Usage

### Simple Custom Agent (my_agent.py)

```python
from reasoningbank import ReasoningBankAgent, get_config_for_claude

# Create configuration using helper function
config = get_config_for_claude()
config.enable_logging = True

# Define a simple environment (or use the built-in mock)
def my_environment(action: str) -> str:
    """Simple environment that responds to actions."""
    if action.lower().startswith("answer:"):
        return "Task completed."
    elif "calculate" in action.lower():
        return "Calculation result: 42"
    else:
        return f"Executed: {action}"

# Create agent
agent = ReasoningBankAgent(config, environment=my_environment)

# Run a task
result = agent.run(
    query="What is 21 * 2?",
    max_steps=5
)

# Check results
print(f"Query: {result.query}")
print(f"Success: {result.success}")
print(f"Steps Taken: {result.steps_taken}")
print(f"Model Output: {result.model_output}")
print(f"Memory Items: {len(result.memory_items)}")
```

Run it:
```bash
python my_agent.py
```

### Using Custom Environments

```python
from reasoningbank import ReasoningBankAgent, ReasoningBankConfig

# Initialize configuration
config = ReasoningBankConfig(
    llm_provider="anthropic",
    agent_model="claude-3-5-sonnet-20241022",
    max_steps_per_trajectory=15,
    enable_memory_injection=True
)

# Define custom environment function
def knowledge_environment(action: str) -> str:
    """Custom environment for knowledge questions."""
    if "capital of france" in action.lower():
        return "The capital of France is Paris."
    elif action.lower().startswith("answer:"):
        return "Answer recorded."
    else:
        return f"Executed: {action}"

# Create agent with custom environment
agent = ReasoningBankAgent(config, environment=knowledge_environment)

# Run a query
result = agent.run(
    query="What is the capital of France?",
    max_steps=10,
    enable_memory_injection=True
)

print(f"Success: {result.success}")
print(f"Final Answer: {result.final_state}")
print(f"Memories Extracted: {len(result.memory_items)}")
```

---

## 4. Run Included Examples

### Basic Usage Example

```bash
# Run basic usage example
python examples/basic_usage.py
```

### MaTTS Parallel Example

```bash
# Run parallel scaling example
python examples/matts_parallel_example.py
```

### MaTTS Sequential Example

```bash
# Run sequential refinement example
python examples/matts_sequential_example.py
```

---

## 5. Run Tests

```bash
# Run all tests
python tests/run_all_tests.py
# Or with pytest: python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/test_agent.py -v      # Agent tests
python -m pytest tests/matts/ -v                  # MaTTS scaling tests
python -m pytest tests/unit/ -v                   # Unit tests only
python -m pytest tests/integration/ -v            # Integration tests

# Run with coverage report
python -m pytest --cov=reasoningbank --cov-report=html
# Then open: open htmlcov/index.html
```

---

## 5.5. Expected Output

When you run a task successfully, you should see output similar to this:

```
Query: What is 21 * 2?
Success: True
Steps Taken: 3
Model Output: Answer: 42
Memory Items Extracted: 2

Memory Bank Statistics:
  Total Entries: 1
  Successful: 1
  Failed: 0
  Success Rate: 100.00%
  Avg Steps per Task: 3.00
  Total Memory Items: 2
```

The agent will automatically:
1. **Retrieve** relevant memories from past tasks (if any exist)
2. **Execute** the query using ReAct format (Reasoning + Acting)
3. **Judge** whether the trajectory succeeded or failed
4. **Extract** new memory items (success patterns or failure lessons)
5. **Consolidate** memories into the persistent memory bank (`data/memory_bank.json`)

---

## 6. Configuration Options

Edit `reasoningbank/config.py` or pass custom config:

```python
config = ReasoningBankConfig(
    # LLM Provider Settings
    llm_provider="anthropic",  # "anthropic", "google", or "openai"
    agent_model="claude-3-5-sonnet-20241022",

    # Temperature Settings (from paper)
    agent_temperature=0.7,      # Balanced exploration
    judge_temperature=0.0,      # Deterministic judgments
    extractor_temperature=1.0,  # Diverse extraction
    selector_temperature=0.0,   # Consistent selection

    # Memory Settings
    enable_memory_injection=True,
    retrieval_k=5,  # Top-k memories to retrieve
    embedding_model="text-embedding-3-small",

    # MaTTS Settings
    parallel_k=3,  # Number of parallel trajectories
    sequential_refinement_iterations=2,

    # Execution Settings
    max_steps_per_trajectory=15,
    rate_limit_delay=1.0,
    max_retries=3
)
```

---

## 7. Interactive Session

```bash
# Start Python interactive session
python

# Or use IPython for better experience (if installed)
ipython
```

```python
# In the Python session:
from reasoningbank import ReasoningBankAgent, ReasoningBankConfig

# Define a simple environment function
def mock_environment(action: str) -> str:
    if "calculate" in action.lower():
        return "Calculation result: 850"
    elif action.lower().startswith("answer:"):
        return "Task completed."
    else:
        return f"Executed: {action}"

config = ReasoningBankConfig(llm_provider="anthropic")
agent = ReasoningBankAgent(config, environment=mock_environment)

# Run queries
result = agent.run("What is 25 * 34?", max_steps=10)
print(result.final_state)
```

---

## 8. Docker Alternative

```bash
# Build Docker image
docker build -t reasoningbank:latest .

# Run with environment variables
docker run -e ANTHROPIC_API_KEY=your_key_here \
    reasoningbank:latest \
    python examples/run_hotpotqa.py
```

---

## 9. Troubleshooting

### API Key Issues
```bash
# Verify your .env file exists
ls -la .env

# Test API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('ANTHROPIC_API_KEY'))"
```

### Import Errors
```bash
# Reinstall in editable mode
pip install -e .

# Verify installation
python -c "import reasoningbank; print(reasoningbank.__version__)"
```

### Rate Limit Issues
Increase `rate_limit_delay` in config:
```python
config = ReasoningBankConfig(rate_limit_delay=2.0)  # 2 seconds between calls
```

### Memory Retrieval Slow
For large memory banks (>1000 entries), consider using FAISS indexing (see advanced docs).

---

## 10. Next Steps

### Experiment with Different Tasks

```python
# Try different types of queries
result = agent.run("How do I find the largest file in a directory?", max_steps=10)
print(result.model_output)

# Check memory bank after multiple runs
stats = agent.get_statistics()
print(f"Total memories: {stats['total_entries']}")
print(f"Success rate: {stats['success_rate']:.2%}")
```

### Try MaTTS Scaling (Test-Time Scaling)

**Parallel Mode** (k-trajectory sampling with best-of-n selection):

```python
from reasoningbank import run_matts_parallel, get_config_for_matts_parallel

# Sample k=3 trajectories in parallel, select best
config = get_config_for_matts_parallel(k=3)
result = run_matts_parallel("Complex reasoning query", config)

print(f"Sampled {result.k} trajectories")
print(f"Best trajectory success: {result.success}")
print(f"Best score: {result.best_score}")
```

**Sequential Mode** (iterative refinement):

```python
from reasoningbank import run_matts_sequential, get_config_for_matts_sequential

# Refine answer over multiple iterations
config = get_config_for_matts_sequential(refinement_iterations=2)
result = run_matts_sequential("Complex reasoning query", config)

print(f"Refinement iterations: {result.refinement_iterations}")
print(f"Final success: {result.success}")
```

### Inspect Memory Bank

```python
# Get statistics
stats = agent.get_statistics()
print(stats)

# Get all memory entries
entries = agent.get_memory_bank()
for entry in entries:
    print(f"Task: {entry.query}")
    print(f"Success: {entry.success}")
    print(f"Memories: {len(entry.memory_items)}")
```

### Build Custom Environment

```python
def my_custom_environment(action: str) -> str:
    """Define your own environment for specific use cases."""
    # Your custom logic here
    if "search" in action.lower():
        return "Search results: ..."
    elif "calculate" in action.lower():
        return "Calculation: ..."
    else:
        return f"Executed: {action}"

agent = ReasoningBankAgent(config, environment=my_custom_environment)
```

### Further Resources

- 📖 **Read**: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for advanced setup
- 🏗️ **Explore**: [ARCHITECTURE_ANALYSIS.md](./ARCHITECTURE_ANALYSIS.md) for system design
- 🧪 **Experiment**: Run batch evaluations on different datasets
- 🔧 **Customize**: Create your own environments and tools
- 📊 **Analyze**: Use memory bank statistics to track learning progress

---

## Quick Reference Commands

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Run example
python examples/basic_usage.py

# Run tests
python tests/run_all_tests.py

# Check memory bank stats
python -c "from reasoningbank import create_consolidator; c = create_consolidator(); print(c.get_statistics())"

# Export memory bank
python -c "from reasoningbank import create_consolidator; c = create_consolidator(); c.export_to_json('memory_backup.json')"
```

---

**Questions?** Check [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed documentation.

**Issues?** See [ARCHITECTURE_ANALYSIS.md](./ARCHITECTURE_ANALYSIS.md) for system architecture and known limitations.

---

`★ Insight ─────────────────────────────────────`

**Key Points for Running ReasoningBank:**

1. **Simple Setup**: Just 5 commands and you're running (venv → install → API key → verify → run)

2. **Mock Environment by Default**: You don't need to define an environment function - the agent includes a built-in mock environment for testing

3. **Closed-Loop Learning**: Every task execution automatically:
   - Retrieves relevant memories (if any exist)
   - Executes with ReAct format (Reasoning + Acting)
   - Judges success/failure using LLM-as-a-Judge
   - Extracts new memories (success patterns or failure lessons)
   - Saves to memory bank for future use

4. **Persistent Memory**: The memory bank (`data/memory_bank.json`) persists across runs, so the agent continuously learns from experience

5. **Multiple Providers**: Works with Anthropic Claude, Google Gemini, or OpenAI - just change the config and API key

6. **Test-Time Scaling**: MaTTS provides two strategies:
   - **Parallel**: Sample k trajectories, select best (best-of-n)
   - **Sequential**: Iterative refinement with progressive improvement

`─────────────────────────────────────────────────`
