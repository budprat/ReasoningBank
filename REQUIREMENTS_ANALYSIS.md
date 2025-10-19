# ReasoningBank Requirements Analysis

**🔬 Comprehensive requirements for running and testing ReasoningBank implementation

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Python Version Requirements](#python-version-requirements)
3. [Core Dependencies](#core-dependencies)
4. [LLM Models and Providers](#llm-models-and-providers)
5. [API Keys and Authentication](#api-keys-and-authentication)
6. [Embedding Models](#embedding-models)
7. [Testing Dependencies](#testing-dependencies)
8. [Optional Dependencies](#optional-dependencies)
9. [System Requirements](#system-requirements)
10. [Installation Guide](#installation-guide)
11. [Configuration Requirements](#configuration-requirements)
12. [Quick Start Checklist](#quick-start-checklist)

---

## Executive Summary

ReasoningBank is a self-evolving agent system with reasoning memory that requires:

- **Python**: 3.9+ (recommended: 3.10 or 3.11)
- **Primary LLM SDKs**: Anthropic Claude, OpenAI, Google Generative AI
- **Embeddings**: Google Gemini Embedding (768d) or OpenAI (1536d)
- **Testing**: pytest, numpy, unittest.mock
- **Storage**: Local JSON files (no database required)
- **API Keys**: At least ONE of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY

**Minimum to Run**:
```bash
pip install anthropic openai google-generativeai numpy pytest
export ANTHROPIC_API_KEY="your-key-here"
python -c "from reasoningbank import create_agent; agent = create_agent()"
```

---

## Python Version Requirements

### Minimum Version
- **Python 3.9+** (required)

### Recommended Version
- **Python 3.10** or **Python 3.11** (best compatibility with all dependencies)

### Version-Specific Requirements
```python
# Type hints require Python 3.9+
from typing import Literal, Optional  # Literal added in 3.8

# tuple[bool, float] syntax requires Python 3.9+
def judge_with_confidence(...) -> tuple[bool, float]:  # 3.9+ syntax
    ...
```

### Checking Your Python Version
```bash
python --version  # Should show 3.9 or higher
python3 --version # Alternative command

# If you have multiple versions
python3.10 --version
python3.11 --version
```

---

## Core Dependencies

### 1. LLM Provider SDKs

#### Anthropic Claude SDK
```bash
pip install anthropic
```

**Version**: Latest (as of Oct 2025: 0.34.0+)

**Used For**:
- Agent reasoning (claude-3-5-sonnet-20241022)
- Trajectory judging (temperature=0.0)
- Memory extraction (temperature=1.0)

**Import Pattern**:
```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2000,
    temperature=0.7,
    messages=[{"role": "user", "content": "Your prompt"}]
)
```

**Key Models**:
- `claude-3-5-sonnet-20241022` (recommended) - Best balance of capability and speed
- `claude-3-opus-20240229` - Highest capability (more expensive)
- `claude-3-haiku-20240307` - Fastest, most cost-effective

#### OpenAI SDK
```bash
pip install openai
```

**Version**: Latest (as of Oct 2025: 1.0.0+)

**Used For**:
- Alternative LLM provider for agent, judge, extractor
- Embeddings via text-embedding-3-small (1536 dimensions)

**Import Pattern**:
```python
import openai

client = openai.OpenAI(api_key="your-api-key")

# For LLM calls
response = client.chat.completions.create(
    model="gpt-4",
    max_tokens=2000,
    temperature=0.7,
    messages=[{"role": "user", "content": "Your prompt"}]
)

# For embeddings
embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Your text to embed"
)
```

**Key Models**:
- `gpt-4` - High capability (expensive)
- `gpt-4-turbo` - Faster GPT-4 variant
- `gpt-3.5-turbo` - Cost-effective option
- `text-embedding-3-small` - Embeddings (1536d)

#### Google Generative AI SDK
```bash
pip install google-generativeai
```

**Version**: Latest (as of Oct 2025: 0.3.0+)

**Used For**:
- Paper's original model (gemini-2.5-flash)
- Embeddings via gemini-embedding-001 (768 dimensions) - **Paper default**

**Import Pattern**:
```python
from google import generativeai as genai

genai.configure(api_key="your-api-key")

# For LLM calls
model = genai.GenerativeModel("gemini-2.5-flash")
generation_config = genai.GenerationConfig(
    temperature=0.7,
    max_output_tokens=2000
)
response = model.generate_content("Your prompt", generation_config=generation_config)

# For embeddings
result = genai.embed_content(
    model="gemini-embedding-001",
    content="Your text to embed",
    task_type="retrieval_document"
)
```

**Key Models**:
- `gemini-2.5-flash` - Paper's model (fast, cost-effective)
- `gemini-2.0-pro` - Higher capability
- `gemini-embedding-001` - Embeddings (768d) - **Paper default**

### 2. Data Processing Libraries

#### NumPy
```bash
pip install numpy
```

**Version**: 1.20.0+ (recommended: latest)

**Used For**:
- Cosine similarity computation in memory retrieval
- Vector operations for embeddings

**Import Pattern**:
```python
import numpy as np

# Example from retriever.py
v1 = np.array(embedding1)
v2 = np.array(embedding2)
dot_product = np.dot(v1, v2)
norm1 = np.linalg.norm(v1)
similarity = dot_product / (norm1 * norm2)
```

**Critical Functions**:
- `np.array()` - Convert lists to arrays
- `np.dot()` - Dot product for similarity
- `np.linalg.norm()` - Vector normalization

---

## LLM Models and Providers

### Model Selection Matrix

| Provider | Agent Model | Judge Model | Extractor Model | Embedding Model | Default |
|----------|-------------|-------------|-----------------|-----------------|---------|
| **Anthropic** | claude-3-5-sonnet-20241022 | Same | Same | N/A | ✅ |
| **Google** | gemini-2.5-flash | Same | Same | gemini-embedding-001 | Paper |
| **OpenAI** | gpt-4 | Same | Same | text-embedding-3-small | No |

### Temperature Settings (From Paper Appendix A.2)

| Component | Temperature | Rationale |
|-----------|-------------|-----------|
| **Agent** | 0.7 | Balanced exploration/exploitation |
| **Judge** | 0.0 | Deterministic, consistent judgments |
| **Extractor** | 1.0 | Diverse memory extraction |
| **Selector** | 0.0 | Consistent trajectory selection |

### Model Requirements by Component

#### 1. ReasoningBankAgent (Main Agent)
**Required Capabilities**:
- Long context window (≥30K tokens recommended, 100K ideal)
- ReAct format understanding (thinking + action)
- JSON/structured output parsing
- Tool/function calling (optional but helpful)

**Recommended Models**:
1. **Claude 3.5 Sonnet** (claude-3-5-sonnet-20241022) - Best overall
2. **Gemini 2.5 Flash** (gemini-2.5-flash) - Paper's choice, fast
3. **GPT-4** (gpt-4) - High quality but expensive

**Temperature**: 0.7 (configurable via `agent_temperature`)

#### 2. TrajectoryJudge (LLM-as-a-Judge)
**Required Capabilities**:
- Binary classification (success/failure)
- Understanding of task descriptions and agent traces
- Consistency across multiple calls

**Recommended Models**: Same as agent (shared model in practice)

**Temperature**: 0.0 (deterministic)

**Typical Token Usage**: 10-50 tokens (just "success" or "failure")

#### 3. MemoryExtractor
**Required Capabilities**:
- Summarization and abstraction
- Markdown formatting
- Following strict output format instructions

**Recommended Models**: Same as agent

**Temperature**: 1.0 (diverse extraction)

**Typical Token Usage**: 500-2000 tokens (multiple memory items)

#### 4. Embedding Models (MemoryRetriever)
**Required Capabilities**:
- Text-to-vector conversion
- Consistent dimensions (768 or 1536)
- Fast inference

**Recommended Models**:
1. **gemini-embedding-001** (768d) - **Paper default**, Google
2. **text-embedding-3-small** (1536d) - OpenAI alternative

**No temperature** (embeddings are deterministic)

### Cost Considerations

**Anthropic Claude Pricing (Approximate)**:
- Claude 3.5 Sonnet: $3/MTok input, $15/MTok output
- Per agent run: ~$0.01-0.05 (depends on trajectory length)

**Google Gemini Pricing (Approximate)**:
- Gemini 2.5 Flash: $0.075/MTok input, $0.30/MTok output
- Gemini Embedding: $0.025/MTok
- Per agent run: ~$0.001-0.01 (very cost-effective)

**OpenAI Pricing (Approximate)**:
- GPT-4: $10/MTok input, $30/MTok output
- text-embedding-3-small: $0.02/MTok
- Per agent run: ~$0.03-0.10 (most expensive)

**Cost-Optimized Configuration**:
```python
config = ReasoningBankConfig(
    llm_provider="google",           # Cheapest provider
    llm_model="gemini-2.5-flash",    # Fast and cheap
    embedding_model="gemini-embedding-001"  # Paper default
)
```

**Highest Quality Configuration**:
```python
config = ReasoningBankConfig(
    llm_provider="anthropic",
    llm_model="claude-3-5-sonnet-20241022",  # Best reasoning
    embedding_model="gemini-embedding-001"    # Can mix providers
)
```

---

## API Keys and Authentication

### Required API Keys

You need **AT LEAST ONE** of the following:

#### Option 1: Anthropic (Recommended)
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

**Get API Key**: https://console.anthropic.com/

**Advantages**:
- Best reasoning capabilities
- Long context windows (200K tokens)
- Strong instruction following

#### Option 2: Google (Paper Default)
```bash
export GOOGLE_API_KEY="AIza..."
```

**Get API Key**: https://makersuite.google.com/app/apikey

**Advantages**:
- Paper's original configuration
- Most cost-effective
- Integrated embeddings

#### Option 3: OpenAI
```bash
export OPENAI_API_KEY="sk-..."
```

**Get API Key**: https://platform.openai.com/api-keys

**Advantages**:
- Widely supported
- Good embeddings
- Flexible models

### Setting API Keys

#### Method 1: Environment Variables (Recommended)
```bash
# In ~/.bashrc, ~/.zshrc, or ~/.bash_profile
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"

# Reload shell
source ~/.bashrc
```

#### Method 2: .env File (Development)
```bash
# Create .env in project root
echo "ANTHROPIC_API_KEY=your-key-here" >> .env
echo "GOOGLE_API_KEY=your-key-here" >> .env
echo "OPENAI_API_KEY=your-key-here" >> .env

# Load with python-dotenv
pip install python-dotenv
```

```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env variables
```

#### Method 3: Direct Configuration
```python
from reasoningbank import ReasoningBankConfig

config = ReasoningBankConfig(
    llm_provider="anthropic",
    llm_api_key="your-key-here"  # Direct specification
)
```

### Verifying API Keys

```python
import os

# Check which keys are available
print("ANTHROPIC_API_KEY:", "✓" if os.getenv("ANTHROPIC_API_KEY") else "✗")
print("GOOGLE_API_KEY:", "✓" if os.getenv("GOOGLE_API_KEY") else "✗")
print("OPENAI_API_KEY:", "✓" if os.getenv("OPENAI_API_KEY") else "✗")
```

### API Key Security Best Practices

1. **Never commit API keys to git**:
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   echo "*.key" >> .gitignore
   ```

2. **Use environment variables in production**:
   ```bash
   # In production environment (e.g., Docker, Heroku)
   # Set via platform-specific configuration
   ```

3. **Rotate keys regularly**: Generate new keys every 90 days

4. **Use different keys for dev/prod**: Don't mix development and production keys

---

## Embedding Models

### Embedding Model Comparison

| Model | Provider | Dimensions | Cost | Performance | Paper Default |
|-------|----------|------------|------|-------------|---------------|
| **gemini-embedding-001** | Google | 768 | $0.025/MTok | Fast | ✅ |
| **text-embedding-3-small** | OpenAI | 1536 | $0.02/MTok | High Quality | No |
| **text-embedding-3-large** | OpenAI | 3072 | $0.13/MTok | Best Quality | No |

### Configuration

#### Paper Configuration (Recommended)
```python
config = ReasoningBankConfig(
    embedding_model="gemini-embedding-001",
    embedding_dimension=768
)
```

#### OpenAI Alternative
```python
config = ReasoningBankConfig(
    embedding_model="text-embedding-3-small",
    embedding_dimension=1536
)
```

### Embedding Usage

**Automatic Caching**: ReasoningBank automatically caches embeddings to `./data/embeddings.json` to avoid redundant API calls.

**Cache Location**:
```python
config = ReasoningBankConfig(
    embedding_cache_path="./data/embeddings.json"  # Customizable
)
```

**Cache Benefits**:
- Reduces API costs (avoid re-embedding same text)
- Faster retrieval (no API call for cached embeddings)
- Persistent across sessions

---

## Testing Dependencies

### Core Testing Libraries

#### pytest
```bash
pip install pytest
```

**Version**: 7.0.0+ (recommended: latest)

**Used For**:
- Test discovery and execution
- Fixtures (shared test data)
- Test markers (@pytest.mark.unit, @pytest.mark.integration)
- Parametrized tests

**Usage**:
```bash
pytest                              # Run all tests
pytest tests/unit/                  # Unit tests only
pytest -m integration               # Integration tests only
pytest --cov=reasoningbank          # With coverage
```

#### pytest-cov (Coverage)
```bash
pip install pytest-cov
```

**Used For**:
- Code coverage measurement
- Coverage reports (HTML, terminal, XML)

**Usage**:
```bash
pytest --cov=reasoningbank --cov-report=html
```

### Testing Support Libraries

All standard library (no additional installation):
- **unittest.mock**: Mocking LLM calls and external APIs
- **tempfile**: Temporary directories for test isolation
- **json**: Test data serialization
- **os**: File system operations

### Optional Testing Enhancements

#### pytest-xdist (Parallel Testing)
```bash
pip install pytest-xdist
```

**Benefits**: Run tests in parallel for faster execution

**Usage**:
```bash
pytest -n auto  # Automatic CPU core detection
pytest -n 4     # Use 4 workers
```

#### pytest-timeout (Test Timeouts)
```bash
pip install pytest-timeout
```

**Benefits**: Prevent hanging tests

**Usage**:
```bash
pytest --timeout=60  # 60 second timeout per test
```

---

## Optional Dependencies

### Development Tools

#### Black (Code Formatting)
```bash
pip install black
```

**Usage**:
```bash
black reasoningbank/  # Format code
black --check reasoningbank/  # Check without modifying
```

#### Pylint/Flake8 (Linting)
```bash
pip install pylint flake8
```

**Usage**:
```bash
pylint reasoningbank/
flake8 reasoningbank/
```

#### MyPy (Type Checking)
```bash
pip install mypy
```

**Usage**:
```bash
mypy reasoningbank/
```

### Environment Management

#### python-dotenv
```bash
pip install python-dotenv
```

**Used For**: Loading environment variables from .env file

**Usage**:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## System Requirements

### Hardware Requirements

#### Minimum
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4 GB
- **Storage**: 1 GB free space
- **Network**: Stable internet connection for API calls

#### Recommended
- **CPU**: 4+ cores, 2.5+ GHz
- **RAM**: 8+ GB
- **Storage**: 5+ GB free space (for logs, embeddings, memory bank)
- **Network**: High-speed internet (LLM API calls)

### Operating System

**Supported**:
- **Linux**: Ubuntu 20.04+, Debian 10+, CentOS 8+
- **macOS**: 10.15+ (Catalina or later)
- **Windows**: Windows 10+ (with WSL2 recommended)

**Tested On**:
- Ubuntu 22.04 LTS
- macOS Ventura 13+
- Windows 11 with WSL2 (Ubuntu 22.04)

### Network Requirements

- **Outbound HTTPS** (port 443) for API calls
- **Bandwidth**: ~1-10 KB per agent step (varies by trajectory length)
- **Latency**: <500ms recommended for good UX

### Storage Requirements

**Initial Installation**: ~500 MB
- Python packages: ~200 MB
- ReasoningBank codebase: ~10 MB
- Test fixtures: ~5 MB

**Runtime Storage**:
- Memory bank: 1-100 MB (grows with usage)
- Embedding cache: 10-500 MB (grows with unique queries)
- Logs: 1-10 MB/day (if logging enabled)

**Recommended**: 5 GB free space for comfortable operation

---

## Installation Guide

### Quick Install (Recommended)

#### Step 1: Clone Repository
```bash
git clone https://github.com/your-org/ReasoningBank.git
cd ReasoningBank
```

#### Step 2: Create Virtual Environment
```bash
# Using venv (built-in)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows

# Using conda (alternative)
conda create -n reasoningbank python=3.10
conda activate reasoningbank
```

#### Step 3: Install Dependencies
```bash
# Install all requirements
pip install anthropic openai google-generativeai numpy pytest

# Verify installation
python -c "import anthropic, openai, google.generativeai, numpy, pytest; print('✓ All dependencies installed')"
```

#### Step 4: Set API Key
```bash
# Choose one provider
export ANTHROPIC_API_KEY="your-key-here"
# OR
export GOOGLE_API_KEY="your-key-here"
# OR
export OPENAI_API_KEY="your-key-here"
```

#### Step 5: Install ReasoningBank
```bash
# Install in development mode (editable)
pip install -e .

# Verify installation
python -c "from reasoningbank import create_agent; print('✓ ReasoningBank installed')"
```

#### Step 6: Run Tests
```bash
pytest tests/unit/ -v
```

### Creating requirements.txt

```bash
# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core LLM SDKs
anthropic>=0.34.0
openai>=1.0.0
google-generativeai>=0.3.0

# Data Processing
numpy>=1.20.0

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0

# Optional Development Tools
python-dotenv>=0.19.0
EOF

# Install from requirements.txt
pip install -r requirements.txt
```

### Docker Installation (Advanced)

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY reasoningbank/ ./reasoningbank/
COPY tests/ ./tests/

# Set environment variables
ENV PYTHONPATH=/app

CMD ["python", "-m", "pytest", "tests/"]
```

```bash
# Build and run
docker build -t reasoningbank .
docker run -e ANTHROPIC_API_KEY="your-key" reasoningbank
```

---

## Configuration Requirements

### Minimum Configuration

```python
from reasoningbank import ReasoningBankConfig

config = ReasoningBankConfig(
    llm_api_key="your-api-key",
    memory_bank_path="./data/memory_bank.json"
)
```

### Recommended Configuration

```python
config = ReasoningBankConfig(
    # LLM Settings
    llm_provider="anthropic",
    llm_model="claude-3-5-sonnet-20241022",
    llm_api_key="your-api-key",

    # Temperature Settings (from paper)
    agent_temperature=0.7,
    judge_temperature=0.0,
    extractor_temperature=1.0,

    # Memory Settings
    memory_bank_path="./data/memory_bank.json",
    embedding_model="gemini-embedding-001",
    embedding_dimension=768,
    top_k_retrieval=1,

    # Extraction Limits (from paper)
    max_memory_items_per_trajectory=3,
    max_memory_items_aggregated=5,

    # Agent Settings
    max_steps_per_task=30,
    enable_memory_injection=True,

    # Logging
    enable_logging=True,
    log_file="./data/reasoningbank.log"
)
```

### Production Configuration

```python
import os

config = ReasoningBankConfig(
    # Use environment variables
    llm_api_key=os.getenv("ANTHROPIC_API_KEY"),

    # Production paths
    memory_bank_path="/var/lib/reasoningbank/memory_bank.json",
    embedding_cache_path="/var/cache/reasoningbank/embeddings.json",
    log_file="/var/log/reasoningbank/app.log",

    # Metrics for monitoring
    track_metrics=True,
    metrics_file="/var/lib/reasoningbank/metrics.json",

    # Conservative settings
    max_steps_per_task=15,  # Prevent runaway costs
    top_k_retrieval=1       # Reduce token usage
)
```

### Configuration Presets

#### Paper Replication
```python
from reasoningbank import get_config_for_paper_replication

config = get_config_for_paper_replication()
# Uses: Gemini 2.5 Flash, Gemini embeddings, paper settings
```

#### Claude-Based
```python
from reasoningbank import get_config_for_claude

config = get_config_for_claude()
# Uses: Claude 3.5 Sonnet, optimized settings
```

#### MaTTS Parallel
```python
from reasoningbank import get_config_for_matts_parallel

config = get_config_for_matts_parallel(k=3)
# Uses: Parallel scaling with k=3 trajectories
```

#### MaTTS Sequential
```python
from reasoningbank import get_config_for_matts_sequential

config = get_config_for_matts_sequential(k=3)
# Uses: Sequential refinement with k=3 iterations
```

---

## Quick Start Checklist

### Prerequisites ✅

- [ ] Python 3.9+ installed
- [ ] pip or conda available
- [ ] Virtual environment created and activated
- [ ] At least one LLM API key obtained

### Installation ✅

- [ ] Cloned ReasoningBank repository
- [ ] Installed core dependencies: `pip install anthropic openai google-generativeai numpy pytest`
- [ ] Set API key environment variable
- [ ] Installed ReasoningBank: `pip install -e .`
- [ ] Verified installation: `python -c "from reasoningbank import create_agent"`

### Configuration ✅

- [ ] Created data directory: `mkdir -p data`
- [ ] Configured memory bank path: `./data/memory_bank.json`
- [ ] Set appropriate temperatures (agent=0.7, judge=0.0, extractor=1.0)
- [ ] Chosen embedding model (gemini-embedding-001 or text-embedding-3-small)

### Testing ✅

- [ ] Ran unit tests: `pytest tests/unit/ -v`
- [ ] Ran integration tests: `pytest tests/integration/ -v` (optional, requires LLM calls)
- [ ] Checked test coverage: `pytest --cov=reasoningbank`

### First Agent Run ✅

```python
from reasoningbank import create_agent

# Create agent with default configuration
agent = create_agent()

# Run a simple task
def mock_env(action):
    if "calculate" in action.lower():
        return "100"
    return "Action executed"

result = agent.run(
    query="Calculate 25 * 4",
    max_steps=10,
    enable_memory_injection=False  # First run, no prior memories
)

print(f"Success: {result.success}")
print(f"Output: {result.model_output}")
print(f"Memories extracted: {len(result.memory_items)}")
```

### Verification Commands

```bash
# Check Python version
python --version  # Should be 3.9+

# Check packages
pip list | grep -E "anthropic|openai|google-generativeai|numpy|pytest"

# Check API key
python -c "import os; print('API Key:', '✓' if os.getenv('ANTHROPIC_API_KEY') else '✗')"

# Run smoke test
python -c "from reasoningbank import create_agent; agent = create_agent(); print('✓ Agent created successfully')"

# Run tests
pytest tests/unit/test_models.py -v  # Quick test
```

---

## Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'anthropic'
**Solution**: Install missing package
```bash
pip install anthropic
```

#### 2. API Key Not Found
**Error**: `ValueError: LLM API key not found`
**Solution**: Set environment variable
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

#### 3. Type Hints Error (Python 3.8)
**Error**: `TypeError: 'type' object is not subscriptable`
**Solution**: Upgrade to Python 3.9+
```bash
python3.10 -m venv venv
source venv/bin/activate
```

#### 4. Numpy Import Error
**Solution**: Install numpy
```bash
pip install numpy
```

#### 5. Test Failures Due to Missing pytest
**Solution**: Install test dependencies
```bash
pip install pytest pytest-cov
```

---

## Additional Resources

- **Paper**: "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory" (Google Cloud AI Research + UIUC, September 2025)
- **Testing Guide**: `README_TESTING.md`
- **Project README**: `README.md`
- **API Documentation**: `docs/API.md` (if available)

---

## Summary Table

| Requirement | Minimum | Recommended | Notes |
|-------------|---------|-------------|-------|
| **Python** | 3.9 | 3.10/3.11 | Type hints require 3.9+ |
| **anthropic** | 0.34.0 | Latest | For Claude models |
| **openai** | 1.0.0 | Latest | For GPT models |
| **google-generativeai** | 0.3.0 | Latest | For Gemini models |
| **numpy** | 1.20.0 | Latest | For embeddings |
| **pytest** | 7.0.0 | Latest | For testing |
| **RAM** | 4 GB | 8 GB+ | Depends on usage |
| **Storage** | 1 GB | 5 GB | Grows with usage |
| **API Key** | 1 provider | All 3 | More flexibility |

---

**Last Updated**: October 2025
**Version**: 1.0.0
**Status**: ✅ Production Ready
