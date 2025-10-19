# ReasoningBank Installation Guide

Quick installation guide for ReasoningBank. For comprehensive requirements analysis, see [REQUIREMENTS_ANALYSIS.md](REQUIREMENTS_ANALYSIS.md).

## Quick Install (5 Minutes)

### 1. Prerequisites

- **Python 3.9+** (check: `python --version`)
- **pip** (check: `pip --version`)
- **At least one LLM API key** (Anthropic, Google, or OpenAI)

### 2. Clone Repository

```bash
git clone https://github.com/your-org/ReasoningBank.git
cd ReasoningBank
```

### 3. Create Virtual Environment

```bash
# Using venv (built-in)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS

# OR using conda
conda create -n reasoningbank python=3.10
conda activate reasoningbank
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `anthropic` - Claude SDK
- `openai` - OpenAI SDK
- `google-generativeai` - Gemini SDK
- `numpy` - For embeddings
- `pytest` - For testing

### 5. Install ReasoningBank

```bash
pip install -e .
```

### 6. Set API Key

Choose **ONE** provider:

**Option A: Anthropic (Recommended)**
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
```

**Option B: Google (Paper Default)**
```bash
export GOOGLE_API_KEY="AIza-your-key-here"
```

**Option C: OpenAI**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### 7. Verify Installation

```bash
# Quick verification
python -c "from reasoningbank import create_agent; print('✓ Installation successful!')"

# Run tests (optional)
pytest tests/unit/test_models.py -v
```

## Configuration

### Create .env File (Optional)

```bash
# Copy example environment file
cp .env.example .env

# Edit with your API keys
nano .env  # or use your preferred editor
```

### Create Data Directory

```bash
mkdir -p data
```

## First Run

```python
from reasoningbank import create_agent

# Create agent with default configuration
agent = create_agent()

# Define simple environment
def mock_environment(action):
    if "calculate" in action.lower():
        return "100"
    return "Action executed"

# Run agent
result = agent.run(
    query="Calculate 25 * 4",
    max_steps=10,
    enable_memory_injection=False
)

print(f"Success: {result.success}")
print(f"Output: {result.model_output}")
print(f"Memories: {len(result.memory_items)} items extracted")
```

## Troubleshooting

### Import Error
```bash
# If you get "No module named 'reasoningbank'"
pip install -e .
```

### API Key Not Found
```bash
# Verify API key is set
echo $ANTHROPIC_API_KEY

# Set permanently in ~/.bashrc or ~/.zshrc
echo 'export ANTHROPIC_API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

### Python Version Error
```bash
# Check Python version
python --version  # Should be 3.9+

# If too old, install newer version
# On Ubuntu/Debian:
sudo apt install python3.10

# Create venv with specific version:
python3.10 -m venv venv
```

## Next Steps

- Read [REQUIREMENTS_ANALYSIS.md](REQUIREMENTS_ANALYSIS.md) for detailed requirements
- Read [README_TESTING.md](README_TESTING.md) for testing guide
- Explore `examples/` directory for usage examples
- Check `docs/` for API documentation

## Getting API Keys

### Anthropic Claude
1. Visit https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create new key
5. Copy and save securely

### Google Gemini
1. Visit https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy and save securely

### OpenAI
1. Visit https://platform.openai.com/api-keys
2. Sign up or log in
3. Click "Create new secret key"
4. Copy and save securely (shown only once!)

## Installation Methods

### Method 1: Development Install (Recommended)
```bash
git clone https://github.com/your-org/ReasoningBank.git
cd ReasoningBank
pip install -e .
```

**Benefits**:
- Code changes reflected immediately
- Easy to modify and experiment
- Can run tests

### Method 2: Direct Install (Future)
```bash
pip install reasoningbank  # When published to PyPI
```

### Method 3: Docker Install
```bash
# Build image
docker build -t reasoningbank .

# Run with API key
docker run -e ANTHROPIC_API_KEY="your-key" reasoningbank
```

## Minimal Installation

If you only want core functionality:

```bash
# Minimal dependencies (choose one LLM provider)
pip install anthropic numpy  # Anthropic only
# OR
pip install google-generativeai numpy  # Google only
# OR
pip install openai numpy  # OpenAI only
```

## Full Development Installation

For contributors and developers:

```bash
# Install with all development tools
pip install -e ".[dev]"

# Or manually install dev dependencies
pip install -r requirements.txt
pip install black pylint flake8 mypy pytest-xdist
```

## Verification Checklist

After installation, verify:

- [ ] Python 3.9+ installed: `python --version`
- [ ] Virtual environment activated: `which python`
- [ ] Dependencies installed: `pip list | grep anthropic`
- [ ] ReasoningBank installed: `python -c "import reasoningbank"`
- [ ] API key set: `python -c "import os; print('✓' if os.getenv('ANTHROPIC_API_KEY') else '✗')"`
- [ ] Tests pass: `pytest tests/unit/test_models.py -v`
- [ ] Agent creation works: `python -c "from reasoningbank import create_agent; create_agent()"`

## System-Specific Instructions

### macOS
```bash
# Install Python via Homebrew
brew install python@3.10

# Install dependencies
python3.10 -m pip install -r requirements.txt
```

### Ubuntu/Debian
```bash
# Install Python
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# Install dependencies
python3.10 -m pip install -r requirements.txt
```

### Windows (WSL2 Recommended)
```bash
# In WSL2 Ubuntu
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
python3.10 -m pip install -r requirements.txt
```

### Windows (Native)
```powershell
# Download Python from python.org
# Install with "Add to PATH" option checked

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Support

For issues:
- Check [REQUIREMENTS_ANALYSIS.md](REQUIREMENTS_ANALYSIS.md) for detailed requirements
- Review [Troubleshooting](#troubleshooting) section above
- Create an issue on GitHub
- Check existing issues for solutions

---

**Quick Reference**:
```bash
# Complete installation in 5 commands:
git clone <repo-url> && cd ReasoningBank
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
export ANTHROPIC_API_KEY="your-key"
python -c "from reasoningbank import create_agent; print('✓ Success')"
```
