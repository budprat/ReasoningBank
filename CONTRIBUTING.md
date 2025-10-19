# Contributing to ReasoningBank

First off, thank you for considering contributing to ReasoningBank! It's people like you that make ReasoningBank such a great tool for the AI community.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps to reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed and expected**
* **Include Python version and package versions**
* **Include any error messages or logs**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a detailed description of the suggested enhancement**
* **Provide specific examples to demonstrate the enhancement**
* **Describe the current behavior and expected behavior**
* **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes: `pytest tests/`
5. Make sure your code follows the style guidelines
6. Issue that pull request!

## Development Process

### Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/your-username/ReasoningBank.git
cd ReasoningBank

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=reasoningbank --cov-report=html

# Run specific test file
pytest tests/unit/test_config.py

# Run with verbose output
pytest -v
```

### Code Style

We use Black for code formatting and follow PEP 8 guidelines:

```bash
# Format code
black reasoningbank tests

# Check linting
flake8 reasoningbank tests

# Type checking
mypy reasoningbank
```

### Writing Tests

* Write unit tests for all new functionality
* Place tests in appropriate directories:
  * `tests/unit/` - Unit tests for individual components
  * `tests/integration/` - Integration tests
  * `tests/e2e/` - End-to-end tests
* Use descriptive test names that explain what is being tested
* Include both positive and negative test cases
* Mock external API calls to avoid rate limits and costs

### Documentation

* Update README.md if you change functionality
* Add docstrings to all public functions and classes
* Use Google-style docstrings
* Update API documentation if you change interfaces

## Project Structure

```
ReasoningBank/
├── reasoningbank/       # Main package code
│   ├── __init__.py
│   ├── agent.py        # Main agent implementation
│   ├── config.py       # Configuration management
│   ├── models.py       # Data models
│   ├── judge.py        # Trajectory evaluation
│   ├── extractor.py    # Memory extraction
│   ├── retriever.py    # Memory retrieval
│   ├── consolidator.py # Memory storage
│   └── matts/          # MaTTS implementation
├── tests/              # Test suite
├── examples/           # Usage examples
├── docs/               # Documentation
└── data/               # Data storage
```

## Commit Message Guidelines

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

Types:
* **feat**: A new feature
* **fix**: A bug fix
* **docs**: Documentation only changes
* **style**: Changes that don't affect code meaning
* **refactor**: Code change that neither fixes a bug nor adds a feature
* **test**: Adding missing tests or correcting existing tests
* **chore**: Changes to the build process or auxiliary tools

## Review Process

All submissions require review. We use GitHub pull requests for this purpose. Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

### Review Criteria

* **Correctness**: Does the code do what it's supposed to?
* **Testing**: Is the code adequately tested?
* **Documentation**: Is the code documented?
* **Style**: Does the code follow our style guidelines?
* **Design**: Is the code well-designed and maintainable?

## Community

* Join our [Discord](https://discord.gg/reasoningbank) for discussions
* Follow [@ReasoningBank](https://twitter.com/reasoningbank) on Twitter
* Read our [blog](https://blog.reasoningbank.ai) for updates

## Recognition

Contributors who make significant contributions will be:
* Added to the AUTHORS file
* Mentioned in release notes
* Given credit in conference presentations
* Eligible for ReasoningBank swag!

## Questions?

Feel free to open an issue with the "question" label or reach out on Discord!

Thank you for contributing! 🚀