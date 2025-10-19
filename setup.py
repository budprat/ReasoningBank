"""
Setup script for ReasoningBank
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return "ReasoningBank: Self-Evolving Agent with Reasoning Memory"

# Core dependencies
REQUIRED_PACKAGES = [
    "anthropic>=0.34.0",
    "openai>=1.0.0",
    "google-generativeai>=0.3.0",
    "numpy>=1.20.0",
]

# Testing dependencies
TEST_PACKAGES = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

# Development dependencies
DEV_PACKAGES = [
    "python-dotenv>=0.19.0",
    "black>=23.0.0",
    "pylint>=2.15.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]

# Optional dependencies
EXTRAS = {
    "test": TEST_PACKAGES,
    "dev": DEV_PACKAGES + TEST_PACKAGES,
    "all": REQUIRED_PACKAGES + TEST_PACKAGES + DEV_PACKAGES,
}

setup(
    name="reasoningbank",
    version="0.1.0",
    author="Google Cloud AI Research + UIUC",
    description="Self-Evolving Agent with Reasoning Memory",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ReasoningBank",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRAS,
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "reasoningbank=reasoningbank.cli:main",  # Future CLI entry point
        ],
    },
)
