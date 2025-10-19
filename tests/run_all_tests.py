#!/usr/bin/env python3
"""
ABOUTME: Comprehensive test runner for ReasoningBank test suite
ABOUTME: Provides multiple test execution modes and coverage reporting
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def main():
    """Main test runner with command-line arguments"""

    parser = argparse.ArgumentParser(
        description="ReasoningBank Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_tests.py --all                 # Run all tests
  python run_all_tests.py --unit                # Unit tests only
  python run_all_tests.py --integration         # Integration tests only
  python run_all_tests.py --matts               # MaTTS tests only
  python run_all_tests.py --fast                # Skip slow tests
  python run_all_tests.py --coverage            # Generate coverage report
  python run_all_tests.py --verbose             # Verbose output
  python run_all_tests.py --file tests/unit/test_judge.py  # Specific file
        """
    )

    # Test selection arguments
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests (default)"
    )

    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only"
    )

    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only"
    )

    parser.add_argument(
        "--matts",
        action="store_true",
        help="Run MaTTS (test-time scaling) tests only"
    )

    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance and benchmarking tests"
    )

    # Test filtering
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow tests (exclude tests marked with @pytest.mark.slow)"
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Run specific test file"
    )

    parser.add_argument(
        "--keyword",
        "-k",
        type=str,
        help="Run tests matching keyword expression"
    )

    # Output control
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet output"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )

    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report (requires --coverage)"
    )

    # Pytest pass-through
    parser.add_argument(
        "--pytest-args",
        type=str,
        help="Additional arguments to pass to pytest (comma-separated)"
    )

    args = parser.parse_args()

    # Build pytest command
    pytest_args = []

    # Determine test directory
    if args.file:
        pytest_args.append(args.file)
    elif args.unit:
        pytest_args.append("tests/unit/")
    elif args.integration:
        pytest_args.append("tests/integration/")
    elif args.matts:
        pytest_args.append("tests/matts/")
    elif args.performance:
        pytest_args.append("tests/performance/")
    else:
        # Default: run all tests
        pytest_args.append("tests/")

    # Add markers
    if args.fast:
        pytest_args.extend(["-m", "not slow"])

    # Add keyword filter
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    # Add verbosity
    if args.verbose:
        pytest_args.append("-v")
    elif args.quiet:
        pytest_args.append("-q")

    # Add coverage
    if args.coverage:
        pytest_args.extend([
            "--cov=reasoningbank",
            "--cov-report=term-missing"
        ])

        if args.html:
            pytest_args.append("--cov-report=html")

    # Add custom pytest args
    if args.pytest_args:
        pytest_args.extend(args.pytest_args.split(","))

    # Always show summary
    pytest_args.append("--tb=short")

    # Print test configuration
    print("=" * 70)
    print("ReasoningBank Test Suite")
    print("=" * 70)
    print(f"Test Selection: {', '.join(pytest_args)}")
    print("=" * 70)
    print()

    # Run pytest
    exit_code = pytest.main(pytest_args)

    # Print summary
    print()
    print("=" * 70)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code {exit_code}")
    print("=" * 70)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
