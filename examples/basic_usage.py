"""
Example: Basic ReasoningBank Usage

Demonstrates:
- Creating a ReasoningBank agent
- Executing tasks with memory integration
- Viewing memory bank statistics
"""

import os
from reasoningbank import (
    ReasoningBankAgent,
    get_config_for_claude,
)


def example_environment(action: str) -> str:
    """
    Simple mock environment for demonstration.

    Args:
        action: Action string

    Returns:
        str: Observation
    """
    if action.lower().startswith("answer:"):
        return "Task completed successfully."
    elif "calculate" in action.lower():
        return "Calculation executed."
    elif "search" in action.lower():
        return "Search results: Found relevant information."
    else:
        return f"Executed action: {action}"


def main():
    """Run basic ReasoningBank example."""
    print("=" * 60)
    print("ReasoningBank Basic Usage Example")
    print("=" * 60)

    # Set API key (replace with your key or use environment variable)
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")

    # Create configuration
    config = get_config_for_claude()
    config.enable_logging = True

    # Create agent with custom environment
    agent = ReasoningBankAgent(config, environment=example_environment)

    # Execute first task
    print("\n" + "=" * 60)
    print("Task 1: Simple Math Problem")
    print("=" * 60)

    result1 = agent.run(
        query="What is 25 * 4 + 15?",
        max_steps=5
    )

    print(f"\nQuery: {result1.query}")
    print(f"Success: {result1.success}")
    print(f"Steps Taken: {result1.steps_taken}")
    print(f"Model Output: {result1.model_output}")
    print(f"Memory Items Extracted: {len(result1.memory_items)}")

    # Execute second task (should use memory from first task)
    print("\n" + "=" * 60)
    print("Task 2: Similar Math Problem")
    print("=" * 60)

    result2 = agent.run(
        query="What is 30 * 3 + 20?",
        max_steps=5
    )

    print(f"\nQuery: {result2.query}")
    print(f"Success: {result2.success}")
    print(f"Steps Taken: {result2.steps_taken}")
    print(f"Model Output: {result2.model_output}")
    print(f"Memory Items Extracted: {len(result2.memory_items)}")

    # View memory bank statistics
    print("\n" + "=" * 60)
    print("Memory Bank Statistics")
    print("=" * 60)

    stats = agent.get_statistics()
    print(f"\nTotal Entries: {stats['total_entries']}")
    print(f"Successful: {stats['successful_entries']}")
    print(f"Failed: {stats['failed_entries']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Avg Items per Entry: {stats['avg_items_per_entry']:.2f}")
    print(f"Avg Steps per Task: {stats['avg_steps_per_task']:.2f}")

    # View all memory items
    print("\n" + "=" * 60)
    print("Memory Items in Bank")
    print("=" * 60)

    memory_bank = agent.get_memory_bank()
    for i, entry in enumerate(memory_bank, 1):
        print(f"\nEntry {i}:")
        print(f"  Query: {entry.task_query}")
        print(f"  Success: {entry.success}")
        print(f"  Items:")
        for j, item in enumerate(entry.memory_items, 1):
            print(f"    {j}. {item.title}")
            print(f"       {item.description}")


if __name__ == "__main__":
    main()
