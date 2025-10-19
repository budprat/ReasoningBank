"""
Example: MaTTS Sequential Scaling

Demonstrates:
- Initial trajectory execution
- Iterative self-refinement
- Progressive improvement tracking
"""

import os
from reasoningbank import (
    get_config_for_matts_sequential,
    run_matts_sequential,
)


def example_environment(action: str) -> str:
    """Simple mock environment."""
    if action.lower().startswith("answer:"):
        return "Task completed."
    else:
        return f"Executed: {action}"


def main():
    """Run MaTTS sequential refinement example."""
    print("=" * 60)
    print("MaTTS Sequential Refinement Example")
    print("=" * 60)

    # Set API key
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")

    # Create configuration for sequential refinement with k=3
    config = get_config_for_matts_sequential(k=3)
    config.enable_logging = True

    # Run MaTTS sequential
    print("\n" + "=" * 60)
    print("Running MaTTS with k=3 sequential refinements")
    print("=" * 60)

    result = run_matts_sequential(
        query="Write a function to reverse a linked list in-place with O(1) space complexity.",
        config=config,
        environment=example_environment,
        k=3
    )

    # View results
    print("\n" + "=" * 60)
    print("MaTTS Results")
    print("=" * 60)

    print(f"\nScaling Mode: {result.scaling_mode}")
    print(f"Scaling Factor (k): {result.scaling_factor}")
    print(f"Total Attempts: {len(result.all_trajectories)} (1 initial + {result.scaling_factor} refinements)")

    # Show progression of refinements
    print("\n" + "-" * 40)
    print("Refinement Progression:")
    print("-" * 40)

    for i, traj in enumerate(result.all_trajectories):
        attempt_type = "Initial" if i == 0 else f"Refinement {i}"
        print(f"\n{attempt_type}:")
        print(f"  Success: {traj.success}")
        print(f"  Steps: {traj.steps_taken}")
        print(f"  Output: {traj.model_output[:100]}...")

    # Show improvement metrics
    print("\n" + "-" * 40)
    print("Improvement Metrics:")
    print("-" * 40)

    initial = result.all_trajectories[0]
    final = result.all_trajectories[-1]

    print(f"\nInitial Attempt:")
    print(f"  Success: {initial.success}")
    print(f"  Steps: {initial.steps_taken}")

    print(f"\nFinal Attempt:")
    print(f"  Success: {final.success}")
    print(f"  Steps: {final.steps_taken}")

    if final.steps_taken < initial.steps_taken:
        improvement = ((initial.steps_taken - final.steps_taken) / initial.steps_taken) * 100
        print(f"\nEfficiency Improvement: {improvement:.1f}% fewer steps")

    # Show best trajectory selected
    print("\n" + "-" * 40)
    print("Best Trajectory (Selected):")
    print("-" * 40)
    best_idx = result.all_trajectories.index(result.best_trajectory)
    attempt_type = "Initial" if best_idx == 0 else f"Refinement {best_idx}"
    print(f"Selected: {attempt_type}")
    print(f"Success: {result.best_trajectory.success}")
    print(f"Steps: {result.best_trajectory.steps_taken}")
    print(f"Output: {result.best_trajectory.model_output}")

    # Show extracted memories
    print("\n" + "-" * 40)
    print("Extracted Memories:")
    print("-" * 40)
    print(f"Total: {len(result.aggregated_memories)}")
    for i, mem in enumerate(result.aggregated_memories, 1):
        print(f"\n{i}. {mem.title}")
        print(f"   {mem.description}")
        print(f"   Content: {mem.content[:200]}...")

    # Show benefits of sequential refinement
    print("\n" + "=" * 60)
    print("Sequential Refinement Benefits")
    print("=" * 60)

    refinements_successful = sum(1 for t in result.all_trajectories[1:] if t.success)
    print(f"\nSuccessful Refinements: {refinements_successful}/{result.scaling_factor}")
    print(f"Progressive Improvement: Each refinement learns from previous attempt")
    print(f"Memory Extraction: {len(result.aggregated_memories)} strategies learned")


if __name__ == "__main__":
    main()
