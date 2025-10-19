"""
Example: MaTTS Parallel Scaling

Demonstrates:
- Running multiple trajectories in parallel
- Best-of-n selection
- Self-contrast memory extraction
"""

import os
from reasoningbank import (
    get_config_for_matts_parallel,
    run_matts_parallel,
)


def example_environment(action: str) -> str:
    """Simple mock environment."""
    if action.lower().startswith("answer:"):
        return "Task completed."
    else:
        return f"Executed: {action}"


def main():
    """Run MaTTS parallel scaling example."""
    print("=" * 60)
    print("MaTTS Parallel Scaling Example")
    print("=" * 60)

    # Set API key
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")

    # Create configuration for parallel scaling with k=3
    config = get_config_for_matts_parallel(k=3)
    config.enable_logging = True

    # Run MaTTS parallel
    print("\n" + "=" * 60)
    print("Running MaTTS with k=3 parallel trajectories")
    print("=" * 60)

    result = run_matts_parallel(
        query="Design an efficient algorithm to find the median of two sorted arrays.",
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
    print(f"Total Trajectories: {len(result.all_trajectories)}")

    # Show all trajectories
    print("\n" + "-" * 40)
    print("All Trajectories:")
    print("-" * 40)
    for i, traj in enumerate(result.all_trajectories, 1):
        print(f"\nTrajectory {i}:")
        print(f"  Success: {traj.success}")
        print(f"  Steps: {traj.steps_taken}")
        print(f"  Output: {traj.model_output[:100]}...")

    # Show best trajectory
    print("\n" + "-" * 40)
    print("Best Trajectory (Selected):")
    print("-" * 40)
    print(f"Success: {result.best_trajectory.success}")
    print(f"Steps: {result.best_trajectory.steps_taken}")
    print(f"Output: {result.best_trajectory.model_output}")

    # Show aggregated memories from self-contrast
    print("\n" + "-" * 40)
    print("Aggregated Memories (Self-Contrast):")
    print("-" * 40)
    print(f"Total: {len(result.aggregated_memories)}")
    for i, mem in enumerate(result.aggregated_memories, 1):
        print(f"\n{i}. {mem.title}")
        print(f"   {mem.description}")
        print(f"   Content: {mem.content[:200]}...")

    # Show benefits of parallel scaling
    print("\n" + "=" * 60)
    print("Parallel Scaling Benefits")
    print("=" * 60)

    success_count = sum(1 for t in result.all_trajectories if t.success)
    print(f"\nSuccess Rate: {success_count}/{len(result.all_trajectories)} trajectories")
    print(f"Best-of-n Selection: Chose trajectory with {result.best_trajectory.steps_taken} steps")
    print(f"Self-Contrast Extraction: Found {len(result.aggregated_memories)} robust patterns")


if __name__ == "__main__":
    main()
