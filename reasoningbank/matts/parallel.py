"""
ABOUTME: MaTTS Parallel Scaling implementation
ABOUTME: Samples multiple trajectories in parallel and uses self-contrast extraction
"""

from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import ReasoningBankConfig
from ..models import TrajectoryResult, MemoryItem, MaTTSResult
from ..agent import ReasoningBankAgent


class MaTTSParallel:
    """
    Memory-aware Test-Time Scaling with Parallel Sampling.

    From Section 3.3.1:
    - Samples k trajectories in parallel
    - Uses best-of-n selection
    - Extracts aggregated memories via self-contrast
    """

    def __init__(
        self,
        config: ReasoningBankConfig,
        environment: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize MaTTS Parallel scaler.

        Args:
            config: ReasoningBankConfig with MaTTS settings
            environment: Optional environment function
        """
        self.config = config
        self.environment = environment
        self.k = config.scaling_factor_k  # Number of trajectories

        # Create base agent (without memory injection for sampling)
        self.agent = ReasoningBankAgent(config, environment)

    def run(
        self,
        query: str,
        max_steps: Optional[int] = None,
        k: Optional[int] = None
    ) -> MaTTSResult:
        """
        Execute task with parallel scaling.

        Full cycle:
        1. Sample k trajectories in parallel
        2. Select best trajectory (best-of-n)
        3. Extract aggregated memories via self-contrast
        4. Consolidate best trajectory with aggregated memories

        Args:
            query: Task description
            max_steps: Maximum steps per trajectory
            k: Scaling factor (default: config.scaling_factor_k)

        Returns:
            MaTTSResult: Complete result with best trajectory and aggregated memories
        """
        if k is None:
            k = self.k

        if max_steps is None:
            max_steps = self.config.max_steps_per_task

        # Step 1: Sample k trajectories in parallel
        if self.config.enable_logging:
            print(f"Sampling {k} trajectories in parallel...")

        trajectories = self._sample_parallel_trajectories(query, k, max_steps)

        # Step 2: Select best trajectory
        if self.config.enable_logging:
            print("Selecting best trajectory...")

        best_trajectory = self._select_best_trajectory(trajectories)

        # Step 3: Extract aggregated memories via self-contrast
        if self.config.enable_logging:
            print("Extracting aggregated memories via self-contrast...")

        aggregated_memories = self._extract_aggregated_memories(query, trajectories)

        # Step 4: Consolidate
        entry_id = self.agent.consolidator.add_from_trajectory(
            query=query,
            trajectory=best_trajectory.full_trajectory,
            final_state=best_trajectory.final_state,
            model_output=best_trajectory.model_output,
            success=best_trajectory.success,
            memory_items=aggregated_memories,
            steps_taken=best_trajectory.steps_taken
        )

        # Return MaTTS result
        return MaTTSResult(
            query=query,
            best_trajectory=best_trajectory,
            all_trajectories=trajectories,
            aggregated_memories=aggregated_memories,
            entry_id=entry_id,
            scaling_mode="parallel",
            scaling_factor=k
        )

    def _sample_parallel_trajectories(
        self,
        query: str,
        k: int,
        max_steps: int
    ) -> List[TrajectoryResult]:
        """
        Sample k trajectories in parallel.

        Args:
            query: Task description
            k: Number of trajectories
            max_steps: Maximum steps per trajectory

        Returns:
            List[TrajectoryResult]: k sampled trajectories
        """
        trajectories = []

        # Use ThreadPoolExecutor for parallel sampling
        with ThreadPoolExecutor(max_workers=min(k, 5)) as executor:
            # Submit k trajectory sampling tasks
            futures = []
            for i in range(k):
                # Create separate agent instance for each trajectory to avoid conflicts
                agent = ReasoningBankAgent(self.config, self.environment)
                future = executor.submit(
                    self._sample_single_trajectory,
                    agent, query, max_steps, i
                )
                futures.append(future)

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    trajectory = future.result()
                    trajectories.append(trajectory)
                except Exception as e:
                    if self.config.enable_logging:
                        print(f"Error sampling trajectory: {e}")

        return trajectories

    def _sample_single_trajectory(
        self,
        agent: ReasoningBankAgent,
        query: str,
        max_steps: int,
        trajectory_num: int
    ) -> TrajectoryResult:
        """
        Sample a single trajectory.

        Args:
            agent: ReasoningBankAgent instance
            query: Task description
            max_steps: Maximum steps
            trajectory_num: Trajectory number for logging

        Returns:
            TrajectoryResult: Sampled trajectory
        """
        if self.config.enable_logging:
            print(f"  Sampling trajectory {trajectory_num + 1}...")

        # Disable memory injection for sampling (use self-contrast instead)
        # Also disable consolidation (we'll consolidate after selection)
        trajectory_steps, final_state, model_output = agent._execute_task(
            query, memories=[], max_steps=max_steps
        )

        # Build full trajectory
        full_trajectory = agent._format_trajectory(trajectory_steps)

        # Judge success/failure
        success = agent.judge.judge_trajectory_success(
            query, full_trajectory, final_state, model_output
        )

        # Don't extract individual memories yet - will use self-contrast

        return TrajectoryResult(
            query=query,
            full_trajectory=full_trajectory,
            final_state=final_state,
            model_output=model_output,
            steps_taken=len(trajectory_steps),
            success=success,
            memory_items=None,
            entry_id=None
        )

    def _select_best_trajectory(self, trajectories: List[TrajectoryResult]) -> TrajectoryResult:
        """
        Select best trajectory using best-of-n strategy.

        Prioritizes:
        1. Successful trajectories over failed ones
        2. Fewer steps (more efficient)
        3. First successful trajectory if tied

        Args:
            trajectories: List of trajectories

        Returns:
            TrajectoryResult: Best trajectory
        """
        if not trajectories:
            raise ValueError("No trajectories to select from")

        # Separate successful and failed trajectories
        successful = [t for t in trajectories if t.success]
        failed = [t for t in trajectories if not t.success]

        # Prefer successful trajectories
        if successful:
            # Among successful, prefer fewer steps
            best = min(successful, key=lambda t: t.steps_taken)
        else:
            # If all failed, prefer fewer steps
            best = min(failed, key=lambda t: t.steps_taken)

        return best

    def _extract_aggregated_memories(
        self,
        query: str,
        trajectories: List[TrajectoryResult]
    ) -> List[MemoryItem]:
        """
        Extract aggregated memories via self-contrast.

        Compares k trajectories to extract robust patterns.

        Args:
            query: Task query
            trajectories: List of all trajectories

        Returns:
            List[MemoryItem]: Aggregated memory items
        """
        # Prepare trajectories for self-contrast
        trajectory_tuples = [
            (t.full_trajectory, t.final_state, t.model_output, t.query, t.success)
            for t in trajectories
        ]

        # Use extractor's self-contrast method
        aggregated_memories = self.agent.extractor.extract_with_self_contrast(
            trajectory_tuples, query
        )

        return aggregated_memories


def run_matts_parallel(
    query: str,
    config: ReasoningBankConfig,
    environment: Optional[Callable[[str], str]] = None,
    k: Optional[int] = None
) -> MaTTSResult:
    """
    Convenience function to run MaTTS parallel scaling.

    Args:
        query: Task description
        config: ReasoningBankConfig
        environment: Optional environment function
        k: Scaling factor

    Returns:
        MaTTSResult: Complete result
    """
    matts = MaTTSParallel(config, environment)
    return matts.run(query, k=k)
