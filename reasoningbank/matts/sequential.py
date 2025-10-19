"""
ABOUTME: MaTTS Sequential Scaling implementation
ABOUTME: Iteratively refines trajectory using self-refinement prompts
"""

from typing import List, Optional, Callable

from ..config import ReasoningBankConfig
from ..models import TrajectoryResult, MemoryItem, MaTTSResult, ReActStep
from ..agent import ReasoningBankAgent


class MaTTSSequential:
    """
    Memory-aware Test-Time Scaling with Sequential Refinement.

    From Section 3.3.2:
    - Executes initial trajectory
    - Iteratively refines with self-refinement prompts
    - After k refinements, extracts memories from best trajectory
    """

    def __init__(
        self,
        config: ReasoningBankConfig,
        environment: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize MaTTS Sequential scaler.

        Args:
            config: ReasoningBankConfig with MaTTS settings
            environment: Optional environment function
        """
        self.config = config
        self.environment = environment
        self.k = config.scaling_factor_k  # Number of refinements

        # Get refinement prompts from config
        self.refinement_prompts = config.refinement_prompts

        # Create base agent
        self.agent = ReasoningBankAgent(config, environment)

    def run(
        self,
        query: str,
        max_steps: Optional[int] = None,
        k: Optional[int] = None
    ) -> MaTTSResult:
        """
        Execute task with sequential refinement.

        Full cycle:
        1. Execute initial trajectory
        2. Iteratively refine trajectory k times
        3. Select best trajectory from all attempts
        4. Extract memories from best trajectory
        5. Consolidate

        Args:
            query: Task description
            max_steps: Maximum steps per trajectory
            k: Number of refinements (default: config.scaling_factor_k)

        Returns:
            MaTTSResult: Complete result with all refinement attempts
        """
        if k is None:
            k = self.k

        if max_steps is None:
            max_steps = self.config.max_steps_per_task

        trajectories = []

        # Step 1: Execute initial trajectory
        if self.config.enable_logging:
            print("Executing initial trajectory...")

        initial_trajectory = self._execute_initial_trajectory(query, max_steps)
        trajectories.append(initial_trajectory)

        # Step 2: Iteratively refine k times
        current_trajectory = initial_trajectory

        for refinement_num in range(k):
            if self.config.enable_logging:
                print(f"Refinement {refinement_num + 1}/{k}...")

            # Get refinement prompt
            refinement_prompt = self._get_refinement_prompt(refinement_num)

            # Execute refined trajectory
            refined_trajectory = self._execute_refinement(
                query, current_trajectory, refinement_prompt, max_steps
            )

            trajectories.append(refined_trajectory)

            # Update current trajectory for next refinement
            current_trajectory = refined_trajectory

        # Step 3: Select best trajectory
        if self.config.enable_logging:
            print("Selecting best trajectory...")

        best_trajectory = self._select_best_trajectory(trajectories)

        # Step 4: Extract memories from best trajectory
        if self.config.enable_logging:
            print("Extracting memories from best trajectory...")

        memory_items = self.agent.extractor.extract_memories(
            query=query,
            trajectory=best_trajectory.full_trajectory,
            final_state=best_trajectory.final_state,
            model_output=best_trajectory.model_output,
            success=best_trajectory.success
        )

        # Step 5: Consolidate
        entry_id = self.agent.consolidator.add_from_trajectory(
            query=query,
            trajectory=best_trajectory.full_trajectory,
            final_state=best_trajectory.final_state,
            model_output=best_trajectory.model_output,
            success=best_trajectory.success,
            memory_items=memory_items,
            steps_taken=best_trajectory.steps_taken
        )

        # Return MaTTS result
        return MaTTSResult(
            query=query,
            best_trajectory=best_trajectory,
            all_trajectories=trajectories,
            aggregated_memories=memory_items,
            entry_id=entry_id,
            scaling_mode="sequential",
            scaling_factor=k
        )

    def _execute_initial_trajectory(
        self,
        query: str,
        max_steps: int
    ) -> TrajectoryResult:
        """
        Execute initial trajectory without refinement.

        Args:
            query: Task description
            max_steps: Maximum steps

        Returns:
            TrajectoryResult: Initial trajectory
        """
        # Disable memory injection for initial execution
        trajectory_steps, final_state, model_output = self.agent._execute_task(
            query, memories=[], max_steps=max_steps
        )

        # Build full trajectory
        full_trajectory = self.agent._format_trajectory(trajectory_steps)

        # Judge success/failure
        success = self.agent.judge.judge_trajectory_success(
            query, full_trajectory, final_state, model_output
        )

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

    def _get_refinement_prompt(self, refinement_num: int) -> str:
        """
        Get refinement prompt for current iteration.

        Uses prompts from config.refinement_prompts.

        Args:
            refinement_num: Current refinement number (0-indexed)

        Returns:
            str: Refinement prompt
        """
        if refinement_num < len(self.refinement_prompts):
            return self.refinement_prompts[refinement_num]
        else:
            # Use last prompt if we've exhausted the list
            return self.refinement_prompts[-1]

    def _execute_refinement(
        self,
        query: str,
        previous_trajectory: TrajectoryResult,
        refinement_prompt: str,
        max_steps: int
    ) -> TrajectoryResult:
        """
        Execute refinement given previous trajectory.

        Injects previous trajectory and refinement prompt.

        Args:
            query: Original task query
            previous_trajectory: Previous attempt
            refinement_prompt: Refinement instruction
            max_steps: Maximum steps

        Returns:
            TrajectoryResult: Refined trajectory
        """
        # Build enhanced prompt with previous trajectory and refinement instruction
        enhanced_system_prompt = self._build_refinement_system_prompt(
            query, previous_trajectory, refinement_prompt
        )

        # Execute with enhanced prompt
        trajectory_steps = []
        current_observation = "Ready to refine previous attempt."

        for step_num in range(1, max_steps + 1):
            # Build step message
            if step_num == 1:
                user_message = f"Task: {query}\n\n{refinement_prompt}\n\nWhat is your first step?"
            else:
                user_message = self.agent._build_step_message(
                    query, trajectory_steps, current_observation, step_num
                )

            # Call LLM
            response = self.agent._call_agent_llm(enhanced_system_prompt, user_message)

            # Parse response
            thinking, action = self.agent._parse_react_response(response)

            # Execute action
            observation = self.environment(action)

            # Record step
            step = ReActStep(
                step_num=step_num,
                think=thinking,
                action=action,
                observation=observation
            )
            trajectory_steps.append(step)

            # Check for terminal action
            if action.lower().startswith("answer:") or action.lower().startswith("final answer"):
                model_output = action
                final_state = observation
                break

            current_observation = observation
        else:
            # Max steps reached
            model_output = "Maximum steps reached without answer."
            final_state = current_observation

        # Build full trajectory
        full_trajectory = self.agent._format_trajectory(trajectory_steps)

        # Judge success/failure
        success = self.agent.judge.judge_trajectory_success(
            query, full_trajectory, final_state, model_output
        )

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

    def _build_refinement_system_prompt(
        self,
        query: str,
        previous_trajectory: TrajectoryResult,
        refinement_prompt: str
    ) -> str:
        """
        Build system prompt for refinement with previous trajectory.

        Args:
            query: Task query
            previous_trajectory: Previous attempt
            refinement_prompt: Refinement instruction

        Returns:
            str: Enhanced system prompt
        """
        base_prompt = """You are a helpful AI agent that solves tasks step-by-step using reasoning and actions.

Use the ReAct format:
1. Think about the next step
2. Take an action
3. Observe the result
4. Repeat until task is solved

Format your response as:
<think>Your reasoning about what to do next</think>
<action>The action to take</action>

When you have the final answer, use:
<action>Answer: [your final answer]</action>"""

        # Add previous attempt section
        previous_section = f"""

## Previous Attempt

Task: {query}

Previous Trajectory:
{previous_trajectory.full_trajectory}

Final State: {previous_trajectory.final_state}
Model Output: {previous_trajectory.model_output}
Result: {'SUCCESS' if previous_trajectory.success else 'FAILURE'}

## Refinement Instructions

{refinement_prompt}"""

        return base_prompt + previous_section

    def _select_best_trajectory(self, trajectories: List[TrajectoryResult]) -> TrajectoryResult:
        """
        Select best trajectory from all attempts.

        Prioritizes:
        1. Successful trajectories over failed ones
        2. More recent attempts (later refinements)
        3. Fewer steps if tied

        Args:
            trajectories: List of all trajectories (initial + refinements)

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
            # Among successful, prefer most recent (later refinements)
            # If multiple recent successes, prefer fewer steps
            best = min(successful, key=lambda t: (trajectories.index(t) * -1, t.steps_taken))
        else:
            # If all failed, prefer most recent attempt
            best = trajectories[-1]

        return best


def run_matts_sequential(
    query: str,
    config: ReasoningBankConfig,
    environment: Optional[Callable[[str], str]] = None,
    k: Optional[int] = None
) -> MaTTSResult:
    """
    Convenience function to run MaTTS sequential refinement.

    Args:
        query: Task description
        config: ReasoningBankConfig
        environment: Optional environment function
        k: Number of refinements

    Returns:
        MaTTSResult: Complete result
    """
    matts = MaTTSSequential(config, environment)
    return matts.run(query, k=k)
