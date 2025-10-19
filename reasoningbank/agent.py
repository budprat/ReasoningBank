"""
ABOUTME: ReasoningBank Agent with complete closed-loop memory integration
ABOUTME: Implements Retrieve → Act → Judge → Extract → Consolidate cycle
"""

from typing import List, Optional, Dict, Any, Callable
import anthropic
import openai

from .config import ReasoningBankConfig

# Lazy import for google-generativeai to avoid import errors when not installed
try:
    from google import generativeai as genai

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None
from .models import MemoryItem, MemoryEntry, TrajectoryResult, ReActStep
from .judge import TrajectoryJudge
from .extractor import MemoryExtractor
from .retriever_optimized import OptimizedMemoryRetriever  # Gap 22 optimization
from .consolidator import MemoryConsolidator


class ReasoningBankAgent:
    """
    Agent with closed-loop memory integration.

    Implements the complete ReasoningBank cycle:
    1. Retrieve relevant memories
    2. Act with memory-augmented prompts (ReAct format)
    3. Judge trajectory success/failure
    4. Extract new memory items
    5. Consolidate into memory bank

    Uses ReAct (Reasoning + Acting) format for agent execution.
    """

    def __init__(
        self,
        config: ReasoningBankConfig,
        environment: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize ReasoningBank agent.

        Args:
            config: ReasoningBankConfig with all settings
            environment: Optional environment function (action -> observation)
                        If None, uses mock environment
        """
        self.config = config
        self.environment = environment or self._mock_environment

        # Initialize all components
        self.judge = TrajectoryJudge(config)
        self.extractor = MemoryExtractor(config)
        self.retriever = OptimizedMemoryRetriever(config)
        self.consolidator = MemoryConsolidator(config)

        # Initialize LLM client for agent
        self.llm_provider = config.llm_provider
        self.llm_model = config.llm_model
        self.temperature = config.agent_temperature  # 0.7 for balanced exploration

        if self.llm_provider == "anthropic":
            self.llm_client = anthropic.Anthropic(api_key=config.llm_api_key)
        elif self.llm_provider == "openai":
            self.llm_client = openai.OpenAI(api_key=config.llm_api_key)
        elif self.llm_provider == "google":
            if not GOOGLE_AVAILABLE:
                raise ImportError(
                    "google-generativeai is not installed. "
                    "Install it with: pip install google-generativeai>=0.3.0"
                )
            genai.configure(api_key=config.llm_api_key)
            self.llm_client = genai.GenerativeModel(self.llm_model)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def run(
        self,
        query: str,
        max_steps: Optional[int] = None,
        enable_memory_injection: Optional[bool] = None,
    ) -> TrajectoryResult:
        """
        Execute task with complete ReasoningBank integration.

        Full cycle:
        1. Retrieve relevant memories
        2. Execute task with memory-augmented prompt
        3. Judge success/failure
        4. Extract new memories
        5. Consolidate into memory bank

        Args:
            query: Task description
            max_steps: Maximum steps (default: config.max_steps_per_task)
            enable_memory_injection: Whether to use memories (default: config.enable_memory_injection)

        Returns:
            TrajectoryResult: Complete execution result with memories
        """
        if max_steps is None:
            max_steps = self.config.max_steps_per_task

        if enable_memory_injection is None:
            enable_memory_injection = self.config.enable_memory_injection

        # Step 1: Retrieve relevant memories
        retrieved_memories = []
        if enable_memory_injection:
            memory_bank = self.consolidator.get_all_entries()
            retrieved_memories = self.retriever.retrieve(
                query, memory_bank, k=self.config.top_k_retrieval
            )

        # Step 2: Act (execute task with memory-augmented prompt)
        trajectory_steps, final_state, model_output = self._execute_task(
            query, retrieved_memories, max_steps
        )

        # Build full trajectory string
        full_trajectory = self._format_trajectory(trajectory_steps)

        # Step 3: Judge success/failure
        success = self.judge.judge_trajectory_success(
            query, full_trajectory, final_state, model_output
        )

        # Step 4: Extract memories (respecting extract_from_failures config)
        memory_items = []
        should_extract = success or self.config.extract_from_failures

        if should_extract:
            memory_items = self.extractor.extract_memories(
                query, full_trajectory, final_state, model_output, success
            )

        # Step 5: Consolidate
        entry_id = self.consolidator.add_from_trajectory(
            query,
            full_trajectory,
            final_state,
            model_output,
            success,
            memory_items,
            steps_taken=len(trajectory_steps),
        )

        # Return complete result
        return TrajectoryResult(
            query=query,
            full_trajectory=full_trajectory,
            final_state=final_state,
            model_output=model_output,
            steps_taken=len(trajectory_steps),
            success=success,
            memory_items=memory_items,
            entry_id=entry_id,
        )

    def _execute_task(
        self, query: str, memories: List[MemoryItem], max_steps: int
    ) -> tuple[List[ReActStep], str, str]:
        """
        Execute task using ReAct format with memory injection.

        Args:
            query: Task description
            memories: Retrieved memories to inject
            max_steps: Maximum number of steps

        Returns:
            tuple: (trajectory_steps, final_state, model_output)
        """
        # Build initial prompt with memory injection
        system_prompt = self._build_system_prompt(query, memories)

        trajectory_steps = []
        current_observation = "Ready to begin task."

        for step_num in range(1, max_steps + 1):
            # Build user message for this step
            user_message = self._build_step_message(
                query, trajectory_steps, current_observation, step_num
            )

            # Call LLM
            response = self._call_agent_llm(system_prompt, user_message)

            # Parse response into thinking and action
            thinking, action = self._parse_react_response(response)

            # Execute action in environment
            observation = self.environment(action)

            # Record step
            step = ReActStep(
                step_num=step_num,
                think=thinking,
                action=action,
                observation=observation,
            )
            trajectory_steps.append(step)

            # Check for terminal action
            if action.lower().startswith("answer:") or action.lower().startswith(
                "final answer"
            ):
                model_output = action
                final_state = observation
                break

            current_observation = observation
        else:
            # Max steps reached
            model_output = "Maximum steps reached without answer."
            final_state = current_observation

        return trajectory_steps, final_state, model_output

    def _build_system_prompt(self, query: str, memories: List[MemoryItem]) -> str:
        """
        Build system prompt with memory injection.

        Args:
            query: Task query
            memories: Retrieved memories

        Returns:
            str: System prompt with injected memories
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

        # Inject memories if available
        if memories:
            memory_section = "\n\n## Relevant Past Experience\n\n"
            memory_section += "Here are relevant strategies from similar tasks:\n\n"

            for i, mem in enumerate(memories, 1):
                memory_section += f"### Memory {i}: {mem.title}\n"
                memory_section += f"{mem.description}\n\n"
                memory_section += f"{mem.content}\n\n"

            prompt = base_prompt + memory_section
        else:
            prompt = base_prompt

        return prompt

    def _build_step_message(
        self,
        query: str,
        trajectory_steps: List[ReActStep],
        current_observation: str,
        step_num: int,
    ) -> str:
        """
        Build user message for current step.

        Args:
            query: Task query
            trajectory_steps: Previous steps
            current_observation: Latest observation
            step_num: Current step number

        Returns:
            str: User message
        """
        if step_num == 1:
            # First step
            message = f"Task: {query}\n\nObservation: {current_observation}\n\nWhat is your first step?"
        else:
            # Subsequent steps - include recent trajectory
            message = f"Task: {query}\n\n"

            # Include last few steps for context
            recent_steps = trajectory_steps[-3:]  # Last 3 steps
            for step in recent_steps:
                message += f"Step {step.step_num}:\n"
                message += f"<think>{step.think}</think>\n"
                message += f"<action>{step.action}</action>\n"
                message += f"<observation>{step.observation}</observation>\n\n"

            message += (
                f"Current Observation: {current_observation}\n\nWhat is your next step?"
            )

        return message

    def _call_agent_llm(self, system_prompt: str, user_message: str) -> str:
        """
        Call LLM for agent execution.

        Args:
            system_prompt: System prompt with memory injection
            user_message: User message for current step

        Returns:
            str: LLM response
        """
        if self.llm_provider == "anthropic":
            response = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=1000,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text

        elif self.llm_provider == "openai":
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                max_tokens=1000,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content

        elif self.llm_provider == "google":
            # Combine system and user messages for Gemini
            full_prompt = f"{system_prompt}\n\n{user_message}"
            generation_config = genai.GenerationConfig(
                temperature=self.temperature, max_output_tokens=1000
            )
            response = self.llm_client.generate_content(
                full_prompt, generation_config=generation_config
            )
            return response.text

        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _parse_react_response(self, response: str) -> tuple[str, str]:
        """
        Parse ReAct response into thinking and action.

        Args:
            response: LLM response

        Returns:
            tuple: (thinking, action)
        """
        thinking = ""
        action = ""

        # Extract thinking
        if "<think>" in response and "</think>" in response:
            start = response.index("<think>") + len("<think>")
            end = response.index("</think>")
            thinking = response[start:end].strip()

        # Extract action
        if "<action>" in response and "</action>" in response:
            start = response.index("<action>") + len("<action>")
            end = response.index("</action>")
            action = response[start:end].strip()
        elif "<action>" in response:
            # Handle case where closing tag is missing
            start = response.index("<action>") + len("<action>")
            action = response[start:].strip()

        # Fallback if parsing fails
        if not action:
            action = response

        return thinking, action

    def _format_trajectory(self, steps: List[ReActStep]) -> str:
        """
        Format trajectory steps into string.

        Args:
            steps: List of ReAct steps

        Returns:
            str: Formatted trajectory
        """
        trajectory = ""
        for step in steps:
            trajectory += f"Step {step.step_num}:\n"
            trajectory += step.to_string() + "\n\n"
        return trajectory

    def _mock_environment(self, action: str) -> str:
        """
        Mock environment for testing.

        Args:
            action: Action string

        Returns:
            str: Observation
        """
        if action.lower().startswith("answer:") or action.lower().startswith(
            "final answer"
        ):
            return "Task completed."
        else:
            return f"Executed: {action}"

    def get_memory_bank(self) -> List[MemoryEntry]:
        """
        Get all entries in memory bank.

        Returns:
            List[MemoryEntry]: All memory entries
        """
        return self.consolidator.get_all_entries()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory bank statistics.

        Returns:
            Dict[str, Any]: Statistics
        """
        return self.consolidator.get_statistics()


def create_agent(
    config: ReasoningBankConfig, environment: Optional[Callable[[str], str]] = None
) -> ReasoningBankAgent:
    """
    Convenience function to create a ReasoningBank agent.

    Args:
        config: ReasoningBankConfig
        environment: Optional environment function

    Returns:
        ReasoningBankAgent: Initialized agent
    """
    return ReasoningBankAgent(config, environment)
