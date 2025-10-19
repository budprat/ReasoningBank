"""
ABOUTME: LLM-as-a-Judge implementation for trajectory success/failure classification
ABOUTME: Implements self-evaluation without ground truth labels
"""

from typing import Literal, Optional
import re
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


class TrajectoryJudge:
    """
    LLM-as-a-Judge for classifying trajectory success/failure.

    Uses temperature=0.0 for deterministic judgments as specified in paper.
    Implements the exact prompt from Figure 9 (Appendix A.1).
    """

    def __init__(self, config: ReasoningBankConfig):
        """
        Initialize the judge with configuration.

        Args:
            config: ReasoningBankConfig with LLM settings
        """
        self.config = config
        self.llm_provider = config.llm_provider
        self.llm_model = config.llm_model
        self.temperature = config.judge_temperature  # 0.0 for deterministic

        # Initialize the appropriate LLM client
        if self.llm_provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=config.llm_api_key)
        elif self.llm_provider == "openai":
            self.client = openai.OpenAI(api_key=config.llm_api_key)
        elif self.llm_provider == "google":
            if not GOOGLE_AVAILABLE:
                raise ImportError(
                    "google-generativeai is not installed. "
                    "Install it with: pip install google-generativeai>=0.3.0"
                )
            genai.configure(api_key=config.llm_api_key)
            self.client = genai.GenerativeModel(self.llm_model)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def judge_trajectory_success(
        self,
        query: str,
        trajectory: str,
        final_state: str,
        model_output: str
    ) -> bool:
        """
        Determine if agent successfully completed the task.

        Implements the judge prompt from Figure 9 of the paper.
        Uses temperature=0.0 for deterministic, consistent judgments.

        Args:
            query: Original task query
            trajectory: Full agent execution trace (thinking + actions + observations)
            final_state: Final environment state after execution
            model_output: Final model output/answer

        Returns:
            bool: True if successful, False if failed

        Raises:
            ValueError: If LLM response cannot be parsed
        """
        # Construct the judge prompt (exact format from Figure 9)
        prompt = self._build_judge_prompt(query, trajectory, final_state, model_output)

        # Call LLM with temperature=0.0
        response = self._call_llm(prompt)

        # Parse response to extract success/failure
        success = self._parse_judgment(response)

        return success

    def _build_judge_prompt(
        self,
        query: str,
        trajectory: str,
        final_state: str,
        model_output: str
    ) -> str:
        """
        Build the judge prompt from Figure 9 (Appendix A.1).

        Args:
            query: Task query
            trajectory: Execution trace
            final_state: Final state
            model_output: Final output

        Returns:
            str: Complete judge prompt
        """
        prompt = f"""You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, map info, comparison of map routes, etc. The bot's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the bot encounters an exception and respond with the error content, the task is considered a failure. Besides, be careful about the sufficiency of the agent's actions. For example, when asked to list the top-searched items in a shop, the agent should order the items by the number of searches, and then return the top items. If the ordering action is missing, the task is likely to fail.

2. Site navigation: The user wants to navigate to a specific page. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.

3. Content modification: The user wants to modify the content of a webpage or configuration. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.

*IMPORTANT*
Format your response into two lines as shown below:
Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"

User Intent: {query}
Trajectory: {trajectory}
The detailed final state of the webpage: ```md {final_state} ```
Bot response to the user: {model_output if model_output else "N/A"}"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call the configured LLM with the judge prompt.

        Args:
            prompt: Judge prompt

        Returns:
            str: LLM response
        """
        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.llm_model,
                max_tokens=200,  # Need space for "Thoughts:" line + "Status:" line
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif self.llm_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.llm_model,
                max_tokens=200,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        elif self.llm_provider == "google":
            generation_config = genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=200
            )
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text

        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _parse_judgment(self, response: str) -> bool:
        """
        Parse LLM response to extract success/failure judgment.

        Args:
            response: LLM response text

        Returns:
            bool: True if success, False if failure

        Raises:
            ValueError: If response cannot be parsed
        """
        # Normalize response
        response_lower = response.strip().lower()

        # Check for explicit "success" or "failure"
        if "success" in response_lower:
            return True
        elif "failure" in response_lower or "fail" in response_lower:
            return False
        else:
            raise ValueError(
                f"Cannot parse judge response. Expected 'success' or 'failure', got: {response}"
            )

    def judge_with_confidence(
        self,
        query: str,
        trajectory: str,
        final_state: str,
        model_output: str,
        num_samples: int = 3
    ) -> tuple[bool, float]:
        """
        Judge trajectory with confidence estimation.

        Runs multiple judge calls and computes agreement rate.
        Useful for borderline cases or quality assessment.

        Args:
            query: Task query
            trajectory: Execution trace
            final_state: Final state
            model_output: Final output
            num_samples: Number of independent judgments (default: 3)

        Returns:
            tuple[bool, float]: (majority_vote, confidence_score)
                - majority_vote: True if majority say success
                - confidence_score: Agreement rate (0.0 to 1.0)
        """
        judgments = []
        for _ in range(num_samples):
            try:
                success = self.judge_trajectory_success(
                    query, trajectory, final_state, model_output
                )
                judgments.append(success)
            except ValueError:
                # If parse fails, treat as failure
                judgments.append(False)

        # Compute majority vote
        success_count = sum(judgments)
        majority_vote = success_count > (num_samples / 2)

        # Compute confidence as agreement rate
        confidence = success_count / num_samples if majority_vote else (num_samples - success_count) / num_samples

        return majority_vote, confidence


def judge_trajectory_success(
    query: str,
    trajectory: str,
    final_state: str,
    model_output: str,
    config: ReasoningBankConfig
) -> bool:
    """
    Convenience function for one-off trajectory judgment.

    Args:
        query: Task query
        trajectory: Execution trace
        final_state: Final state
        model_output: Final output
        config: ReasoningBankConfig

    Returns:
        bool: True if successful, False if failed
    """
    judge = TrajectoryJudge(config)
    return judge.judge_trajectory_success(query, trajectory, final_state, model_output)
