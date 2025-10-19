"""
ABOUTME: Memory extraction implementation for ReasoningBank
ABOUTME: Extracts generalizable strategies from successes and preventative lessons from failures
"""

from typing import List, Optional
import json
import anthropic
import openai
from datetime import datetime

from .config import ReasoningBankConfig
from .models import MemoryItem

# Lazy import for google-generativeai to avoid import errors when not installed
try:
    from google import generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None


class MemoryExtractor:
    """
    Extract structured memory items from agent trajectories.

    Uses dual-prompt approach:
    - Success trajectories: Extract generalizable strategies
    - Failure trajectories: Extract preventative lessons

    Uses temperature=1.0 for diverse extraction (per paper).
    """

    def __init__(self, config: ReasoningBankConfig):
        """
        Initialize the memory extractor with configuration.

        Args:
            config: ReasoningBankConfig with LLM settings
        """
        self.config = config
        self.llm_provider = config.llm_provider
        self.llm_model = config.llm_model
        self.temperature = config.extractor_temperature  # 1.0 for diversity
        self.max_items = config.max_memory_items_per_trajectory  # Max 3

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

    def extract_memories(
        self,
        query: str,
        trajectory: str,
        final_state: str,
        model_output: str,
        success: bool,
        source_task_id: Optional[str] = None
    ) -> List[MemoryItem]:
        """
        Extract memory items from a trajectory.

        Automatically selects the appropriate prompt based on success/failure.

        Args:
            query: Original task query
            trajectory: Full agent execution trace
            final_state: Final environment state
            model_output: Final model output
            success: Whether the trajectory was successful
            source_task_id: Optional task identifier

        Returns:
            List[MemoryItem]: Extracted memory items (max 3)

        Raises:
            ValueError: If LLM response cannot be parsed
        """
        # Select appropriate prompt
        if success:
            prompt = self._build_success_prompt(query, trajectory, final_state, model_output)
        else:
            prompt = self._build_failure_prompt(query, trajectory, final_state, model_output)

        # Call LLM with temperature=1.0
        response = self._call_llm(prompt)

        # Parse response to extract memory items
        memory_items = self._parse_extraction_response(
            response,
            success_signal=success,
            source_task_id=source_task_id
        )

        # Enforce max items limit
        if len(memory_items) > self.max_items:
            memory_items = memory_items[:self.max_items]

        return memory_items

    def _build_success_prompt(
        self,
        query: str,
        trajectory: str,
        final_state: str,
        model_output: str
    ) -> str:
        """
        Build the success extraction prompt from Figure 8 (Appendix A.1).

        Extracts generalizable strategies that contributed to success.

        Args:
            query: Task query
            trajectory: Execution trace
            final_state: Final state
            model_output: Final output

        Returns:
            str: Complete success extraction prompt
        """
        prompt = f"""You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents how an agent successfully accomplished the task.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
- You must first think why the trajectory is successful, and then summarize the insights.
- You can extract at most {self.max_items} memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:
```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task>
```

Query: {query}
Trajectory: {trajectory}"""

        return prompt

    def _build_failure_prompt(
        self,
        query: str,
        trajectory: str,
        final_state: str,
        model_output: str
    ) -> str:
        """
        Build the failure extraction prompt from Figure 8 (Appendix A.1).

        Extracts preventative lessons about what went wrong.

        Args:
            query: Task query
            trajectory: Execution trace
            final_state: Final state
            model_output: Final output

        Returns:
            str: Complete failure extraction prompt
        """
        prompt = f"""You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents how an agent attempted to resolve the task but failed.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
- You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
- You can extract at most {self.max_items} memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:
```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task>
```

Query: {query}
Trajectory: {trajectory}"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call the configured LLM with the extraction prompt.

        Args:
            prompt: Extraction prompt (success or failure)

        Returns:
            str: LLM response
        """
        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.llm_model,
                max_tokens=2000,  # Room for multiple detailed items
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif self.llm_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.llm_model,
                max_tokens=2000,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        elif self.llm_provider == "google":
            generation_config = genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=2000
            )
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text

        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _parse_extraction_response(
        self,
        response: str,
        success_signal: bool,
        source_task_id: Optional[str] = None
    ) -> List[MemoryItem]:
        """
        Parse LLM response to extract memory items from Markdown format.

        Args:
            response: LLM response text (Markdown format as per paper Figure 8)
            success_signal: Whether this was a success or failure trajectory
            source_task_id: Optional task identifier

        Returns:
            List[MemoryItem]: Parsed memory items

        Raises:
            ValueError: If response cannot be parsed as Markdown
        """
        # Clean response
        response = response.strip()

        # Remove markdown code fences if present
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first line (```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)

        # Split response into memory item sections
        # Each section starts with "# Memory Item"
        memory_items = []
        timestamp = datetime.now().timestamp()

        # Split by "# Memory Item" headers
        sections = []
        current_section = []

        for line in response.split("\n"):
            if line.strip().startswith("# Memory Item"):
                if current_section:
                    sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        # Don't forget the last section
        if current_section:
            sections.append("\n".join(current_section))

        # Parse each section
        for section in sections:
            if not section.strip():
                continue

            # Extract title, description, and content
            title = None
            description = None
            content = None

            lines = section.split("\n")
            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # Look for ## Title
                if line.startswith("## Title"):
                    # Extract title (can be on same line or next line)
                    title_text = line[8:].strip()  # Remove "## Title"
                    if not title_text and i + 1 < len(lines):
                        # Title on next line
                        title = lines[i + 1].strip()
                        i += 1
                    else:
                        title = title_text

                # Look for ## Description
                elif line.startswith("## Description"):
                    # Extract description (can be on same line or next line)
                    desc_text = line[14:].strip()  # Remove "## Description"
                    if not desc_text and i + 1 < len(lines):
                        # Description on next line
                        description = lines[i + 1].strip()
                        i += 1
                    else:
                        description = desc_text

                # Look for ## Content
                elif line.startswith("## Content"):
                    # Extract content (can be on same line or next lines)
                    content_text = line[10:].strip()  # Remove "## Content"
                    if not content_text and i + 1 < len(lines):
                        # Content on following lines
                        content_lines = []
                        i += 1
                        while i < len(lines) and not lines[i].strip().startswith("##") and not lines[i].strip().startswith("#"):
                            if lines[i].strip():
                                content_lines.append(lines[i].strip())
                            i += 1
                        content = " ".join(content_lines)
                        i -= 1  # Back up one since loop will increment
                    else:
                        content = content_text

                i += 1

            # Validate and create MemoryItem
            if title and description and content:
                memory_item = MemoryItem(
                    title=title,
                    description=description,
                    content=content,
                    source_task_id=source_task_id,
                    success_signal=success_signal,
                    extraction_timestamp=timestamp
                )
                memory_items.append(memory_item)

        if not memory_items:
            raise ValueError(f"No valid memory items found in response. Response: {response[:500]}...")

        return memory_items

    def extract_with_self_contrast(
        self,
        trajectories: List[tuple[str, str, str, str, bool]],
        query: str,
        source_task_id: Optional[str] = None
    ) -> List[MemoryItem]:
        """
        Extract memories by comparing multiple trajectories (MaTTS parallel).

        This implements the self-contrast approach from Section 3.3.1.
        Compares multiple trajectories to identify robust patterns.

        Args:
            trajectories: List of (trajectory, final_state, model_output, query, success) tuples
            query: Original task query
            source_task_id: Optional task identifier

        Returns:
            List[MemoryItem]: Aggregated memory items from comparison
        """
        # Build self-contrast prompt
        prompt = self._build_self_contrast_prompt(trajectories, query)

        # Call LLM
        response = self._call_llm(prompt)

        # Parse response
        memory_items = self._parse_extraction_response(
            response,
            success_signal=None,  # Mixed success/failure
            source_task_id=source_task_id
        )

        # Enforce max items limit (for MaTTS, use max_memory_items_aggregated)
        max_aggregated = self.config.max_memory_items_aggregated
        if len(memory_items) > max_aggregated:
            memory_items = memory_items[:max_aggregated]

        return memory_items

    def _build_self_contrast_prompt(
        self,
        trajectories: List[tuple[str, str, str, str, bool]],
        query: str
    ) -> str:
        """
        Build self-contrast prompt for comparing multiple trajectories.

        From Section 3.3.1: "comparing across the k sampled trajectories to
        extract more generalized memory items".

        Args:
            trajectories: List of trajectory tuples
            query: Task query

        Returns:
            str: Self-contrast extraction prompt
        """
        # Format trajectories for comparison
        trajectory_texts = []
        for i, (traj, final_state, output, _, success) in enumerate(trajectories, 1):
            status = "SUCCESS" if success else "FAILURE"
            trajectory_texts.append(
                f"Trajectory {i} ({status}):\n{traj}\n"
                f"Final State: {final_state}\n"
                f"Output: {output}\n"
            )

        trajectories_section = "\n---\n".join(trajectory_texts)
        max_items = self.config.max_memory_items_aggregated

        prompt = f"""You are an expert in web navigation. You will be given a user query and multiple trajectories showing how an agent attempted the task. Some trajectories may be successful, and others may have failed.

## Guidelines
Your goal is to compare and contrast these trajectories to identify the most useful and generalizable strategies as memory items.
Use self-contrast reasoning:
- Identify patterns and strategies that consistently led to success.
- Identify mistakes or inefficiencies from failed trajectories and formulate preventative strategies.
- Prefer strategies that generalize beyond specific pages or exact wording.

## Important notes
- Think first: Why did some trajectories succeed while others failed?
- You can extract at most {max_items} memory items from all trajectories combined.
- Do not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents — focus on generalizable behaviors and reasoning patterns.
- Make sure each memory item captures actionable and transferable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:
```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-5 sentences describing the insights learned to successfully accomplishing the task>
```

Query: {query}
Trajectories: {trajectories_section}"""

        return prompt


def extract_memories(
    query: str,
    trajectory: str,
    final_state: str,
    model_output: str,
    success: bool,
    config: ReasoningBankConfig,
    source_task_id: Optional[str] = None
) -> List[MemoryItem]:
    """
    Convenience function for one-off memory extraction.

    Args:
        query: Task query
        trajectory: Execution trace
        final_state: Final state
        model_output: Final output
        success: Success/failure signal
        config: ReasoningBankConfig
        source_task_id: Optional task identifier

    Returns:
        List[MemoryItem]: Extracted memory items
    """
    extractor = MemoryExtractor(config)
    return extractor.extract_memories(
        query, trajectory, final_state, model_output, success, source_task_id
    )
