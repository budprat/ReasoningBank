"""
ABOUTME: Configuration system for ReasoningBank
ABOUTME: Manages LLM settings, memory parameters, and scaling options
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import os


@dataclass
class ReasoningBankConfig:
    """
    Complete configuration for ReasoningBank system.

    Based on paper's implementation details from Appendix A.2
    """

    # ============= LLM Settings =============
    llm_provider: Literal["anthropic", "openai", "google"] = "anthropic"
    llm_model: str = "claude-3-5-sonnet-20241022"  # or "gemini-2.5-flash", "gpt-4"
    llm_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))

    # Separate API keys for different providers
    google_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    # Temperature settings (from paper)
    agent_temperature: float = 0.7  # Balanced exploration
    judge_temperature: float = 0.0  # Deterministic judgments
    extractor_temperature: float = 1.0  # Diverse extraction
    selector_temperature: float = 0.0  # Consistent selection

    # ============= Memory Settings =============
    memory_bank_path: str = "./data/memory_bank.json"
    embedding_cache_path: str = "./data/embeddings.json"

    # Retrieval settings
    top_k_retrieval: int = 1  # Default from paper (see Appendix C.1)
    embedding_model: str = "gemini-embedding-001"  # Paper uses Gemini embeddings
    embedding_dimension: int = 768  # 768 for Gemini, 1536 for OpenAI

    # Extraction settings (from Appendix A.1)
    max_memory_items_per_trajectory: int = 3  # Paper uses max 3
    max_memory_items_aggregated: int = 5  # MaTTS uses max 5
    extract_from_failures: bool = True  # Learn from both successes AND failures (Gap 24)

    # ============= Agent Settings =============
    max_steps_per_task: int = 30  # From paper Section 4.1
    react_format: bool = True  # Use ReAct (Reasoning + Acting)
    enable_memory_injection: bool = True

    # ============= MaTTS Settings =============
    enable_matts: bool = False
    matts_mode: Literal["parallel", "sequential"] = "parallel"
    scaling_factor_k: int = 3  # Number of trajectories/refinements

    # Parallel scaling
    use_self_contrast: bool = True  # Extract via comparing trajectories
    use_best_of_n: bool = True  # Select best trajectory

    # Sequential scaling (exact prompts from Figure 10)
    refinement_prompts: list = field(default_factory=lambda: [
        "Important: Let's carefully re-examine the previous trajectory, including your reasoning steps and actions taken. Pay special attention to whether you used the correct elements on the page, and whether your response addresses the user query. If you find inconsistencies, correct them. If everything seems correct, confirm your final answer. Output must stay in the same \"<think>...</think><action></action>\" format as previous trajectories.",
        "Let's check again. Output must stay in the same \"<think>...</think><action></action>\" format as previous trajectories."
    ])

    # ============= Environment Settings =============
    environment: Literal["mock", "browsergym", "bash"] = "mock"
    max_context_length: int = 100000  # Model context window

    # ============= Logging & Monitoring =============
    enable_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = "./data/reasoningbank.log"

    # Metrics
    track_metrics: bool = True
    metrics_file: str = "./data/metrics.json"

    def validate(self) -> None:
        """Validate configuration settings"""
        if self.top_k_retrieval < 1:
            raise ValueError("top_k_retrieval must be >= 1")

        if self.max_memory_items_per_trajectory not in [1, 2, 3]:
            raise ValueError("max_memory_items_per_trajectory should be 1-3 per paper")

        if self.scaling_factor_k < 1:
            raise ValueError("scaling_factor_k must be >= 1")

        if self.agent_temperature < 0 or self.agent_temperature > 2:
            raise ValueError("agent_temperature must be in [0, 2]")

        # Ensure API key is set
        if not self.llm_api_key:
            raise ValueError(f"LLM API key not found. Set {self.llm_provider.upper()}_API_KEY environment variable")

    def __post_init__(self):
        """Post-initialization validation"""
        # Create data directories if they don't exist
        os.makedirs(os.path.dirname(self.memory_bank_path), exist_ok=True)

        # Set appropriate API key env var
        if self.llm_provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            if self.llm_api_key:
                os.environ["ANTHROPIC_API_KEY"] = self.llm_api_key

        # Validate configuration
        self.validate()


# Preset configurations for common use cases
def get_config_for_paper_replication() -> ReasoningBankConfig:
    """
    Configuration matching paper's experiments.

    Uses Gemini-2.5-flash as in paper's WebArena experiments.
    """
    return ReasoningBankConfig(
        llm_provider="google",
        llm_model="gemini-2.5-flash",
        agent_temperature=0.7,
        judge_temperature=0.0,
        extractor_temperature=1.0,
        top_k_retrieval=1,
        max_steps_per_task=30,
        environment="browsergym"
    )


def get_config_for_claude() -> ReasoningBankConfig:
    """Configuration for Claude-based implementation"""
    return ReasoningBankConfig(
        llm_provider="anthropic",
        llm_model="claude-3-5-sonnet-20241022",
        agent_temperature=0.7,
        top_k_retrieval=1,
        max_steps_per_task=30
    )


def get_config_for_matts_parallel(k: int = 3) -> ReasoningBankConfig:
    """Configuration for MaTTS parallel scaling"""
    config = get_config_for_claude()
    config.enable_matts = True
    config.matts_mode = "parallel"
    config.scaling_factor_k = k
    return config


def get_config_for_matts_sequential(k: int = 3) -> ReasoningBankConfig:
    """Configuration for MaTTS sequential scaling"""
    config = get_config_for_claude()
    config.enable_matts = True
    config.matts_mode = "sequential"
    config.scaling_factor_k = k
    return config
