"""
ABOUTME: ReasoningBank - Self-Evolving Agent with Reasoning Memory
ABOUTME: Implementation based on Google Cloud AI Research + UIUC paper (September 2025)
"""

from .config import (
    ReasoningBankConfig,
    get_config_for_paper_replication,
    get_config_for_claude,
    get_config_for_matts_parallel,
    get_config_for_matts_sequential,
)

from .models import (
    MemoryItem,
    MemoryEntry,
    TrajectoryResult,
    ReActStep,
    MaTTSResult,
)

from .judge import TrajectoryJudge, judge_trajectory_success

from .extractor import MemoryExtractor, extract_memories

from .retriever import MemoryRetriever, retrieve_memories

from .consolidator import MemoryConsolidator, create_consolidator

from .agent import ReasoningBankAgent, create_agent

from .matts import (
    MaTTSParallel,
    MaTTSSequential,
    run_matts_parallel,
    run_matts_sequential,
)

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "ReasoningBankConfig",
    "get_config_for_paper_replication",
    "get_config_for_claude",
    "get_config_for_matts_parallel",
    "get_config_for_matts_sequential",
    # Data Models
    "MemoryItem",
    "MemoryEntry",
    "TrajectoryResult",
    "ReActStep",
    "MaTTSResult",
    # Core Components
    "TrajectoryJudge",
    "judge_trajectory_success",
    "MemoryExtractor",
    "extract_memories",
    "MemoryRetriever",
    "retrieve_memories",
    "MemoryConsolidator",
    "create_consolidator",
    # Agent
    "ReasoningBankAgent",
    "create_agent",
    # MaTTS
    "MaTTSParallel",
    "MaTTSSequential",
    "run_matts_parallel",
    "run_matts_sequential",
]
