"""
ABOUTME: MaTTS (Memory-aware Test-Time Scaling) package
ABOUTME: Implements parallel and sequential scaling strategies
"""

from .parallel import MaTTSParallel, run_matts_parallel
from .sequential import MaTTSSequential, run_matts_sequential

__all__ = [
    "MaTTSParallel",
    "MaTTSSequential",
    "run_matts_parallel",
    "run_matts_sequential",
]
