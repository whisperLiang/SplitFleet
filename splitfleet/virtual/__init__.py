"""Native Flower virtual clients executing SplitFleet's real split training."""

from .config import VirtualConfig
from .runtime import TrainingSpec, run_virtual

__all__ = ["TrainingSpec", "VirtualConfig", "run_virtual"]
