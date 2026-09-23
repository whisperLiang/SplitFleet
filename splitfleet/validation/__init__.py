"""Download-free task validation with explicit per-boundary outcomes.

The examples are correctness fixtures, not benchmark datasets or accuracy claims.
Optional backends are imported only when their fixtures are requested.
"""

from splitfleet.validation.matrix import TASKS, TASK_BACKENDS, build_case, validate_case
from splitfleet.validation.runner import run_matrix

__all__ = ["TASKS", "TASK_BACKENDS", "build_case", "validate_case", "run_matrix"]
