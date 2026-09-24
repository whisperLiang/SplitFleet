"""Configuration for the CoSplit-UCB placement policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoSplitUCBConfig:
    """Tunable CoSplit-UCB parameters.

    The defaults are conservative starting points, not claimed optima.
    """

    ridge_lambda: float = 1.0
    discount_gamma: float = 0.98

    alpha_edge: float = 1.0
    alpha_network: float = 1.0
    alpha_server: float = 1.0
    alpha_switch: float = 0.5

    safe_exploration_epsilon: float = 0.05
    max_explorations_per_round: int = 1

    min_residence_rounds: int = 2
    forced_probe_interval: int = 10
    failure_cooldown_rounds: int = 3

    target_scale_ms: float = 1000.0
    server_concurrency: int = 1
    max_coordinate_passes: int = 3
    max_candidates: int | None = None
    warm_start: bool = True
    state_path: str | None = None
    seed: int = 233
    debug: bool = False

    def __post_init__(self) -> None:
        if self.ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be positive")
        if not 0 < self.discount_gamma <= 1:
            raise ValueError("discount_gamma must be in (0, 1]")
        for name in ("alpha_edge", "alpha_network", "alpha_server", "alpha_switch"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.safe_exploration_epsilon < 0:
            raise ValueError("safe_exploration_epsilon must be non-negative")
        if self.max_explorations_per_round < 0:
            raise ValueError("max_explorations_per_round must be non-negative")
        if self.min_residence_rounds < 0 or self.forced_probe_interval < 1:
            raise ValueError("residence/probe intervals are invalid")
        if self.failure_cooldown_rounds < 1:
            raise ValueError("failure_cooldown_rounds must be positive")
        if self.target_scale_ms <= 0:
            raise ValueError("target_scale_ms must be positive")
        if self.server_concurrency < 1 or self.max_coordinate_passes < 0:
            raise ValueError("solver concurrency/passes are invalid")
        if self.max_candidates is not None and self.max_candidates < 1:
            raise ValueError("max_candidates must be positive or None")


__all__ = ["CoSplitUCBConfig"]
