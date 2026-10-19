"""Configuration of SplitFleet's measured Greedy placement policy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GreedyConfig:
    server_concurrency: int = 1
    failure_cooldown_rounds: int = 3
    state_path: str | None = None
    warm_start: bool = True
    seed: int = 233
    debug: bool = False

    def __post_init__(self):
        if isinstance(self.server_concurrency, bool) or self.server_concurrency < 1:
            raise ValueError("server_concurrency must be a positive integer")
        if int(self.server_concurrency) != self.server_concurrency:
            raise ValueError("server_concurrency must be a positive integer")
        if self.failure_cooldown_rounds < 1:
            raise ValueError("failure_cooldown_rounds must be positive")
