"""Physical multi-host CoSplit-UCB experiment integration."""


def __getattr__(name: str):
    if name in {"build_cosplit_policy", "build_cosplit_strategy"}:
        from . import run

        return getattr(run, name)
    raise AttributeError(name)

__all__ = ["build_cosplit_policy", "build_cosplit_strategy"]
