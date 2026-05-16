"""Ariadne owns tracing for the current autosplit backend."""

from __future__ import annotations


class ModelTracer:
    """Compatibility placeholder for callers that still construct a planner with a tracer."""

    def __init__(self, *args, **kwargs) -> None:
        _ = (args, kwargs)
