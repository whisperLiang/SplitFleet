"""Model tracing facade built on vendored TorchLens and Plank-road logic."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from model_management.universal_model_split import UniversalModelSplitter
from torchlens import compile_execution_plan, enumerate_frontier_splits
from torchlens.replay_plan import ExecutionPlan, FrontierSplit

_TRACE_LOCK = threading.RLock()


@dataclass
class TracedModel:
    """Bundle together the reusable tracing artefacts."""

    execution_plan: ExecutionPlan
    splitter: UniversalModelSplitter
    sample_inputs: Any
    sample_kwargs: Dict[str, Any]


class ModelTracer:
    """Compile reusable execution plans for autosplit planning."""

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device

    def trace(
        self,
        model,
        sample_inputs: Any,
        *,
        sample_kwargs: Optional[Dict[str, Any]] = None,
    ) -> TracedModel:
        kwargs = dict(sample_kwargs or {})
        with _TRACE_LOCK:
            execution_plan = compile_execution_plan(
                model,
                sample_inputs,
                input_kwargs=kwargs or None,
                device=self.device,
                strict=False,
            )
            splitter = UniversalModelSplitter(device=self.device)
            try:
                splitter.trace(model, sample_inputs, sample_kwargs=kwargs or None)
            except RuntimeError as exc:
                if "OR-Tools" not in str(exc):
                    raise
        return TracedModel(
            execution_plan=execution_plan,
            splitter=splitter,
            sample_inputs=sample_inputs,
            sample_kwargs=kwargs,
        )

    def enumerate_frontiers(
        self,
        execution_plan: ExecutionPlan,
        *,
        max_frontier_size: int = 4,
        max_splits: int = 24,
        mode: str = "minimal",
    ) -> list[FrontierSplit]:
        return enumerate_frontier_splits(
            execution_plan,
            mode=mode,
            max_frontier_size=max_frontier_size,
            max_splits=max_splits,
        )
