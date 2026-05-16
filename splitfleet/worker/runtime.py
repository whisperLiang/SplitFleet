"""Compatibility worker runtime for removed node-level stage execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from splitfleet.autosplit.runtime import AutoSplitSession
from splitfleet.autosplit.types import AriadnePlacementPlan, WorkerSpec


REMOTE_STAGE_ERROR = (
    "Ariadne backend currently supports coordinator-local suffix execution only; "
    "old node-level remote stage execution has been removed."
)


@dataclass
class RegisteredPlacement:
    """Worker-local Ariadne placement metadata."""

    descriptor: Dict[str, Any]
    placement_plan: AriadnePlacementPlan
    model: Any


class StageWorkerRuntime:
    """Remote node-level workers are no longer part of the Ariadne backend."""

    def __init__(
        self,
        *,
        worker_spec: WorkerSpec,
        model,
        sample_inputs: Any,
        sample_kwargs: Optional[Dict[str, Any]] = None,
        autosplit_session: Optional[AutoSplitSession] = None,
    ) -> None:
        self.worker_spec = worker_spec
        self.model = model
        self.sample_inputs = sample_inputs
        self.sample_kwargs = dict(sample_kwargs or {})
        self.autosplit_session = autosplit_session or AutoSplitSession(device=worker_spec.device)
        self._placements: Dict[str, RegisteredPlacement] = {}

    def register_plan(
        self,
        descriptor: Dict[str, Any],
        *,
        model_state: bytes = b"",
    ) -> AriadnePlacementPlan:
        _ = model_state
        placement_plan = AriadnePlacementPlan(
            plan_id=str(descriptor["plan_id"]),
            split_id=str(descriptor.get("split_id", "")),
            graph_signature=str(descriptor.get("graph_signature", "")),
            boundary=str(descriptor.get("boundary", "50%")),
            mode=str(descriptor.get("mode", "generated_eager")),
            prefix_worker_id="client",
            suffix_worker_id=self.worker_spec.worker_id,
            score=float(descriptor.get("score", 0.0)),
            metadata=dict(descriptor.get("metadata", {})),
            worker_specs={self.worker_spec.worker_id: self.worker_spec},
        )
        self._placements[placement_plan.plan_id] = RegisteredPlacement(
            descriptor=descriptor,
            placement_plan=placement_plan,
            model=self.model,
        )
        return placement_plan

    def execute_stage(self, *args, **kwargs):
        _ = (args, kwargs)
        raise NotImplementedError(REMOTE_STAGE_ERROR)

    def backward_stage(self, *args, **kwargs):
        _ = (args, kwargs)
        raise NotImplementedError(REMOTE_STAGE_ERROR)

    def clear_execution(self, execution_id: str) -> bool:
        _ = execution_id
        return False
