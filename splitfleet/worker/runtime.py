"""Compatibility worker runtime for inactive node-level stage execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from splitfleet.autosplit.runtime import AutoSplitSession
from splitfleet.autosplit.types import SplitPlan, WorkerSpec


REMOTE_STAGE_ERROR = (
    "TorchLens autosplit backend supports coordinator-local suffix execution only; "
    "node-level remote stage execution is not active."
)


@dataclass
class RegisteredPlacement:
    """Worker-local placement metadata."""

    descriptor: Dict[str, Any]
    placement_plan: SplitPlan
    model: Any


class StageWorkerRuntime:
    """Remote node-level workers are not active for the TorchLens autosplit backend."""

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
    ) -> SplitPlan:
        _ = model_state
        if descriptor.get("engine") != "torchlens" or descriptor.get("backend") != "torchlens":
            raise ValueError("Worker requires an explicit TorchLens placement descriptor")
        placement_plan = SplitPlan(
            plan_id=str(descriptor["plan_id"]),
            split_id=str(descriptor.get("split_id", "")),
            graph_signature=str(descriptor.get("graph_signature", "")),
            boundary=str(descriptor.get("boundary", "50%")),
            mode=str(descriptor.get("mode", "generated_eager")),
            prefix_worker_id="client",
            suffix_worker_id=self.worker_spec.worker_id,
            score=float(descriptor.get("score", 0.0)),
            engine=str(descriptor["engine"]),
            backend=str(descriptor["backend"]),
            runtime_backend=str(descriptor["runtime_backend"]),
            candidate_id=str(descriptor.get("candidate_id", "")),
            boundary_tensor_labels=[
                str(label) for label in list(descriptor.get("boundary_tensor_labels") or [])
            ],
            payload_bytes=int(descriptor.get("payload_bytes", 0) or 0),
            runtime_contract=dict(descriptor.get("runtime_contract") or {}),
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
