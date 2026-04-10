"""Worker-side runtime for stage-level autosplit execution."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

from slbd.autosplit.planner import build_partition_plan
from slbd.autosplit.runtime import AutoSplitSession, StageBackwardResult, StageForwardResult, StageTrace
from slbd.autosplit.serde import load_model_state
from slbd.autosplit.types import PlacementPlan, WorkerSpec


@dataclass
class RegisteredPlacement:
    """Worker-local placement metadata."""

    descriptor: Dict[str, Any]
    placement_plan: PlacementPlan
    model: Any


class StageWorkerRuntime:
    """Stateful worker runtime that executes individual autosplit stages."""

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
        self._execution_plan = None
        self._placements: Dict[str, RegisteredPlacement] = {}
        self._stage_traces: Dict[str, Dict[str, StageTrace]] = {}

    def _ensure_execution_plan(self):
        if self._execution_plan is None:
            traced = self.autosplit_session.planner.tracer.trace(
                self.model,
                self.sample_inputs,
                sample_kwargs=self.sample_kwargs or None,
            )
            self._execution_plan = traced.execution_plan
        self._execution_plan.set_model(self.model)
        return self._execution_plan

    def register_plan(
        self,
        descriptor: Dict[str, Any],
        *,
        model_state: bytes = b"",
    ) -> PlacementPlan:
        existing = self._placements.get(descriptor["plan_id"])
        if existing is not None:
            plan_model = existing.model
            if model_state:
                load_model_state(plan_model, model_state, map_location=self.worker_spec.device)
            plan_model.zero_grad(set_to_none=True)
            self._placements[descriptor["plan_id"]] = RegisteredPlacement(
                descriptor=descriptor,
                placement_plan=existing.placement_plan,
                model=plan_model,
            )
            return existing.placement_plan

        plan_model = copy.deepcopy(self.model).to(self.worker_spec.device)
        if model_state:
            load_model_state(plan_model, model_state, map_location=self.worker_spec.device)
        plan_model.zero_grad(set_to_none=True)

        execution_plan = copy.copy(self._ensure_execution_plan())
        execution_plan.set_model(plan_model)
        expected_signature = descriptor.get("graph_signature")
        if expected_signature and execution_plan.graph_signature != expected_signature:
            raise ValueError(
                "Worker graph signature mismatch: "
                f"{execution_plan.graph_signature} != {expected_signature}"
            )

        partition_plan = build_partition_plan(
            execution_plan,
            descriptor.get("cutoffs", []),
            model_name=descriptor.get("model_name") or execution_plan.model_name,
        )
        placement_plan = PlacementPlan(
            partition_plan=partition_plan,
            stage_to_worker=dict(descriptor.get("stage_to_worker", {})),
            worker_specs={self.worker_spec.worker_id: self.worker_spec},
            score=float(descriptor.get("score", 0.0)),
            metadata=dict(descriptor.get("metadata", {})),
        )
        self._placements[descriptor["plan_id"]] = RegisteredPlacement(
            descriptor=descriptor,
            placement_plan=placement_plan,
            model=plan_model,
        )
        return placement_plan

    def execute_stage(
        self,
        *,
        plan_id: str,
        stage_id: str,
        execution_id: str,
        seeded_values: Dict[int, Any],
        differentiable: bool,
        detach_boundary: bool,
    ) -> StageForwardResult:
        placement_plan = self._placements[plan_id].placement_plan
        stage = next(
            stage
            for stage in placement_plan.partition_plan.stages
            if stage.stage_id == stage_id
        )
        stage_trace, result = self.autosplit_session.run_stage_forward(
            placement_plan,
            stage,
            seeded_values,
            differentiable=differentiable,
            detach_boundary=detach_boundary,
            output_indices=self.autosplit_session.get_output_indices(placement_plan),
        )
        self._stage_traces.setdefault(execution_id, {})[stage_id] = stage_trace
        return result

    def backward_stage(
        self,
        *,
        plan_id: str,
        stage_id: str,
        execution_id: str,
        upstream_grads: Dict[int, Any],
    ) -> StageBackwardResult:
        placement_plan = self._placements[plan_id].placement_plan
        stage_trace = self._stage_traces.get(execution_id, {}).pop(stage_id)
        result = self.autosplit_session.run_stage_backward(
            placement_plan,
            stage_trace,
            upstream_grads,
        )
        if execution_id in self._stage_traces and not self._stage_traces[execution_id]:
            self._stage_traces.pop(execution_id, None)
        return result

    def clear_execution(self, execution_id: str) -> bool:
        return self._stage_traces.pop(execution_id, None) is not None
