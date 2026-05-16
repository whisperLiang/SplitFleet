"""Ariadne-backed two-stage autosplit placement planning."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional, Sequence

from splitfleet.autosplit.ariadne_adapter import (
    AriadneRuntimeHandle,
    prepare_ariadne_runtime,
)
from splitfleet.autosplit.cache import PlanCacheEntry, PlanCacheStore
from splitfleet.autosplit.types import (
    AriadnePlacementPlan,
    PlacementConstraint,
    PlacementObjective,
    WorkerSpec,
)


TWO_STAGE_ERROR = "Ariadne backend currently supports prefix/suffix two-stage split only."
ONE_CLIENT_STAGE_ERROR = (
    "Ariadne backend currently supports exactly one client-local prefix stage."
)


def _validate_stage_counts(
    *,
    preferred_stage_count: Optional[int],
    client_stage_count: Optional[int],
) -> None:
    if preferred_stage_count not in (None, 2):
        raise ValueError(TWO_STAGE_ERROR)
    if client_stage_count not in (None, 1):
        raise ValueError(ONE_CLIENT_STAGE_ERROR)


def _coordinator_worker(worker_specs: Sequence[WorkerSpec]) -> WorkerSpec:
    online = [worker for worker in worker_specs if worker.online]
    for worker in online:
        labels = {worker.worker_id.lower(), *(tag.lower() for tag in worker.tags)}
        if worker.address is None and labels & {"local", "server", "coordinator"}:
            return worker
    for worker in online:
        if worker.address is None:
            return worker
    return WorkerSpec(worker_id="coordinator", device="cpu")


def _score(runtime_handle: AriadneRuntimeHandle, worker: WorkerSpec, objective: PlacementObjective) -> float:
    bandwidth_bytes_per_s = max(float(worker.bandwidth_mbps), 1.0) * 125_000.0
    bandwidth_cost = runtime_handle.plan.boundary_bytes / bandwidth_bytes_per_s
    suffix_nodes = max(runtime_handle.plan.suffix_node_count, 1)
    latency_cost = suffix_nodes / 1_000.0
    return objective.bandwidth_weight * bandwidth_cost + objective.latency_weight * latency_cost


def _build_placement(
    runtime_handle: AriadneRuntimeHandle,
    *,
    boundary: str,
    worker_specs: Sequence[WorkerSpec],
    constraints: PlacementConstraint,
    objective: PlacementObjective,
    model_name: Optional[str],
) -> AriadnePlacementPlan:
    if runtime_handle.plan.boundary_bytes > constraints.max_payload_bytes:
        raise RuntimeError(
            "Ariadne split boundary exceeds max_payload_bytes: "
            f"{runtime_handle.plan.boundary_bytes} > {constraints.max_payload_bytes}."
        )
    suffix_worker = _coordinator_worker(worker_specs)
    if constraints.max_stage_memory_bytes and suffix_worker.memory_bytes:
        suffix_memory = runtime_handle.plan.metadata.get("suffix_memory_bytes") or 0
        if suffix_memory and int(suffix_memory) > constraints.max_stage_memory_bytes:
            raise RuntimeError("Ariadne suffix stage exceeds max_stage_memory_bytes.")
    score = _score(runtime_handle, suffix_worker, objective)
    plan = AriadnePlacementPlan(
        plan_id=runtime_handle.plan.plan_id,
        split_id=runtime_handle.plan.split_id,
        graph_signature=runtime_handle.plan.graph_signature,
        boundary=boundary,
        mode=runtime_handle.plan.mode,
        prefix_worker_id="client",
        suffix_worker_id=suffix_worker.worker_id,
        score=score,
        worker_specs={suffix_worker.worker_id: suffix_worker},
        objective=objective,
        constraints=constraints,
        metadata={
            "backend": "ariadne",
            "model_name": model_name,
            "trainable": runtime_handle.plan.trainable,
            "dynamic_batch": runtime_handle.plan.dynamic_batch,
            "trace_batch_mode": runtime_handle.plan.trace_batch_mode,
            "boundary_bytes": runtime_handle.plan.boundary_bytes,
            "prefix_node_count": runtime_handle.plan.prefix_node_count,
            "suffix_node_count": runtime_handle.plan.suffix_node_count,
            "trainable_suffix": runtime_handle.plan.trainable_suffix,
            "boundary_nodes": runtime_handle.plan.metadata.get("boundary_nodes", ()),
            "requested_boundary": boundary,
            "_runtime_handle": runtime_handle,
        },
    )
    return plan


class AutoSplitPlanner:
    """Plan Ariadne client-prefix/server-suffix split execution."""

    def plan(
        self,
        model,
        sample_inputs: Any,
        *,
        sample_kwargs: Optional[dict[str, Any]] = None,
        worker_specs: Optional[Sequence[WorkerSpec]] = None,
        constraints: Optional[PlacementConstraint] = None,
        objective: Optional[PlacementObjective] = None,
        preferred_stage_count: Optional[int] = None,
        client_stage_count: Optional[int] = 1,
        cache_store: Optional[PlanCacheStore] = None,
        model_name: Optional[str] = None,
        boundary: str = "50%",
        mode: str = "generated_eager",
        trainable: bool = True,
        dynamic_batch: tuple[int, int] | None = None,
        trace_batch_mode: str | None = None,
        compile_options: Any = None,
    ) -> AriadnePlacementPlan:
        if sample_kwargs:
            raise ValueError("Ariadne backend currently accepts positional model inputs only.")
        _validate_stage_counts(
            preferred_stage_count=preferred_stage_count,
            client_stage_count=client_stage_count,
        )
        constraints = constraints or PlacementConstraint()
        objective = objective or PlacementObjective()
        workers = list(worker_specs or [WorkerSpec(worker_id="coordinator", device="cpu")])

        runtime_handle = prepare_ariadne_runtime(
            model,
            sample_inputs,
            boundary=boundary,
            mode=mode,
            trainable=trainable,
            dynamic_batch=dynamic_batch,
            trace_batch_mode=trace_batch_mode,
            objective=None,
            compile_options=compile_options,
        )
        if (
            trainable
            and constraints.require_trainable_tail
            and not runtime_handle.plan.trainable_suffix
        ):
            if boundary == "auto":
                raise RuntimeError("Ariadne did not find a trainable suffix for boundary='auto'.")
            runtime_handle = prepare_ariadne_runtime(
                model,
                sample_inputs,
                boundary="auto",
                mode=mode,
                trainable=trainable,
                dynamic_batch=dynamic_batch,
                trace_batch_mode=trace_batch_mode,
                objective=None,
                compile_options=compile_options,
            )
            boundary = "auto"
            if not runtime_handle.plan.trainable_suffix:
                raise RuntimeError("Ariadne did not find a trainable suffix for this model.")

        placement = _build_placement(
            runtime_handle,
            boundary=boundary,
            worker_specs=workers,
            constraints=constraints,
            objective=objective,
            model_name=model_name or model.__class__.__name__,
        )

        if cache_store is not None:
            cache_store.save(
                PlanCacheEntry(
                    model_name=model_name or model.__class__.__name__,
                    graph_signature=placement.graph_signature,
                    boundary=placement.boundary,
                    split_id=placement.split_id,
                    score=placement.score,
                    worker_signature=PlanCacheStore.worker_signature(workers),
                    constraint_signature=asdict(constraints),
                    objective_signature=asdict(objective),
                    metadata={
                        "plan_id": placement.plan_id,
                        "mode": placement.mode,
                    },
                )
            )
        return placement


def validate_ariadne_stage_counts(
    *,
    preferred_stage_count: Optional[int],
    client_stage_count: Optional[int],
) -> None:
    _validate_stage_counts(
        preferred_stage_count=preferred_stage_count,
        client_stage_count=client_stage_count,
    )
