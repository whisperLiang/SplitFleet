"""TorchLens-backed two-stage autosplit placement planning."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional, Sequence

from splitfleet.autosplit.cache import PlanCacheEntry, PlanCacheStore
from splitfleet.autosplit.torchlens_backend import (
    TorchLensRuntimeHandle,
    TorchLensSplitBackend,
)
from splitfleet.autosplit.torchlens_candidate import SplitCandidate
from splitfleet.autosplit.types import (
    PlacementConstraint,
    PlacementObjective,
    SplitPlacementPlan,
    WorkerSpec,
)
from splitfleet.split_engine import graph_contract_for_runtime_handle
from splitfleet.tasks import ModelInputs


TWO_STAGE_ERROR = "TorchLens autosplit backend supports prefix/suffix two-stage split only."
ONE_CLIENT_STAGE_ERROR = "TorchLens autosplit backend supports exactly one client-local prefix stage."


def validate_stage_counts(
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


def _candidate_satisfies_constraints(
    candidate: SplitCandidate,
    validation: dict[str, Any],
    constraints: PlacementConstraint,
) -> bool:
    if constraints.max_frontier_size is not None and candidate.boundary_count > constraints.max_frontier_size:
        return False
    if candidate.estimated_payload_bytes > constraints.max_payload_bytes:
        return False
    if constraints.require_trainable_tail and not candidate.is_trainable_tail:
        return False
    if constraints.max_privacy_leakage is not None and (
        candidate.privacy_leakage > float(constraints.max_privacy_leakage)
    ):
        return False
    if constraints.max_layer_freezing_ratio is not None and (
        candidate.layer_freezing_ratio > float(constraints.max_layer_freezing_ratio)
    ):
        return False
    if not bool(validation.get("success")):
        return False
    return True


def _rejection_reason(
    candidate: SplitCandidate,
    validation: dict[str, Any],
    constraints: PlacementConstraint,
) -> str:
    if constraints.max_frontier_size is not None and candidate.boundary_count > constraints.max_frontier_size:
        return "max_frontier_size"
    if candidate.estimated_payload_bytes > constraints.max_payload_bytes:
        return "max_payload_bytes"
    if constraints.require_trainable_tail and not candidate.is_trainable_tail:
        return "suffix_not_trainable"
    if constraints.max_privacy_leakage is not None and (
        candidate.privacy_leakage > float(constraints.max_privacy_leakage)
    ):
        return "max_privacy_leakage"
    if constraints.max_layer_freezing_ratio is not None and (
        candidate.layer_freezing_ratio > float(constraints.max_layer_freezing_ratio)
    ):
        return "max_layer_freezing_ratio"
    if not bool(validation.get("success")):
        return "replay_validation_failed"
    return "unknown"


def _build_placement(
    runtime_handle: TorchLensRuntimeHandle,
    *,
    candidate: SplitCandidate,
    validation: dict[str, Any],
    worker_specs: Sequence[WorkerSpec],
    constraints: PlacementConstraint,
    objective: PlacementObjective,
    model_name: Optional[str],
) -> SplitPlacementPlan:
    suffix_worker = _coordinator_worker(worker_specs)
    if constraints.max_stage_memory_bytes and suffix_worker.memory_bytes:
        suffix_memory = runtime_handle.plan.metadata.get("suffix_memory_bytes") or 0
        if suffix_memory and int(suffix_memory) > constraints.max_stage_memory_bytes:
            raise RuntimeError("TorchLens suffix stage exceeds max_stage_memory_bytes.")
    contract = dict(runtime_handle.plan.runtime_contract)
    metadata = {
        "backend": "torchlens",
        "runtime_backend": "torchlens_native",
        "model_name": model_name,
        "trainable": runtime_handle.plan.trainable,
        "dynamic_batch": runtime_handle.plan.dynamic_batch,
        "trace_batch_mode": runtime_handle.plan.trace_batch_mode,
        "trace_batch_size": runtime_handle.plan.trace_batch_size,
        "boundary_bytes": runtime_handle.plan.boundary_bytes,
        "prefix_node_count": runtime_handle.plan.prefix_node_count,
        "suffix_node_count": runtime_handle.plan.suffix_node_count,
        "trainable_suffix": runtime_handle.plan.trainable_suffix,
        "boundary_nodes": tuple(runtime_handle.plan.boundary_tensor_labels),
        "candidate_descriptor": candidate.to_dict(),
        "validation": dict(validation),
        "runtime_contract": contract,
        "feature_abi_id": contract.get("feature_abi_id", ""),
        "torchlens_version": runtime_handle.plan.torchlens_version,
        "_runtime_handle": runtime_handle,
    }
    graph_contract = graph_contract_for_runtime_handle(runtime_handle)
    return SplitPlacementPlan(
        plan_id=runtime_handle.plan.plan_id,
        split_id=runtime_handle.plan.split_id,
        graph_signature=runtime_handle.plan.graph_signature,
        boundary=runtime_handle.plan.boundary,
        mode=runtime_handle.plan.mode,
        prefix_worker_id="client",
        suffix_worker_id=suffix_worker.worker_id,
        # Static/manual planning does not predict latency. Dynamic cost
        # learning and ranking belong exclusively to CoSplit-UCB.
        score=0.0,
        backend="torchlens",
        engine="torchlens",
        split_request={"boundary": runtime_handle.plan.boundary, "mode": runtime_handle.plan.mode},
        canonical_graph_hash=graph_contract.canonical_graph_hash,
        boundary_schema_hash=graph_contract.boundary_schema_hash,
        capabilities={"contract_digest": graph_contract.capability_hash},
        runtime_backend="torchlens_native",
        torchlens_version=runtime_handle.plan.torchlens_version,
        model_name=model_name or runtime_handle.plan.model_name,
        model_family=runtime_handle.plan.model_family,
        candidate_id=candidate.candidate_id,
        split_label=candidate.split_label,
        boundary_tensor_labels=list(candidate.boundary_tensor_labels),
        payload_bytes=candidate.estimated_payload_bytes,
        validation=dict(validation),
        candidate_descriptor=candidate.to_dict(),
        trace_signature=runtime_handle.plan.graph_signature,
        trace_batch_mode=runtime_handle.plan.trace_batch_mode,
        dynamic_batch=runtime_handle.plan.dynamic_batch,
        trace_batch_size=runtime_handle.plan.trace_batch_size,
        canonical_split_key=candidate.boundary,
        feature_abi_id=str(contract.get("feature_abi_id", "")),
        runtime_contract=contract,
        worker_specs={suffix_worker.worker_id: suffix_worker},
        objective=objective,
        constraints=constraints,
        metadata=metadata,
    )


class AutoSplitPlanner:
    """Plan TorchLens client-prefix/server-suffix split execution."""

    def plan(
        self,
        model,
        sample_inputs: Any,
        *,
        sample_kwargs: Optional[dict[str, Any]] = None,
        batch_axes: dict[str, int] | None = None,
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
    ) -> SplitPlacementPlan:
        validate_stage_counts(
            preferred_stage_count=preferred_stage_count,
            client_stage_count=client_stage_count,
        )
        constraints = constraints or PlacementConstraint()
        objective = objective or PlacementObjective()
        workers = list(worker_specs or [WorkerSpec(worker_id="coordinator", device="cpu")])

        call = ModelInputs.from_value(sample_inputs)
        backend = TorchLensSplitBackend(model_name=model_name or model.__class__.__name__)
        backend.trace(
            model,
            call.args,
            sample_kwargs=dict(call.kwargs) if sample_kwargs is None else sample_kwargs,
            batch_axes=batch_axes,
            boundary=boundary,
            mode=mode,
            trainable=trainable,
            dynamic_batch=dynamic_batch,
            trace_batch_mode=trace_batch_mode,
            model_name=model_name or model.__class__.__name__,
        )
        # Explicit points (including percentages) are promises to the caller.
        # Only ``auto`` is allowed to search for a different placement.
        candidates = (
            backend.enumerate_candidates(
                max_boundary_count=constraints.max_frontier_size,
                max_payload_bytes=constraints.max_payload_bytes,
                max_candidates=None,
                kinds=("before", "after"),
            )
            if boundary == "auto"
            else [backend.current_candidate] if backend.current_candidate is not None else []
        )
        if not candidates:
            raise RuntimeError("TorchLens did not enumerate any legal split candidates for this model.")
        # ``trace`` resolves explicit module names/percentages to a concrete
        # TorchLens candidate.  Preserve that caller choice when it satisfies
        # constraints; enumeration order is graph order and must not silently
        # replace every requested placement with the earliest legal cut.
        requested = backend.current_candidate
        if requested is not None:
            candidates = sorted(
                candidates,
                key=lambda item: item.candidate_id != requested.candidate_id,
            )

        checked = 0
        selected: tuple[SplitCandidate, dict[str, Any]] | None = None
        rejected: list[dict[str, Any]] = []
        for candidate in candidates:
            validation = backend.validate_candidate(candidate)
            checked += 1
            if _candidate_satisfies_constraints(candidate, validation, constraints):
                selected = (candidate, validation)
                break
            rejected.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "boundary": candidate.boundary,
                    "estimated_payload_bytes": candidate.estimated_payload_bytes,
                    "boundary_count": candidate.boundary_count,
                    "validation_error": validation.get("error"),
                    "max_abs_diff": validation.get("max_abs_diff"),
                    "max_rel_diff": validation.get("max_rel_diff"),
                    "reason": _rejection_reason(candidate, validation, constraints),
                }
            )
            if constraints.max_candidates and checked >= constraints.max_candidates:
                break
        if selected is None:
            raise RuntimeError(
                "TorchLens did not find a split candidate satisfying constraints. "
                f"checked={checked}, rejected={rejected[:5]!r}"
            )
        selected_candidate, validation = selected
        if backend.current_candidate is None or (
            backend.current_candidate.candidate_id != selected_candidate.candidate_id
        ):
            backend.split(selected_candidate)
        runtime_handle = backend.make_handle()
        placement = _build_placement(
            runtime_handle,
            candidate=runtime_handle.backend.current_candidate or selected_candidate,
            validation=validation,
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
                        "backend": "torchlens",
                    },
                )
            )
        return placement
