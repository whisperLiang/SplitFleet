"""Partition and placement planning for autosplit execution."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from torchlens.replay_plan import ExecutionPlan, FrontierSplit, ParamRef

from slbd.autosplit.cache import PlanCacheEntry, PlanCacheStore
from slbd.autosplit.tracer import ModelTracer, TracedModel
from slbd.autosplit.types import (
    PartitionPlan,
    PartitionStage,
    PlacementConstraint,
    PlacementObjective,
    PlacementPlan,
    WorkerSpec,
)

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover - exercised in environments without ortools
    cp_model = None


def _estimate_dtype_size(dtype_name: Any) -> int:
    text = str(dtype_name or "").lower()
    if "float16" in text or "bfloat16" in text or "int16" in text:
        return 2
    if "float64" in text or "int64" in text or "complex64" in text:
        return 8
    if "complex128" in text:
        return 16
    if "int8" in text or "uint8" in text or "bool" in text:
        return 1
    return 4


def _shape_numel(shape: Any) -> int:
    if not shape:
        return 0
    total = 1
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            return 0
        total *= dim
    return total


def _node_activation_bytes(node) -> int:
    return _shape_numel(node.meta.get("tensor_shape")) * _estimate_dtype_size(
        node.meta.get("tensor_dtype")
    )


def _iter_param_addresses(template: Any) -> set[str]:
    addresses: set[str] = set()
    if isinstance(template, ParamRef):
        addresses.add(template.address)
        return addresses
    if isinstance(template, dict):
        for value in template.values():
            addresses.update(_iter_param_addresses(value))
        return addresses
    if isinstance(template, (list, tuple)):
        for value in template:
            addresses.update(_iter_param_addresses(value))
        return addresses
    if hasattr(template, "__dataclass_fields__"):
        for field_name in template.__dataclass_fields__:
            addresses.update(_iter_param_addresses(getattr(template, field_name)))
    return addresses


def _node_parameter_bytes(node, model=None) -> int:
    if model is not None:
        named_parameters = dict(model.named_parameters())
        param_addresses = _iter_param_addresses(node.const_args_template)
        param_addresses.update(_iter_param_addresses(node.const_kwargs_template))
        total = 0
        for address in param_addresses:
            parameter = named_parameters.get(address)
            if parameter is None:
                continue
            total += int(parameter.numel()) * int(parameter.element_size())
        if total:
            return total

    count = (
        node.meta.get("parameter_numel")
        or node.meta.get("num_parameters")
        or node.meta.get("parameter_count")
        or 0
    )
    if not count:
        return 0
    return int(count) * _estimate_dtype_size(node.meta.get("parameter_dtype"))


def _node_compute_cost(node) -> float:
    for key in ("estimated_flops", "flops", "compute_cost"):
        if key in node.meta and node.meta[key]:
            return float(node.meta[key])
    return 1.0


def _build_children(plan: ExecutionPlan) -> dict[int, set[int]]:
    children: dict[int, set[int]] = {node.idx: set() for node in plan.nodes}
    for node in plan.nodes:
        for parent in node.parents:
            children[parent].add(node.idx)
    return children


def _graph_signature_payload(plan: ExecutionPlan, cutoffs: Sequence[int]) -> str:
    return hashlib.sha1(
        "|".join([plan.graph_signature, *(str(cutoff) for cutoff in cutoffs)]).encode("utf-8")
    ).hexdigest()


def _make_stage_id(plan: ExecutionPlan, stage_index: int, cutoffs: Sequence[int]) -> str:
    stage_hash = _graph_signature_payload(plan, [*cutoffs, stage_index])
    return f"stage_{stage_index}_{stage_hash[:10]}"


def build_partition_plan(
    plan: ExecutionPlan,
    cutoffs: Sequence[int],
    *,
    model_name: Optional[str] = None,
) -> PartitionPlan:
    """Convert ordered cutoffs into a reusable multi-stage partition plan."""

    normalized_cutoffs = sorted({int(value) for value in cutoffs if 0 <= int(value) < len(plan.nodes) - 1})
    children = _build_children(plan)
    input_set = set(plan.input_node_indices)
    output_set = set(plan.output_node_indices)
    model = plan.model

    stage_assignment: dict[int, int] = {}
    for index in range(len(plan.nodes)):
        stage_index = 0
        for cutoff in normalized_cutoffs:
            if index > cutoff:
                stage_index += 1
        stage_assignment[index] = stage_index

    stages: list[PartitionStage] = []
    total_stages = len(normalized_cutoffs) + 1
    for stage_index in range(total_stages):
        node_indices = [
            index for index in range(len(plan.nodes))
            if stage_assignment[index] == stage_index
        ]
        stage_set = set(node_indices)
        input_indices = sorted(
            {
                parent
                for index in node_indices
                for parent in plan.nodes[index].parents
                if parent not in stage_set
            }
        )
        passthrough_input_indices = sorted(index for index in input_indices if index in input_set)
        output_indices = sorted(
            {
                index
                for index in node_indices
                if (
                    index in output_set
                    or (
                        index not in input_set
                        and any(stage_assignment[child] > stage_index for child in children[index])
                    )
                )
            }
        )
        stage = PartitionStage(
            stage_id=_make_stage_id(plan, stage_index, normalized_cutoffs),
            node_indices=node_indices,
            input_indices=input_indices,
            output_indices=output_indices,
            input_labels=[plan.nodes[index].label for index in input_indices],
            output_labels=[plan.nodes[index].label for index in output_indices],
            passthrough_input_indices=passthrough_input_indices,
            passthrough_input_labels=[plan.nodes[index].label for index in passthrough_input_indices],
            estimated_compute_cost=sum(_node_compute_cost(plan.nodes[index]) for index in node_indices),
            estimated_activation_bytes=sum(
                _node_activation_bytes(plan.nodes[index]) for index in output_indices
            ),
            estimated_parameter_bytes=sum(
                _node_parameter_bytes(plan.nodes[index], model) for index in node_indices
            ),
            metadata={
                "stage_index": stage_index,
                "cutoffs": list(normalized_cutoffs),
            },
        )
        stages.append(stage)

    plan_id = _graph_signature_payload(plan, normalized_cutoffs)
    return PartitionPlan(
        plan_id=f"partition_{plan_id[:12]}",
        model_name=model_name or plan.model_name,
        execution_plan=plan,
        stages=stages,
        graph_signature=plan.graph_signature,
        metadata={"cutoffs": list(normalized_cutoffs)},
    )


def _score_stage_for_worker(
    stage: PartitionStage,
    worker: WorkerSpec,
    objective: PlacementObjective,
) -> float:
    bandwidth_bytes_per_s = max(worker.bandwidth_mbps, 1.0) * 125_000.0
    device_factor = 0.35 if "cuda" in worker.device.lower() else 1.0
    latency_term = (stage.estimated_compute_cost / 1_000_000.0) * device_factor
    payload_term = stage.estimated_activation_bytes / bandwidth_bytes_per_s
    if worker.memory_bytes:
        memory_ratio = (
            float(stage.estimated_activation_bytes + stage.estimated_parameter_bytes)
            / float(worker.memory_bytes)
        )
    else:
        memory_ratio = 0.0
    return (
        objective.latency_weight * latency_term
        + objective.bandwidth_weight * payload_term
        + objective.memory_weight * memory_ratio
    )


def _privacy_metric(partition_plan: PartitionPlan) -> float:
    non_input_nodes = [
        node.idx for node in partition_plan.execution_plan.nodes
        if node.idx not in partition_plan.execution_plan.input_node_indices
    ]
    if not non_input_nodes:
        return 1.0
    first_stage = partition_plan.stages[0]
    covered = [index for index in first_stage.node_indices if index in non_input_nodes]
    return min(1.0, max(0.0, len(covered) / float(len(non_input_nodes))))


def _compatible_workers(
    stage: PartitionStage,
    worker_specs: Sequence[WorkerSpec],
    constraints: PlacementConstraint,
) -> list[WorkerSpec]:
    compatible = []
    stage_bytes = stage.estimated_activation_bytes + stage.estimated_parameter_bytes
    for worker in worker_specs:
        if not worker.online:
            continue
        if constraints.max_stage_memory_bytes and stage_bytes > constraints.max_stage_memory_bytes:
            continue
        if worker.memory_bytes and stage_bytes > worker.memory_bytes:
            continue
        compatible.append(worker)
    return compatible


def _assign_workers_exact(
    partition_plan: PartitionPlan,
    worker_specs: Sequence[WorkerSpec],
    constraints: PlacementConstraint,
    objective: PlacementObjective,
) -> Optional[tuple[dict[str, str], dict[str, float], float]]:
    if cp_model is None:
        return None

    model = cp_model.CpModel()
    assignment: dict[tuple[int, int], Any] = {}
    costs: dict[tuple[int, int], int] = {}

    for stage_index, stage in enumerate(partition_plan.stages):
        compatible = _compatible_workers(stage, worker_specs, constraints)
        if not compatible:
            return None
        for worker_index, worker in enumerate(worker_specs):
            var = model.NewBoolVar(f"stage_{stage_index}_worker_{worker_index}")
            assignment[(stage_index, worker_index)] = var
            if worker in compatible:
                cost = int(_score_stage_for_worker(stage, worker, objective) * 1_000_000)
                costs[(stage_index, worker_index)] = cost
            else:
                model.Add(var == 0)
        model.Add(sum(assignment[(stage_index, worker_index)] for worker_index in range(len(worker_specs))) == 1)

    model.Minimize(
        sum(
            assignment[(stage_index, worker_index)] * costs.get((stage_index, worker_index), 0)
            for stage_index in range(len(partition_plan.stages))
            for worker_index in range(len(worker_specs))
        )
    )

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    stage_to_worker: dict[str, str] = {}
    stage_scores: dict[str, float] = {}
    total = 0.0
    for stage_index, stage in enumerate(partition_plan.stages):
        for worker_index, worker in enumerate(worker_specs):
            if solver.BooleanValue(assignment[(stage_index, worker_index)]):
                stage_score = _score_stage_for_worker(stage, worker, objective)
                stage_to_worker[stage.stage_id] = worker.worker_id
                stage_scores[stage.stage_id] = stage_score
                total += stage_score
                break
    return stage_to_worker, stage_scores, total


def _assign_workers_heuristic(
    partition_plan: PartitionPlan,
    worker_specs: Sequence[WorkerSpec],
    constraints: PlacementConstraint,
    objective: PlacementObjective,
) -> Optional[tuple[dict[str, str], dict[str, float], float]]:
    stage_to_worker: dict[str, str] = {}
    stage_scores: dict[str, float] = {}
    total = 0.0
    for stage in partition_plan.stages:
        compatible = _compatible_workers(stage, worker_specs, constraints)
        if not compatible:
            return None
        worker = min(
            compatible,
            key=lambda spec: (_score_stage_for_worker(stage, spec, objective), spec.worker_id),
        )
        stage_score = _score_stage_for_worker(stage, worker, objective)
        stage_to_worker[stage.stage_id] = worker.worker_id
        stage_scores[stage.stage_id] = stage_score
        total += stage_score
    return stage_to_worker, stage_scores, total


def _placement_penalty(
    partition_plan: PartitionPlan,
    stage_score_total: float,
    objective: PlacementObjective,
) -> float:
    total_payload = sum(stage.estimated_activation_bytes for stage in partition_plan.stages[:-1])
    privacy_penalty = 1.0 - _privacy_metric(partition_plan)
    return (
        stage_score_total
        + objective.bandwidth_weight * (total_payload / float(32 * 1024 * 1024))
        + objective.privacy_weight * privacy_penalty
    )


class AutoSplitPlanner:
    """End-to-end autosplit planner with cache-aware selection."""

    def __init__(self, tracer: Optional[ModelTracer] = None) -> None:
        self.tracer = tracer or ModelTracer()

    def enumerate_partition_plans(
        self,
        traced: TracedModel,
        *,
        constraints: PlacementConstraint,
        preferred_stage_count: Optional[int] = None,
    ) -> list[PartitionPlan]:
        frontiers = self.tracer.enumerate_frontiers(
            traced.execution_plan,
            max_frontier_size=constraints.max_frontier_size,
            max_splits=max(8, constraints.max_candidates * max(1, constraints.max_stages - 1)),
            mode="minimal",
        )
        cutoffs = sorted(
            {
                int(split.meta["execution_cutoff"])
                for split in frontiers
                if split.meta.get("execution_cutoff") is not None
            }
        )
        plans: list[PartitionPlan] = []
        max_stages = max(1, constraints.max_stages)
        stage_counts = (
            [preferred_stage_count]
            if preferred_stage_count is not None
            else list(range(2, max_stages + 1))
        )
        for stage_count in stage_counts:
            if stage_count is None or stage_count <= 1:
                plans.append(build_partition_plan(traced.execution_plan, [], model_name=traced.execution_plan.model_name))
                continue
            combo_count = 0
            for combo in itertools.combinations(cutoffs, stage_count - 1):
                plans.append(
                    build_partition_plan(
                        traced.execution_plan,
                        combo,
                        model_name=traced.execution_plan.model_name,
                    )
                )
                combo_count += 1
                if combo_count >= constraints.max_candidates:
                    break
        if not plans:
            plans.append(build_partition_plan(traced.execution_plan, [], model_name=traced.execution_plan.model_name))
        return plans[: max(1, constraints.max_candidates)]

    def place_partition_plan(
        self,
        partition_plan: PartitionPlan,
        worker_specs: Sequence[WorkerSpec],
        *,
        constraints: PlacementConstraint,
        objective: PlacementObjective,
    ) -> Optional[PlacementPlan]:
        active_workers = [spec for spec in worker_specs if spec.online]
        if not active_workers:
            active_workers = [WorkerSpec(worker_id="local", device="cpu")]

        privacy_metric = _privacy_metric(partition_plan)
        if privacy_metric < constraints.privacy_metric_lower_bound:
            return None

        assignment = _assign_workers_exact(
            partition_plan,
            active_workers,
            constraints,
            objective,
        )
        if assignment is None:
            assignment = _assign_workers_heuristic(
                partition_plan,
                active_workers,
                constraints,
                objective,
            )
        if assignment is None:
            return None

        stage_to_worker, stage_scores, stage_score_total = assignment
        score = _placement_penalty(partition_plan, stage_score_total, objective)
        return PlacementPlan(
            partition_plan=partition_plan,
            stage_to_worker=stage_to_worker,
            worker_specs={spec.worker_id: spec for spec in active_workers},
            score=score,
            stage_scores=stage_scores,
            objective=objective,
            constraints=constraints,
            metadata={"privacy_metric": privacy_metric},
        )

    def plan(
        self,
        model,
        sample_inputs: Any,
        *,
        sample_kwargs: Optional[Dict[str, Any]] = None,
        worker_specs: Optional[Sequence[WorkerSpec]] = None,
        constraints: Optional[PlacementConstraint] = None,
        objective: Optional[PlacementObjective] = None,
        preferred_stage_count: Optional[int] = None,
        cache_store: Optional[PlanCacheStore] = None,
        model_name: Optional[str] = None,
    ) -> PlacementPlan:
        constraints = constraints or PlacementConstraint()
        objective = objective or PlacementObjective()
        traced = self.tracer.trace(model, sample_inputs, sample_kwargs=sample_kwargs)
        workers = list(worker_specs or [WorkerSpec(worker_id="local", device=str(self.tracer.device))])
        worker_signature = PlanCacheStore.worker_signature(workers)
        cache_key_name = model_name or traced.execution_plan.model_name

        if cache_store is not None:
            cached = cache_store.load(cache_key_name)
            if cached is not None and cached.matches(
                model_name=cache_key_name,
                graph_signature=traced.execution_plan.graph_signature,
                worker_signature=worker_signature,
                constraints=constraints,
                objective=objective,
            ):
                cached_partition = build_partition_plan(
                    traced.execution_plan,
                    cached.cutoffs,
                    model_name=cache_key_name,
                )
                placement = self.place_partition_plan(
                    cached_partition,
                    workers,
                    constraints=constraints,
                    objective=objective,
                )
                if placement is not None:
                    return placement

        partition_plans = self.enumerate_partition_plans(
            traced,
            constraints=constraints,
            preferred_stage_count=preferred_stage_count,
        )
        placements = [
            placement
            for placement in (
                self.place_partition_plan(
                    partition_plan,
                    workers,
                    constraints=constraints,
                    objective=objective,
                )
                for partition_plan in partition_plans
            )
            if placement is not None
        ]
        if not placements:
            raise RuntimeError("No feasible partition placement satisfied the provided constraints.")

        best = min(
            placements,
            key=lambda placement: (
                placement.score,
                placement.partition_plan.stage_count,
                placement.partition_plan.plan_id,
            ),
        )
        if cache_store is not None:
            cache_store.save(
                PlanCacheEntry(
                    model_name=cache_key_name,
                    graph_signature=traced.execution_plan.graph_signature,
                    cutoffs=list(best.partition_plan.metadata.get("cutoffs", [])),
                    stage_to_worker=dict(best.stage_to_worker),
                    score=best.score,
                    worker_signature=worker_signature,
                    constraint_signature=asdict(constraints),
                    objective_signature=asdict(objective),
                    metadata={"plan_id": best.plan_id},
                )
            )
        return best
