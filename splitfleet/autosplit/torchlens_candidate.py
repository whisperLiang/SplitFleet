"""TorchLens split candidate descriptors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import math
import re

import numpy as np
from torchlens.split.shape_program import ShapeBinding


@dataclass(frozen=True)
class SplitCandidate:
    candidate_id: str
    split_label: str
    boundary: str
    boundary_tensor_labels: list[str]
    boundary_count: int
    estimated_payload_bytes: int
    layer_index: int | None = None
    node_index: int | None = None
    edge_parameter_count: int = 0
    suffix_parameter_count: int = 0
    total_parameter_count: int = 0
    layer_freezing_ratio: float = 0.0
    privacy_leakage: float = 0.0
    is_trainable_tail: bool = True
    graph_signature: str = ""
    trace_batch_mode: str = "batch_gt1"
    dynamic_batch: tuple[int, int] | None = None
    descriptor: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def payload_bytes_from_plan(runtime: Any, plan: Any) -> int:
    """Count boundary tensor bytes at the native capture batch.

    ABI shapes collapse expressions such as ``B * 10`` to ``B``. The shape
    program retains those expressions, and boundary bindings resolve container
    keys to the graph values used by that program.
    """
    program = runtime.trace_graph.shape_program
    if program is None:
        raise RuntimeError("TorchLens payload estimation requires a captured shape program.")
    binding = ShapeBinding({program.batch_symbol: program.traced_batch_size}, {})
    total = 0
    for label, spec in plan.boundary_spec.items():
        shape = program.value_shapes[plan.boundary_bindings[label]].evaluate(binding)
        dtype_name = str(spec.dtype).rsplit(".", 1)[-1]
        match = re.fullmatch(r"<dtype: '([^']+)'>", dtype_name)
        if match:
            dtype_name = match.group(1)
        itemsize = 2 if dtype_name == "bfloat16" else np.dtype(dtype_name).itemsize
        total += math.prod(shape) * itemsize
    return total


def boundary_schema_summary(specs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for label, spec in specs.items():
        summary[str(label)] = {
            "canonical_id": spec.value_id,
            "torchlens_label": spec.label,
            "module_path": spec.module_path or "",
            "op_type": spec.op_type,
            "symbolic_shape": [str(dim) for dim in spec.shape],
            "dtype": str(spec.dtype),
            "requires_grad": bool(spec.requires_grad),
            "role": spec.role,
            "output_index": spec.output_index,
            "device_policy": spec.device_policy,
        }
    return summary


def feature_layout_from_specs(specs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    layout: dict[str, dict[str, Any]] = {}
    for label, spec in specs.items():
        shape = list(spec.shape)
        layout[str(label)] = {
            "dtype": str(spec.dtype),
            "shape_without_batch": [str(dim) for dim in shape[1:]],
            "rank": len(shape),
            "requires_grad": bool(spec.requires_grad),
            "device_policy": spec.device_policy,
        }
    return layout


def parameter_count_for_nodes(
    runtime: Any,
    node_names: Iterable[str],
    *,
    trainable_only: bool = False,
) -> int:
    selected = set(node_names)
    seen: set[int] = set()
    total = 0
    for node in runtime.trace_graph.nodes:
        if node.canonical_id not in selected:
            continue
        for parameter in node.param_refs:
            if trainable_only and not parameter.is_trainable:
                continue
            value = parameter.handle
            identity = id(value) if value is not None else hash((parameter.address, parameter.shape))
            if identity not in seen:
                seen.add(identity)
                total += math.prod(value.shape if value is not None else parameter.shape)
    return int(total)


def candidate_from_plan(
    runtime: Any,
    split_spec: Any,
    plan: Any,
    *,
    node_index: int | None = None,
    graph_signature: str = "",
) -> SplitCandidate:
    graph = runtime.trace_graph
    prefix_nodes = list(plan.prefix_node_ids)
    suffix_nodes = list(plan.suffix_node_ids)
    boundary_labels = list(plan.boundary_node_ids)
    specs = plan.boundary_spec
    payload_bytes = payload_bytes_from_plan(runtime, plan)
    all_nodes = [
        node.canonical_id
        for node in graph.nodes
    ]
    edge_params = parameter_count_for_nodes(runtime, prefix_nodes)
    suffix_params = parameter_count_for_nodes(runtime, suffix_nodes)
    trainable_suffix_params = parameter_count_for_nodes(
        runtime,
        suffix_nodes,
        trainable_only=True,
    )
    total_params = parameter_count_for_nodes(runtime, all_nodes)
    freezing_ratio = float(edge_params) / float(total_params) if total_params else 0.0
    privacy_leakage = 1.0 / float(edge_params) if edge_params > 0 else float("inf")
    nodes = {node.canonical_id: node for node in graph.nodes}
    target_node = nodes[plan.target_node_id]
    suffix_compute_nodes = [node_id for node_id in suffix_nodes if not nodes[node_id].is_output]
    split_label = target_node.label
    boundary = f"{plan.boundary_kind}:{plan.target_node_id}"
    candidate_id = boundary
    descriptor = {
        "candidate_id": candidate_id,
        "split_label": split_label,
        "boundary": boundary,
        "boundary_tensor_labels": boundary_labels,
        "boundary_schema": boundary_schema_summary(specs),
        "feature_layout": feature_layout_from_specs(specs),
        "prefix_node_count": len(prefix_nodes),
        "suffix_node_count": len(suffix_compute_nodes),
        "trainable_suffix_parameter_count": trainable_suffix_params,
        "torchlens_split_id": plan.split_id,
        "torchlens_split_label": split_label,
        "runtime_backend": "torchlens_native",
        "payload_batch_size": runtime.traced_batch_size,
    }
    return SplitCandidate(
        candidate_id=candidate_id,
        split_label=split_label,
        boundary=boundary,
        boundary_tensor_labels=boundary_labels,
        boundary_count=len(boundary_labels),
        estimated_payload_bytes=payload_bytes,
        layer_index=node_index,
        node_index=node_index,
        edge_parameter_count=edge_params,
        suffix_parameter_count=suffix_params,
        total_parameter_count=total_params,
        layer_freezing_ratio=freezing_ratio,
        privacy_leakage=privacy_leakage,
        is_trainable_tail=split_spec.features.training
        and trainable_suffix_params > 0,
        graph_signature=graph_signature,
        descriptor=descriptor,
    )


def build_candidate_descriptor(candidate: SplitCandidate) -> dict[str, Any]:
    return candidate.to_dict()
