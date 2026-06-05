"""TorchLens split candidate descriptors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch


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


def torch_dtype_size(dtype: torch.dtype | None) -> int:
    if dtype is None:
        return 4
    try:
        return int(torch.empty((), dtype=dtype).element_size())
    except Exception:
        return 4


def symbolic_dim_size(dim: Any) -> int:
    if isinstance(dim, int):
        return max(1, int(dim))
    text = str(dim)
    if text == "B":
        return 1
    if text.startswith("B*"):
        try:
            return max(1, int(text[2:]))
        except ValueError:
            return 1
    try:
        return max(1, int(text))
    except ValueError:
        return 1


def shape_numel(shape: Sequence[Any] | Any) -> int:
    total = 1
    for dim in list(shape or ()):
        total *= symbolic_dim_size(dim)
    return int(total)


def payload_bytes_from_specs(specs: Mapping[str, Any]) -> int:
    total = 0
    for spec in dict(specs or {}).values():
        total += shape_numel(getattr(spec, "shape", None) or ()) * torch_dtype_size(
            getattr(spec, "dtype", None)
        )
    return int(total)


def boundary_schema_summary(specs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for label, spec in dict(specs or {}).items():
        summary[str(label)] = {
            "canonical_id": str(getattr(spec, "canonical_id", "") or ""),
            "torchlens_label": str(getattr(spec, "torchlens_label", label) or label),
            "module_path": str(getattr(spec, "module_path", "") or ""),
            "op_type": str(getattr(spec, "op_type", "") or ""),
            "symbolic_shape": [str(dim) for dim in list(getattr(spec, "shape", ()) or ())],
            "dtype": str(getattr(spec, "dtype", "") or ""),
            "requires_grad": bool(getattr(spec, "requires_grad", False)),
            "role": str(getattr(spec, "role", "") or ""),
            "output_index": getattr(spec, "output_index", None),
            "device_policy": str(getattr(spec, "device_policy", "runtime") or "runtime"),
        }
    return summary


def feature_layout_from_specs(specs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    layout: dict[str, dict[str, Any]] = {}
    for label, spec in dict(specs or {}).items():
        shape = list(getattr(spec, "shape", ()) or ())
        layout[str(label)] = {
            "dtype": str(getattr(spec, "dtype", "") or ""),
            "shape_without_batch": [str(dim) for dim in shape[1:]],
            "rank": len(shape),
        }
    return layout


def _parameter_logs_for_node(node: Any) -> list[Any]:
    return list(getattr(getattr(node, "layer", None), "parent_param_logs", []) or [])


def _parameter_from_log(
    log: Any,
    named_parameters: Mapping[str, torch.nn.Parameter],
) -> torch.nn.Parameter | None:
    param = getattr(log, "_param_ref", None)
    if isinstance(param, torch.nn.Parameter):
        return param
    address = str(getattr(log, "address", "") or "")
    if address and address in named_parameters:
        return named_parameters[address]
    return None


def parameter_count_for_nodes(
    runtime: Any,
    node_names: Iterable[str],
    *,
    trainable_only: bool = False,
) -> int:
    graph = getattr(runtime, "trace_graph", None)
    model = getattr(runtime, "model", None)
    named_parameters = dict(model.named_parameters()) if isinstance(model, torch.nn.Module) else {}
    selected = {str(name) for name in node_names}
    seen: set[int | str] = set()
    total = 0
    if graph is None:
        return 0
    for node in graph.ordered_nodes():
        if str(getattr(node, "torchlens_label", "")) not in selected:
            continue
        for log in _parameter_logs_for_node(node):
            param = _parameter_from_log(log, named_parameters)
            if param is not None:
                if trainable_only and not bool(param.requires_grad):
                    continue
                key: int | str = id(param)
                numel = int(param.numel())
            else:
                if trainable_only and not bool(
                    getattr(log, "requires_grad", getattr(log, "trainable", True))
                ):
                    continue
                key = str(getattr(log, "address", "") or id(log))
                numel = shape_numel(getattr(log, "shape", None) or getattr(log, "tensor_shape", None) or ())
            if key in seen:
                continue
            seen.add(key)
            total += numel
    return int(total)


def candidate_from_plan(
    runtime: Any,
    split_spec: Any,
    plan: Any,
    *,
    node_index: int | None = None,
    graph_signature: str = "",
) -> SplitCandidate:
    graph = getattr(runtime, "trace_graph", None)
    prefix_nodes = [str(node) for node in list(getattr(plan, "prefix_nodes", ()) or ())]
    suffix_nodes = [str(node) for node in list(getattr(plan, "suffix_nodes", ()) or ())]
    boundary_labels = [str(node) for node in list(getattr(plan, "boundary_nodes", ()) or ())]
    specs = dict(getattr(plan, "boundary_specs", {}) or {})
    payload_bytes = payload_bytes_from_specs(specs)
    all_nodes = [
        str(getattr(node, "torchlens_label", ""))
        for node in graph.ordered_nodes()
    ] if graph is not None else []
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
    split_label = str(getattr(plan, "split_label", "") or "")
    boundary = str(getattr(plan, "split_id", "") or getattr(split_spec, "boundary", "") or "")
    if boundary and not boundary.startswith("after:") and split_label:
        boundary = f"after:{split_label}"
    candidate_id = boundary or f"after:{split_label}"
    descriptor = {
        "candidate_id": candidate_id,
        "split_label": split_label,
        "boundary": boundary,
        "boundary_tensor_labels": boundary_labels,
        "boundary_schema": boundary_schema_summary(specs),
        "feature_layout": feature_layout_from_specs(specs),
        "prefix_node_count": len(prefix_nodes),
        "suffix_node_count": len(suffix_nodes),
        "trainable_suffix_parameter_count": trainable_suffix_params,
        "torchlens_split_id": getattr(plan, "split_id", None),
        "torchlens_split_label": getattr(plan, "split_label", None),
        "runtime_backend": "torchlens_native",
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
        is_trainable_tail=bool(getattr(split_spec, "trainable", True))
        and trainable_suffix_params > 0,
        graph_signature=graph_signature,
        trace_batch_mode=str(getattr(split_spec, "trace_batch_mode", "batch_gt1")),
        dynamic_batch=getattr(split_spec, "dynamic_batch", None),
        descriptor=descriptor,
    )


def build_candidate_descriptor(candidate: SplitCandidate) -> dict[str, Any]:
    return candidate.to_dict()
