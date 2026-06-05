"""Runtime contract and feature ABI helpers for TorchLens autosplit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


FEATURE_ABI_VERSION = "feature-abi.v1"
RUNTIME_CONTRACT_VERSION = "splitfleet-torchlens-runtime-contract.v1"


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalise_dtype(value: object) -> str:
    return str(value or "").replace("torch.", "")


def _normalise_shape_dim(value: object) -> int | str:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def _symbolize_batch_shape(shape: object, *, batch_symbol: str = "B") -> list[int | str]:
    dims = list(shape or []) if isinstance(shape, (list, tuple)) else []
    if not dims:
        return []
    return [batch_symbol, *[_normalise_shape_dim(dim) for dim in dims[1:]]]


def _normalise_boundary_schema(
    boundary_schema: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    normalised: dict[str, dict[str, Any]] = {}
    for label, spec in dict(boundary_schema or {}).items():
        payload = dict(spec) if isinstance(spec, Mapping) else {
            "canonical_id": getattr(spec, "canonical_id", label),
            "torchlens_label": getattr(spec, "torchlens_label", label),
            "module_path": getattr(spec, "module_path", ""),
            "op_type": getattr(spec, "op_type", ""),
            "symbolic_shape": getattr(spec, "shape", None) or (),
            "dtype": getattr(spec, "dtype", ""),
            "requires_grad": getattr(spec, "requires_grad", False),
            "role": getattr(spec, "role", "primary"),
            "output_index": getattr(spec, "output_index", None),
            "device_policy": getattr(spec, "device_policy", "runtime"),
        }
        normalised[str(label)] = {
            "canonical_id": str(payload.get("canonical_id") or label),
            "torchlens_label": str(payload.get("torchlens_label") or label),
            "module_path": str(payload.get("module_path") or ""),
            "op_type": str(payload.get("op_type") or payload.get("op") or ""),
            "symbolic_shape": _symbolize_batch_shape(
                payload.get("symbolic_shape") or payload.get("shape") or ()
            ),
            "dtype": _normalise_dtype(payload.get("dtype")),
            "requires_grad": bool(payload.get("requires_grad", False)),
            "role": str(payload.get("role") or "primary"),
            "output_index": payload.get("output_index"),
            "device_policy": str(payload.get("device_policy") or "runtime"),
        }
    return normalised


def _ordered_boundary_tensors(
    feature_layout: Mapping[str, Mapping[str, Any]] | None,
    labels: list[str],
) -> list[dict[str, Any]]:
    layout = {
        str(label): dict(spec)
        for label, spec in dict(feature_layout or {}).items()
        if isinstance(spec, Mapping)
    }
    ordered = [label for label in labels if label in layout]
    seen = set(ordered)
    ordered.extend(label for label in sorted(layout) if label not in seen)
    tensors: list[dict[str, Any]] = []
    for label in ordered:
        spec = dict(layout.get(label) or {})
        shape_without_batch = [
            _normalise_shape_dim(dim)
            for dim in list(spec.get("shape_without_batch") or [])
        ]
        tensors.append(
            {
                "label": str(label),
                "dtype": _normalise_dtype(spec.get("dtype")),
                "rank": int(spec.get("rank") or len(shape_without_batch) + 1),
                "shape_without_batch": shape_without_batch,
            }
        )
    return tensors


@dataclass(frozen=True)
class FeatureAbiSpec:
    version: str
    model_family: str
    canonical_split_key: str
    graph_signature: str
    boundary_tensor_labels: list[str]
    boundary_tensors: list[dict[str, Any]]
    boundary_schema: dict[str, dict[str, Any]]
    preprocessing_abi: dict[str, Any]
    passthrough_specs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_feature_abi_spec(
    *,
    model_family: str = "",
    adapter_version: str = "",
    runtime_version: str = "",
    canonical_split_key: str = "",
    graph_signature: str = "",
    boundary_tensor_labels: list[str] | tuple[str, ...] | None = None,
    boundary_schema: Mapping[str, Any] | None = None,
    feature_layout: Mapping[str, Mapping[str, Any]] | None = None,
    preprocessing_abi: Mapping[str, Any] | None = None,
    passthrough_specs: Mapping[str, Any] | None = None,
) -> FeatureAbiSpec:
    _ = (adapter_version, runtime_version)
    labels = [str(label) for label in list(boundary_tensor_labels or [])]
    preprocessing = dict(preprocessing_abi or {})
    if "input_tensor_shape" in preprocessing:
        preprocessing["input_tensor_shape"] = _symbolize_batch_shape(preprocessing["input_tensor_shape"])
    return FeatureAbiSpec(
        version=FEATURE_ABI_VERSION,
        model_family=str(model_family or ""),
        canonical_split_key=str(canonical_split_key or ""),
        graph_signature=str(graph_signature or ""),
        boundary_tensor_labels=labels,
        boundary_tensors=_ordered_boundary_tensors(feature_layout, labels),
        boundary_schema=_normalise_boundary_schema(boundary_schema),
        preprocessing_abi=preprocessing,
        passthrough_specs=dict(passthrough_specs or {}),
    )


def feature_abi_id(spec: FeatureAbiSpec | Mapping[str, Any]) -> str:
    payload = spec.to_dict() if isinstance(spec, FeatureAbiSpec) else dict(spec)
    return hashlib.sha1(stable_json(payload).encode("utf-8")).hexdigest()


def feature_layout_id(layout: Mapping[str, Any]) -> str:
    return hashlib.sha1(stable_json(dict(layout)).encode("utf-8")).hexdigest()


def runtime_identity_id(identity: Mapping[str, Any]) -> str:
    return hashlib.sha1(stable_json(dict(identity)).encode("utf-8")).hexdigest()


def build_runtime_contract(
    *,
    model_family: str,
    canonical_split_key: str,
    graph_signature: str,
    boundary_tensor_labels: list[str] | tuple[str, ...],
    boundary_schema: Mapping[str, Any] | None,
    feature_layout: Mapping[str, Mapping[str, Any]] | None = None,
    preprocessing_abi: Mapping[str, Any] | None = None,
    passthrough_specs: Mapping[str, Any] | None = None,
    runtime_backend: str = "torchlens_native",
    adapter_version: str = "",
    runtime_version: str = "",
    trace_batch_mode: str = "",
    dynamic_batch: tuple[int, int] | None = None,
    trace_batch_size: int | None = None,
) -> dict[str, Any]:
    labels = [str(label) for label in list(boundary_tensor_labels or [])]
    abi_spec = build_feature_abi_spec(
        model_family=model_family,
        adapter_version=adapter_version,
        runtime_version=runtime_version,
        canonical_split_key=canonical_split_key,
        graph_signature=graph_signature,
        boundary_tensor_labels=labels,
        boundary_schema=boundary_schema,
        feature_layout=feature_layout,
        preprocessing_abi=preprocessing_abi,
        passthrough_specs=passthrough_specs,
    )
    layout = {
        str(label): dict(spec)
        for label, spec in dict(feature_layout or {}).items()
        if isinstance(spec, Mapping)
    }
    identity = {
        "runtime_backend": runtime_backend,
        "adapter_version": adapter_version,
        "runtime_version": runtime_version,
        "canonical_split_key": canonical_split_key,
        "graph_signature": graph_signature,
        "trace_batch_mode": trace_batch_mode,
        "dynamic_batch": dynamic_batch,
        "trace_batch_size": trace_batch_size,
    }
    return {
        "contract_version": RUNTIME_CONTRACT_VERSION,
        "runtime_backend": runtime_backend,
        "canonical_split_key": canonical_split_key,
        "graph_signature": graph_signature,
        "boundary_tensor_labels": labels,
        "boundary_schema": abi_spec.boundary_schema,
        "feature_layout": layout,
        "feature_layout_id": feature_layout_id(layout) if layout else "",
        "feature_abi_spec": abi_spec.to_dict(),
        "feature_abi_id": feature_abi_id(abi_spec),
        "runtime_identity": identity,
        "runtime_identity_id": runtime_identity_id(identity),
        "trace_batch_mode": trace_batch_mode,
        "dynamic_batch": dynamic_batch,
        "trace_batch_size": trace_batch_size,
    }


def _contract_payload(contract: Mapping[str, Any] | object | None) -> dict[str, Any]:
    if contract is None:
        return {}
    if isinstance(contract, Mapping):
        return dict(contract)
    to_dict = getattr(contract, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return dict(value) if isinstance(value, Mapping) else {}
    return {}


def _payload_feature_abi_id(payload: Mapping[str, Any]) -> str:
    abi_id = str(payload.get("feature_abi_id") or "")
    if abi_id:
        return abi_id
    abi_spec = payload.get("feature_abi_spec")
    if isinstance(abi_spec, Mapping) and abi_spec:
        return feature_abi_id(abi_spec)
    layout_id = str(payload.get("feature_layout_id") or "")
    return layout_id


def _payload_runtime_identity_id(payload: Mapping[str, Any]) -> str:
    identity_id = str(payload.get("runtime_identity_id") or "")
    if identity_id:
        return identity_id
    identity = payload.get("runtime_identity")
    if isinstance(identity, Mapping) and identity:
        return runtime_identity_id(identity)
    return ""


def classify_contract_compatibility(
    edge_contract: Mapping[str, Any] | object | None,
    cloud_contract: Mapping[str, Any] | object | None,
) -> dict[str, Any]:
    edge = _contract_payload(edge_contract)
    cloud = _contract_payload(cloud_contract)
    edge_abi_id = _payload_feature_abi_id(edge)
    cloud_abi_id = _payload_feature_abi_id(cloud)
    edge_layout_id = str(edge.get("feature_layout_id") or "")
    cloud_layout_id = str(cloud.get("feature_layout_id") or "")
    compatible = False
    reason = "feature_abi_id"
    if not edge:
        reason = "missing_edge_runtime_contract"
    elif not cloud:
        reason = "missing_cloud_runtime_contract"
    elif edge_abi_id and cloud_abi_id:
        compatible = edge_abi_id == cloud_abi_id
        reason = "compatible" if compatible else "feature_abi_id"
    else:
        edge_spec = edge.get("feature_abi_spec")
        cloud_spec = cloud.get("feature_abi_spec")
        if isinstance(edge_spec, Mapping) and isinstance(cloud_spec, Mapping):
            compatible = stable_json(edge_spec) == stable_json(cloud_spec)
            reason = "compatible" if compatible else "feature_abi_spec"
        else:
            compatible = bool(edge_layout_id and cloud_layout_id and edge_layout_id == cloud_layout_id)
            reason = "legacy_feature_layout_id_compatible" if compatible else "feature_layout_id"
    edge_runtime_id = _payload_runtime_identity_id(edge)
    cloud_runtime_id = _payload_runtime_identity_id(cloud)
    if compatible and edge_runtime_id and cloud_runtime_id and edge_runtime_id != cloud_runtime_id:
        reason = "runtime_identity_changed_but_feature_abi_compatible"
    return {
        "compatible": compatible,
        "reason": reason,
        "edge_feature_abi_id": edge_abi_id,
        "cloud_feature_abi_id": cloud_abi_id,
        "edge_runtime_identity_id": edge_runtime_id,
        "cloud_runtime_identity_id": cloud_runtime_id,
        "edge_feature_layout_id": edge_layout_id,
        "cloud_feature_layout_id": cloud_layout_id,
        "edge_boundary_tensor_labels": [str(label) for label in list(edge.get("boundary_tensor_labels") or [])],
        "cloud_boundary_tensor_labels": [str(label) for label in list(cloud.get("boundary_tensor_labels") or [])],
    }
