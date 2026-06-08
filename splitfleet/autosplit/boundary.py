"""SplitFleet boundary payload types for TorchLens autosplit."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import torch
from torchlens.split import BoundaryTensorSpec, ReplayBoundary


@dataclass
class BoundarySpec:
    split_id: str
    boundary: str
    boundary_tensor_labels: list[str]
    boundary_tensor_specs: dict[str, Any] = field(default_factory=dict)
    torchlens_spec_dict: dict[str, Any] = field(default_factory=dict)
    trace_batch_mode: str = "batch_gt1"
    dynamic_batch: tuple[int, int] | None = None
    batch_symbol: str = "B"
    feature_layout_id: str = ""
    feature_abi_id: str = ""
    feature_layout: dict[str, Any] = field(default_factory=dict)
    passthrough_specs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        boundary: str | None = None,
        feature_layout_id: str = "",
        feature_abi_id: str = "",
    ) -> "BoundarySpec":
        plan = getattr(runtime, "plan", None)
        split_spec = getattr(runtime, "split_spec", None)
        raw_specs = dict(
            getattr(plan, "boundary_specs", None)
            or getattr(runtime, "boundary_spec", None)
            or {}
        )
        labels = _ordered_labels(
            list(getattr(plan, "boundary_nodes", ()) or ()),
            raw_specs,
        )
        split_id = str(getattr(runtime, "split_id", "") or getattr(plan, "split_id", "") or "")
        boundary_value = str(boundary or getattr(split_spec, "boundary", "") or split_id)
        stable_specs = _stable_spec_dict(raw_specs)
        return cls(
            split_id=split_id or boundary_value,
            boundary=boundary_value,
            boundary_tensor_labels=labels,
            boundary_tensor_specs=dict(raw_specs),
            torchlens_spec_dict=stable_specs,
            trace_batch_mode=str(getattr(split_spec, "trace_batch_mode", "batch_gt1")),
            dynamic_batch=_normalise_dynamic_batch(getattr(split_spec, "dynamic_batch", None)),
            batch_symbol=str(getattr(split_spec, "batch_symbol", "B") or "B"),
            feature_layout_id=str(feature_layout_id or ""),
            feature_abi_id=str(feature_abi_id or ""),
            feature_layout=_feature_layout_from_specs(raw_specs),
            passthrough_specs={},
        )

    def to_stable_dict(self) -> dict[str, Any]:
        return {
            "split_id": self.split_id,
            "boundary": self.boundary,
            "boundary_tensor_labels": list(self.boundary_tensor_labels),
            "boundary_tensor_specs": _stable_spec_dict(self.boundary_tensor_specs),
            "torchlens_spec_dict": _stable_spec_dict(self.torchlens_spec_dict),
            "trace_batch_mode": self.trace_batch_mode,
            "dynamic_batch": list(self.dynamic_batch) if self.dynamic_batch is not None else None,
            "batch_symbol": self.batch_symbol,
            "feature_layout_id": self.feature_layout_id,
            "feature_abi_id": self.feature_abi_id,
            "feature_layout": _json_safe(self.feature_layout),
            "passthrough_specs": _json_safe(self.passthrough_specs),
        }

    def to_torchlens_spec(self) -> dict[str, BoundaryTensorSpec]:
        raw_specs = self.boundary_tensor_specs or self.torchlens_spec_dict
        specs: dict[str, BoundaryTensorSpec] = {}
        for label in self.boundary_tensor_labels or list(raw_specs):
            if label not in raw_specs:
                continue
            specs[str(label)] = _coerce_boundary_tensor_spec(label, raw_specs[label])
        for label, spec in raw_specs.items():
            label = str(label)
            if label not in specs:
                specs[label] = _coerce_boundary_tensor_spec(label, spec)
        return specs


@dataclass
class BoundaryPayload:
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)
    passthrough_inputs: tuple[Any, ...] = ()
    batch_size: int | None = None
    spec: BoundarySpec | None = None
    native: Any | None = None

    @classmethod
    def from_torchlens_boundary(
        cls,
        boundary: ReplayBoundary | "BoundaryPayload",
        *,
        runtime: Any | None = None,
        boundary_name: str | None = None,
        feature_layout_id: str = "",
        feature_abi_id: str = "",
    ) -> "BoundaryPayload":
        return from_torchlens_boundary(
            boundary,
            runtime=runtime,
            boundary_name=boundary_name,
            feature_layout_id=feature_layout_id,
            feature_abi_id=feature_abi_id,
        )

    def to_torchlens_boundary(self) -> ReplayBoundary:
        return to_torchlens_boundary(self)

    def to(self, device: torch.device | str) -> "BoundaryPayload":
        native = self.native.to(device) if hasattr(self.native, "to") else self.native
        return replace(
            self,
            tensors={key: tensor.to(device) for key, tensor in self.tensors.items()},
            native=native,
        )

    def cpu(self) -> "BoundaryPayload":
        return self.to("cpu")

    def detach(self) -> "BoundaryPayload":
        native = self.native.detach() if hasattr(self.native, "detach") else self.native
        return replace(
            self,
            tensors={key: tensor.detach() for key, tensor in self.tensors.items()},
            native=native,
        )


def _normalise_dynamic_batch(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    low, high = list(value)
    return int(low), int(high)


def _ordered_labels(preferred: list[Any], specs: Mapping[str, Any]) -> list[str]:
    labels = [str(label) for label in preferred if str(label) in dict(specs)]
    seen = set(labels)
    labels.extend(str(label) for label in dict(specs) if str(label) not in seen)
    return labels


def _spec_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _dtype_to_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text if text.startswith("torch.") else f"torch.{text}" if hasattr(torch, text) else text


def _dtype_from_string(value: Any) -> torch.dtype | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    text = str(value)
    if not text:
        return None
    name = text.replace("torch.", "")
    dtype = getattr(torch, name, None)
    return dtype if isinstance(dtype, torch.dtype) else None


def _normalise_shape(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    dims = []
    for dim in list(value):
        try:
            dims.append(int(dim))
        except (TypeError, ValueError):
            dims.append(str(dim))
    return tuple(dims)


def _stable_spec(label: str, spec: Any) -> dict[str, Any]:
    shape = (
        _spec_value(spec, "shape", None)
        or _spec_value(spec, "symbolic_shape", None)
        or None
    )
    normalised_shape = _normalise_shape(shape)
    return {
        "canonical_id": str(_spec_value(spec, "canonical_id", label) or label),
        "torchlens_label": str(_spec_value(spec, "torchlens_label", label) or label),
        "module_path": str(_spec_value(spec, "module_path", "") or ""),
        "op_type": str(_spec_value(spec, "op_type", "") or ""),
        "shape": list(normalised_shape) if normalised_shape is not None else None,
        "symbolic_shape": list(normalised_shape) if normalised_shape is not None else None,
        "dtype": _dtype_to_string(_spec_value(spec, "dtype", None)),
        "requires_grad": bool(_spec_value(spec, "requires_grad", False)),
        "role": str(_spec_value(spec, "role", "primary") or "primary"),
        "output_index": _spec_value(spec, "output_index", None),
        "device_policy": str(_spec_value(spec, "device_policy", "runtime") or "runtime"),
    }


def _stable_spec_dict(specs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(label): _stable_spec(str(label), spec) for label, spec in dict(specs or {}).items()}


def _coerce_boundary_tensor_spec(label: str, spec: Any) -> BoundaryTensorSpec:
    if isinstance(spec, BoundaryTensorSpec):
        return spec
    stable = _stable_spec(label, spec)
    return BoundaryTensorSpec(
        canonical_id=stable["canonical_id"],
        torchlens_label=stable["torchlens_label"],
        module_path=stable["module_path"],
        op_type=stable["op_type"],
        shape=_normalise_shape(stable.get("shape")),
        dtype=_dtype_from_string(stable.get("dtype")),
        requires_grad=bool(stable.get("requires_grad", False)),
        role=str(stable.get("role") or "primary"),
        output_index=stable.get("output_index"),
        device_policy=str(stable.get("device_policy") or "runtime"),
    )


def _feature_layout_from_specs(specs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    layout: dict[str, dict[str, Any]] = {}
    for label, spec in dict(specs or {}).items():
        stable = _stable_spec(str(label), spec)
        shape = list(stable.get("shape") or [])
        layout[str(label)] = {
            "dtype": stable["dtype"],
            "shape_without_batch": list(shape[1:]),
            "rank": len(shape),
            "requires_grad": bool(stable.get("requires_grad", False)),
            "device_policy": str(stable.get("device_policy") or "runtime"),
        }
    return layout


def _infer_batch_size(tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]) -> int | None:
    value = metadata.get("batch_size")
    if value is not None:
        return int(value)
    for tensor in tensors.values():
        if isinstance(tensor, torch.Tensor) and tensor.ndim > 0:
            return int(tensor.shape[0])
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return _dtype_to_string(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def from_torchlens_boundary(
    boundary: ReplayBoundary | BoundaryPayload,
    *,
    runtime: Any | None = None,
    boundary_name: str | None = None,
    feature_layout_id: str = "",
    feature_abi_id: str = "",
) -> BoundaryPayload:
    if isinstance(boundary, BoundaryPayload):
        return boundary
    metadata = dict(getattr(boundary, "metadata", {}) or {})
    tensors = dict(getattr(boundary, "tensors", {}) or {})
    labels = [
        str(label)
        for label in list(metadata.get("boundary_order") or getattr(boundary, "spec", {}).keys() or tensors.keys())
    ]
    tensors = {label: tensors[label] for label in labels if label in tensors}
    if runtime is not None:
        boundary_spec = BoundarySpec.from_runtime(
            runtime,
            boundary=boundary_name,
            feature_layout_id=feature_layout_id,
            feature_abi_id=feature_abi_id,
        )
    else:
        raw_specs = dict(getattr(boundary, "spec", {}) or {})
        boundary_spec = BoundarySpec(
            split_id=str(metadata.get("split_id") or boundary_name or ""),
            boundary=str(boundary_name or metadata.get("split_id") or ""),
            boundary_tensor_labels=labels or list(raw_specs),
            boundary_tensor_specs=raw_specs,
            torchlens_spec_dict=_stable_spec_dict(raw_specs),
            feature_layout_id=feature_layout_id,
            feature_abi_id=feature_abi_id,
            feature_layout=_feature_layout_from_specs(raw_specs),
        )
    metadata.setdefault("split_id", boundary_spec.split_id)
    metadata["boundary_order"] = list(boundary_spec.boundary_tensor_labels)
    if boundary_spec.feature_abi_id:
        metadata["feature_abi_id"] = boundary_spec.feature_abi_id
    return BoundaryPayload(
        tensors=tensors,
        metadata=metadata,
        passthrough_inputs=tuple(getattr(boundary, "passthrough_inputs", ()) or ()),
        batch_size=_infer_batch_size(tensors, metadata),
        spec=boundary_spec,
        native=boundary,
    )


def to_torchlens_boundary(payload: ReplayBoundary | BoundaryPayload) -> ReplayBoundary:
    if isinstance(payload, ReplayBoundary):
        return payload
    if isinstance(payload.native, ReplayBoundary):
        native = payload.native
        spec = dict(getattr(native, "spec", {}) or {})
    elif payload.spec is not None:
        spec = payload.spec.to_torchlens_spec()
    else:
        spec = {}
    if not spec:
        raise RuntimeError(
            "BoundaryPayload cannot be converted to TorchLens ReplayBoundary without BoundarySpec tensor specs."
        )
    labels = list(payload.spec.boundary_tensor_labels if payload.spec is not None else spec.keys())
    tensors = {label: payload.tensors[label] for label in labels if label in payload.tensors}
    for label, tensor in payload.tensors.items():
        if label not in tensors:
            tensors[label] = tensor
    metadata = dict(payload.metadata)
    if payload.spec is not None:
        metadata.setdefault("split_id", payload.spec.split_id)
        metadata["boundary_order"] = list(payload.spec.boundary_tensor_labels)
        if payload.spec.feature_abi_id:
            metadata["feature_abi_id"] = payload.spec.feature_abi_id
    if payload.batch_size is not None:
        metadata["batch_size"] = int(payload.batch_size)
    return ReplayBoundary(tensors=tensors, spec=spec, metadata=metadata)


def move_boundary_to_device(
    payload: ReplayBoundary | BoundaryPayload,
    device: torch.device | str | None,
) -> ReplayBoundary | BoundaryPayload:
    if device is None:
        return payload
    if hasattr(payload, "to"):
        return payload.to(device)
    return payload
