"""SplitFleet boundary payload types for TorchLens autosplit."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import torch
from torchlens.split import ReplayBoundary


@dataclass
class BoundarySpec:
    boundary: str
    boundary_tensor_labels: list[str]
    feature_layout: dict[str, Any] = field(default_factory=dict)
    passthrough_specs: dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryPayload:
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)
    passthrough_inputs: tuple[Any, ...] = ()
    batch_size: int | None = None
    spec: BoundarySpec | None = None
    native: Any | None = None

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


def _spec_value(value: Any, name: str, default: Any = "") -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _boundary_schema(specs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for label, tensor_spec in dict(specs or {}).items():
        schema[str(label)] = {
            "canonical_id": str(_spec_value(tensor_spec, "canonical_id", label) or label),
            "torchlens_label": str(_spec_value(tensor_spec, "torchlens_label", label) or label),
            "module_path": str(_spec_value(tensor_spec, "module_path", "") or ""),
            "op_type": str(_spec_value(tensor_spec, "op_type", "") or ""),
            "symbolic_shape": [
                str(dim)
                for dim in list(
                    _spec_value(tensor_spec, "shape", None)
                    or _spec_value(tensor_spec, "symbolic_shape", None)
                    or ()
                )
            ],
            "dtype": str(_spec_value(tensor_spec, "dtype", "") or ""),
            "requires_grad": bool(_spec_value(tensor_spec, "requires_grad", False)),
            "role": str(_spec_value(tensor_spec, "role", "primary") or "primary"),
            "output_index": _spec_value(tensor_spec, "output_index", None),
            "device_policy": str(_spec_value(tensor_spec, "device_policy", "runtime") or "runtime"),
        }
    return schema


def _infer_batch_size(tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]) -> int | None:
    value = metadata.get("batch_size")
    if value is not None:
        return int(value)
    for tensor in tensors.values():
        if isinstance(tensor, torch.Tensor) and tensor.ndim > 0:
            return int(tensor.shape[0])
    return None


def from_torchlens_boundary(boundary: ReplayBoundary | BoundaryPayload) -> BoundaryPayload:
    if isinstance(boundary, BoundaryPayload):
        return boundary
    metadata = dict(getattr(boundary, "metadata", {}) or {})
    tensors = dict(getattr(boundary, "tensors", {}) or {})
    torchlens_spec = dict(getattr(boundary, "spec", {}) or {})
    labels = [
        str(label)
        for label in list(metadata.get("boundary_order") or torchlens_spec.keys() or tensors.keys())
    ]
    metadata.setdefault("_torchlens_spec", torchlens_spec)
    boundary_spec = BoundarySpec(
        boundary=str(metadata.get("split_id") or ""),
        boundary_tensor_labels=labels,
        feature_layout={
            "boundary_schema": _boundary_schema(torchlens_spec),
            "tensors": {
                str(label): {
                    "dtype": str(tensor.dtype),
                    "shape_without_batch": [int(dim) for dim in tensor.shape[1:]],
                }
                for label, tensor in tensors.items()
                if isinstance(tensor, torch.Tensor)
            },
        },
        passthrough_specs={},
    )
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
    native = payload.native
    if isinstance(native, ReplayBoundary):
        spec = dict(getattr(native, "spec", {}) or {})
    else:
        spec = dict(payload.metadata.get("_torchlens_spec") or {})
    if not spec:
        raise RuntimeError(
            "BoundaryPayload cannot be converted to TorchLens ReplayBoundary without native spec metadata."
        )
    metadata = dict(payload.metadata)
    # A remote suffix owns a different, independently updated model replica.
    # TorchLens' value-sensitive full-model fingerprint is therefore local-only;
    # SplitFleet validates graph/schema and model versions at the wire boundary.
    metadata.pop("state_fingerprint", None)
    if payload.batch_size is not None:
        metadata["batch_size"] = int(payload.batch_size)
    backend = str(metadata.get("backend") or getattr(native, "backend", "torch"))
    return ReplayBoundary(backend=backend, tensors=dict(payload.tensors), spec=spec, metadata=metadata)


def move_boundary_to_device(
    payload: ReplayBoundary | BoundaryPayload,
    device: torch.device | str | None,
) -> ReplayBoundary | BoundaryPayload:
    if device is None:
        return payload
    if hasattr(payload, "to"):
        return payload.to(device)
    return payload
