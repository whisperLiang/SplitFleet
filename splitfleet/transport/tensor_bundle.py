"""Pickle-free encoding for nested tensor/scalar targets and outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch

from splitfleet.transport.envelopes import (
    BoundaryEnvelope,
    EnvelopeLimits,
    DEFAULT_LIMITS,
    TensorEnvelope,
    decode_boundary,
    decode_tensor,
    encode_boundary,
    encode_tensor,
)


@dataclass(frozen=True)
class TensorBundle:
    tensors: tuple[TensorEnvelope, ...]
    structure: bytes


def encode_bundle(value: Any) -> TensorBundle:
    tensors: list[TensorEnvelope] = []

    def visit(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            index = len(tensors)
            tensors.append(encode_tensor(str(index), item))
            return {"tensor": index}
        if item is None or isinstance(item, (bool, int, float, str)):
            return {"scalar": item}
        if isinstance(item, dict):
            return {"dict": [[str(key), visit(child)] for key, child in item.items()]}
        if isinstance(item, tuple):
            return {"tuple": [visit(child) for child in item]}
        if isinstance(item, list):
            return {"list": [visit(child) for child in item]}
        raise TypeError(f"Unsupported wire bundle value {type(item).__name__}")

    structure = json.dumps(visit(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return TensorBundle(tuple(tensors), structure)


def decode_bundle(bundle: TensorBundle, device: str | torch.device = "cpu") -> Any:
    if len(bundle.structure) > 1024 * 1024:
        raise ValueError("Tensor bundle structure exceeds size limit")
    tree = json.loads(bundle.structure.decode("utf-8"))

    def visit(node: Any) -> Any:
        if "tensor" in node:
            index = int(node["tensor"])
            if index < 0 or index >= len(bundle.tensors):
                raise ValueError("Tensor bundle references missing tensor")
            return decode_tensor(bundle.tensors[index], device)
        if "scalar" in node:
            return node["scalar"]
        if "dict" in node:
            return {key: visit(value) for key, value in node["dict"]}
        if "tuple" in node:
            return tuple(visit(value) for value in node["tuple"])
        if "list" in node:
            return [visit(value) for value in node["list"]]
        raise ValueError("Invalid tensor bundle structure")

    return visit(tree)


def encode_bundle_wire(value: Any, *, compression: str = "none") -> bytes:
    bundle = encode_bundle(value)
    return encode_boundary(
        BoundaryEnvelope(
            tensors=bundle.tensors,
            engine="bundle",
            backend="torch",
            metadata={"structure": bundle.structure.decode("utf-8")},
        ),
        compression=compression,
    )


def decode_bundle_wire(
    payload: bytes,
    device: str | torch.device = "cpu",
    *,
    limits: EnvelopeLimits = DEFAULT_LIMITS,
) -> Any:
    envelope = decode_boundary(payload, limits=limits)
    if envelope.engine != "bundle":
        raise ValueError("Expected SplitFleet tensor bundle")
    structure = envelope.metadata.get("structure")
    if not isinstance(structure, str):
        raise ValueError("Tensor bundle is missing structure metadata")
    return decode_bundle(TensorBundle(envelope.tensors, structure.encode("utf-8")), device)
