"""Portable contracts exchanged between independently prepared split runtimes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


GRAPH_CONTRACT_VERSION = "splitfleet-graph.v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def contract_hash(value: Any) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GraphContract:
    protocol_version: str = GRAPH_CONTRACT_VERSION
    torchlens_version: str = ""
    backend: str = ""
    model_profile_id: str = ""
    model_revision: str = ""
    model_state_schema_hash: str = ""
    input_schema_hash: str = ""
    split_request_hash: str = ""
    canonical_graph_hash: str = ""
    split_id: str = ""
    boundary_schema_hash: str = ""
    capability_hash: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> bytes:
        return canonical_json(self.to_dict()).encode("utf-8")

    @classmethod
    def from_json(cls, value: bytes | str) -> "GraphContract":
        payload = json.loads(value.decode("utf-8") if isinstance(value, bytes) else value)
        return cls(**payload)

    @property
    def digest(self) -> str:
        payload = self.to_dict()
        payload.pop("metadata", None)
        return contract_hash(payload)


@dataclass(frozen=True)
class ModelVersionContract:
    round_model_version: int
    prefix_state_version: int
    suffix_state_version: int
    state_schema_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> bytes:
        return canonical_json(self.to_dict()).encode("utf-8")

    @classmethod
    def from_json(cls, value: bytes | str) -> "ModelVersionContract":
        payload = json.loads(value.decode("utf-8") if isinstance(value, bytes) else value)
        return cls(**payload)


@dataclass(frozen=True)
class ContractMismatch:
    field: str
    expected: Any
    actual: Any


class ContractValidationError(ValueError):
    def __init__(self, mismatch: ContractMismatch):
        self.mismatch = mismatch
        super().__init__(
            f"Graph contract mismatch at {mismatch.field}: "
            f"expected {mismatch.expected!r}, got {mismatch.actual!r}"
        )


_IDENTITY_FIELDS = (
    "protocol_version",
    "torchlens_version",
    "backend",
    "model_profile_id",
    "model_revision",
    "model_state_schema_hash",
    "input_schema_hash",
    "split_request_hash",
    "canonical_graph_hash",
    "split_id",
    "boundary_schema_hash",
    "capability_hash",
)


def compare_contracts(expected: GraphContract, actual: GraphContract) -> ContractMismatch | None:
    """Return the first actionable difference; device metadata is excluded."""
    for name in _IDENTITY_FIELDS:
        expected_value = getattr(expected, name)
        actual_value = getattr(actual, name)
        if expected_value != actual_value:
            return ContractMismatch(name, expected_value, actual_value)
    return None


def validate_contract(expected: GraphContract, actual: GraphContract) -> None:
    mismatch = compare_contracts(expected, actual)
    if mismatch is not None:
        raise ContractValidationError(mismatch)
