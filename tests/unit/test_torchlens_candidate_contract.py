from __future__ import annotations

import torch
from torch import nn

from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend
from splitfleet.autosplit.torchlens_contract import (
    build_feature_abi_spec,
    build_runtime_contract,
    classify_contract_compatibility,
    feature_abi_id,
)


class ToyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tl_user_marker = "keep-me"
        self.fc1 = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _traced_backend() -> TorchLensSplitBackend:
    backend = TorchLensSplitBackend()
    backend.trace(
        ToyNet().eval(),
        torch.randn(2, 4),
        boundary="50%",
        dynamic_batch=(2, 8),
    )
    return backend


def test_torchlens_candidate_enumeration_is_sorted_and_filterable() -> None:
    backend = _traced_backend()
    candidates = backend.enumerate_candidates()

    assert candidates
    assert all(candidate.boundary_tensor_labels for candidate in candidates)
    assert all(candidate.boundary.startswith("after:") for candidate in candidates)
    assert candidates == sorted(
        candidates,
        key=lambda item: (
            item.estimated_payload_bytes,
            item.boundary_count,
            item.node_index,
            item.candidate_id,
        ),
    )
    assert all("input" not in candidate.split_label for candidate in candidates)
    assert all("output" not in candidate.split_label for candidate in candidates)
    final_candidates = [
        candidate
        for candidate in candidates
        if candidate.descriptor.get("suffix_node_count") == 0
    ]
    assert final_candidates
    assert all(not candidate.is_trainable_tail for candidate in final_candidates)

    first_payload = candidates[0].estimated_payload_bytes
    limited = backend.enumerate_candidates(max_payload_bytes=first_payload)
    assert limited
    assert all(candidate.estimated_payload_bytes <= first_payload for candidate in limited)
    assert backend.enumerate_candidates(max_boundary_count=0) == []


def test_torchlens_candidate_replay_validation_executes_runtime() -> None:
    backend = _traced_backend()
    candidate = backend.enumerate_candidates()[0]
    report = backend.validate_candidate(candidate)

    assert report["runtime"] == "torchlens_native"
    assert report["candidate_id"] == candidate.candidate_id
    assert report["success"] is True
    assert report["max_abs_diff"] <= 1e-5
    assert report["max_rel_diff"] <= 1e-4


def test_torchlens_trace_preserves_user_tl_prefixed_attributes() -> None:
    model = ToyNet().eval()
    backend = TorchLensSplitBackend()

    backend.trace(
        model,
        torch.randn(2, 4),
        boundary="50%",
        dynamic_batch=(2, 8),
    )

    assert model.tl_user_marker == "keep-me"


def test_feature_abi_is_batch_symbolic_and_runtime_identity_tolerant() -> None:
    schema_b1 = {
        "x": {
            "symbolic_shape": [1, 8],
            "dtype": "torch.float32",
            "requires_grad": True,
        }
    }
    schema_b2 = {
        "x": {
            "symbolic_shape": [2, 8],
            "dtype": "torch.float32",
            "requires_grad": True,
        }
    }
    layout = {"x": {"dtype": "torch.float32", "shape_without_batch": [8], "rank": 2}}
    spec_b1 = build_feature_abi_spec(
        model_family="toy",
        canonical_split_key="after:x",
        graph_signature="graph",
        boundary_tensor_labels=["x"],
        boundary_schema=schema_b1,
        feature_layout=layout,
    )
    spec_b2 = build_feature_abi_spec(
        model_family="toy",
        canonical_split_key="after:x",
        graph_signature="graph",
        boundary_tensor_labels=["x"],
        boundary_schema=schema_b2,
        feature_layout=layout,
    )
    assert feature_abi_id(spec_b1) == feature_abi_id(spec_b2)

    edge_contract = build_runtime_contract(
        model_family="toy",
        canonical_split_key="after:x",
        graph_signature="graph",
        boundary_tensor_labels=["x"],
        boundary_schema=schema_b1,
        feature_layout=layout,
        runtime_version="2.17.0",
        trace_batch_size=1,
    )
    cloud_contract = build_runtime_contract(
        model_family="toy",
        canonical_split_key="after:x",
        graph_signature="graph",
        boundary_tensor_labels=["x"],
        boundary_schema=schema_b2,
        feature_layout=layout,
        runtime_version="2.17.1",
        trace_batch_size=2,
    )
    compatibility = classify_contract_compatibility(edge_contract, cloud_contract)
    assert compatibility["compatible"] is True
    assert compatibility["reason"] == "runtime_identity_changed_but_feature_abi_compatible"


def test_feature_abi_rejects_label_order_dtype_and_shape_changes() -> None:
    base_schema = {
        "a": {"symbolic_shape": ["B", 4], "dtype": "torch.float32"},
        "b": {"symbolic_shape": ["B", 2], "dtype": "torch.float32"},
    }
    base_layout = {
        "a": {"dtype": "torch.float32", "shape_without_batch": [4], "rank": 2},
        "b": {"dtype": "torch.float32", "shape_without_batch": [2], "rank": 2},
    }
    base = build_feature_abi_spec(
        model_family="toy",
        canonical_split_key="after:split",
        graph_signature="graph",
        boundary_tensor_labels=["a", "b"],
        boundary_schema=base_schema,
        feature_layout=base_layout,
    )
    reordered = build_feature_abi_spec(
        model_family="toy",
        canonical_split_key="after:split",
        graph_signature="graph",
        boundary_tensor_labels=["b", "a"],
        boundary_schema=base_schema,
        feature_layout=base_layout,
    )
    dtype_changed = build_feature_abi_spec(
        model_family="toy",
        canonical_split_key="after:split",
        graph_signature="graph",
        boundary_tensor_labels=["a", "b"],
        boundary_schema={**base_schema, "a": {"symbolic_shape": ["B", 4], "dtype": "torch.float16"}},
        feature_layout={**base_layout, "a": {"dtype": "torch.float16", "shape_without_batch": [4], "rank": 2}},
    )
    shape_changed = build_feature_abi_spec(
        model_family="toy",
        canonical_split_key="after:split",
        graph_signature="graph",
        boundary_tensor_labels=["a", "b"],
        boundary_schema={**base_schema, "a": {"symbolic_shape": ["B", 5], "dtype": "torch.float32"}},
        feature_layout={**base_layout, "a": {"dtype": "torch.float32", "shape_without_batch": [5], "rank": 2}},
    )

    assert feature_abi_id(base) != feature_abi_id(reordered)
    assert feature_abi_id(base) != feature_abi_id(dtype_changed)
    assert feature_abi_id(base) != feature_abi_id(shape_changed)
