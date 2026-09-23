from __future__ import annotations

import pytest
import torch
from torch import nn

from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend
from splitfleet.autosplit.planner import AutoSplitPlanner
from splitfleet.autosplit.types import PlacementConstraint
from splitfleet.tasks import ModelInputs
from splitfleet.autosplit.torchlens_contract import (
    build_feature_abi_spec,
    build_runtime_contract,
    classify_contract_compatibility,
    feature_abi_id,
    stable_json,
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


class FlattenBatchNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 8)
        self.last = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.last(self.first(x).reshape(-1, 8))


@pytest.mark.parametrize("batch_axes", [None, {}])
def test_reshape_payload_matches_captured_boundary_and_enforces_limit(batch_axes) -> None:
    model = FlattenBatchNet().eval()
    inputs = torch.randn(2, 10, 4)
    boundary = "after:reshape_1_2:1"
    placement = AutoSplitPlanner().plan(
        model, inputs, boundary=boundary, batch_axes=batch_axes, dynamic_batch=(1, 8),
    )
    handle = placement.metadata["_runtime_handle"]
    capture_batch = handle.runtime.traced_batch_size
    payload = handle.backend.run_prefix(inputs[:capture_batch])
    actual_bytes = sum(tensor.numel() * tensor.element_size() for tensor in payload.tensors.values())

    assert placement.payload_bytes == actual_bytes == capture_batch * 10 * 8 * 4
    assert placement.candidate_descriptor["descriptor"]["payload_batch_size"] == capture_batch
    assert all(
        candidate.boundary != boundary
        for candidate in handle.backend.enumerate_candidates(max_payload_bytes=64)
    )
    with pytest.raises(RuntimeError, match="max_payload_bytes"):
        AutoSplitPlanner().plan(
            model, inputs, boundary=boundary, batch_axes=batch_axes,
            constraints=PlacementConstraint(max_payload_bytes=64),
        )


class ResidualKeywordNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.last = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, *, scale: torch.Tensor) -> torch.Tensor:
        return self.last((self.first(x).relu() + x) * scale)


@pytest.mark.parametrize("boundary", ["before:linear_1_1:1", "after:add_1_3:1"])
def test_planner_preserves_explicit_multitensor_cut_or_rejects_it(boundary: str) -> None:
    model = ResidualKeywordNet().eval()
    inputs = ModelInputs((torch.randn(2, 4),), {"scale": torch.randn(2, 1)})
    placement = AutoSplitPlanner().plan(model, inputs, boundary=boundary)

    assert placement.boundary == boundary
    # Before first: x and scale. After the residual add: its result and scale.
    assert len(placement.boundary_tensor_labels) == 2
    handle = placement.metadata["_runtime_handle"]
    kind, node_id = boundary.split(":", 1)
    assert handle.runtime.plan.target_node_id == node_id
    assert (node_id in handle.runtime.plan.prefix_node_ids) == (kind == "after")
    assert (node_id in handle.runtime.plan.suffix_node_ids) == (kind == "before")
    payload = handle.backend.run_prefix(*inputs.args, input_kwargs=dict(inputs.kwargs))
    torch.testing.assert_close(
        handle.backend.run_suffix(payload), model(*inputs.args, **inputs.kwargs),
    )

    with pytest.raises(RuntimeError, match="max_frontier_size"):
        AutoSplitPlanner().plan(
            model, inputs, boundary=boundary,
            constraints=PlacementConstraint(max_frontier_size=1),
        )


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
    cuda_layout = {"x": {"dtype": "torch.float32", "shape_without_batch": [8], "rank": 2, "device": "cuda:0"}}
    spec_b1 = build_feature_abi_spec(
        torchlens_version="2.34.1",
        model_family="toy",
        model_name="ToyNet",
        canonical_split_key="after:x",
        graph_signature="graph",
        boundary="after:x",
        boundary_tensor_labels=["x"],
        boundary_schema=schema_b1,
        feature_layout=layout,
        trace_batch_mode="batch_gt1",
        dynamic_batch=(1, 8),
    )
    spec_b2 = build_feature_abi_spec(
        torchlens_version="2.34.1",
        model_family="toy",
        model_name="ToyNet",
        canonical_split_key="after:x",
        graph_signature="graph",
        boundary="after:x",
        boundary_tensor_labels=["x"],
        boundary_schema=schema_b2,
        feature_layout=cuda_layout,
        trace_batch_mode="batch_gt1",
        dynamic_batch=(1, 8),
    )
    assert feature_abi_id(spec_b1) == feature_abi_id(spec_b2)
    assert "tensor(" not in stable_json(spec_b1)

    sample_a = build_feature_abi_spec(
        model_family="toy",
        canonical_split_key="after:x",
        graph_signature="graph",
        preprocessing_abi={"sample": torch.tensor([[1.0, 2.0]])},
    )
    sample_b = build_feature_abi_spec(
        model_family="toy",
        canonical_split_key="after:x",
        graph_signature="graph",
        preprocessing_abi={"sample": torch.tensor([[9.0, 8.0]])},
    )
    encoded_sample = stable_json(sample_a)
    assert feature_abi_id(sample_a) == feature_abi_id(sample_b)
    assert "tensor(" not in encoded_sample
    assert "1.0" not in encoded_sample
    assert "2.0" not in encoded_sample

    edge_contract = build_runtime_contract(
        model_family="toy",
        model_name="ToyNet",
        canonical_split_key="after:x",
        graph_signature="graph",
        boundary_tensor_labels=["x"],
        boundary_schema=schema_b1,
        feature_layout=layout,
        torchlens_version="2.34.1",
        runtime_version="2.34.1",
        trace_batch_size=1,
    )
    cloud_contract = build_runtime_contract(
        model_family="toy",
        model_name="ToyNet",
        canonical_split_key="after:x",
        graph_signature="graph",
        boundary_tensor_labels=["x"],
        boundary_schema=schema_b2,
        feature_layout=cuda_layout,
        torchlens_version="2.34.1",
        runtime_version="2.34.1",
        trace_batch_size=2,
    )
    compatibility = classify_contract_compatibility(edge_contract, cloud_contract)
    assert compatibility["compatible"] is True
    assert compatibility["reason"] == "runtime_identity_changed_but_feature_abi_compatible"
    assert edge_contract["torchlens_version"] == "2.34.1"


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


@pytest.mark.parametrize(
    "legacy_contract",
    [
        {"feature_layout_id": "matching-legacy-layout"},
        {"feature_abi_spec": {"boundary_tensor_labels": ["x"]}},
        {"feature_abi_spec": {}},
    ],
)
def test_contract_requires_explicit_feature_abi_id(legacy_contract) -> None:
    compatibility = classify_contract_compatibility(legacy_contract, legacy_contract)

    assert compatibility["compatible"] is False
    assert compatibility["reason"] == "feature_abi_id"


@pytest.mark.parametrize("payload", ["invalid-json", "[]", "null"])
def test_malformed_contract_is_rejected(payload: str) -> None:
    with pytest.raises(ValueError):
        classify_contract_compatibility(payload, {"feature_abi_id": "abi"})
