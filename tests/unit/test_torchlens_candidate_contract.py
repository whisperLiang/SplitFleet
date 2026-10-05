from __future__ import annotations

import pytest
import torch
from torch import nn
from dataclasses import replace
from types import SimpleNamespace

from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend
from splitfleet.autosplit.torchlens_candidate import ParameterCountIndex
from splitfleet.autosplit.planner import AutoSplitPlanner
from splitfleet.autosplit.types import PlacementConstraint, PlacementObjective
from splitfleet.server.placement import TorchLensCandidateProvider
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


def test_parameter_index_preserves_shared_parameter_counts() -> None:
    class SharedWeights(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(4, 4, bias=False)
            self.second = nn.Linear(4, 4, bias=False)
            self.second.weight = self.first.weight

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.second(self.first(x))

    backend = TorchLensSplitBackend()
    backend.trace(SharedWeights().eval(), torch.randn(2, 4), boundary="50%")
    runtime = backend.runtime
    index = ParameterCountIndex.from_runtime(runtime)
    nodes = [node.canonical_id for node in runtime.trace_graph.nodes]
    def expected_count(selected, trainable_only=False):
        selected = set(selected)
        unique = {}
        for node in runtime.trace_graph.nodes:
            if node.canonical_id in selected:
                for parameter in node.param_refs:
                    if not trainable_only or parameter.is_trainable:
                        unique[id(parameter.handle)] = parameter.handle.numel()
        return sum(unique.values())

    assert index.total_count == expected_count(nodes)
    for split in range(len(nodes) + 1):
        for selected in (nodes[:split], nodes[split:]):
            for trainable_only in (False, True):
                assert index.count(selected, trainable_only=trainable_only) == expected_count(
                    selected, trainable_only=trainable_only)


def test_torchlens_candidate_enumeration_covers_operation_cuts() -> None:
    backend = _traced_backend()
    candidates = list(backend.iter_candidates(kinds=("after",)))

    assert candidates
    assert all(candidate.boundary_tensor_labels for candidate in candidates)
    assert all(candidate.boundary.startswith("after:") for candidate in candidates)
    assert [item.node_index for item in candidates] == sorted(item.node_index for item in candidates)
    assert all("input" not in candidate.split_label for candidate in candidates)
    assert all("output" not in candidate.split_label for candidate in candidates)
    final_candidates = [
        candidate
        for candidate in candidates
        if candidate.descriptor.get("suffix_node_count") == 0
    ]
    assert final_candidates
    assert all(not candidate.is_trainable_tail for candidate in final_candidates)

def test_selected_candidate_executes_runtime() -> None:
    backend = _traced_backend()
    candidate = list(backend.iter_candidates(kinds=("after",)))[0]
    backend.split(candidate)
    x = backend.trace_sample_input[0]
    torch.testing.assert_close(backend.run_suffix(backend.run_prefix(x)), backend.model(x))


def test_replay_only_native_capability_cannot_enter_training_catalog(monkeypatch) -> None:
    backend = _traced_backend()
    refused = next(candidate.boundary for candidate in backend.iter_candidates() if candidate.is_trainable_tail)
    runtime_type = type(backend.runtime)
    original_analyze = runtime_type.analyze

    def replay_only_report(runtime, point):
        analysis = original_analyze(runtime, point)
        boundary = f"{analysis.plan.boundary_kind}:{analysis.plan.target_node_id}"
        if boundary == refused:
            # An explicit replay-only report must remain replay-only even when
            # suffix parameters exist. No executable runtime is fabricated.
            analysis = replace(analysis, capability_report=SimpleNamespace(
                unsupported_reasons=(), training=SimpleNamespace(supported=False)))
        return analysis

    monkeypatch.setattr(runtime_type, "analyze", replay_only_report)
    candidate = next(value for value in backend.iter_candidates() if value.boundary == refused)
    assert candidate.descriptor["capabilities"]["replay_supported"]
    assert not candidate.descriptor["capabilities"]["training_supported"]
    assert not candidate.is_trainable_tail
    provider = TorchLensCandidateProvider(model=ToyNet().eval(), sample_inputs=torch.randn(2, 4))
    catalog = provider.get_candidates(training=True)
    assert refused not in {value.boundary for value in catalog}
    assert provider.catalog_diagnostics[True]["rejected_candidates"][refused] == "native_training_unsupported"


def test_auto_planner_uses_native_capability_without_numeric_replay(monkeypatch) -> None:
    monkeypatch.setattr(
        TorchLensSplitBackend, "run_prefix",
        lambda *args, **kwargs: pytest.fail("planning ran a numeric replay"),
    )
    placement = AutoSplitPlanner().plan(ToyNet().eval(), torch.randn(2, 4), boundary="auto")
    assert placement.validation["success"]
    assert placement.validation["verification"] == "native_capability_only"
    assert placement.validation["replay_performed"] is False


def test_metadata_only_catalog_cuts_do_not_build_executable_segments(monkeypatch) -> None:
    import torchlens.split.runtime as native_runtime
    backend = _traced_backend()
    original_runtime = backend.runtime
    with monkeypatch.context() as patch:
        patch.setattr(
            native_runtime, "execute_split_runtime",
            lambda *args, **kwargs: pytest.fail("catalog cut built executable segments"),
        )
        candidates = list(backend.iter_candidates())
    assert candidates
    inspected = backend.runtime.analyze(
        backend.runtime.split_points(diagnose=False).candidates[0].point,
    )
    assert inspected.source_graph is original_runtime.trace_graph
    assert not hasattr(inspected, "segments")
    assert backend.runtime is original_runtime
    selected = backend.split(candidates[0])
    assert selected.boundary == candidates[0].boundary
    x = backend.trace_sample_input[0]
    torch.testing.assert_close(backend.run_suffix(backend.run_prefix(x)), backend.model(x))


def test_catalog_plan_matches_materialized_candidate_and_restores_on_close() -> None:
    backend = _traced_backend()
    original_runtime = backend.runtime
    original_plan = backend.make_plan()
    iterator = backend.iter_candidates()
    candidate = next(iterator)
    catalog_plan = backend.make_plan()
    assert catalog_plan.boundary == candidate.boundary
    assert backend.runtime is original_runtime
    with pytest.raises(RuntimeError, match="Select a catalog candidate"):
        backend.make_handle()
    iterator.close()
    assert backend.make_plan().boundary == original_plan.boundary
    assert backend.make_plan().runtime_contract == original_plan.runtime_contract
    backend.split(candidate)
    materialized = backend.make_handle().plan
    assert materialized.boundary == catalog_plan.boundary
    assert materialized.prefix_node_count == catalog_plan.prefix_node_count
    assert materialized.suffix_node_count == catalog_plan.suffix_node_count
    assert materialized.runtime_contract == catalog_plan.runtime_contract


def test_cosplit_provider_caches_all_valid_before_and_after_operations() -> None:
    provider = TorchLensCandidateProvider(
        model=ToyNet().eval(),
        sample_inputs=torch.randn(2, 4),
        dynamic_batch=(2, 8),
    )
    first = provider.get_candidates(training=True)
    second = provider.get_candidates(training=True)

    assert first is second
    assert any(candidate.boundary.startswith("before:") for candidate in first)
    assert any(candidate.boundary.startswith("after:") for candidate in first)
    assert all(candidate.graph_signature for candidate in first)
    assert all(candidate.feature_abi_id for candidate in first)
    assert all(candidate.runtime_contract for candidate in first)
    assert all(candidate.metadata["boundary_tensor_labels"] for candidate in first)
    assert all(candidate.trainable for candidate in first)


def test_candidate_catalog_skips_numeric_replay_and_discloses_verification(monkeypatch) -> None:
    monkeypatch.setattr(
        TorchLensSplitBackend, "run_prefix",
        lambda *args, **kwargs: pytest.fail("catalog must not run numeric replay"),
    )
    provider = TorchLensCandidateProvider(
        model=ToyNet().eval(), sample_inputs=torch.randn(2, 4),
        require_trainable_prefix=True,
    )
    catalog = provider.get_candidates(training=True)
    diagnostics = provider.catalog_diagnostics[True]
    assert diagnostics["numeric_replay_validations"] == 0
    assert diagnostics["structurally_supported_training_candidates"] == len(catalog)

    for descriptor in catalog:
        validation = provider._validations[True][descriptor.boundary]
        assert validation["candidate_id"] == descriptor.metadata["candidate_id"]
        assert validation["split_id"] == descriptor.boundary
        assert validation["success"]
        assert validation["verification"] == "native_capability_only"
        assert validation["replay_performed"] is False
    placement = provider.get_placement_plan(
        catalog[0].boundary, worker_specs=[], constraints=PlacementConstraint(),
        objective=PlacementObjective(),
    )
    assert placement.validation["replay_performed"] is False


def test_evaluation_placement_uses_its_catalog_runtime() -> None:
    provider = TorchLensCandidateProvider(model=ToyNet().train(), sample_inputs=torch.randn(2, 4))
    catalog = provider.get_candidates(training=False)
    placement = provider.get_placement_plan(
        catalog[0].boundary, worker_specs=[], constraints=PlacementConstraint(),
        objective=PlacementObjective(), training=False,
    )
    handle = placement.metadata["_runtime_handle"]

    assert handle.model.training is False
    assert handle.plan.trainable is False
    assert handle.plan.plan_id == placement.plan_id
    assert handle is provider.get_placement_plan(
        catalog[0].boundary, worker_specs=[], constraints=PlacementConstraint(),
        objective=PlacementObjective(), training=False,
    ).metadata["_runtime_handle"]


def test_catalog_excludes_cuts_with_shared_state() -> None:
    from splitfleet.autosplit.state_ownership import state_ownership
    from splitfleet.backends.utils import adapter_for

    class TiedNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(4, 4, bias=False)
            self.second = nn.Linear(4, 4, bias=False)
            self.second.weight = self.first.weight
            self.last = nn.Linear(4, 2, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.last(self.second(torch.relu(self.first(x))))

    model = TiedNet()
    sample = torch.randn(2, 4)
    provider = TorchLensCandidateProvider(
        model=model, sample_inputs=sample, require_trainable_prefix=True,
    )
    catalog = provider.get_candidates(training=True)
    assert any(reason == "state_shared_across_stages" for reason in
               provider.catalog_diagnostics[True]["rejected_candidates"].values())
    diagnostics = provider.catalog_diagnostics[True]
    assert diagnostics["prechecked_training_exclusions"] > 0
    assert not any(
        "training_catalog_rejection:" in str(reason)
        for reason in diagnostics["unsupported_candidates"].values()
    )
    schema = adapter_for(model, sample).state_manifest(model).schema_hash
    for candidate in catalog:
        placement = provider.get_placement_plan(
            candidate.boundary, worker_specs=[], constraints=PlacementConstraint(),
            objective=PlacementObjective(),
        )
        state_ownership(placement.metadata["_runtime_handle"], schema)


def test_full_catalog_covers_native_training_cuts_and_reuses_capture(monkeypatch) -> None:
    provider = TorchLensCandidateProvider(
        model=ToyNet().eval(), sample_inputs=torch.randn(2, 4),
        batch_axes={"/args/0": 0}, dynamic_batch=(1, 2),
        require_trainable_prefix=True,
    )
    catalog = provider.get_candidates(training=True)
    backend = provider._backends[True]
    expected = {
        candidate.boundary
        for candidate in backend.iter_candidates()
        if candidate.is_trainable_tail
        and int(candidate.descriptor["trainable_prefix_parameter_count"]) > 0
    }
    assert {candidate.boundary for candidate in catalog} == expected
    assert len(catalog) > 3
    assert provider.catalog_diagnostics[True]["scope"] == "all_valid_operation_boundaries"
    assert not provider._prepared_handles
    graph = backend.runtime.trace_graph
    monkeypatch.setattr(TorchLensSplitBackend, "trace", lambda *args, **kwargs: pytest.fail("unexpected trace"))
    for candidate in catalog:
        placement = provider.get_placement_plan(
            candidate.boundary, worker_specs=[],
            constraints=PlacementConstraint(), objective=PlacementObjective(),
        )
        handle = placement.metadata["_runtime_handle"]
        assert handle.runtime.trace_graph is graph
        assert placement.boundary == candidate.boundary
        assert placement.feature_abi_id == candidate.feature_abi_id
        assert placement.validation["success"]
        assert handle is provider.get_placement_plan(
            candidate.boundary, worker_specs=[],
            constraints=PlacementConstraint(), objective=PlacementObjective(),
        ).metadata["_runtime_handle"]
    with pytest.raises(ValueError, match="not in the supported catalog"):
        provider.get_placement_plan(
            "after:missing", worker_specs=[],
            constraints=PlacementConstraint(), objective=PlacementObjective(),
        )
    with pytest.raises(RuntimeError, match="max_payload_bytes"):
        provider.get_placement_plan(
            catalog[0].boundary, worker_specs=[],
            constraints=PlacementConstraint(max_payload_bytes=0), objective=PlacementObjective(),
        )


def test_unused_model_parameters_do_not_make_an_empty_prefix_trainable() -> None:
    model = ToyNet().eval()
    model.unused = nn.Parameter(torch.ones(123))
    provider = TorchLensCandidateProvider(
        model=model, sample_inputs=torch.randn(2, 4), require_trainable_prefix=True,
    )
    catalog = provider.get_candidates()
    native = list(provider._backends[True].iter_candidates())
    empty_prefixes = {
        candidate.boundary for candidate in native
        if candidate.descriptor["trainable_prefix_parameter_count"] == 0
    }
    assert empty_prefixes
    assert empty_prefixes.isdisjoint(candidate.boundary for candidate in catalog)
    assert all(candidate.metadata["prefix_trainable_parameter_count"] > 0 for candidate in catalog)


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
        for candidate in handle.backend.iter_candidates()
        if candidate.estimated_payload_bytes <= 64
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
