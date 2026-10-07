"""Online bootstrap, real parameter ownership, and aggregate evidence weight."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.common import ndarrays_to_parameters

from splitfleet.autosplit.torchlens_candidate import ParameterByteIndex
from splitfleet.common.model_state import tensor_state_hash
from splitfleet.server.placement.cosplit_ucb import CoSplitUCBConfig
from splitfleet.server.placement.cosplit_ucb.context import ContextEncoder
from splitfleet.server.placement.cosplit_ucb.residual import ResidualLinUCB
from tests.test_cosplit_calibration import _native_cohort, _execute_instruction
from splitfleet.server.placement.cosplit_ucb.calibration import provider_handle


def test_byte_index_counts_dtype_and_shared_identity_and_excludes_frozen_parameters():
    a = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    b = torch.nn.Parameter(torch.ones(5, dtype=torch.float16))
    frozen = torch.nn.Parameter(torch.ones(7), requires_grad=False)
    ref = lambda value: SimpleNamespace(handle=value, is_trainable=value.requires_grad)
    runtime = SimpleNamespace(trace_graph=SimpleNamespace(nodes=[
        SimpleNamespace(canonical_id="a", param_refs=[ref(a), ref(frozen)]),
        SimpleNamespace(canonical_id="b", param_refs=[ref(a), ref(b)])]))
    index = ParameterByteIndex.from_runtime(runtime)
    assert index.count(["a"]) == 24
    assert index.count(["a", "b"]) == 34


def test_online_native_catalog_features_and_transport_bootstrap_remain_state_preserving():
    cohort = _native_cohort(config=CoSplitUCBConfig())
    client = cohort.clients["a"]
    before, rng = tensor_state_hash(client.model.state_dict()), torch.get_rng_state().clone()
    assert client.get_properties({"cosplit_probe_payload": b"echo"}) == {"cosplit_probe_payload": b"echo"}
    assert tensor_state_hash(client.model.state_dict()) == before
    assert torch.equal(rng, torch.get_rng_state())
    instructions = cohort.strategy.configure_fit(1, ndarrays_to_parameters(cohort.initial), cohort.client_manager)
    encoder = ContextEncoder()
    total = sum(p.numel() * p.element_size() for p in cohort.provider.model.parameters() if p.requires_grad)
    fractions = []
    for candidate in cohort.provider.get_candidates(training=True):
        prefix = candidate.metadata["optimizer_prefix_parameter_bytes"]
        suffix = candidate.metadata["optimizer_suffix_parameter_bytes"]
        assert prefix > 0 and suffix > 0 and prefix + suffix == total
        context = encoder.edge_context(candidate)
        assert context[3] == pytest.approx(prefix / total) and context[-1] == 0
        fractions.append(context[3])
    assert len(set(fractions)) > 1
    receipt = cohort.policy.telemetry_provider.receipt
    assert receipt["network_initialized"] and len(receipt["transport_bootstrap"]["samples"]) == 4
    assert receipt["server_calibration"]["schema"] == "splitfleet.cosplit-calibration"
    requests = len(cohort.requests["a"])
    cohort.strategy.configure_fit(1, ndarrays_to_parameters(cohort.initial), cohort.client_manager)
    assert len(cohort.requests["a"]) == requests
    _execute_instruction(cohort, *instructions[0], training=True, round_id=1)
    state = copy.deepcopy(cohort.policy.state_dict())
    restored = _native_cohort(config=CoSplitUCBConfig())
    restored.policy.load_state_dict(state)
    assert restored.policy.learners.state_dict() == state["state"]["learners"]
    with pytest.raises(ValueError, match="feature"):
        _native_cohort().policy.load_state_dict({**state, "feature_schema": "incompatible-context"})


def test_online_encoder_marks_unknown_ownership_instead_of_fabricating_optimizer_work():
    from tests.unit.test_cosplit_context import _candidate
    candidate = replace(_candidate(), metadata={})
    assert ContextEncoder().edge_context(candidate)[3] == 0
    assert ContextEncoder().edge_context(candidate)[-1] == 1


def test_online_payload_features_match_real_native_envelopes_at_each_batch():
    cohort = _native_cohort(config=CoSplitUCBConfig())
    encoder = ContextEncoder()
    for candidate in cohort.provider.get_candidates(training=True):
        handle = provider_handle(cohort.provider, candidate.boundary)
        for batch in (1, 2, 4):
            inputs = torch.randn(batch, 4)
            payload = handle.backend.run_prefix(inputs, training=True)
            actual = sum(t.numel() * t.element_size() for t in payload.tensors.values())
            context = encoder.network_context(candidate, {"batch_size": batch}, direction="upload")
            assert context[1] * 1048576 == actual
            assert context[2] == context[-1] == 0
        capture = candidate.metadata["payload_batch_size"]
        assert candidate.boundary_forward_bytes == candidate.metadata["boundary_forward_bytes_by_batch_size"][capture]
    candidate = replace(candidate, metadata={})
    context = encoder.network_context(candidate, {"batch_size": 4}, direction="upload")
    assert context[1] == 0 and context[2] == context[-1] == 1


def test_online_payload_shape_program_preserves_flattened_batch_expressions():
    from tests.unit.test_torchlens_candidate_contract import FlattenBatchNet
    from splitfleet.server.placement.cosplit_ucb import TorchLensCandidateProvider
    provider = TorchLensCandidateProvider(model=FlattenBatchNet(), sample_inputs=torch.randn(2, 10, 4),
        batch_axes={"/args/0": 0}, dynamic_batch=(1, 4), require_trainable_prefix=True)
    candidate = next(c for c in provider.get_candidates(training=True) if c.boundary == "after:reshape_1_2:1")
    assert candidate.metadata["boundary_forward_bytes_by_batch_size"][4] == 4 * 10 * 8 * 4


def test_batch_weighted_feedback_ages_once_and_preserves_exact_discounted_mean():
    args = dict(dimension=2, discount_gamma=.5, target_scale=100, feature_schema="online-test")
    model = ResidualLinUCB(**args)
    x = np.array([1., .25])
    model.update(x, 100, round_id=1, sample_weight=1)
    model.update(x, 200, round_id=3, sample_weight=10)
    assert model.num_updates == 2
    assert model.predict(x).mean == pytest.approx((25 + 2000) / 10.25)
    expected = np.eye(2) + 10.25 * np.outer(x, x)
    np.testing.assert_allclose(model.A, expected)
    restored = ResidualLinUCB(**args)
    restored.load_state_dict(model.state_dict())
    assert restored.predict(x) == model.predict(x)
    for weight in (0, -1, float("nan"), float("inf")):
        before = model.state_dict()
        with pytest.raises(ValueError, match="sample_weight"):
            model.update(x, 300, round_id=4, sample_weight=weight)
        assert model.state_dict() == before
