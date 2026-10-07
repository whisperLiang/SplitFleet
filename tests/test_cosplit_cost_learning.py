"""Optimizer-inclusive calibration and clock-independent cost feedback."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from splitfleet.client.autosplit_split_client import _transport_roundtrip_ms
from splitfleet.common.constants import TRANSPORT_SERVER_ELAPSED_NS_METADATA_KEY
from splitfleet.common.model_state import tensor_state_hash
from splitfleet.server.placement.cosplit_ucb import CoSplitUCBConfig, GlobalPlacementSolver, TorchLensCandidateProvider
from splitfleet.server.placement.cosplit_ucb.bandit import DiscountedLinUCB
from splitfleet.server.placement.cosplit_ucb.calibration import calibrate_split, calibration_boundaries, provider_handle
from splitfleet.server.placement.cosplit_ucb.residual import ResidualLinUCB
from splitfleet.tasks import ModelInputs
from tests.test_cosplit_calibration import CalibrationNet
from tests.unit.test_cosplit_exploration import _estimate


@pytest.mark.parametrize("stage", ["prefix", "suffix"])
@pytest.mark.parametrize("fail", [False, True])
def test_adam_calibration_restores_state_and_rng_even_after_optimizer_failure(stage, fail):
    torch.set_num_threads(1)
    model = CalibrationNet()
    inputs = ModelInputs(args=(torch.randn(2, 4),))
    targets = torch.randn(2, 2)
    provider = TorchLensCandidateProvider(model=model, sample_inputs=inputs,
        batch_axes={"/args/0": 0}, dynamic_batch=(2, 2), require_trainable_prefix=True)
    cuts = calibration_boundaries(provider.get_candidates(training=True))
    model.drop.eval()
    before = tensor_state_hash(model.state_dict())
    for p in model.parameters():
        p.grad = torch.full_like(p, .125)
    gradients = [p.grad.clone() for p in model.parameters()]
    modes = {name: m.training for name, m in model.named_modules()}
    rng = torch.get_rng_state().clone()
    calls = []

    class MeasuredAdam(torch.optim.Adam):
        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            calls.append(len(self.state))
            if fail:
                raise RuntimeError("optimizer calibration failure")
            return result

    def run():
        return calibrate_split(model, inputs, targets, boundaries=cuts,
            make_handle=lambda cut: provider_handle(provider, cut),
            loss_fn=torch.nn.functional.mse_loss, device="cpu",
            source="server_shape_matched_current_deployment" if stage == "suffix" else "client_private_sample_current_deployment",
            optimizer_fn=lambda module: MeasuredAdam(module.parameters(), lr=.01))

    if fail:
        with pytest.raises(RuntimeError, match="optimizer calibration failure"):
            run()
    else:
        receipt = run()
        assert receipt["schema"] == "splitfleet.cosplit-calibration"
        assert receipt["persistent_optimizer_steps"] == 0
        assert receipt["optimizer_steps"] == len(cuts) * 2 == len(calls)
        assert all(r["optimizer_stage"] == stage and r["measured_batches"] == 1 for r in receipt["records"])
    assert calls and all(value > 0 for value in calls)
    assert tensor_state_hash(model.state_dict()) == before
    assert modes == {name: m.training for name, m in model.named_modules()}
    assert torch.equal(rng, torch.get_rng_state())
    for p, grad in zip(model.parameters(), gradients):
        torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)


def test_local_residual_preserves_measured_cost_and_shared_model_uncertainty():
    args = dict(dimension=2, ridge_lambda=1, discount_gamma=.5, alpha=1,
                target_scale=100, feature_schema="test")
    model, linear = ResidualLinUCB(**args), DiscountedLinUCB(**args)
    x = np.array([1., .25])
    for item in (model, linear):
        item.update(x, 100, round_id=1)
    assert model.predict(x).mean == 100
    assert model.predict(x).uncertainty >= linear.predict(x).uncertainty
    model.advance_round(3)
    model.update(x, 200, round_id=3)
    assert model.predict(x).mean == pytest.approx(180)
    restored = ResidualLinUCB(**args)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    assert restored.predict(x) == model.predict(x)
    model.advance_round(100000)
    assert np.isfinite(model.predict(x).mean)


@pytest.mark.parametrize("metadata,elapsed,expected", [
    ({TRANSPORT_SERVER_ELAPSED_NS_METADATA_KEY: "7000000", "splitfleet_server_receive_ns": "999999999999999"}, 12., 5.),
    ({}, 12., None),
    ({TRANSPORT_SERVER_ELAPSED_NS_METADATA_KEY: "13000000"}, 12., None),
    ({TRANSPORT_SERVER_ELAPSED_NS_METADATA_KEY: "-1"}, 12., None),
])
def test_roundtrip_feedback_uses_duration_not_cross_host_timestamp(metadata, elapsed, expected):
    assert _transport_roundtrip_ms(metadata, rpc_elapsed_ms=elapsed) == expected


def test_solver_counts_combined_transport_once_without_inventing_one_way_phases():
    a = replace(_estimate("a", "cut", 0, 0), server_service_mean_ms=5, network_roundtrip_mean_ms=3)
    b = replace(a, client_id="b")
    result = GlobalPlacementSolver().simulate({"a": a, "b": b}, batch_counts={"a": 2, "b": 2})
    assert a.mean_total_without_queue_ms == 8
    assert result.max_client_completion_ms == 23


def test_full_catalog_is_available_from_the_first_round_and_preserves_feasibility():
    from splitfleet.server.placement.cosplit_ucb import CoSplitUCBPlacementPolicy, StaticCandidateProvider
    from tests.unit.test_cosplit_policy import _candidate
    candidates = [_candidate(name, position) for name, position in [
        ("b-min", .1), ("c-quarter", .25), ("d-half", .5),
        ("e-three-quarter", .75), ("f-max", .9), ("a-interior", .33)]]
    config = CoSplitUCBConfig(min_residence_rounds=0)
    policy = CoSplitUCBPlacementPolicy(candidate_provider=StaticCandidateProvider(candidates), config=config)
    assert policy.plan_round(round_id=1, client_ids=["a"], training=True)["a"] == "a-interior"
    assert policy.plan_round(round_id=3, client_ids=["a"], training=True)["a"] == "a-interior"
    other = CoSplitUCBPlacementPolicy(candidate_provider=StaticCandidateProvider(candidates), config=config)
    for candidate in candidates[:-1]:
        other.observe_failure(round_id=0, client_id="a", boundary=candidate.boundary, kind="oom")
    assert other.plan_round(round_id=1, client_ids=["a"], training=True)["a"] == "a-interior"
