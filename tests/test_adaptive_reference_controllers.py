from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.baselines.adaptive_reference import (
    AdaptiveReference, LinUCBE, graph_workload_features,
)
from splitfleet.server.placement.greedy import PlacementFeedback, TorchLensCandidateProvider
from splitfleet.tasks import ModelInputs
from experiments.physical_adaptive_baselines import reference_feedback, initial_admitted_boundary


@pytest.mark.parametrize('algorithm', ['L2S', 'Greedy'])
def test_evaluation_planning_does_not_initialize_or_advance_reference_training(algorithm):
    # Centralized evaluation produces an empty plan every round, but the
    # placement interface also supports actual client evaluation requests.
    from tests.test_greedy_placement import make_policy
    fixture = make_policy()
    controller = AdaptiveReference(candidate_provider=fixture.candidate_provider,
        config=fixture.config, algorithm=algorithm, horizon=10, output="unused.json")
    controller.telemetry_provider = fixture.telemetry_provider
    def unexpected_initialization():
        pytest.fail('Evaluation must not initialize the training controller')
    controller._initialize_reference = unexpected_initialization
    assert controller.plan_round(round_id=1, client_ids=[], training=False) == {}
    assert controller.plan_round(round_id=1, client_ids=['a'], training=False) == {'a': 'early'}
    assert controller.profiles is None and not controller.cost_tables



def test_l2s_escapes_zero_feedback_fully_local_action():
    learner = LinUCBE(2, horizon=16)
    features = np.array([[0., 0.], [1., 0.]])
    frontend = [0., 3.]
    assert learner.choose(features, frontend, round_id=1, fully_local_index=0) == 0
    assert learner.choose(features, frontend, round_id=2, fully_local_index=0) == 1
    learner.observe(features[1], 2.)
    assert learner.updates == 1
    assert learner.choose(features, frontend, round_id=3, fully_local_index=0) == 0


def test_l2s_latency_lcb_and_statistical_update_are_finite():
    learner = LinUCBE(2, alpha=.5)
    learner.observe([1., 0.], 8.)
    learner.observe([0., 1.], 1.)
    assert learner.choose(np.eye(2), [.1, .1], round_id=1) == 1
    np.testing.assert_allclose(learner.a, 2*np.eye(2))
    for invalid in [float('nan'), -1.]:
        with pytest.raises(ValueError):
            learner.observe([1., 0.], invalid)








def test_reference_feedback_uses_whole_fit_rpc_and_mean_batch_components():
    metrics = {'boundary':'after:op', 'coordinator_fit_rpc_ms': 900.,
        'client_fit_handler_ms': 600., 'fit_duration_ms': 400.,
        'client_forward_mean_ms': 25., 'client_backward_mean_ms': 35.,
        'network_roundtrip_mean_ms': 40., 'server_service_mean_ms': 15., 'num_batches': 3}
    observations = reference_feedback(2, [(SimpleNamespace(cid='a'),
        SimpleNamespace(metrics=metrics, num_examples=6))])
    assert observations[0].completion_ms == 900
    assert observations[0].completion_scope == "coordinator_fit_rpc"
    assert observations[0].state_exchange_ms == 300
    assert observations[0].client_forward_ms == 25
    assert observations[0].num_batches == 3
    with pytest.raises(ValueError):
        reference_feedback(2, [(SimpleNamespace(cid='a'), SimpleNamespace(metrics={}))])


def test_initial_bootstrap_boundary_belongs_to_admitted_training_catalog():
    candidates = [SimpleNamespace(boundary='after:early', graph_position_ratio=.2),
        SimpleNamespace(boundary='after:mid', graph_position_ratio=.49),
        SimpleNamespace(boundary='after:late', graph_position_ratio=.8)]
    policy = SimpleNamespace(candidate_provider=SimpleNamespace(get_candidates=lambda **kwargs: candidates))
    assert initial_admitted_boundary(policy, 'before:outside-catalog') == 'after:mid'
    assert initial_admitted_boundary(policy, 'after:early') == 'after:early'


def test_workload_features_cover_every_admitted_cut():
    torch.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(), torch.nn.Linear(4, 2))
    original = TorchLensCandidateProvider(model=model,
        sample_inputs=ModelInputs(args=(torch.randn(1, 3, 8, 8),)),
        dynamic_batch=(1, 2), batch_axes={'/args/0': 0}, require_trainable_prefix=True)
    full = original.get_candidates(training=True)
    assert full
    features = graph_workload_features(original)
    assert set(features) == {candidate.boundary for candidate in full}
    assert all(len(profile['l2s_structural']) == 8 and 0 <= profile['prefix_mac_fraction'] <= 1 for profile in features.values())
    assert all(profile['gradient_volume_proxy_bytes_per_sample'] >= 0 for profile in features.values())
