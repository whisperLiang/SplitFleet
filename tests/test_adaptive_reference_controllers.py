from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.baselines import adaptive_reference
from experiments.baselines.adaptive_reference import (
    AdaptiveReference, LayerScopedProvider, LinUCBE, graph_workload_features,
    normalized_fedadapt_reward, workload_action,
)
from splitfleet.server.placement.cosplit_ucb import PlacementFeedback, TorchLensCandidateProvider
from splitfleet.server.placement.cosplit_ucb.calibration import CalibratedTelemetry
from splitfleet.tasks import ModelInputs
from experiments.physical_adaptive_baselines import reference_feedback, scoped_initial_boundary
from tests.test_cosplit_calibration import _bootstrap_policy, _receipt


@pytest.mark.parametrize('algorithm', ['L2S-LinUCB-E', 'Profile-Greedy', 'FedAdapt-PPO-training'])
def test_evaluation_planning_does_not_initialize_or_advance_reference_training(algorithm):
    # Centralized evaluation produces an empty plan every round, but the
    # placement interface also supports actual client evaluation requests.
    controller = AdaptiveReference.__new__(AdaptiveReference)
    controller.algorithm = algorithm
    controller.state_download_bytes = 123
    calls = []
    def evaluation_plan(**kwargs):
        calls.append(kwargs)
        assert kwargs['training'] is False
        return {'a': 'after:evaluation-cut'}
    controller.clients = {'a': SimpleNamespace(plan_round=evaluation_plan)}
    def unexpected_initialization():
        pytest.fail('Evaluation must not initialize or sample the training controller')
    controller._initialize_reference = unexpected_initialization
    controller.memory = SimpleNamespace(rewards=[], actions=[], states=[])
    controller.pending_features = {}
    assert controller.plan_round(round_id=1, client_ids=[], training=False) == {}
    assert controller.plan_round(round_id=1, client_ids=['a'], training=False) == {'a': 'after:evaluation-cut'}
    assert calls == [{'round_id': 1, 'client_ids': ['a'], 'training': False}]
    assert controller.memory.rewards == controller.memory.actions == controller.memory.states == []
    assert controller.pending_features == {}


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


def test_fedadapt_reward_bounds_and_both_normalization_branches():
    assert normalized_fedadapt_reward([5., 20.], [10., 10.]) == 0
    assert normalized_fedadapt_reward([10., 10.], [10., 10.]) == 0
    assert 0 < normalized_fedadapt_reward([1., 2.], [10., 10.]) < 2
    with pytest.raises(ValueError):
        normalized_fedadapt_reward([0.], [10.])


def test_fedadapt_loaded_actor_can_train_and_save_its_original_reference(tmp_path, monkeypatch):
    base, worker, _ = _bootstrap_policy(tmp_path)
    provider = base.candidate_provider
    catalog = provider.get_candidates(training=True)
    profiles = {}
    for candidate in catalog:
        candidate.metadata.update(boundary_forward_bytes_by_batch_size={1: 1000},
                                  optimizer_prefix_parameter_bytes=1000)
        profiles[candidate.boundary] = {'l2s_structural': [1.] * 8,
            'prefix_mac_fraction': candidate.graph_position_ratio,
            'gradient_volume_proxy_bytes_per_sample': 1000}
    monkeypatch.setattr(adaptive_reference, 'graph_workload_features', lambda _: profiles)

    memory = SimpleNamespace(rewards=[], is_terminals=[])
    def clear_memory():
        memory.rewards.clear()
        memory.is_terminals.clear()
    memory.clear_memory = clear_memory
    updates = []
    agent = SimpleNamespace(policy=torch.nn.Linear(2, 1, bias=False),
        policy_old=torch.nn.Linear(2, 1, bias=False),
        select_action=lambda state, memory: (np.array([.25]), None, None))
    def update(memory):
        updates.append((list(memory.rewards), list(memory.is_terminals)))
        with torch.no_grad():
            agent.policy.weight.add_(.1)
    agent.update = update
    monkeypatch.setattr(adaptive_reference, 'load_upstream_ppo', lambda *args, **kwargs: (agent, memory))

    original_state = {'weight': torch.tensor([[.1, .2]])}
    checkpoint = tmp_path / 'actor.pt'
    torch.save({'schema': 'splitfleet.fedadapt-controller.v1',
        'policy_state': original_state, 'logical_client_order': ['a'],
        'reference_batch_seconds': [.6], 'reference_workloads': [.75]}, checkpoint)
    controller = AdaptiveReference(candidate_provider=provider, config=base.config,
        algorithm='FedAdapt-PPO-training', horizon=1, output=tmp_path / 'result.json',
        actor_checkpoint=checkpoint)
    controller.telemetry_provider = CalibratedTelemetry(initial_model_hash='initial',
        server_receipt=_receipt(catalog, server=True), policy=controller.bootstrap)
    controller.bind_clients([worker], 1)
    assignment = controller.plan_round(round_id=1, client_ids=['a'], training=True)
    assert assignment == {'a': 'mid'}
    torch.testing.assert_close(agent.policy.weight, original_state['weight'])
    torch.testing.assert_close(agent.policy_old.weight, original_state['weight'])
    controller.observe_round(round_id=1, feedback=[PlacementFeedback(
        round_id=1, client_id='a', boundary=assignment['a'], num_batches=3,
        client_forward_ms=25, client_backward_ms=35, server_service_ms=15,
        completion_ms=900)])

    saved = torch.load(tmp_path / 'result.ppo.pt', map_location='cpu', weights_only=False)
    assert updates == [([.5], [True])]
    assert saved['reference_batch_seconds'] == [.6]
    assert saved['reference_workloads'] == [.75]
    assert saved['logical_client_order'] == ['a']
    assert saved['physical_transitions'] == 1
    torch.testing.assert_close(saved['policy_state']['weight'], original_state['weight'] + .1)
    np.testing.assert_array_equal(controller.previous_workloads, [.25])
    np.testing.assert_array_equal(controller.reference_workloads, [.75])
    assert memory.rewards == memory.is_terminals == []


def test_fedadapt_maps_flops_instead_of_graph_depth():
    candidates = [SimpleNamespace(boundary='early', graph_position_ratio=.1),
                  SimpleNamespace(boundary='late', graph_position_ratio=.8)]
    profiles = {'early': {'prefix_mac_fraction': .6}, 'late': {'prefix_mac_fraction': .9}}
    assert workload_action(candidates, profiles, .55).boundary == 'early'
    assert workload_action(candidates, profiles, 2.).boundary == 'late'
    with pytest.raises(ValueError):
        workload_action(candidates, profiles, float('nan'))


def test_reference_feedback_uses_whole_fit_rpc_and_mean_batch_components():
    metrics = {'boundary':'after:op', 'coordinator_fit_rpc_ms': 900.,
        'client_fit_handler_ms': 600., 'fit_duration_ms': 400.,
        'client_forward_mean_ms': 25., 'client_backward_mean_ms': 35.,
        'network_roundtrip_mean_ms': 40., 'server_service_mean_ms': 15., 'num_batches': 3}
    observations = reference_feedback(2, [(SimpleNamespace(cid='a'),
        SimpleNamespace(metrics=metrics, num_examples=6))])
    assert observations[0].completion_ms == 900
    assert observations[0].state_exchange_ms == 300
    assert observations[0].client_forward_ms == 25
    assert observations[0].num_batches == 3
    with pytest.raises(ValueError):
        reference_feedback(2, [(SimpleNamespace(cid='a'), SimpleNamespace(metrics={}))])


def test_initial_bootstrap_boundary_belongs_to_restricted_training_catalog():
    candidates = [SimpleNamespace(boundary='after:layer1', graph_position_ratio=.2),
        SimpleNamespace(boundary='after:layer2', graph_position_ratio=.49),
        SimpleNamespace(boundary='after:layer3', graph_position_ratio=.8)]
    policy = SimpleNamespace(candidate_provider=SimpleNamespace(get_candidates=lambda **kwargs: candidates))
    assert scoped_initial_boundary(policy, 'before:outside-layer-catalog') == 'after:layer2'
    assert scoped_initial_boundary(policy, 'after:layer1') == 'after:layer1'


def test_layer_catalog_and_workload_features_preserve_real_capture_contract():
    torch.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(), torch.nn.Linear(4, 2))
    original = TorchLensCandidateProvider(model=model,
        sample_inputs=ModelInputs(args=(torch.randn(1, 3, 8, 8),)),
        dynamic_batch=(1, 2), batch_axes={'/args/0': 0}, require_trainable_prefix=True)
    full = original.get_candidates(training=True)
    scoped = LayerScopedProvider(original)
    layers = scoped.get_candidates(training=True)
    assert 0 < len(layers) <= len(full)
    assert {value.boundary for value in layers} <= {value.boundary for value in full}
    assert scoped._backends[True] is original._backends[True]
    assert len(original.get_candidates(training=True)) == len(full)
    features = graph_workload_features(scoped)
    assert set(features) == {candidate.boundary for candidate in layers}
    assert all(len(profile['l2s_structural']) == 8 and 0 <= profile['prefix_mac_fraction'] <= 1 for profile in features.values())
    assert all(profile['gradient_volume_proxy_bytes_per_sample'] >= 0 for profile in features.values())
