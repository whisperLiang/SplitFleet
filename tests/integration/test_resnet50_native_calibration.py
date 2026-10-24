"""Full native ResNet calibration, receipt binding and production placement.

Run in the backend's dedicated environment with
SPLITFLEET_NATIVE_CALIBRATION_BACKEND and SPLITFLEET_NATIVE_CALIBRATION_DEVICE.
These are numerical/initialization controls, not benchmark timings.
"""

import copy
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.integration.test_torchlens_optional_frameworks import _run_isolated


@pytest.mark.integration
def test_native_resnet_calibration_initializes_the_physical_policy_before_placement(request):
    backend = os.environ.get('SPLITFLEET_NATIVE_CALIBRATION_BACKEND')
    if backend is None:
        pytest.skip('Select a dedicated native backend environment')
    if _run_isolated(request):
        return
    device = os.environ.get('SPLITFLEET_NATIVE_CALIBRATION_DEVICE', 'cpu')
    from splitfleet.virtual.backend_ops import FrameworkOps
    ops = FrameworkOps(backend)
    ops.prepare_process(isolate_unused_frameworks=True)
    from experiments.resnet50_quality.model_contract import canonical_state
    from tests.integration.resnet50_reference.native import build_model, call_inputs, configure_backend, native_batch
    from tests.integration.resnet50_reference.workloads import StageCandidateProvider
    from splitfleet.autosplit import AutoSplitSession
    from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
    from splitfleet.common.model_state import model_state_hash
    from splitfleet.server.placement.greedy import GreedyPlacementPolicy
    from splitfleet.server.placement.greedy.calibration import calibration_boundaries
    from splitfleet.server.strategy import AutoSplitStrategy
    from splitfleet.tasks import ImageClassificationTask, ModelInputs, TaskBatch
    from splitfleet.virtual.client_adapter import OptimizerFactory
    from flwr.common import ndarrays_to_parameters

    configure_backend(backend)
    initial = canonical_state(233)
    model = build_model(backend, initial, device)
    client_model = build_model(backend, initial, device)
    ops.set_training(model, True)
    ops.set_training(client_model, True)
    images, labels = native_batch(backend, np.random.default_rng(4).normal(size=(16, 3, 64, 64)).astype(np.float32),
                                 np.arange(16, dtype=np.int64) % 10, device)
    batch = TaskBatch(ModelInputs(args=call_inputs(backend, model, images)), labels, num_examples=16)
    client_batch = TaskBatch(ModelInputs(args=call_inputs(backend, client_model, images)), labels, num_examples=16)
    provider = StageCandidateProvider(model=model, sample_inputs=batch.inputs, backend=backend,
        batch_axes={'/args/0': 0}, dynamic_batch=(1, 16))
    catalog = provider.get_candidates(training=True)
    before = model_state_hash(model, ops.adapter)
    assert model_state_hash(client_model, ops.adapter) == before
    policy = GreedyPlacementPolicy(candidate_provider=provider)
    # Model calibration is exercised in-process here; split-channel probing
    # is covered by the active-network and paced virtual integration tests.
    policy.uses_active_network_probes = False
    optimizer = OptimizerFactory('sgd', .05, ops.backend)
    client = AutoSplitSplitLearningClient(model=client_model, sample_inputs=client_batch.inputs,
        calibration_batch=client_batch, train_data=[client_batch], task=ImageClassificationTask(),
        batch_axes={'/args/0': 0}, device=device,
        optimizer_fn=optimizer, autosplit_session=AutoSplitSession(device=device))
    strategy = AutoSplitStrategy(model=model, sample_inputs=batch, task=ImageClassificationTask(),
        placement_policy=policy, batch_axes=provider.batch_axes, dynamic_batch=(1, 16),
        runtime_device=device, optimizer_fn=optimizer, owned_state_exchange=True,
        min_fit_clients=1, min_available_clients=1)
    requests = []

    def get_properties(ins, timeout, group_id):
        requests.append(dict(ins.config))
        if 'splitfleet_calibration_boundaries' in ins.config:
            result = client.get_properties(ins.config)
            receipt = client.online_calibration_receipt
            assert receipt['state_and_torch_rng_preserved'] and receipt['model_hash_before'] == receipt['model_hash_after']
            return SimpleNamespace(properties=result)
        return SimpleNamespace(properties=client.get_properties(ins.config))

    worker = SimpleNamespace(cid='native-client', get_properties=get_properties)
    manager = SimpleNamespace(num_available=lambda: 1, sample=lambda **kwargs: [worker])
    # Flower client sampling can consume host randomness. Check preservation
    # around the native calibration itself, independently of selection/probes.
    from splitfleet.server.placement.greedy import calibration
    original = calibration.calibrate_split

    def checked_calibration(*args, **kwargs):
        rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), copy.deepcopy(np.random.get_state())
        receipt = original(*args, **kwargs)
        assert torch.equal(torch.get_rng_state(), rng) and random.getstate() == python_rng
        actual_numpy = np.random.get_state()
        assert actual_numpy[0] == numpy_rng[0] and actual_numpy[2:] == numpy_rng[2:]
        np.testing.assert_array_equal(actual_numpy[1], numpy_rng[1])
        return receipt

    calibration.calibrate_split = checked_calibration
    try:
        fit = strategy.configure_fit(1, ndarrays_to_parameters(ops.export_arrays(model)), manager)
    finally:
        calibration.calibrate_split = original
    assert len(fit) == 1
    anchors = calibration_boundaries(catalog)
    assert policy.telemetry_provider.initialized
    assert len(policy.cost_tables["native-client"]["anchors"]) == len(anchors)
    assert policy.telemetry_provider.receipt['source'] == 'current_deployment_restored_training_cost_calibration'
    assert any('splitfleet_calibration_boundaries' in config for config in requests)
    assert any('splitfleet_probe_payload' in config for config in requests)
    assert model_state_hash(model, ops.adapter) == model_state_hash(client_model, ops.adapter) == before
    assert policy.round_diagnostics[1]['assignment']['native-client'] in {row.boundary for row in catalog}
    assert set(policy.round_diagnostics[1]['estimates']['native-client']) == {row.boundary for row in catalog}
    for receipt in (policy.telemetry_provider.server_receipt, client.online_calibration_receipt):
        assert receipt['persistent_optimizer_steps'] == 0
        stage = receipt['records'][0]['optimizer_stage']
        candidates = {row.boundary: row for row in catalog}
        expected_steps = sum(0 if candidates[cut].metadata[f'optimizer_{stage}_parameter_bytes'] == 0 else 2
                             for cut in anchors)
        assert receipt['optimizer_steps'] == expected_steps
        assert {row['boundary'] for row in receipt['records']} == set(anchors)
    if backend == 'tensorflow':
        # Reuse native programs while repeating every real measurement and
        # restoring a different logical client's input/state.
        assert client.online_calibration_receipt['native_runtime_cache_hits'] == 0
        first_steps = client.online_calibration_receipt['optimizer_steps']
        first_losses = {row['boundary']: row['loss'] for row in client.online_calibration_receipt['records']}
        first = dict(client._calibration_runtime_cache)
        images2, targets2 = native_batch(backend,
            np.random.default_rng(19).normal(size=(16, 3, 64, 64)).astype(np.float32),
            (np.arange(16, dtype=np.int64) + 2) % 10, device)
        client._calibration_inputs = ModelInputs(call_inputs(backend, client_model, images2))
        repeated = client.calibrate_runtime(targets2, boundaries=anchors)
        assert repeated['native_runtime_cache_hits'] == len(anchors)
        assert repeated['optimizer_steps'] == first_steps
        assert repeated['state_and_rng_preserved']
        assert repeated['model_hash_before'] == repeated['model_hash_after'] == before
        assert set(first) == set(client._calibration_runtime_cache)
        assert all(first[cut] is client._calibration_runtime_cache[cut] for cut in anchors)
        assert model_state_hash(client_model, ops.adapter) == before
        assert any(not np.isclose(row['loss'], first_losses[row['boundary']]) for row in repeated['records'])
        client.retire_execution()
        for handle in client._calibration_runtime_cache.values():
            assert handle.backend.trace_sample_input == ()
            assert handle.plan.metadata['_example_inputs'] == ()
        client.close_execution()
        assert not client._calibration_runtime_cache and client._calibration_runtime_source is None
