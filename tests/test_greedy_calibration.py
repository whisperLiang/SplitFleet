"""Measured calibration restores model and optimizer/RNG state."""
import copy
import json
import random
from threading import Barrier, Lock
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.common.identity import tensor_state_hash
from splitfleet.common.constants import *
from splitfleet.server.placement.greedy.calibration import CalibratedTelemetry, ClientTelemetry, calibrate_split, calibration_boundaries, provider_handle
from experiments.physical_multitask import _make_candidate_provider, _make_placement_policy
from splitfleet.tasks import ModelInputs
from splitfleet.server.placement import PlacementFeedback, StaticCandidateProvider
from splitfleet.server.placement.greedy import TorchLensCandidateProvider
from tests.unit.placement_fixtures import _candidate

def test_virtual_calibration_rpcs_use_the_admitted_pool_without_sharing_receipts():
    barrier, lock = Barrier(2), Lock()
    active, peak = 0, 0

    class Client:
        def __init__(self, cid):
            self.cid = str(cid)

        def get_properties(self, ins, timeout, group_id):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait(timeout=5)
            with lock:
                active -= 1
            return SimpleNamespace(properties={'logical_client_id': self.cid,
                'num_batches': 3, 'batch_size': 16, 'private_receipt': self.cid})

    telemetry = ClientTelemetry(policy=SimpleNamespace(client_concurrency=2))
    telemetry.bind_clients([Client(cid) for cid in range(4)], round_id=1)
    assert peak == 2
    assert {cid: props['private_receipt'] for cid, props in telemetry.clients.items()} == {
        str(cid): str(cid) for cid in range(4)}

def test_failed_calibration_metadata_does_not_partially_update_bound_clients():
    class Client:
        def __init__(self, cid, batches):
            self.cid, self.batches = cid, batches

        def get_properties(self, *args, **kwargs):
            return SimpleNamespace(properties={'logical_client_id': self.cid,
                'num_batches': self.batches, 'batch_size': 16})

    telemetry = ClientTelemetry(policy=SimpleNamespace(client_concurrency=2))
    with pytest.raises(ValueError, match='positive num_batches'):
        telemetry.bind_clients([Client('good', 1), Client('bad', 0)], round_id=1)
    assert telemetry.clients == {}

class CalibrationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.drop = nn.Dropout(.2)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(self.drop(self.bn(self.fc1(x))))

@pytest.mark.parametrize("fail", [False, True])
def test_actual_split_calibration_restores_weights_buffers_gradients_modes_and_rng(fail):
    torch.set_num_threads(1)
    model = CalibrationNet().train()
    inputs = ModelInputs(args=(torch.randn(2, 4),))
    targets = torch.randn(2, 2)
    provider = TorchLensCandidateProvider(model=model, sample_inputs=inputs,
        batch_axes={"/args/0": 0}, dynamic_batch=(2, 2), require_trainable_prefix=True)
    cuts = calibration_boundaries(provider.get_candidates(training=True))
    model.drop.eval()  # Mixed modes must survive even when calibration trains.
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .125)
    initial_hash = tensor_state_hash(model.state_dict())
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    modes = {name: m.training for name, m in model.named_modules()}
    cpu_rng = torch.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), copy.deepcopy(np.random.get_state())

    def loss(outputs, labels):
        random.random()
        np.random.random()
        if fail:
            raise RuntimeError("deliberate calibration failure")
        return nn.functional.mse_loss(outputs, labels)

    def execute():
        return calibrate_split(model, inputs, targets, boundaries=cuts,
            make_handle=lambda cut: provider_handle(provider, cut), loss_fn=loss,
            device="cpu", source="client_private_sample_current_deployment",
            optimizer_fn=lambda module: torch.optim.SGD(module.parameters(), lr=.01))

    if fail:
        with pytest.raises(RuntimeError, match="deliberate calibration failure"):
            execute()
    else:
        receipt = execute()
        assert receipt["optimizer_steps"] == len(cuts) * 2
        assert receipt["model_hash_before"] == receipt["model_hash_after"] == initial_hash
        assert {row["boundary"] for row in receipt["records"]} == set(cuts)
        assert all(row["client_forward_ms"] > 0 and row["client_backward_ms"] > 0
                   and row["local_tail_service_ms"] > 0 for row in receipt["records"])
    assert tensor_state_hash(model.state_dict()) == initial_hash
    assert {name: m.training for name, m in model.named_modules()} == modes
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, gradients[name], rtol=0, atol=0)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert random.getstate() == python_rng
    restored = np.random.get_state()
    assert restored[0] == numpy_rng[0] and restored[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(restored[1], numpy_rng[1])

@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_boundary_only_calibration_matches_full_backward_and_prefix_sgd(device):
    from splitfleet.autosplit.state_ownership import state_ownership

    torch.set_num_threads(1)
    model = CalibrationNet().to(device).train()
    inputs = ModelInputs(args=(torch.randn(2, 4, device=device),))
    labels = torch.randn(2, 2, device=device)
    provider = TorchLensCandidateProvider(model=model, sample_inputs=inputs,
        batch_axes={"/args/0": 0}, dynamic_batch=(2, 2), require_trainable_prefix=True)
    cuts = calibration_boundaries(provider.get_candidates(training=True))
    initial = copy.deepcopy(model.state_dict())
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith("cuda") else None
    for cut in cuts:
        handle = provider_handle(provider, cut)
        ownership = state_ownership(handle, "calibration")
        prefix = {name for name, stage in zip(ownership["names"], ownership["owners"])
                  if stage == "prefix"}
        parameters = [p for name, p in model.named_parameters() if name in prefix]
        reference = None
        for only_boundary in (False, True):
            model.load_state_dict(initial)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            optimizer = torch.optim.SGD(parameters, lr=.01, momentum=.9)
            steps = []
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                boundary = handle.backend.run_prefix(*inputs.args, training=True)
                loss, gradients = handle.backend.train_suffix(boundary, labels,
                    loss_fn=nn.functional.mse_loss, boundary_gradients_only=only_boundary)
                if only_boundary:
                    assert all(p.grad is None for p in model.parameters())
                handle.backend.backward_prefix(boundary, boundary_grads=gradients, optimizer=optimizer)
                steps.append((loss.detach().clone(), {name: value.clone() for name, value in gradients.items()},
                    {name: p.grad.detach().clone() for name, p in model.named_parameters() if name in prefix}))
            actual = (steps, copy.deepcopy(model.state_dict()))
            if reference is None:
                reference = actual
            else:
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)

def test_boundary_only_calibration_cannot_silently_skip_suffix_updates():
    model = CalibrationNet().train()
    inputs = ModelInputs(args=(torch.randn(2, 4),))
    provider = TorchLensCandidateProvider(model=model, sample_inputs=inputs,
        batch_axes={"/args/0": 0}, dynamic_batch=(2, 2), require_trainable_prefix=True)
    handle = provider_handle(provider, calibration_boundaries(provider.get_candidates(training=True))[0])
    boundary = handle.backend.run_prefix(*inputs.args, training=True)
    initial = tensor_state_hash(model.state_dict())
    with pytest.raises(ValueError, match="cannot update a suffix optimizer"):
        handle.backend.train_suffix(boundary, torch.randn(2, 2), loss_fn=nn.functional.mse_loss,
            optimizer=torch.optim.SGD(model.parameters(), lr=.01), boundary_gradients_only=True)
    assert tensor_state_hash(model.state_dict()) == initial

def _receipt(catalog, *, server=False):
    return dict(schema="splitfleet.training-calibration", device="cpu",
        source="server_shape_matched_current_deployment" if server else "client_private_sample_current_deployment",
        model_hash_before="initial", model_hash_after="initial", elapsed_sec=.5,
        state_and_torch_rng_preserved=True, optimizer_steps=len(catalog)*2,
        persistent_optimizer_steps=0, warmup_batches=1,
        records=[dict(boundary=c.boundary, graph_signature=c.graph_signature,
            feature_abi_id=c.feature_abi_id, client_forward_ms=100*c.graph_position_ratio,
            client_backward_ms=200*c.graph_position_ratio,
            local_tail_service_ms=50*(1-c.graph_position_ratio), measured_batches=1, optimizer_steps=2,
            optimizer_stage="suffix" if server else "prefix")
            for c in catalog])

def test_server_calibration_restores_the_model_owned_by_the_captured_provider():
    model = CalibrationNet().train()
    inputs = ModelInputs(args=(torch.randn(2, 4),))
    targets = torch.randn(2, 2)
    provider = _make_candidate_provider(model=model, sample_inputs=inputs,
        bundle={"task": "image_classification", "model_id": "resnet50_pretrained", "batch_size": 2})
    assert provider.model is not model
    cuts = calibration_boundaries(provider.get_candidates(training=True))
    captured = provider.model
    for parameter in captured.parameters():
        parameter.grad = torch.full_like(parameter, .125)
    captured.drop.eval()
    initial = tensor_state_hash(captured.state_dict())
    modes = {name: module.training for name, module in captured.named_modules()}
    gradients = {name: parameter.grad.clone() for name, parameter in captured.named_parameters()}
    receipt = calibrate_split(captured, inputs, targets, boundaries=cuts,
        make_handle=lambda cut: provider_handle(provider, cut), loss_fn=nn.functional.mse_loss,
        device="cpu", source="server_shape_matched_current_deployment",
        optimizer_fn=lambda module: torch.optim.SGD(module.parameters(), lr=.01))
    assert receipt["model_hash_before"] == receipt["model_hash_after"] == initial
    assert tensor_state_hash(model.state_dict()) == initial
    assert {name: module.training for name, module in captured.named_modules()} == modes
    for name, parameter in captured.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())

def _native_cohort(*, count=1, config=None, with_targets=True, require_trainable_prefix=True):
    from flwr.common import serde
    from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
    from splitfleet.server.placement.greedy import GreedyPlacementPolicy
    from splitfleet.server.stage_runtime.manager import StageRuntimeManager
    from splitfleet.server.strategy import AutoSplitStrategy
    from splitfleet.tasks import ImageClassificationTask, TaskBatch
    from tests.test_flower_autosplit_round_loop import DeepNet

    model, inputs, labels = DeepNet(), torch.randn(2, 4), torch.tensor([0, 1])
    sample = TaskBatch(ModelInputs(args=(inputs,)), labels) if with_targets else inputs
    task = ImageClassificationTask()
    provider = TorchLensCandidateProvider(model=copy.deepcopy(model), sample_inputs=inputs,
        batch_axes={"/args/0": 0}, dynamic_batch=(1, 4), require_trainable_prefix=require_trainable_prefix)
    policy = GreedyPlacementPolicy(candidate_provider=provider, config=config)
    # This fixture executes calibration directly in-process and has no RPC
    # channel. Active link measurement has separate real gRPC regressions.
    policy.uses_active_network_probes = False
    strategy = AutoSplitStrategy(model=model, sample_inputs=sample, task=task,
        batch_axes={"/args/0": 0}, dynamic_batch=(1, 4), placement_policy=policy,
        aggregation_policy="splitfed", fraction_fit=1/count, fraction_evaluate=1,
        min_fit_clients=1, min_evaluate_clients=1, min_available_clients=count,
        optimizer_fn=lambda module: torch.optim.SGD(module.parameters(), lr=.01))
    runtime_manager = StageRuntimeManager(autosplit_session=strategy.autosplit_session)
    strategy.bind_stage_runtime_manager(runtime_manager)
    clients, workers, requests = {}, [], {}
    for index in range(count):
        cid = chr(ord("a") + index)
        client = AutoSplitSplitLearningClient(model=DeepNet(), sample_inputs=sample, task=task,
            train_data=[(inputs, labels)], batch_axes={"/args/0": 0},
            optimizer_fn=lambda module: torch.optim.SGD(module.parameters(), lr=.01))
        clients[cid], requests[cid] = client, []

        def get_properties(ins, *, _client=client, _cid=cid, **kwargs):
            requests[_cid].append(dict(ins.config))
            # Exercise Flower's scalar encoding for the parameter handshake.
            decoded = serde.get_properties_ins_from_proto(serde.get_properties_ins_to_proto(ins))
            return SimpleNamespace(properties=_client.get_properties(decoded.config))

        workers.append(SimpleNamespace(cid=cid, get_properties=get_properties))
    manager = SimpleNamespace(num_available=lambda: count,
        sample=lambda num_clients, **kw: workers[:num_clients])
    return SimpleNamespace(strategy=strategy, policy=policy, provider=provider, clients=clients,
        workers=workers, requests=requests, client_manager=manager, runtime_manager=runtime_manager,
        initial=[value.detach().numpy().copy() for value in model.state_dict().values()],
        inputs=ModelInputs(args=(inputs,)), targets=labels, task=task)

def _execute_instruction(cohort, worker, instruction, *, training, round_id):
    from flwr.common import parameters_to_ndarrays
    from splitfleet.common import ServerModelFitIns, ServerModelEvaluateIns
    from splitfleet.server.server_model.autosplit_tail_server_model import AutoSplitTailServerModel
    from tests.test_flower_autosplit_round_loop import InProcessServerModelProxy

    parameters = parameters_to_ndarrays(instruction.parameters)
    client = cohort.clients[worker.cid]
    server = AutoSplitTailServerModel(runtime_manager=cohort.runtime_manager,
        model=copy.deepcopy(cohort.strategy.model), loss_fn=cohort.task.loss,
        optimizer_fn=lambda module: torch.optim.SGD(module.parameters(), lr=.01))
    client.server_model_proxy = InProcessServerModelProxy(server_model=server, cid=worker.cid)
    if training:
        server.configure_fit(ServerModelFitIns(parameters=parameters, config=instruction.config, sid=worker.cid))
        _, examples, metrics = client.fit(parameters, instruction.config)
        cohort.policy.observe_round(round_id=round_id, feedback=[PlacementFeedback(
            round_id=round_id, client_id=worker.cid, boundary=metrics["boundary"],
            client_forward_ms=metrics["client_forward_ms"], client_backward_ms=metrics["client_backward_ms"],
            server_service_ms=metrics["server_service_ms"], num_batches=metrics["num_batches"], num_examples=examples)])
    else:
        server.configure_evaluate(ServerModelEvaluateIns(parameters=parameters, config=instruction.config, sid=worker.cid))
        loss, examples, metrics = client.evaluate(parameters, instruction.config)
        assert np.isfinite(loss)
        with torch.no_grad():
            expected = cohort.task.loss(server.model(*cohort.inputs.args), cohort.targets)
        assert loss == pytest.approx(float(expected), rel=1e-5, abs=1e-6)
    assert examples == 2 and metrics["num_batches"] == 1

@pytest.mark.parametrize('matching', [True, False])
def test_hash_handshake_uses_local_state_or_loads_the_full_model_before_native_calibration(matching):
    from flwr.common import ndarrays_to_parameters, serde

    cohort = _native_cohort()
    client, worker = cohort.clients['a'], cohort.workers[0]
    worker.properties = {PLACEMENT_CALIBRATION_HASH_HANDSHAKE_KEY: True}
    if matching:
        client.model.load_state_dict(cohort.strategy.model.state_dict())
    original = tensor_state_hash(client.model.state_dict())
    global_hash = tensor_state_hash(cohort.strategy.model.state_dict())
    real_properties = worker.get_properties

    def properties(ins, **kwargs):
        cpu_rng = torch.get_rng_state().clone()
        result = real_properties(ins, **kwargs)
        if result.properties.get(PLACEMENT_CALIBRATION_PARAMETERS_REQUIRED_KEY):
            assert tensor_state_hash(client.model.state_dict()) == original
            assert client._prewarmed_runtime is None
            assert torch.equal(cpu_rng, torch.get_rng_state())
        return result

    worker.get_properties = properties
    parameters = ndarrays_to_parameters(cohort.initial)
    size = len(serde.parameters_to_proto(parameters).SerializeToString())
    instructions = cohort.strategy.configure_fit(1, parameters, cohort.client_manager)
    calibration = [request for request in cohort.requests['a'] if request.get('splitfleet_calibration_boundaries')]
    assert len(calibration) == (1 if matching else 2)
    assert calibration[0][PLACEMENT_CALIBRATION_MODEL_HASH_CONFIG_KEY] == global_hash
    assert PLACEMENT_CALIBRATION_PARAMETERS_CONFIG_KEY not in calibration[0]
    if not matching:
        assert len(calibration[1][PLACEMENT_CALIBRATION_PARAMETERS_CONFIG_KEY]) == size
    assert client.online_calibration_receipt['model_hash_before'] == global_hash
    assert client.online_calibration_receipt['model_hash_after'] == global_hash
    assert tensor_state_hash(client.model.state_dict()) == global_hash
    transfer = cohort.policy.telemetry_provider.receipt['parameter_handshake']
    assert transfer['requests'] == (1 if matching else 2)
    assert transfer['hash_matched_clients'] == int(matching)
    assert transfer['fallback_clients'] == int(not matching)
    assert transfer['parameter_payload_bytes'] == (0 if matching else size)
    assert transfer['avoided_parameter_payload_bytes'] == (size if matching else 0)
    _execute_instruction(cohort, *instructions[0], training=True, round_id=1)

def test_metadata_binding_reads_keyword_batch_axes_without_tracing():
    from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient

    class SequenceNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    client = AutoSplitSplitLearningClient(model=SequenceNet(), train_data=[object()],
        sample_inputs=ModelInputs(kwargs={"x": torch.randn(5, 2, 4)}), batch_axes={"/kwargs/x": 1})
    assert client.get_properties({})["batch_size"] == 2
    assert client._prewarmed_runtime is None
    assert client.online_calibration_receipt is None
