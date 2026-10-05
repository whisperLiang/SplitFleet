"""Bootstrap execution preserves training state and leaves online feedback active."""

import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.common.identity import tensor_state_hash
from splitfleet.server.placement.cosplit_ucb.calibration import (
    CalibratedTelemetry, calibrate_split, calibration_boundaries, provider_handle,
)
from experiments.physical_multitask import _make_candidate_provider, _make_placement_policy
from splitfleet.tasks import ModelInputs
from splitfleet.server.placement import PlacementFeedback, StaticCandidateProvider
from splitfleet.server.placement.cosplit_ucb import GlobalPlacementSolver, SafeExplorationController, TorchLensCandidateProvider
from tests.unit.test_cosplit_exploration import _estimate
from tests.unit.test_cosplit_policy import _candidate


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
            device="cpu", source="client_private_sample_current_deployment")

    if fail:
        with pytest.raises(RuntimeError, match="deliberate calibration failure"):
            execute()
    else:
        receipt = execute()
        assert receipt["optimizer_steps"] == 0
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


def _receipt(catalog, *, server=False):
    return dict(schema="splitfleet.online-calibration.v1", device="cpu",
        source="server_shape_matched_current_deployment" if server else "client_private_sample_current_deployment",
        model_hash_before="initial", model_hash_after="initial", elapsed_sec=.5,
        state_and_torch_rng_preserved=True, optimizer_steps=0,
        records=[dict(boundary=c.boundary, graph_signature=c.graph_signature,
            feature_abi_id=c.feature_abi_id, client_forward_ms=100*c.graph_position_ratio,
            client_backward_ms=200*c.graph_position_ratio,
            local_tail_service_ms=50*(1-c.graph_position_ratio), measured_batches=1, optimizer_steps=0)
            for c in catalog])


def _bootstrap_policy(tmp_path, model_id="rfdetr_nano"):
    catalog = tuple(_candidate(name, position) for name, position in (("early", .1), ("mid", .25), ("late", .75)))
    policy = _make_placement_policy(provider=StaticCandidateProvider(catalog),
        bundle=dict(seed=7401, model_id=model_id, initial_model_hash="initial"),
        args=SimpleNamespace(min_residence_rounds=2, output=tmp_path / "result.json"))
    telemetry = CalibratedTelemetry(initial_model_hash="initial", server_receipt=_receipt(catalog, server=True))
    telemetry.policy = policy
    policy.telemetry_provider = telemetry
    props = dict(logical_client_id="a", num_batches=3, batch_size=1,
        framework_backend="torch", runtime_backend="torchlens_native", device_type="cpu",
        accelerator="aarch64", precision="fp32", online_calibration_receipt=json.dumps(_receipt(catalog)))
    worker = SimpleNamespace(cid="a", get_properties=lambda *a, **k: SimpleNamespace(properties=props))
    return policy, worker, props


@pytest.mark.parametrize("model_id", ["rfdetr_nano", "resnet50_pretrained", "bert_base", "deeplabv3_resnet50"])
def test_current_calibration_seeds_uncertain_learners_then_online_feedback_updates_them(tmp_path, model_id):
    policy, worker, _ = _bootstrap_policy(tmp_path, model_id)
    policy.bind_clients([worker], 1)
    assert "device_cost_prior" not in vars(policy)
    assert policy.learners.server.model.num_updates == 3
    assert policy.learners.network.state_dict() == {}
    assert policy._last_boundary == policy._last_switch_round == {}
    assert policy.exploration_controller.last_probe_round == {}
    assert policy.exploration_controller.candidate_last_explored_round == {}
    initial_b = policy.learners.server.model.b.copy()
    policy.bind_clients([worker], 1)  # No repeated calibration samples.
    assert policy.learners.server.model.num_updates == 3
    assignment = policy.plan_round(round_id=1, client_ids=["a"], training=True)
    estimates = policy.round_diagnostics[1]["estimates"]["a"].values()
    assert all(row["mean_total_without_queue_ms"] > 0 for row in estimates)
    assert all(row["uncertainty_total_ms"] > 0 for row in estimates)
    policy.observe_round(round_id=1, feedback=[PlacementFeedback(
        round_id=1, client_id="a", boundary=assignment["a"], client_forward_ms=40,
        client_backward_ms=80, server_service_ms=10, network_upload_ms=5,
        network_download_ms=5, num_batches=3, num_examples=3)])
    assert policy.learners.server.model.num_updates == 4
    assert not np.array_equal(initial_b, policy.learners.server.model.b)
    assert policy.learners.server.model.discount_gamma == .98
    assert policy.telemetry_provider.receipt["prediction_override"] is False


@pytest.mark.parametrize("change, match", [
    (lambda r: r.update(model_hash_after="wrong"), "frozen initial model"),
    (lambda r: r.update(device="cuda:0"), "device differs"),
    (lambda r: r["records"][-1].update(feature_abi_id="wrong"), "ABI mismatch"),
    (lambda r: r["records"][-1].update(client_forward_ms=float("nan")), "finite"),
    (lambda r: r.update(source="historical_profile"), "current deployment"),
])
def test_bad_receipt_is_rejected_before_any_cost_updates(tmp_path, change, match):
    policy, worker, props = _bootstrap_policy(tmp_path)
    receipt = json.loads(props["online_calibration_receipt"])
    change(receipt)
    props["online_calibration_receipt"] = json.dumps(receipt)
    with pytest.raises(ValueError, match=match):
        policy.bind_clients([worker], 1)
    assert policy.learners.server.model.num_updates == 0
    assert policy.learners.edge.state_dict() == {}


def test_calibration_refuses_a_preexisting_learner(tmp_path):
    policy, worker, _ = _bootstrap_policy(tmp_path)
    policy.learners.server.model.update(np.ones(policy.context_encoder.server_dimension), 10, round_id=0)
    with pytest.raises(ValueError, match="fresh, unplanned"):
        policy.bind_clients([worker], 1)


def test_mean_budget_blocks_a_probe_admitted_by_a_loose_ucb_even_when_forced():
    base, expensive, cheap = (_estimate("a", name, mean, uncertainty) for name, mean, uncertainty
                              in (("base", 100, 1000), ("expensive", 300, 800), ("cheap", 104, 500)))
    arguments = dict(round_id=20, baseline={"a": base}, estimates={"a": [base, expensive]},
                     solver=GlobalPlacementSolver(), batch_counts={"a": 3})
    new, decisions = SafeExplorationController().apply(**arguments)
    assert new["a"].boundary == "base" and decisions == []
    arguments["estimates"]["a"].append(cheap)
    new, decisions = SafeExplorationController().apply(**arguments)
    assert new["a"].boundary == "cheap" and decisions[0].reason == "forced_safe_probe"
    new, decisions = SafeExplorationController().apply(**arguments, residence_locked={"a"})
    assert new["a"].boundary == "base" and decisions == []


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
        device="cpu", source="server_shape_matched_current_deployment")
    assert receipt["model_hash_before"] == receipt["model_hash_after"] == initial
    assert tensor_state_hash(model.state_dict()) == initial
    assert {name: module.training for name, module in captured.named_modules()} == modes
    for name, parameter in captured.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("scenario", ["matching", "independent", "heterogeneous_cold", "heterogeneous_prewarm"])
def test_native_strategy_calibrates_before_placement_and_trains_without_experiment_flags(device, scenario):
    from flwr.common import ndarrays_to_parameters
    from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
    from splitfleet.common import ServerModelFitIns
    from splitfleet.server.placement.cosplit_ucb import CoSplitUCBPlacementPolicy
    from splitfleet.server.server_model.autosplit_tail_server_model import AutoSplitTailServerModel
    from splitfleet.server.stage_runtime.manager import StageRuntimeManager
    from splitfleet.server.strategy import AutoSplitStrategy
    from splitfleet.tasks import ImageClassificationTask, TaskBatch
    from tests.test_flower_autosplit_round_loop import DeepNet, InProcessServerModelProxy

    model = DeepNet().to(device)
    inputs, labels = torch.randn(2, 4, device=device), torch.tensor([0, 1], device=device)
    sample = TaskBatch(ModelInputs(args=(inputs,)), labels)
    task = ImageClassificationTask()
    provider = TorchLensCandidateProvider(model=DeepNet().to(device) if scenario == "independent" else copy.deepcopy(model), sample_inputs=sample.inputs,
        batch_axes={"/args/0": 0}, dynamic_batch=(1, 4), require_trainable_prefix=True)
    policy = CoSplitUCBPlacementPolicy(candidate_provider=provider)
    strategy = AutoSplitStrategy(model=model, sample_inputs=sample, task=task,
        batch_axes={"/args/0": 0}, dynamic_batch=(1, 4), placement_policy=policy,
        aggregation_policy="splitfed", runtime_device=device,
        optimizer_fn=lambda module: torch.optim.SGD(module.parameters(), lr=.01),
        min_fit_clients=1, min_available_clients=1)
    manager = StageRuntimeManager(autosplit_session=strategy.autosplit_session)
    strategy.bind_stage_runtime_manager(manager)
    client_batch = 1 if scenario.startswith("heterogeneous") else 2
    client_sample = TaskBatch(ModelInputs(args=(inputs[:client_batch],)), labels[:client_batch])
    client = AutoSplitSplitLearningClient(model=DeepNet().to(device) if scenario == "independent" else model,
        sample_inputs=client_sample, task=task,
        train_data=[(inputs[:client_batch], labels[:client_batch])], batch_axes={"/args/0": 0}, device=device,
        optimizer_fn=lambda module: torch.optim.SGD(module.parameters(), lr=.01))
    global_model = DeepNet().to(device) if scenario == "independent" else model
    original = tensor_state_hash(global_model.state_dict())
    if scenario == "independent":
        assert original != tensor_state_hash(client.model.state_dict())
        assert original != tensor_state_hash(provider.model.state_dict())
    if scenario == "heterogeneous_prewarm":
        client.prewarm_runtime(dynamic_batch=(1, 4))
        client.calibrate_runtime(client_sample.targets,
            boundaries=calibration_boundaries(provider.get_candidates(training=True)))
        assert client._prewarmed_runtime.plan.trace_batch_mode == "batch_1"
    worker = SimpleNamespace(cid="a", get_properties=lambda ins, **kw: SimpleNamespace(
        properties=client.get_properties(ins.config)))
    client_manager = SimpleNamespace(num_available=lambda: 1, sample=lambda **kw: [worker])
    initial = [value.detach().cpu().numpy().copy() for value in global_model.state_dict().values()]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith("cuda") else None
    instructions = strategy.configure_fit(1, ndarrays_to_parameters(initial), client_manager)
    config = instructions[0][1].config

    assert tensor_state_hash(client.model.state_dict()) == original
    assert tensor_state_hash(provider.model.state_dict()) == original
    assert client._prewarmed_runtime.plan.dynamic_batch == (1, 4)
    assert client._prewarmed_runtime.plan.trace_batch_mode == provider.trace_batch_mode == "batch_gt1"
    assert client.online_calibration_receipt["optimizer_steps"] == 0
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    anchors = len(calibration_boundaries(provider.get_candidates(training=True)))
    assert policy.learners.server.model.num_updates == anchors
    server = AutoSplitTailServerModel(runtime_manager=manager, model=copy.deepcopy(model), device=device,
        loss_fn=task.loss, optimizer_fn=lambda module: torch.optim.SGD(module.parameters(), lr=.01))
    server.configure_fit(ServerModelFitIns(parameters=initial, config=config, sid="a"))
    client.server_model_proxy = InProcessServerModelProxy(server_model=server, cid="a")
    _, examples, metrics = client.fit(initial, config)
    assert examples == client_batch and metrics["num_batches"] == 1
    assert tensor_state_hash(client.model.state_dict()) != original
    boundary = policy.round_diagnostics[1]["assignment"]["a"]
    policy.observe_round(round_id=1, feedback=[PlacementFeedback(
        round_id=1, client_id="a", boundary=boundary,
        client_forward_ms=metrics["client_forward_ms"], client_backward_ms=metrics["client_backward_ms"],
        server_service_ms=metrics["server_service_ms"], num_batches=1, num_examples=client_batch)])
    assert policy.learners.server.model.num_updates == anchors + 1
    policy.bind_clients([worker], 2)
    assert policy.learners.server.model.num_updates == anchors + 1


def test_native_cosplit_does_not_silently_start_an_empty_learner_without_sample_targets():
    from flwr.common import ndarrays_to_parameters
    from splitfleet.server.placement.cosplit_ucb import CoSplitUCBPlacementPolicy
    from splitfleet.server.strategy import AutoSplitStrategy
    from splitfleet.tasks import ImageClassificationTask
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    sample = torch.randn(2, 4)
    policy = CoSplitUCBPlacementPolicy(candidate_provider=TorchLensCandidateProvider(
        model=model, sample_inputs=sample))
    strategy = AutoSplitStrategy(model=model, sample_inputs=sample, task=ImageClassificationTask(),
        placement_policy=policy, min_fit_clients=1, min_available_clients=1)
    manager = SimpleNamespace(num_available=lambda: 1, sample=lambda **kwargs: [SimpleNamespace(cid="a")])
    parameters = ndarrays_to_parameters([value.detach().numpy() for value in model.state_dict().values()])
    with pytest.raises(ValueError, match="targets in a TaskBatch"):
        strategy.configure_fit(1, parameters, manager)
    assert policy.learners.server.model.num_updates == 0


def _native_cohort(*, count=1, config=None, with_targets=True):
    from flwr.common import serde
    from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
    from splitfleet.server.placement.cosplit_ucb import CoSplitUCBPlacementPolicy
    from splitfleet.server.stage_runtime.manager import StageRuntimeManager
    from splitfleet.server.strategy import AutoSplitStrategy
    from splitfleet.tasks import ImageClassificationTask, TaskBatch
    from tests.test_flower_autosplit_round_loop import DeepNet

    model, inputs, labels = DeepNet(), torch.randn(2, 4), torch.tensor([0, 1])
    sample = TaskBatch(ModelInputs(args=(inputs,)), labels) if with_targets else inputs
    task = ImageClassificationTask()
    provider = TorchLensCandidateProvider(model=copy.deepcopy(model), sample_inputs=inputs,
        batch_axes={"/args/0": 0}, dynamic_batch=(1, 4), require_trainable_prefix=True)
    policy = CoSplitUCBPlacementPolicy(candidate_provider=provider, config=config)
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


@pytest.mark.parametrize("initial_evaluation", [False, True])
def test_evaluation_binds_all_workers_without_calibrating_or_changing_learned_costs(initial_evaluation):
    from flwr.common import ndarrays_to_parameters

    cohort = _native_cohort(count=2)
    strategy, policy = cohort.strategy, cohort.policy
    parameters = ndarrays_to_parameters(cohort.initial)
    if initial_evaluation:
        for worker, instruction in strategy.configure_evaluate(0, parameters, cohort.client_manager):
            _execute_instruction(cohort, worker, instruction, training=False, round_id=0)
        assert not policy.learners.has_observations
        assert all(client.online_calibration_receipt is None for client in cohort.clients.values())
        assert all("cosplit_calibration_boundaries" not in request
                   for requests in cohort.requests.values() for request in requests)

    instructions = strategy.configure_fit(1, parameters, cohort.client_manager)
    assert [worker.cid for worker, _ in instructions] == ["a"]
    _execute_instruction(cohort, *instructions[0], training=True, round_id=1)
    learned = copy.deepcopy(policy.learners.state_dict())
    # A later evaluation receives updated weights. Worker b must never be
    # calibrated against the original frozen model when first seen here.
    parameters = ndarrays_to_parameters([value + .01 for value in cohort.initial])
    evaluations = strategy.configure_evaluate(1, parameters, cohort.client_manager)
    assert [worker.cid for worker, _ in evaluations] == ["a", "b"]
    for worker, instruction in evaluations:
        _execute_instruction(cohort, worker, instruction, training=False, round_id=1)
    after = policy.learners.state_dict()
    assert after["server"] == learned["server"]
    assert after["switch"] == learned["switch"]
    assert after["edge"] == learned["edge"]
    assert after["network"]["a"] == learned["network"]["a"]
    assert all(model["num_updates"] == 0 for model in after["network"]["b"].values())
    assert cohort.clients["b"].online_calibration_receipt is None
    assert all("cosplit_calibration_boundaries" not in request for request in cohort.requests["b"])

    cohort.client_manager.sample = lambda **kw: [cohort.workers[1]]
    next_fit = strategy.configure_fit(2, parameters, cohort.client_manager)
    _execute_instruction(cohort, *next_fit[0], training=True, round_id=2)
    assert policy.learners.server.model.num_updates == learned["server"]["num_updates"] + 1
    assert cohort.clients["b"].online_calibration_receipt is None


@pytest.mark.parametrize("restore", ["file", "explicit", "explicit_after_calibration"])
def test_native_restore_preserves_learned_statistics_without_injecting_calibration(tmp_path, monkeypatch, restore):
    from flwr.common import ndarrays_to_parameters
    from splitfleet.server.placement.cosplit_ucb import BanditStateStore, CoSplitUCBConfig
    from splitfleet.server.placement.cosplit_ucb import calibration

    original = _native_cohort()
    instructions = original.strategy.configure_fit(1, ndarrays_to_parameters(original.initial), original.client_manager)
    _execute_instruction(original, *instructions[0], training=True, round_id=1)
    state = original.policy.state_dict()
    path = tmp_path / "learned.json"
    BanditStateStore(path).save(state)
    restored = _native_cohort(
        config=CoSplitUCBConfig(state_path=str(path)) if restore == "file" else None,
        with_targets=restore == "explicit_after_calibration")
    policy = restored.policy
    if restore == "explicit_after_calibration":
        policy.prepare_calibration(restored.inputs, restored.targets, loss_fn=restored.task.loss,
                                   device="cpu", parameters=restored.initial)
    if restore != "file":
        policy.load_state_dict(state)

    def unexpected_calibration(*args, **kwargs):
        pytest.fail("A restored learner must not execute fresh calibration")

    monkeypatch.setattr(calibration, "calibrate_split", unexpected_calibration)
    policy.prepare_calibration(restored.inputs, None, loss_fn=restored.task.loss,
                               device="cpu", parameters=restored.initial)
    policy.bind_clients(restored.workers, 2)
    assert policy.learners.state_dict() == state["state"]["learners"]
    assert policy.telemetry_provider.receipt is None
    assert restored.clients["a"].online_calibration_receipt is None
    assert restored.requests["a"] == [{}]
    restored_updates = policy.learners.server.model.num_updates
    instructions = restored.strategy.configure_fit(2, ndarrays_to_parameters(restored.initial), restored.client_manager)
    assert policy.learners.server.model.num_updates == restored_updates
    assert restored.requests["a"] == [{}]
    _execute_instruction(restored, *instructions[0], training=True, round_id=2)
    assert policy.learners.server.model.num_updates == restored_updates + 1


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
