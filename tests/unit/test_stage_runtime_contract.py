from __future__ import annotations

import copy
import json

import pytest
import torch
from torch import nn

from splitfleet.autosplit import AutoSplitSession
from splitfleet.autosplit.types import PlacementConstraint, PlacementObjective
from splitfleet.server.stage_runtime.manager import StageRuntimeManager


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _manager_with_plan():
    torch.manual_seed(71)
    model = TinyNet().eval()
    session = AutoSplitSession(device="cpu")
    placement = session.plan(model, torch.randn(2, 4), dynamic_batch=(1, 8))
    manager = StageRuntimeManager(autosplit_session=session)
    manager.set_placement_plan(placement)
    return manager, model, placement


def test_clone_runtime_preserves_feature_abi_and_cache_key() -> None:
    manager, model, placement = _manager_with_plan()
    clone = manager.clone_runtime_for_model(copy.deepcopy(model), suffix="client1")

    assert clone.feature_abi_id == placement.feature_abi_id
    assert clone.plan.runtime_contract["feature_abi_id"] == placement.feature_abi_id

    key = json.loads(
        manager._clone_runtime_cache_key(
            manager._require_runtime_handle(),
            model=model,
            suffix="client1",
        )
    )
    assert key["plan_id"] == placement.plan_id
    assert key["split_id"] == placement.split_id
    assert key["feature_abi_id"] == placement.feature_abi_id
    assert key["boundary"] == placement.boundary
    assert key["graph_signature"] == placement.graph_signature
    assert key["torchlens_version"] == "2.34.1"
    assert key["runtime_backend"] == "torchlens_native"
    assert key["trace_batch_mode"] == placement.trace_batch_mode
    assert key["dynamic_batch"] == list(placement.dynamic_batch)
    assert key["module_mode"] == "eval"
    assert key["runtime_contract_digest"]


def test_binding_runtime_handle_clears_clone_cache() -> None:
    manager, model, _placement = _manager_with_plan()
    manager.clone_runtime_for_model(copy.deepcopy(model), suffix="client1")

    assert manager._clone_runtime_cache
    manager.bind_runtime_handle(manager._require_runtime_handle())

    assert manager._clone_runtime_cache == {}


def test_registered_placements_clone_the_requested_client_abi() -> None:
    manager, model, default = _manager_with_plan()
    alternate = manager.autosplit_session.plan(
        model,
        torch.randn(2, 4),
        boundary="after:fc1",
        dynamic_batch=(1, 8),
    )
    manager.register_placement_plan(alternate)

    default_clone = manager.clone_runtime_for_model(
        copy.deepcopy(model), suffix="default-client", plan_id=default.plan_id
    )
    alternate_clone = manager.clone_runtime_for_model(
        copy.deepcopy(model), suffix="alternate-client", plan_id=alternate.plan_id
    )

    assert default_clone.plan.boundary == default.boundary
    assert alternate_clone.plan.boundary == alternate.boundary
    assert default_clone.plan.feature_abi_id != alternate_clone.plan.feature_abi_id


def test_clone_runtime_rejects_feature_abi_mismatch(monkeypatch) -> None:
    manager, model, _placement = _manager_with_plan()
    base = manager._require_runtime_handle()
    bad = copy.copy(base)
    bad.plan = copy.copy(base.plan)
    bad.plan.runtime_contract = dict(base.plan.runtime_contract)
    bad.plan.runtime_contract["feature_abi_id"] = "bad-abi"
    bad.plan.feature_abi_id = "bad-abi"
    bad.feature_abi_id = "bad-abi"

    monkeypatch.setattr("splitfleet.server.stage_runtime.manager.clone_runtime_handle", lambda *args: bad)

    with pytest.raises(RuntimeError, match="feature ABI is incompatible"):
        manager.clone_runtime_for_model(copy.deepcopy(model), suffix="bad")


def test_incompatible_torch_replica_does_not_retrace(monkeypatch) -> None:
    manager, model, _placement = _manager_with_plan()
    replica = copy.deepcopy(model)
    replica.fc1.out_features = 9
    monkeypatch.setattr(
        manager.autosplit_session, "prepare_runtime",
        lambda *args, **kwargs: pytest.fail("unexpected second trace"),
    )

    with pytest.raises(RuntimeError, match="cannot bind this model replica"):
        manager.clone_runtime_for_model(replica)


def test_clone_reuses_capture_with_independent_parameters_and_gradients(monkeypatch) -> None:
    manager, model, _placement = _manager_with_plan()
    first_model = copy.deepcopy(model)
    second_model = copy.deepcopy(model)
    with torch.no_grad():
        first_model.fc2.weight.add_(0.5)

    def unexpected_trace(*args, **kwargs):
        pytest.fail("compatible model replicas must not be traced again")

    monkeypatch.setattr(manager.autosplit_session, "prepare_runtime", unexpected_trace)
    first = manager.clone_runtime_for_model(first_model, suffix="first")
    second = manager.clone_runtime_for_model(second_model, suffix="second")
    inputs = torch.randn(2, 4)
    torch.testing.assert_close(first.runtime.replay(inputs), first_model(inputs))
    torch.testing.assert_close(second.runtime.replay(inputs), second_model(inputs))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    first.runtime.replay(inputs).square().sum().backward()
    assert first_model.fc2.weight.grad is not None
    assert model.fc2.weight.grad is None
    assert second_model.fc2.weight.grad is None
    torch.optim.SGD(first_model.parameters(), lr=0.01).step()
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before[name])
        torch.testing.assert_close(dict(second_model.named_parameters())[name], before[name])
    torch.testing.assert_close(first.runtime.replay(inputs), first_model(inputs))


class BufferedNet(TinyNet):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.ones(2))

    def forward(self, x):
        return super().forward(x) * self.scale


def test_clone_rebinds_replaced_buffers(monkeypatch) -> None:
    model = BufferedNet().eval()
    session = AutoSplitSession()
    placement = session.plan(model, torch.randn(2, 4), dynamic_batch=(1, 8))
    manager = StageRuntimeManager(autosplit_session=session)
    manager.set_placement_plan(placement)
    replica = copy.deepcopy(model)
    monkeypatch.setattr(session, "prepare_runtime", lambda *args, **kwargs: pytest.fail("unexpected trace"))
    handle = manager.clone_runtime_for_model(replica)
    replica.scale = torch.tensor([3.0, 4.0])
    inputs = torch.randn(2, 4)
    torch.testing.assert_close(handle.runtime.replay(inputs), replica(inputs))
    torch.testing.assert_close(model.scale, torch.ones(2))


class StatefulTiedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 4, bias=False)
        self.norm = nn.BatchNorm1d(4)
        self.last = nn.Linear(4, 4, bias=False)
        self.last.weight = self.first.weight

    def forward(self, inputs):
        return self.last(torch.relu(self.norm(self.first(inputs))))


def test_clone_preserves_tied_weights_and_isolates_training_buffers(monkeypatch) -> None:
    model = StatefulTiedNet().train()
    inputs = torch.randn(2, 4)
    session = AutoSplitSession()
    placement = session.plan(model, inputs, dynamic_batch=(1, 8))
    manager = StageRuntimeManager(autosplit_session=session)
    manager.set_placement_plan(placement)
    replica = copy.deepcopy(model)
    full = copy.deepcopy(model)
    before = {name: value.clone() for name, value in model.named_buffers()}
    monkeypatch.setattr(session, "prepare_runtime", lambda *args, **kwargs: pytest.fail("unexpected trace"))
    handle = manager.clone_runtime_for_model(replica)
    expected = full(inputs)
    actual = handle.runtime.run_suffix(handle.runtime.run_training_prefix(inputs))
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert replica.first.weight is replica.last.weight
    for name, value in replica.named_parameters():
        torch.testing.assert_close(value.grad, dict(full.named_parameters())[name].grad)
    for name, value in replica.named_buffers():
        torch.testing.assert_close(value, dict(full.named_buffers())[name])
        torch.testing.assert_close(dict(model.named_buffers())[name], before[name])


def test_rebind_declines_changed_configuration_and_aliases() -> None:
    from splitfleet.autosplit.torchlens_clone import rebind_torch_runtime

    model = StatefulTiedNet().train()
    handle = AutoSplitSession().prepare_runtime(model, torch.randn(2, 4))
    replica = copy.deepcopy(model)
    replica.norm.eps = 0.1
    assert rebind_torch_runtime(handle.runtime, replica) is None
    replica = copy.deepcopy(model)
    replica.last.weight = nn.Parameter(replica.last.weight.detach().clone())
    assert rebind_torch_runtime(handle.runtime, replica) is None


def test_full_candidates_match_client_contract_and_survive_global_evaluation(monkeypatch) -> None:
    from experiments.rfdetr_nano_physical import RFDETRCandidateProvider
    from splitfleet.split_engine import graph_contract_for_runtime_handle
    from splitfleet.split_engine.contracts import validate_contract

    model = nn.Sequential(*[layer for _ in range(6) for layer in (nn.Linear(4, 4), nn.ReLU())]).train()
    sample = torch.randn(1, 4)
    provider = RFDETRCandidateProvider(model=model, sample_inputs=sample)
    candidates = provider.get_candidates()
    assert len(candidates) > 3
    assert any(candidate.boundary.startswith("before:") for candidate in candidates)
    assert any(candidate.boundary.startswith("after:") for candidate in candidates)
    session = AutoSplitSession()
    client = session.prepare_runtime(model, sample, batch_axes={"/args/0": 0},
                                     boundary=candidates[0].boundary, dynamic_batch=(1, 1))
    model.eval()
    manager = StageRuntimeManager(autosplit_session=session)
    monkeypatch.setattr(session, "prepare_runtime", lambda *args, **kwargs: pytest.fail("unexpected trace"))
    for candidate in candidates:
        placement = provider.get_placement_plan(
            candidate.boundary, worker_specs=[],
            constraints=PlacementConstraint(), objective=PlacementObjective(),
        )
        base = placement.metadata["_runtime_handle"]
        client = client.backend.repartition(candidate.boundary)
        validate_contract(graph_contract_for_runtime_handle(base), graph_contract_for_runtime_handle(client))
        manager.bind_runtime_handle(base)
        clone = manager.clone_runtime_for_model(copy.deepcopy(model).train())
        assert clone.plan.metadata["_reused_capture"]
        validate_contract(graph_contract_for_runtime_handle(base), graph_contract_for_runtime_handle(clone))


@pytest.mark.parametrize("method,args", [
    ("get_server_model", ("client",)),
    ("initialize_server_models", ([],)),
    ("collect_server_fit_results", ()),
    ("get_server_model_ids", ()),
    ("end_round", ()),
])
def test_server_lifecycle_requires_a_model_factory(method, args) -> None:
    manager = StageRuntimeManager()
    with pytest.raises(RuntimeError, match="init_server_model_fn"):
        getattr(manager, method)(*args)
