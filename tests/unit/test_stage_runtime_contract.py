from __future__ import annotations

import copy
import json

import pytest
import torch
from torch import nn

from splitfleet.autosplit import AutoSplitSession
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
    assert key["torchlens_version"] == "2.31.0"
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


def test_clone_runtime_rejects_feature_abi_mismatch(monkeypatch) -> None:
    manager, model, _placement = _manager_with_plan()
    base = manager._require_runtime_handle()
    bad = copy.copy(base)
    bad.plan = copy.copy(base.plan)
    bad.plan.runtime_contract = dict(base.plan.runtime_contract)
    bad.plan.runtime_contract["feature_abi_id"] = "bad-abi"
    bad.plan.feature_abi_id = "bad-abi"
    bad.feature_abi_id = "bad-abi"

    monkeypatch.setattr(manager.autosplit_session, "prepare_runtime", lambda *args, **kwargs: bad)

    with pytest.raises(RuntimeError, match="feature ABI is incompatible"):
        manager.clone_runtime_for_model(copy.deepcopy(model), suffix="bad")
