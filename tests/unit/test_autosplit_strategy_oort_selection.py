from __future__ import annotations

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters
from torch import nn

from splitfleet.common import ServerModelFitRes
from splitfleet.common.constants import AUTOSPLIT_PLAN_ID_CONFIG_KEY
from splitfleet.server.client_selection import OortSelector
from splitfleet.server.server_model.server_model import ServerModel
from splitfleet.server.strategy import AutoSplitStrategy


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(x)))


class DummyServerModel(ServerModel):
    def get_parameters(self):
        return []

    def configure_fit(self, ins):
        self.fit_config = ins

    def get_fit_result(self):
        return ServerModelFitRes(parameters=[], config={"num_examples": 1})

    def configure_evaluate(self, ins):
        self.eval_config = ins


class FakeClientProxy:
    def __init__(self, cid: str) -> None:
        self.cid = cid


class FakeClientManager:
    def __init__(self, cids: list[str]) -> None:
        self.clients = {cid: FakeClientProxy(cid) for cid in cids}
        self.sample_calls = 0

    def all(self):
        return dict(self.clients)

    def num_available(self) -> int:
        return len(self.clients)

    def sample(self, num_clients: int, min_num_clients: int | None = None):
        self.sample_calls += 1
        if min_num_clients is not None and len(self.clients) < min_num_clients:
            return []
        return list(self.clients.values())[:num_clients]


def _strategy(**kwargs) -> AutoSplitStrategy:
    return AutoSplitStrategy(
        model=TinyNet().eval(),
        sample_inputs=torch.randn(2, 4),
        worker_specs=[],
        init_server_model_fn=lambda: DummyServerModel(),
        min_fit_clients=2,
        min_available_clients=2,
        fraction_fit=0.5,
        **kwargs,
    )


def _parameters(value: float = 0.0):
    return ndarrays_to_parameters([np.array([value], dtype=np.float32)])


def _fit_res(value: float, *, loss: float = 1.0, duration: float = 1.0) -> FitRes:
    return FitRes(
        status=Status(Code.OK, ""),
        parameters=_parameters(value),
        num_examples=5,
        metrics={
            "loss": loss,
            "fit_duration_sec": duration,
            "num_examples": 5,
        },
    )


def test_oort_configure_fit_uses_oort_selector() -> None:
    strategy = _strategy(client_selection="oort")
    manager = FakeClientManager(["c1", "c2", "c3", "c4"])

    instructions = strategy.configure_fit(1, _parameters(), manager)

    assert isinstance(strategy.client_selector, OortSelector)
    assert manager.sample_calls == 0
    assert len(instructions) == 2


def test_oort_configure_fit_returns_correct_number_of_clients() -> None:
    strategy = _strategy(client_selection="oort")
    manager = FakeClientManager(["c1", "c2", "c3", "c4"])

    instructions = strategy.configure_fit(1, _parameters(), manager)

    assert len(instructions) == 2
    assert len({client.cid for client, _ in instructions}) == 2


def test_oort_configure_fit_keeps_ariadne_autosplit_metadata() -> None:
    strategy = _strategy(client_selection="oort")
    manager = FakeClientManager(["c1", "c2", "c3", "c4"])

    instructions = strategy.configure_fit(1, _parameters(), manager)

    for _, fit_ins in instructions:
        assert AUTOSPLIT_PLAN_ID_CONFIG_KEY in fit_ins.config


def test_oort_aggregate_fit_updates_selector_state() -> None:
    strategy = _strategy(client_selection="oort")
    manager = FakeClientManager(["c1", "c2", "c3", "c4"])
    instructions = strategy.configure_fit(1, _parameters(), manager)

    results = [
        (client, _fit_res(index, loss=1.0 + index, duration=1.0 + index))
        for index, (client, _) in enumerate(instructions, start=1)
    ]
    strategy.aggregate_fit(1, results, [])

    for client, _ in instructions:
        state = strategy.client_selector.clients[client.cid]
        assert state.selected_count == 1
        assert state.reward > 0
        assert state.duration >= 1.0


def test_flower_default_selection_uses_original_client_manager_sample() -> None:
    strategy = _strategy(client_selection="flower_default")
    manager = FakeClientManager(["c1", "c2", "c3", "c4"])

    instructions = strategy.configure_fit(1, _parameters(), manager)

    assert strategy.client_selector is None
    assert manager.sample_calls == 1
    assert [client.cid for client, _ in instructions] == ["c1", "c2"]


def test_invalid_client_selection_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported client_selection"):
        _strategy(client_selection="definitely_not_a_selector")
