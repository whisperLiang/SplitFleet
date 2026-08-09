"""The real server round loop must adopt the reassembled logical model."""

from __future__ import annotations

import numpy as np
import torch
from flwr.common import (
    Code,
    FitRes,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from torch import nn

from splitfleet.common import ServerModelFitRes
from splitfleet.server.server import Server
from splitfleet.server.server_model.manager.manager import ServerModelManager
from splitfleet.server.strategy import AutoSplitStrategy


PREFIX_DELTA = 1.0
SUFFIX_DELTA = 2.0


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(x)))


class PrefixOnlyClientProxy(ClientProxy):
    """A client that returns a fresh prefix and an untouched suffix half."""

    def __init__(self, cid: str) -> None:
        super().__init__(cid)

    def fit(self, ins, timeout=None, group_id=None) -> FitRes:
        state = [np.array(value, copy=True) for value in parameters_to_ndarrays(ins.parameters)]
        state[0] = state[0] + PREFIX_DELTA
        return FitRes(
            status=Status(Code.OK, ""),
            parameters=ndarrays_to_parameters(state),
            num_examples=4,
            metrics={},
        )

    def get_parameters(self, ins, timeout=None, group_id=None) -> GetParametersRes:
        raise NotImplementedError

    def get_properties(self, ins, timeout=None, group_id=None):
        raise NotImplementedError

    def evaluate(self, ins, timeout=None, group_id=None):
        raise NotImplementedError

    def reconnect(self, ins, timeout=None, group_id=None):
        raise NotImplementedError


class SuffixOnlyServerModelManager(ServerModelManager):
    """A manager that returns one suffix result per configured replica."""

    def __init__(self) -> None:
        super().__init__()
        self.configs: list = []

    def initialize_server_models(self, configs) -> None:
        self.configs = list(configs)

    def collect_server_fit_results(self) -> list[ServerModelFitRes]:
        results = []
        for config in self.configs:
            state = [np.array(value, copy=True) for value in config.parameters]
            state[-1] = state[-1] + SUFFIX_DELTA
            result = ServerModelFitRes(parameters=state, config={"num_examples": 4})
            result.sid = config.sid
            results.append(result)
        return results

    def get_server_model(self, sid):
        raise NotImplementedError

    def get_server_model_ids(self):
        return [config.sid for config in self.configs]

    def end_round(self) -> None:
        self.configs = []


def test_fit_round_publishes_the_reassembled_model_not_the_stale_average() -> None:
    torch.manual_seed(11)
    model = TinyNet().eval()
    strategy = AutoSplitStrategy(
        model=model,
        sample_inputs=torch.randn(2, 4),
        aggregation_policy="splitfed",
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    client_manager = SimpleClientManager()
    for cid in ("device-a", "device-b"):
        client_manager.register(PrefixOnlyClientProxy(cid))
    server = Server(
        server_model_manager=SuffixOnlyServerModelManager(),
        client_manager=client_manager,
        strategy=strategy,
    )
    initial = strategy.initialize_server_parameters()
    server.client_parameters = ndarrays_to_parameters(initial)
    server.server_parameters = initial

    client_parameters, server_parameters, _, (results, failures) = server.fit_round(
        server_round=1, timeout=None
    )

    assert not failures and len(results) == 2
    published = parameters_to_ndarrays(client_parameters)
    # Both halves of the split are present exactly once: the prefix update the
    # clients produced and the suffix update the replicas produced.
    assert np.allclose(published[0], initial[0] + PREFIX_DELTA)
    assert np.allclose(published[-1], initial[-1] + SUFFIX_DELTA)
    assert all(
        np.allclose(client_value, server_value)
        for client_value, server_value in zip(published, server_parameters)
    )
    assert strategy._pending_client_updates == {}
    assert strategy._reassembled_client_states == {}
