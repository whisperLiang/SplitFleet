from __future__ import annotations

import copy

import numpy as np
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from torch import nn

from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.server.client_selection import OortSelector, OortSelectorConfig
from splitfleet.server.server_model.autosplit_tail_server_model import AutoSplitTailServerModel
from splitfleet.server.server_model.proxy.server_model_proxy import ServerModelProxy
from splitfleet.server.stage_runtime.manager import StageRuntimeManager
from splitfleet.server.strategy import AutoSplitStrategy


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class FakeClientProxy:
    def __init__(self, cid: str) -> None:
        self.cid = cid


class FakeClientManager:
    def __init__(self, cids: list[str]) -> None:
        self.clients = {cid: FakeClientProxy(cid) for cid in cids}

    def all(self):
        return dict(self.clients)

    def num_available(self) -> int:
        return len(self.clients)

    def sample(self, num_clients: int, min_num_clients: int | None = None):
        if min_num_clients is not None and len(self.clients) < min_num_clients:
            return []
        return list(self.clients.values())[:num_clients]


class InProcessServerModelProxy(ServerModelProxy):
    def __init__(self, *, server_model, cid: str) -> None:
        super().__init__(cid=cid)
        self.server_model = server_model

    def _blocking_request(self, method, batch_data, _streams_, _timeout_):
        result = getattr(self.server_model, method)([batch_data])
        return result[0]

    def _streaming_request(self, method, batch_data, _timeout_, _streams_):
        _ = (method, batch_data, _timeout_, _streams_)
        raise NotImplementedError

    def _nonblocking_request(self, method, batch_data, _timeout_, _streams_):
        _ = (method, batch_data, _timeout_, _streams_)
        raise NotImplementedError

    def close_stream(self):
        return None

    def get_pending_batches_count(self) -> int:
        return 0


def _model_to_ndarrays(model: nn.Module):
    return [tensor.detach().cpu().numpy() for tensor in model.state_dict().values()]


def test_torchlens_split_training_smoke_with_oort_selection() -> None:
    torch.manual_seed(101)
    base_model = TinyNet()
    strategy_model = copy.deepcopy(base_model)
    manager = FakeClientManager(["c1", "c2", "c3", "c4"])

    strategy = AutoSplitStrategy(
        model=strategy_model,
        sample_inputs=torch.randn(2, 4),
        boundary="50%",
        loss_fn=nn.MSELoss(),
        optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.02),
        min_fit_clients=2,
        min_available_clients=2,
        fraction_fit=0.5,
        client_selection="oort",
        oort_config=OortSelectorConfig(seed=31),
    )
    runtime_manager = StageRuntimeManager(autosplit_session=strategy.autosplit_session)
    strategy.bind_stage_runtime_manager(runtime_manager)
    runtime_manager.set_placement_plan(strategy.get_or_create_placement_plan())

    clients: dict[str, AutoSplitSplitLearningClient] = {}
    for index, cid in enumerate(manager.clients):
        inputs = torch.randn(3, 4) + index * 0.1
        targets = torch.randn(3, 2)
        clients[cid] = AutoSplitSplitLearningClient(
            model=base_model,
            train_data=[(inputs, targets)],
            sample_inputs=torch.randn(2, 4),
            optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.02),
        )

    client_parameters = ndarrays_to_parameters(_model_to_ndarrays(base_model))
    server_parameters = _model_to_ndarrays(strategy_model)
    observed_losses: list[float] = []
    selected_counts: list[int] = []

    for server_round in (1, 2):
        instructions = strategy.configure_fit(server_round, client_parameters, manager)
        selected_counts.append(len(instructions))
        assert len(instructions) == 2

        selected_cids = [proxy.cid for proxy, _ in instructions]
        server_configs = strategy.configure_server_fit(
            server_round,
            server_parameters,
            selected_cids,
        )
        server_model = AutoSplitTailServerModel(
            runtime_manager=runtime_manager,
            model=strategy_model,
            optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.02),
            loss_fn=nn.MSELoss(),
        )
        server_model.configure_fit(server_configs[0])

        results = []
        for proxy, fit_ins in instructions:
            client = clients[proxy.cid]
            client.server_model_proxy = InProcessServerModelProxy(
                server_model=server_model,
                cid=proxy.cid,
            )
            parameters, num_examples, metrics = client.fit(
                parameters_to_ndarrays(fit_ins.parameters),
                fit_ins.config,
            )
            observed_losses.append(metrics["loss"])
            results.append(
                (
                    proxy,
                    FitRes(
                        status=Status(Code.OK, ""),
                        parameters=ndarrays_to_parameters(parameters),
                        num_examples=num_examples,
                        metrics=metrics,
                    ),
                )
            )

        server_result = server_model.get_fit_result()
        client_parameters, _ = strategy.aggregate_fit(server_round, results, [])
        server_parameters = strategy.aggregate_server_fit(server_round, [server_result])
        client_parameters, server_parameters = strategy.finalize_round(
            server_round, client_parameters, server_parameters
        )

    assert selected_counts == [2, 2]
    assert all(np.isfinite(loss) and loss >= 0 for loss in observed_losses)
    assert isinstance(strategy.client_selector, OortSelector)
    updated_states = [
        state
        for state in strategy.client_selector.clients.values()
        if state.selected_count > 0
    ]
    assert updated_states
    assert all(state.reward >= 0 for state in updated_states)
    assert all(state.duration > 0 for state in updated_states)
