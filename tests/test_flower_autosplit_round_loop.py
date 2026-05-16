import copy

import numpy as np
import torch
from torch import nn

from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.common import ServerModelFitIns
from splitfleet.server.server_model.ariadne_tail_server_model import AriadneTailServerModel
from splitfleet.server.server_model.proxy.server_model_proxy import ServerModelProxy
from splitfleet.server.stage_runtime.manager import StageRuntimeManager
from splitfleet.server.strategy import AutoSplitStrategy


class DeepNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 8)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


class BatchNormNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4, bias=False)
        self.bn = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.bn(self.fc1(x)))


class InProcessServerModelProxy(ServerModelProxy):
    def __init__(self, *, server_model, cid: str = "client-1") -> None:
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


def test_ariadne_split_learning_client_and_tail_exchange_boundary_payloads() -> None:
    torch.manual_seed(23)
    base_model = DeepNet()
    strategy_model = copy.deepcopy(base_model)
    x = torch.randn(3, 4)
    targets = torch.randn(3, 2)

    strategy = AutoSplitStrategy(
        model=strategy_model,
        sample_inputs=torch.randn(2, 4),
        boundary="50%",
        loss_fn=nn.MSELoss(),
        optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.05),
        init_server_model_fn=lambda: AriadneTailServerModel(
            runtime_manager=StageRuntimeManager(),
            model=strategy_model,
        ),
    )
    manager = StageRuntimeManager(autosplit_session=strategy.autosplit_session)
    strategy.bind_stage_runtime_manager(manager)
    manager.set_placement_plan(strategy.get_or_create_placement_plan())
    config = strategy._autosplit_config()

    server_model = AriadneTailServerModel(
        runtime_manager=manager,
        model=strategy_model,
        optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.05),
        loss_fn=nn.MSELoss(),
    )
    server_model.configure_fit(
        ServerModelFitIns(
            parameters=_model_to_ndarrays(strategy_model),
            config=config,
            sid="",
        )
    )

    client = AutoSplitSplitLearningClient(
        model=base_model,
        train_data=[(x, targets)],
        evaluate_data=[(x, targets)],
        sample_inputs=torch.randn(2, 4),
        optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.05),
    )
    client.server_model_proxy = InProcessServerModelProxy(server_model=server_model)

    client_params, num_examples, metrics = client.fit(_model_to_ndarrays(base_model), config)
    server_result = server_model.get_fit_result()

    assert num_examples == 3
    assert metrics["loss"] > 0
    assert server_result.config["num_examples"] == 3
    assert any(
        not np.allclose(before, after)
        for before, after in zip(_model_to_ndarrays(base_model), client_params)
    )
    assert any(
        not np.allclose(before, after)
        for before, after in zip(_model_to_ndarrays(strategy_model), server_result.parameters)
    )


def test_ariadne_splitfed_keeps_per_client_tail_semantics() -> None:
    strategy = AutoSplitStrategy(
        model=DeepNet(),
        sample_inputs=torch.randn(2, 4),
        aggregation_policy="splitfed",
        client_stage_count=1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    assert strategy.replica_scope_policy.resolve().value == "per_client"
    assert strategy.common_server_model is False


def test_ariadne_tail_server_model_prepares_runtime_after_train_mode() -> None:
    torch.manual_seed(37)
    base_model = BatchNormNet().eval()
    strategy = AutoSplitStrategy(
        model=base_model,
        sample_inputs=torch.randn(2, 4),
        boundary="after:fc1",
        loss_fn=nn.MSELoss(),
    )
    manager = StageRuntimeManager(autosplit_session=strategy.autosplit_session)
    strategy.bind_stage_runtime_manager(manager)
    manager.set_placement_plan(strategy.get_or_create_placement_plan())
    server_model = AriadneTailServerModel(
        runtime_manager=manager,
        model=base_model,
        loss_fn=nn.MSELoss(),
    )

    server_model.configure_fit(
        ServerModelFitIns(
            parameters=_model_to_ndarrays(base_model),
            config=strategy._autosplit_config(),
            sid="",
        )
    )

    inputs = torch.randn(3, 4)
    expected = copy.deepcopy(server_model.model).train()(inputs)
    boundary = server_model.runtime_handle.runtime.run_prefix(inputs)
    split = server_model.runtime_handle.runtime.run_suffix(boundary)

    assert torch.allclose(split, expected, atol=1e-5, rtol=1e-5)
