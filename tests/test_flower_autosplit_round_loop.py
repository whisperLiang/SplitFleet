import copy

import numpy as np
import torch
from torch import nn

from flwr.client.client import maybe_call_evaluate, maybe_call_fit
from flwr.common import (
    Code,
    GetParametersRes,
    GetPropertiesRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import DisconnectRes
from flwr.server.app import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

from splitfleet.autosplit import ReplicaScope
from splitfleet.client.autosplit_client import AutoSplitNumPyClient
from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.client.numpy_client import NumPyClient
from splitfleet.server.app import init_defaults
from splitfleet.server.server_model.proxy.server_model_proxy import ServerModelProxy
from splitfleet.server.strategy import AutoSplitStrategy
from splitfleet.worker import start_worker


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


class InProcessServerModelProxy(ServerModelProxy):
    def __init__(self, *, server_model_manager, strategy, cid: str) -> None:
        super().__init__(cid=cid)
        self.server_model_manager = server_model_manager
        self.strategy = strategy

    def _blocking_request(self, method, batch_data, _streams_, _timeout_):
        _ = (_streams_, _timeout_)
        sid = self.strategy._cid_to_sid_mapping[self.cid]
        server_model = self.server_model_manager.get_server_model(sid)
        result = getattr(server_model, method)([batch_data])
        return result[0]

    def _streaming_request(self, method, batch_data, _timeout_, _streams_):
        _ = (method, batch_data, _timeout_, _streams_)
        raise NotImplementedError("Streaming is not used by the in-process autosplit test proxy.")

    def _nonblocking_request(self, method, batch_data, _timeout_, _streams_):
        _ = (method, batch_data, _timeout_, _streams_)
        raise NotImplementedError("Future requests are not used by the in-process autosplit test proxy.")

    def close_stream(self):
        return None

    def get_pending_batches_count(self) -> int:
        return 0


class InProcessAutoSplitClientProxy(ClientProxy):
    def __init__(self, *, cid: str, numpy_client: NumPyClient, server_model_manager, strategy) -> None:
        super().__init__(cid)
        self._client = numpy_client.to_client()
        self._server_model_manager = server_model_manager
        self._strategy = strategy

    def get_properties(self, ins, timeout, group_id):
        _ = (ins, timeout, group_id)
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="ok"),
            properties={},
        )

    def get_parameters(self, ins, timeout, group_id):
        _ = (ins, timeout, group_id)
        return GetParametersRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters([]),
        )

    def fit(self, ins, timeout, group_id):
        _ = (timeout, group_id)
        self._client.set_server_model_proxy(
            InProcessServerModelProxy(
                server_model_manager=self._server_model_manager,
                strategy=self._strategy,
                cid=self.cid,
            )
        )
        return maybe_call_fit(self._client, ins)

    def evaluate(self, ins, timeout, group_id):
        _ = (timeout, group_id)
        self._client.set_server_model_proxy(
            InProcessServerModelProxy(
                server_model_manager=self._server_model_manager,
                strategy=self._strategy,
                cid=self.cid,
            )
        )
        return maybe_call_evaluate(self._client, ins)

    def reconnect(self, ins, timeout, group_id):
        _ = (ins, timeout, group_id)
        return DisconnectRes(reason="")


def _load_model_from_ndarrays(model: nn.Module, ndarrays) -> None:
    state_dict = model.state_dict()
    loaded = {}
    for (name, reference), array in zip(state_dict.items(), ndarrays):
        loaded[name] = torch.as_tensor(np.array(array), dtype=reference.dtype)
    model.load_state_dict(loaded, strict=True)


def _merge_split_states(base_model: nn.Module, client_ndarrays, server_ndarrays) -> nn.Module:
    merged_model = copy.deepcopy(base_model)
    base_state = list(base_model.state_dict().values())
    merged_state = {}
    for (name, reference), base_tensor, client_array, server_array in zip(
        merged_model.state_dict().items(),
        base_state,
        client_ndarrays,
        server_ndarrays,
    ):
        base_array = base_tensor.detach().cpu().numpy()
        client_changed = not np.allclose(client_array, base_array, atol=1e-7, rtol=1e-7)
        server_changed = not np.allclose(server_array, base_array, atol=1e-7, rtol=1e-7)
        if client_changed and server_changed:
            raise AssertionError(f"Parameter {name} was updated on both split sides.")
        if client_changed:
            chosen = client_array
        elif server_changed:
            chosen = server_array
        else:
            chosen = base_array
        merged_state[name] = torch.as_tensor(np.array(chosen), dtype=reference.dtype)
    merged_model.load_state_dict(merged_state, strict=True)
    return merged_model


def _run_single_batch_reference_step(
    base_model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    lr: float = 0.1,
) -> nn.Module:
    reference_model = copy.deepcopy(base_model)
    optimizer = torch.optim.SGD(reference_model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.MSELoss()(reference_model(inputs), targets)
    loss.backward()
    optimizer.step()
    return reference_model


def _average_reference_models(models: list[nn.Module], weights: list[int]) -> nn.Module:
    averaged_model = copy.deepcopy(models[0])
    total_weight = float(sum(weights))
    averaged_state = {}
    for name, reference in averaged_model.state_dict().items():
        weighted_tensor = sum(
            model.state_dict()[name].detach().cpu() * weight
            for model, weight in zip(models, weights)
        ) / total_weight
        averaged_state[name] = weighted_tensor.to(dtype=reference.dtype)
    averaged_model.load_state_dict(averaged_state, strict=True)
    return averaged_model


def test_autosplit_strategy_runs_full_flower_round_with_remote_workers() -> None:
    torch.manual_seed(13)
    base_model = DeepNet()
    strategy_model = copy.deepcopy(base_model)
    reference_model = copy.deepcopy(base_model)
    x = torch.randn(6, 4)
    targets = torch.randn(6, 2)

    worker_a = start_worker(
        worker_id="round-worker-a",
        model=strategy_model,
        sample_inputs=(x,),
        server_address="127.0.0.1:0",
        register_with_registry=False,
    )
    worker_b = start_worker(
        worker_id="round-worker-b",
        model=strategy_model,
        sample_inputs=(x,),
        server_address="127.0.0.1:0",
        register_with_registry=False,
    )

    try:
        strategy = AutoSplitStrategy(
            model=strategy_model,
            sample_inputs=x,
            worker_specs=[worker_a.worker_spec, worker_b.worker_spec],
            loss_fn=nn.MSELoss(),
            optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )

        client_manager = SimpleClientManager()
        server, config = init_defaults(
            server=None,
            config=ServerConfig(num_rounds=1),
            strategy=strategy,
            client_manager=client_manager,
        )

        client = AutoSplitNumPyClient(
            train_data=[(x, targets)],
            evaluate_data=[(x, targets)],
        )
        client_manager.register(
            InProcessAutoSplitClientProxy(
                cid="client-1",
                numpy_client=client,
                server_model_manager=server.server_model_manager,
                strategy=strategy,
            )
        )

        history, _ = server.fit(num_rounds=config.num_rounds, timeout=30.0)

        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = nn.MSELoss()(reference_model(x), targets)
        reference_loss.backward()
        reference_optimizer.step()

        trained_model = DeepNet()
        _load_model_from_ndarrays(trained_model, server.server_parameters)

        for trained_param, reference_param in zip(
            trained_model.parameters(),
            reference_model.parameters(),
        ):
            assert torch.allclose(
                trained_param,
                reference_param,
                atol=1e-6,
                rtol=1e-6,
            )

        assert history.losses_distributed
    finally:
        worker_a.stop()
        worker_b.stop()


def test_autosplit_strategy_runs_split_learning_round_with_client_prefix() -> None:
    torch.manual_seed(23)
    base_model = DeepNet()
    strategy_model = copy.deepcopy(base_model)
    reference_model = copy.deepcopy(base_model)
    x = torch.randn(6, 4)
    targets = torch.randn(6, 2)

    worker = start_worker(
        worker_id="split-tail-worker",
        model=strategy_model,
        sample_inputs=(x,),
        server_address="127.0.0.1:0",
        register_with_registry=False,
    )

    try:
        strategy = AutoSplitStrategy(
            model=strategy_model,
            sample_inputs=x,
            worker_specs=[worker.worker_spec],
            preferred_stage_count=2,
            client_stage_count=1,
            loss_fn=nn.MSELoss(),
            optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )

        client_manager = SimpleClientManager()
        server, config = init_defaults(
            server=None,
            config=ServerConfig(num_rounds=1),
            strategy=strategy,
            client_manager=client_manager,
        )

        client = AutoSplitSplitLearningClient(
            model=base_model,
            train_data=[(x, targets)],
            evaluate_data=[(x, targets)],
            sample_inputs=x,
            optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        )
        client_manager.register(
            InProcessAutoSplitClientProxy(
                cid="split-client-1",
                numpy_client=client,
                server_model_manager=server.server_model_manager,
                strategy=strategy,
            )
        )

        history, _ = server.fit(num_rounds=config.num_rounds, timeout=30.0)

        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = nn.MSELoss()(reference_model(x), targets)
        reference_loss.backward()
        reference_optimizer.step()

        merged_model = _merge_split_states(
            base_model,
            parameters_to_ndarrays(server.client_parameters),
            server.server_parameters,
        )

        for merged_param, reference_param in zip(
            merged_model.parameters(),
            reference_model.parameters(),
        ):
            assert torch.allclose(
                merged_param,
                reference_param,
                atol=1e-6,
                rtol=1e-6,
            )

        assert history.losses_distributed
    finally:
        worker.stop()


def test_autosplit_strategy_runs_split_learning_round_with_multi_stage_client_prefix() -> None:
    torch.manual_seed(31)
    base_model = DeepNet()
    strategy_model = copy.deepcopy(base_model)
    x = torch.randn(6, 4)
    targets = torch.randn(6, 2)

    worker = start_worker(
        worker_id="split-multi-prefix-worker",
        model=strategy_model,
        sample_inputs=(x,),
        server_address="127.0.0.1:0",
        register_with_registry=False,
    )

    try:
        strategy = AutoSplitStrategy(
            model=strategy_model,
            sample_inputs=x,
            worker_specs=[worker.worker_spec],
            preferred_stage_count=3,
            client_stage_count=2,
            loss_fn=nn.MSELoss(),
            optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )

        client_manager = SimpleClientManager()
        server, config = init_defaults(
            server=None,
            config=ServerConfig(num_rounds=1),
            strategy=strategy,
            client_manager=client_manager,
        )

        client = AutoSplitSplitLearningClient(
            model=base_model,
            train_data=[(x, targets)],
            evaluate_data=[(x, targets)],
            sample_inputs=x,
            optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        )
        client_manager.register(
            InProcessAutoSplitClientProxy(
                cid="split-client-2-local-stages",
                numpy_client=client,
                server_model_manager=server.server_model_manager,
                strategy=strategy,
            )
        )

        history, _ = server.fit(num_rounds=config.num_rounds, timeout=30.0)

        reference_model = _run_single_batch_reference_step(base_model, x, targets)
        merged_model = _merge_split_states(
            base_model,
            parameters_to_ndarrays(server.client_parameters),
            server.server_parameters,
        )

        for merged_param, reference_param in zip(
            merged_model.parameters(),
            reference_model.parameters(),
        ):
            assert torch.allclose(
                merged_param,
                reference_param,
                atol=1e-6,
                rtol=1e-6,
            )

        assert history.losses_distributed
    finally:
        worker.stop()


def test_autosplit_strategy_runs_splitfed_round_with_per_client_tails() -> None:
    torch.manual_seed(41)
    base_model = DeepNet()
    strategy_model = copy.deepcopy(base_model)
    x_a = torch.randn(4, 4)
    y_a = torch.randn(4, 2)
    x_b = torch.randn(2, 4)
    y_b = torch.randn(2, 2)

    worker = start_worker(
        worker_id="splitfed-tail-worker",
        model=strategy_model,
        sample_inputs=(x_a,),
        server_address="127.0.0.1:0",
        register_with_registry=False,
    )

    try:
        strategy = AutoSplitStrategy(
            model=strategy_model,
            sample_inputs=x_a,
            worker_specs=[worker.worker_spec],
            preferred_stage_count=2,
            client_stage_count=1,
            aggregation_policy="splitfed",
            loss_fn=nn.MSELoss(),
            optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
        )

        client_manager = SimpleClientManager()
        server, config = init_defaults(
            server=None,
            config=ServerConfig(num_rounds=1),
            strategy=strategy,
            client_manager=client_manager,
        )

        client_manager.register(
            InProcessAutoSplitClientProxy(
                cid="splitfed-client-a",
                numpy_client=AutoSplitSplitLearningClient(
                    model=base_model,
                    train_data=[(x_a, y_a)],
                    evaluate_data=[(x_a, y_a)],
                    sample_inputs=x_a,
                    optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
                ),
                server_model_manager=server.server_model_manager,
                strategy=strategy,
            )
        )
        client_manager.register(
            InProcessAutoSplitClientProxy(
                cid="splitfed-client-b",
                numpy_client=AutoSplitSplitLearningClient(
                    model=base_model,
                    train_data=[(x_b, y_b)],
                    evaluate_data=[(x_b, y_b)],
                    sample_inputs=x_b,
                    optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
                ),
                server_model_manager=server.server_model_manager,
                strategy=strategy,
            )
        )

        history, _ = server.fit(num_rounds=config.num_rounds, timeout=30.0)

        reference_model = _average_reference_models(
            [
                _run_single_batch_reference_step(base_model, x_a, y_a),
                _run_single_batch_reference_step(base_model, x_b, y_b),
            ],
            [len(x_a), len(x_b)],
        )
        merged_model = _merge_split_states(
            base_model,
            parameters_to_ndarrays(server.client_parameters),
            server.server_parameters,
        )

        for merged_param, reference_param in zip(
            merged_model.parameters(),
            reference_model.parameters(),
        ):
            assert torch.allclose(
                merged_param,
                reference_param,
                atol=1e-6,
                rtol=1e-6,
            )

        assert strategy.replica_scope_policy.resolve() == ReplicaScope.PER_CLIENT
        assert strategy.common_server_model is False
        assert history.losses_distributed
    finally:
        worker.stop()
