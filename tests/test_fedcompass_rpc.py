"""Real local Flower RPCs with tensor fixtures; no paper performance population."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import socket
import time
from types import SimpleNamespace

import pytest
import torch
from flwr.client import start_client
from flwr.common import Code, FitRes, GetPropertiesRes, Status
from flwr.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server, run_fl
from flwr.server.superlink.fleet.grpc_bidi.grpc_server import start_grpc_server

from experiments.baselines.fedcompass import FedCompassConfig
from experiments.common.fedcompass_server import FedCompassServer
from experiments.baselines.async_fl import AsyncFLConfig
from experiments.common.async_fl_server import AsyncFLServer
from experiments.common.identity import tensor_state_hash
from experiments.common.time_budget_server import with_time_budget
from experiments.orchestrate_physical_multitask import RoundBarrier, validate_result
from experiments.physical_multitask import FullClient, RecordingFedAvg
from experiments.unified_multitask.data import _build_workload
from experiments.analysis.communication import communication_summary


class ClassifierFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 10)
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))

    def forward(self, images):
        return self.fc(images.mean(dim=(-2, -1)))


def unused_address():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{s.getsockname()[1]}"


def fixture_population():
    torch.manual_seed(19)
    torch.set_num_threads(1)
    model = ClassifierFixture()
    # Load optimizer internals before starting the short timed RPC test.
    torch.optim.SGD(model.parameters(), lr=.01)
    images, labels = torch.randn(27, 3, 2, 2), torch.arange(27) % 10
    workload = _build_workload("image_classification", ClassifierFixture,
        torch.utils.data.TensorDataset(images[:15], labels[:15]),
        torch.utils.data.TensorDataset(images[15:], labels[15:]), None, source="unit")
    ids = ["fast-cpu", "medium-cpu", "slow-cpu"]
    bundle = {"task": "image_classification", "seed": 19, "initial_model_state": model.state_dict(),
              "initial_model_hash": tensor_state_hash(model.state_dict()), "partition_hash": "unit-partitions",
              "batch_size": 3, "train_size": 15, "test_size": 12, "worker_ids": ids,
              "partition_sizes": {str(i): 5 for i in range(3)},
              "assignments": {str(i): list(range(i * 5, (i + 1) * 5)) for i in range(3)},
              "data_content_hash": workload.data_content_hash}
    return model, workload, bundle


class DelayedClient(FullClient):
    def __init__(self, *, step_delay, **kwargs):
        super().__init__(**kwargs)
        self.step_delay = step_delay

    def fit(self, parameters, config):
        result = super().fit(parameters, config)
        time.sleep(self.step_delay * config.get("local_steps", 1))
        return result


@pytest.mark.parametrize("method,budget", [("fedcompass", None), ("fedcompass", 1.5),
    ("fedavg", .8), ("fedasync", 1.5), ("fedbuff", 1.5)])
def test_real_rpc_dynamic_work_readiness_budget_and_communication(method, budget):
    model, workload, bundle = fixture_population()
    ids = bundle["worker_ids"]
    async_run = method in ("fedcompass", "fedasync", "fedbuff")
    barrier = RoundBarrier(unused_address(), set(ids), 1 if async_run else 2**31 - 1)
    barrier_address = f"127.0.0.1:{barrier.listener.getsockname()[1]}"
    strategy = RecordingFedAvg(workload=workload, model=model, device="cpu", fraction_fit=1,
        fraction_evaluate=0, min_fit_clients=3, min_evaluate_clients=0, min_available_clients=3)
    strategy.partition_hash, strategy.model_initial_hash = bundle["partition_hash"], bundle["initial_model_hash"]
    manager = SimpleClientManager()
    if method in ("fedasync", "fedbuff"):
        server = AsyncFLServer(strategy=strategy, client_manager=manager, worker_ids=ids,
            partition_sizes=dict.fromkeys(ids, 5), config=AsyncFLConfig(method, local_steps=6),
            time_budget_sec=budget, evaluation_interval_sec=.15, rpc_timeout_sec=5)
    elif method == "fedcompass":
        server = FedCompassServer(strategy=strategy, client_manager=manager, worker_ids=ids,
            partition_sizes=dict.fromkeys(ids, 5), config=FedCompassConfig(min_local_steps=1, max_local_steps=6),
            time_budget_sec=budget, evaluation_interval_sec=.15, rpc_timeout_sec=5)
    else:
        server = with_time_budget(Server(strategy=strategy, client_manager=manager), budget, .15)
    address = unused_address()
    clients = [DelayedClient(workload=replace(workload,
        train_dataset=torch.utils.data.Subset(workload.train_dataset, bundle["assignments"][str(i)])),
        bundle=bundle, identity=cid, client_index=i, method=method, device="cpu", learning_rate=.01,
        barrier=barrier_address, step_delay=delay) for i, (cid, delay) in enumerate(zip(ids, [.005, .018, .04]))]
    grpc_server = start_grpc_server(client_manager=manager, server_address=address)
    barrier.start()
    pool = ThreadPoolExecutor(max_workers=3)
    connections = [pool.submit(start_client, server_address=address, client=c.to_client()) for c in clients]
    try:
        run_fl(server=server, config=ServerConfig(num_rounds=12, round_timeout=5))
        for future in connections:
            future.result(timeout=10)
    finally:
        server.disconnect_all_clients(timeout=5)
        grpc_server.stop(grace=0)
        barrier.stop()
        pool.shutdown(wait=True)
    metadata = server.result_metadata() if async_run else server.budget_metadata
    result = {**{k: v for k, v in bundle.items() if k != "initial_model_state"}, **metadata,
              "method": method, "expected_clients": 3, "fit_records": strategy.fit_records,
              "fit_failures": strategy.fit_failures, "server_fit_records": [],
              "evaluation_records": strategy.evaluation_records, "state_download_records": strategy.state_download_records,
              "final_model_hash": tensor_state_hash(model.state_dict()), "owned_state_exchange": False,
              "communication_accounting": "training_client_application_buffers_v1"}
    hosts = [{"id": name, "workers": ["cpu"]} for name in ("fast", "medium", "slow")]
    report = validate_result(result, task=bundle["task"], method=method, rounds=12, hosts=hosts,
                             batch_size=3, bundle=bundle, barrier_records=barrier.records)
    assert report["valid"], report["errors"]
    assert communication_summary(result)["complete"]
    assert metadata["global_updates"] > 0
    if async_run:
        assert len(barrier.records) == 1
        assert {fit["metrics"]["logical_client_id"] for fit in strategy.fit_records} == set(ids)
        assert all(fit["metrics"]["num_batches"] == fit["metrics"]["assigned_local_steps"]
                   for fit in strategy.fit_records)
        assert any(fit["num_examples"] > 5 for fit in strategy.fit_records)
        if budget is None:
            assert metadata["global_updates"] == 12
        else:
            assert metadata["stop_reason"] == "time_budget"
    if budget:
        assert metadata["last_committed_model_elapsed_sec"] < budget
        assert all(fit["aggregation_elapsed_sec"] < budget for fit in strategy.fit_records
                   if fit["update_status"] == "aggregated")
        assert any(fit["update_status"] in ("discarded_at_stop", "unaggregated_at_stop")
                   for fit in strategy.fit_records)
        assert strategy.evaluation_records[-1]["evaluation_kind"] == "final_frozen_model"


def test_incompatible_assignment_fails_the_run_and_preserves_failure(monkeypatch):
    import experiments.physical_multitask as physical

    model = ClassifierFixture()
    monkeypatch.setattr(physical, "_evaluate_physical", lambda *args, **kwargs: {"accuracy": .5})
    strategy = RecordingFedAvg(workload=object(), model=model, device="cpu")
    strategy.partition_hash = "partition"
    strategy.model_initial_hash = tensor_state_hash(model.state_dict())
    props = {"logical_client_id": "a", "client_index": 0, "partition_size": 5,
             "partition_hash": "partition", "initial_model_hash": strategy.model_initial_hash}

    def fit(ins, timeout, group_id):
        return FitRes(Status(Code.OK, ""), ins.parameters, 5,
                      {"logical_client_id": "a", "num_batches": 100, "model_version": 999,
                       "num_examples": 5, "partition_hash": "partition"})

    manager = SimpleClientManager()
    manager.register(SimpleNamespace(cid="connection-a", fit=fit,
        get_properties=lambda *args: GetPropertiesRes(Status(Code.OK, ""), props)))
    server = FedCompassServer(strategy=strategy, client_manager=manager, worker_ids=["a"],
        partition_sizes={"a": 5}, config=FedCompassConfig(), time_budget_sec=None,
        evaluation_interval_sec=30, rpc_timeout_sec=5)
    with pytest.raises(ValueError, match="assignment receipt"):
        server.fit(num_rounds=1, timeout=5)
    assert len(strategy.fit_failures) == 1
    assert server.controller.version == 0
    assert not server.controller.active
