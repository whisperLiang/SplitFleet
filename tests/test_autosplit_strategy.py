from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from flwr.common import Code, EvaluateRes, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from torch import nn

from splitfleet.autosplit import ReplicaScope
from splitfleet.common import ServerModelFitRes
from splitfleet.common.constants import (
    AUTOSPLIT_BACKEND_CONFIG_KEY,
    AUTOSPLIT_BACKEND_VALUE_TORCHLENS,
    AUTOSPLIT_BOUNDARY_CONFIG_KEY,
    AUTOSPLIT_BOUNDARY_TENSOR_LABELS_CONFIG_KEY,
    AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY,
    AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY,
    AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY,
    AUTOSPLIT_MODE_CONFIG_KEY,
    AUTOSPLIT_PLAN_ID_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_BACKEND_VALUE_TORCHLENS_NATIVE,
    AUTOSPLIT_RUNTIME_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_CONTRACT_DIGEST_CONFIG_KEY,
    AUTOSPLIT_SPLIT_ID_CONFIG_KEY,
    AUTOSPLIT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_TORCHLENS_VERSION_CONFIG_KEY,
    AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY,
)
from splitfleet.autosplit.torchlens_contract import runtime_contract_digest
from splitfleet.server.server_model.server_model import ServerModel
from splitfleet.server.stage_runtime.manager import StageRuntimeManager
from splitfleet.server.strategy import AutoSplitStrategy
from splitfleet.server.placement import (
    CandidateEstimate,
    CoSplitUCBConfig,
    CoSplitUCBPlacementPolicy,
    SplitCandidateDescriptor,
    StaticCandidateProvider,
)


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
        self.clients = [FakeClientProxy(cid) for cid in cids]

    def num_available(self) -> int:
        return len(self.clients)

    def sample(self, num_clients: int, min_num_clients: int | None = None):
        assert min_num_clients is None or len(self.clients) >= min_num_clients
        return self.clients[:num_clients]


def test_successful_evaluation_reports_executed_cut_to_policy() -> None:
    class TrackingPolicy:
        def __init__(self) -> None:
            self.executions = None

        def observe_evaluation(self, *, round_id, executions) -> None:
            self.executions = (round_id, executions)

    policy = TrackingPolicy()
    strategy = AutoSplitStrategy(
        model=TinyNet(), sample_inputs=torch.randn(2, 4),
        placement_policy=policy, init_server_model_fn=lambda: DummyServerModel(),
    )
    strategy._round_placement_assignments[(1, False)] = {"a": "after:first"}
    result = EvaluateRes(
        status=Status(Code.OK, ""), loss=0.5, num_examples=2,
        metrics={"boundary": "after:first"},
    )
    strategy.aggregate_evaluate(1, [(FakeClientProxy("a"), result)], [])
    assert policy.executions == (1, {"a": "after:first"})


def test_autosplit_strategy_generates_torchlens_metadata() -> None:
    model = TinyNet().eval()
    sample_inputs = torch.randn(2, 4)
    strategy = AutoSplitStrategy(
        model=model,
        sample_inputs=sample_inputs,
        worker_specs=[],
        init_server_model_fn=lambda: DummyServerModel(),
    )

    config = strategy._autosplit_config()

    assert config[AUTOSPLIT_BACKEND_CONFIG_KEY] == AUTOSPLIT_BACKEND_VALUE_TORCHLENS
    assert config[AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY] == AUTOSPLIT_RUNTIME_BACKEND_VALUE_TORCHLENS_NATIVE
    assert config[AUTOSPLIT_TORCHLENS_VERSION_CONFIG_KEY] == "2.34.1"
    assert config[AUTOSPLIT_PLAN_ID_CONFIG_KEY].startswith("torchlens_")
    assert config[AUTOSPLIT_SPLIT_ID_CONFIG_KEY]
    assert config[AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY]
    assert config[AUTOSPLIT_BOUNDARY_CONFIG_KEY].startswith("after:")
    assert json.loads(config[AUTOSPLIT_BOUNDARY_TENSOR_LABELS_CONFIG_KEY])
    assert config[AUTOSPLIT_MODE_CONFIG_KEY] == "generated_eager"
    assert config[AUTOSPLIT_STAGE_COUNT_CONFIG_KEY] == 2
    assert config[AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY] == 1
    assert config[AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY]
    contract = json.loads(config[AUTOSPLIT_RUNTIME_CONTRACT_CONFIG_KEY])
    assert runtime_contract_digest(contract) == config[AUTOSPLIT_RUNTIME_CONTRACT_DIGEST_CONFIG_KEY]
    assert config[AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY] in {"batch_gt1", "batch_1"}
    assert json.loads(config[AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY]) is not None
    json.dumps(config, sort_keys=True)


def test_autosplit_strategy_splitfed_defaults_to_per_client_tail() -> None:
    model = TinyNet().eval()
    sample_inputs = torch.randn(2, 4)
    strategy = AutoSplitStrategy(
        model=model,
        sample_inputs=sample_inputs,
        worker_specs=[],
        aggregation_policy="splitfed",
        client_stage_count=1,
        init_server_model_fn=lambda: DummyServerModel(),
    )

    assert strategy.replica_scope_policy.resolve() == ReplicaScope.PER_CLIENT
    assert strategy.common_server_model is False
    assert strategy._autosplit_config()[AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY] == 1


def test_autosplit_strategy_rejects_splitfed_with_shared_replicas() -> None:
    model = TinyNet().eval()
    sample_inputs = torch.randn(2, 4)

    with pytest.raises(ValueError, match="SplitFed aggregation requires"):
        AutoSplitStrategy(
            model=model,
            sample_inputs=sample_inputs,
            worker_specs=[],
            aggregation_policy="splitfed",
            replica_scope_policy=ReplicaScope.SHARED,
            client_stage_count=1,
            init_server_model_fn=lambda: DummyServerModel(),
        )


def test_autosplit_strategy_rejects_non_two_stage_requests() -> None:
    with pytest.raises(ValueError, match="two-stage"):
        AutoSplitStrategy(
            model=TinyNet(),
            sample_inputs=torch.randn(2, 4),
            preferred_stage_count=3,
            init_server_model_fn=lambda: DummyServerModel(),
        )


def test_per_client_placement_routes_matching_client_and_tail_plans_across_rounds() -> None:
    model = TinyNet().eval()
    wall_times = []

    class PlacementPolicy:
        def plan_round(self, *, round_id, client_ids, training):
            _ = training
            return {
                cid: "after:first" if (round_id, cid) in {(1, "a"), (2, "b")} else "50%"
                for cid in client_ids
            }

        def observe_round(self, **kwargs):
            _ = kwargs

        def observe_failure(self, **kwargs):
            _ = kwargs

        def observe_round_wall_time(self, **kwargs):
            wall_times.append(kwargs)

    strategy = AutoSplitStrategy(
        model=model,
        sample_inputs=torch.randn(2, 4),
        worker_specs=[],
        placement_policy=PlacementPolicy(),
        min_fit_clients=2,
        min_available_clients=2,
        init_server_model_fn=lambda: DummyServerModel(),
    )
    manager = StageRuntimeManager(autosplit_session=strategy.autosplit_session)
    strategy.bind_stage_runtime_manager(manager)
    clients = FakeClientManager(["a", "b"])
    initial = strategy.backend_adapter.export_ndarrays(model)
    flower_parameters = ndarrays_to_parameters(initial)

    round_one = strategy.configure_fit(1, flower_parameters, clients)
    client_plans_one = {client.cid: ins.config[AUTOSPLIT_PLAN_ID_CONFIG_KEY] for client, ins in round_one}
    server_one = strategy.configure_server_fit(1, initial, ["a", "b"])
    server_plans_one = {ins.sid: ins.config[AUTOSPLIT_PLAN_ID_CONFIG_KEY] for ins in server_one}

    assert strategy.common_server_model is False
    assert client_plans_one == server_plans_one
    assert client_plans_one["a"] != client_plans_one["b"]
    assert all(manager.get_placement_plan(plan_id) is not None for plan_id in client_plans_one.values())

    round_two = strategy.configure_fit(2, flower_parameters, clients)
    client_plans_two = {client.cid: ins.config[AUTOSPLIT_PLAN_ID_CONFIG_KEY] for client, ins in round_two}
    assert client_plans_two["a"] == client_plans_one["b"]
    assert client_plans_two["b"] == client_plans_one["a"]
    strategy.aggregate_fit(2, [], [])
    assert len(wall_times) == 1
    assert wall_times[0]["round_id"] == 2
    assert wall_times[0]["duration_ms"] >= 0


def test_cosplit_global_solver_assignment_is_shared_by_client_and_server_configs() -> None:
    class LearnedComponents:
        def predict(self, *, client_id, boundary, **kwargs):
            _ = kwargs
            early = boundary == "after:first"
            return CandidateEstimate(
                client_id=client_id,
                boundary=boundary,
                client_forward_mean_ms=0 if early else 60,
                client_backward_mean_ms=0,
                network_upload_mean_ms=0,
                network_download_mean_ms=0,
                server_service_mean_ms=50 if early else 0,
                switch_mean_ms=0,
                client_forward_uncertainty_ms=0,
                client_backward_uncertainty_ms=0,
                network_upload_uncertainty_ms=0,
                network_download_uncertainty_ms=0,
                server_service_uncertainty_ms=0,
                switch_uncertainty_ms=0,
            )

    candidates = [
        SplitCandidateDescriptor(
            boundary=boundary,
            split_id=boundary,
            graph_position_ratio=position,
            prefix_node_count=index,
            suffix_node_count=3 - index,
            total_node_count=3,
            boundary_forward_bytes=64,
            boundary_gradient_bytes=64,
            boundary_tensor_count=1,
            prefix_parameter_bytes=None,
            suffix_parameter_bytes=None,
            client_memory_bytes=None,
            server_memory_bytes=None,
            trainable=True,
            feature_abi_id=boundary,
            graph_signature="tiny",
            framework_backend="pytorch",
        )
        for index, (boundary, position) in enumerate(
            (("after:first", 0.25), ("50%", 0.75)), start=1
        )
    ]
    learned = LearnedComponents()
    policy = CoSplitUCBPlacementPolicy(
        candidate_provider=StaticCandidateProvider(candidates),
        learners=learned,
        config=CoSplitUCBConfig(max_explorations_per_round=0, min_residence_rounds=0),
    )
    model = TinyNet().eval()
    strategy = AutoSplitStrategy(
        model=model,
        sample_inputs=torch.randn(2, 4),
        worker_specs=[],
        placement_policy=policy,
        min_fit_clients=2,
        min_available_clients=2,
        init_server_model_fn=lambda: DummyServerModel(),
    )
    manager = StageRuntimeManager(autosplit_session=strategy.autosplit_session)
    strategy.bind_stage_runtime_manager(manager)
    clients = FakeClientManager(["a", "b"])
    initial = strategy.backend_adapter.export_ndarrays(model)
    client_instructions = strategy.configure_fit(1, ndarrays_to_parameters(initial), clients)
    server_instructions = strategy.configure_server_fit(1, initial, ["a", "b"])
    client_boundaries = {
        client.cid: ins.config[AUTOSPLIT_BOUNDARY_CONFIG_KEY]
        for client, ins in client_instructions
    }
    server_boundaries = {
        ins.sid: ins.config[AUTOSPLIT_BOUNDARY_CONFIG_KEY]
        for ins in server_instructions
    }
    assert client_boundaries == server_boundaries
    assert len(set(client_boundaries.values())) == 2
    independent = {
        cid: learned.predict(client_id=cid, boundary="after:first")
        for cid in ("a", "b")
    }
    assert policy.round_diagnostics[1]["predicted_final_makespan_ms"] < (
        policy.solver.simulate(independent).max_client_completion_ms
    )


def test_splitfed_reassembles_named_prefix_and_suffix_updates_before_fedavg() -> None:
    model = TinyNet().eval()
    strategy = AutoSplitStrategy(
        model=model,
        sample_inputs=torch.randn(2, 4),
        aggregation_policy="splitfed",
        init_server_model_fn=lambda: DummyServerModel(),
    )
    initial = [np.array(value, copy=True) for value in strategy.backend_adapter.export_ndarrays(model)]
    strategy._round_initial_client_states[1] = [np.array(value, copy=True) for value in initial]
    strategy._round_initial_server_states[1] = [np.array(value, copy=True) for value in initial]

    client_results = []
    server_results = []
    for cid, weight, prefix_delta, suffix_delta in (
        ("a", 1, 1.0, 2.0),
        ("b", 3, 3.0, 4.0),
    ):
        client_state = [np.array(value, copy=True) for value in initial]
        server_state = [np.array(value, copy=True) for value in initial]
        client_state[0] += prefix_delta
        server_state[-1] += suffix_delta
        client_results.append(
            (
                FakeClientProxy(cid),
                FitRes(
                    status=Status(Code.OK, ""),
                    parameters=ndarrays_to_parameters(client_state),
                    num_examples=weight,
                    metrics={},
                ),
            )
        )
        result = ServerModelFitRes(
            parameters=server_state,
            config={"num_examples": weight},
        )
        result.sid = cid
        server_results.append(result)

    client_parameters, _ = strategy.aggregate_fit(1, client_results, [])
    server_parameters = strategy.aggregate_server_fit(1, server_results)
    # The client-side model of a per-client round is produced once both halves
    # have been aggregated; `aggregate_fit` alone cannot publish one.
    assert client_parameters is None
    client_parameters, _ = strategy.finalize_round(1, client_parameters, server_parameters)
    reassembled_client = parameters_to_ndarrays(client_parameters)

    assert all(
        np.array_equal(client_value, server_value)
        for client_value, server_value in zip(reassembled_client, server_parameters)
    )
    assert np.allclose(reassembled_client[0], initial[0] + 2.5)
    assert np.allclose(reassembled_client[-1], initial[-1] + 3.5)
    assert all(
        np.array_equal(reassembled_client[index], initial[index])
        for index in range(1, len(initial) - 1)
    )
