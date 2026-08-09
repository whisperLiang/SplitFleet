"""Cross-device, cross-batch split federated learning behaviour."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from torch import nn

from splitfleet.autosplit.batch_window import BatchWindowError, normalize_batch_window
from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.common import ServerModelFitIns, ServerModelFitRes
from splitfleet.common.constants import (
    AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY,
    AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY,
    AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY,
)
from splitfleet.server.placement import CapabilityAwarePlacementPolicy, CapabilityPlacementConfig
from splitfleet.server.server_model.autosplit_tail_server_model import AutoSplitTailServerModel
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


class InProcessServerModelProxy(ServerModelProxy):
    def __init__(self, *, server_model, cid: str = "client-1") -> None:
        super().__init__(cid=cid)
        self.server_model = server_model

    def _blocking_request(self, method, batch_data, _streams_, _timeout_):
        return getattr(self.server_model, method)([batch_data])[0]

    def _streaming_request(self, method, batch_data, _timeout_, _streams_):
        raise NotImplementedError

    def _nonblocking_request(self, method, batch_data, _timeout_, _streams_):
        raise NotImplementedError

    def close_stream(self):
        return None

    def get_pending_batches_count(self) -> int:
        return 0


def _ndarrays(model: nn.Module) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy() for tensor in model.state_dict().values()]


class _Fleet:
    """One strategy, one suffix replica, and one prefix device."""

    def __init__(
        self,
        *,
        dynamic_batch=None,
        client_sample_batch: int = 2,
        partial_batch_policy: str = "error",
        train_batches=(3,),
    ) -> None:
        torch.manual_seed(23)
        self.client_model = DeepNet()
        self.strategy_model = copy.deepcopy(self.client_model)
        self.strategy = AutoSplitStrategy(
            model=self.strategy_model,
            sample_inputs=torch.randn(2, 4),
            boundary="50%",
            dynamic_batch=dynamic_batch,
            loss_fn=nn.MSELoss(),
            optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.05),
        )
        self.manager = StageRuntimeManager(autosplit_session=self.strategy.autosplit_session)
        self.strategy.bind_stage_runtime_manager(self.manager)
        self.manager.set_placement_plan(self.strategy.get_or_create_placement_plan())
        self.config = self.strategy._autosplit_config()
        self.server_model = AutoSplitTailServerModel(
            runtime_manager=self.manager,
            model=self.strategy_model,
            optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.05),
            loss_fn=nn.MSELoss(),
        )
        self.server_model.configure_fit(
            ServerModelFitIns(
                parameters=_ndarrays(self.strategy_model),
                config=self.config,
                sid="",
            )
        )
        self.client = AutoSplitSplitLearningClient(
            model=self.client_model,
            train_data=[(torch.randn(rows, 4), torch.randn(rows, 2)) for rows in train_batches],
            sample_inputs=torch.randn(client_sample_batch, 4),
            optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.05),
            partial_batch_policy=partial_batch_policy,
        )
        self.client.server_model_proxy = InProcessServerModelProxy(server_model=self.server_model)

    def fit(self, config=None):
        return self.client.fit(_ndarrays(self.client_model), config or self.config)


def test_strategy_broadcasts_the_negotiated_batch_window() -> None:
    fleet = _Fleet(dynamic_batch=(1, 128))

    assert normalize_batch_window(fleet.config[AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY]) == (1, 128)
    assert fleet.config[AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY] == "batch_gt1"
    assert fleet.strategy.get_or_create_placement_plan().dynamic_batch == (1, 128)


def test_device_prefix_adopts_the_broadcast_window_over_its_own_sample_batch() -> None:
    # The device's local sample batch (5) differs from the strategy's (2); the
    # negotiated window, not the local sample, must decide what the prefix accepts.
    fleet = _Fleet(dynamic_batch=(1, 64), client_sample_batch=5, train_batches=(3,))
    fleet.fit()

    prepared = list(fleet.client._runtime_cache.values())
    assert [item.batch_window for item in prepared] == [(1, 64)]
    assert prepared[0].handle.plan.dynamic_batch == (1, 64)
    assert prepared[0].handle.plan.feature_abi_id == fleet.config[AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY]


def test_heterogeneous_batch_sizes_train_in_one_round() -> None:
    fleet = _Fleet(dynamic_batch=(1, 64), train_batches=(8, 3, 1))
    _, num_examples, metrics = fleet.fit()

    assert num_examples == 12
    assert metrics["num_batches"] == 3
    assert metrics["skipped_batches"] == 0
    assert (metrics["min_batch_size"], metrics["max_batch_size"]) == (1, 8)
    assert metrics["upload_bytes"] > 0
    assert metrics["tail_wait_sec"] >= 0.0
    assert fleet.server_model.get_fit_result().config["num_examples"] == 12


def test_batch_outside_the_window_names_the_window_and_the_fix() -> None:
    fleet = _Fleet(dynamic_batch=(2, 8), train_batches=(8, 1))

    with pytest.raises(BatchWindowError, match=r"batch size 1.*\[2, 8\]"):
        fleet.fit()


def test_skip_policy_reports_the_dropped_partial_batch() -> None:
    fleet = _Fleet(dynamic_batch=(2, 8), train_batches=(8, 1), partial_batch_policy="skip")
    _, num_examples, metrics = fleet.fit()

    assert num_examples == 8
    assert metrics["skipped_batches"] == 1
    assert metrics["skipped_examples"] == 1
    assert metrics["num_batches"] == 1


def test_client_rejects_an_unknown_partial_batch_policy() -> None:
    with pytest.raises(ValueError, match="partial_batch_policy"):
        AutoSplitSplitLearningClient(
            model=DeepNet(),
            train_data=[],
            sample_inputs=torch.randn(2, 4),
            partial_batch_policy="pad",
        )


def test_prefix_refuses_a_round_whose_feature_abi_does_not_match() -> None:
    fleet = _Fleet(dynamic_batch=(2, 16), train_batches=(4,))
    config = dict(fleet.config)
    config[AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY] = "0" * 40

    with pytest.raises(RuntimeError, match="feature ABI"):
        fleet.fit(config)


def test_suffix_rejects_an_upload_outside_the_window() -> None:
    from splitfleet.transport.split_wire import boundary_to_envelope
    from splitfleet.split_engine import graph_contract_for_runtime_handle

    fleet = _Fleet(dynamic_batch=(2, 8), train_batches=(4,))
    handle = fleet.server_model.runtime_handle
    contract = graph_contract_for_runtime_handle(handle)
    payload = handle.backend.run_prefix(torch.randn(4, 4), training=True)
    envelope = boundary_to_envelope(
        payload,
        round_id=0,
        client_id="device",
        step_id="step",
        plan_id=fleet.server_model.plan_id,
        split_id=contract.split_id,
        canonical_graph_hash=contract.canonical_graph_hash,
        boundary_schema_hash=contract.boundary_schema_hash,
        model_version=0,
    )

    fleet.server_model._validate_wire_identity(envelope)
    with pytest.raises(BatchWindowError, match=r"Split suffix.*batch size 4096"):
        fleet.server_model._validate_wire_identity(
            dataclasses.replace(envelope, batch_size=4096)
        )


def test_capability_placement_drives_per_client_boundaries_through_the_strategy() -> None:
    policy = CapabilityAwarePlacementPolicy(
        boundary_ladder=["after:fc1", "after:fc2"],
        start_index=1,
        config=CapabilityPlacementConfig(warmup_rounds=0, min_rounds_between_switches=0),
    )
    strategy = AutoSplitStrategy(
        model=DeepNet(),
        sample_inputs=torch.randn(2, 4),
        client_placement_fn=policy,
        loss_fn=nn.MSELoss(),
    )
    manager = StageRuntimeManager(autosplit_session=strategy.autosplit_session)
    strategy.bind_stage_runtime_manager(manager)

    # Placements are keyed by the requested boundary; the plan itself carries the
    # canonical TorchLens label for that cut.
    assert strategy._placement_for_client(1, "slow", training=True) is (
        strategy._placement_plans["after:fc2"]
    )
    strategy.aggregate_fit(
        1,
        [
            (
                _FakeClientProxy(cid),
                FitRes(
                    status=Status(Code.OK, ""),
                    parameters=ndarrays_to_parameters([np.zeros(1)]),
                    num_examples=8,
                    metrics={"fit_duration_sec": duration},
                ),
            )
            for cid, duration in (("slow", 10.0), ("fast", 1.0))
        ],
        [],
    )

    slow_plan = strategy._placement_for_client(2, "slow", training=True)
    fast_plan = strategy._placement_for_client(2, "fast", training=True)
    assert slow_plan is strategy._placement_plans["after:fc1"]
    assert fast_plan is strategy._placement_plans["after:fc2"]
    assert slow_plan.boundary != fast_plan.boundary
    # Round 1 entries are dropped once round 2 is planned.
    assert all(key[0] == 2 for key in strategy._client_placement_cache)


class _FakeClientProxy:
    def __init__(self, cid: str) -> None:
        self.cid = cid


def test_missing_suffix_replica_does_not_discard_the_whole_round() -> None:
    model = DeepNet().eval()
    strategy = AutoSplitStrategy(
        model=model,
        sample_inputs=torch.randn(2, 4),
        aggregation_policy="splitfed",
    )
    initial = [
        np.array(value, copy=True)
        for value in strategy.backend_adapter.export_ndarrays(model)
    ]
    strategy._round_initial_client_states[1] = [np.array(v, copy=True) for v in initial]
    strategy._round_initial_server_states[1] = [np.array(v, copy=True) for v in initial]

    client_results = []
    for cid in ("survivor", "lost"):
        client_state = [np.array(value, copy=True) for value in initial]
        client_state[0] += 1.0
        client_results.append(
            (
                _FakeClientProxy(cid),
                FitRes(
                    status=Status(Code.OK, ""),
                    parameters=ndarrays_to_parameters(client_state),
                    num_examples=4,
                    metrics={},
                ),
            )
        )
    server_state = [np.array(value, copy=True) for value in initial]
    server_state[-1] += 2.0
    survivor_result = ServerModelFitRes(parameters=server_state, config={"num_examples": 4})
    survivor_result.sid = "survivor"

    client_parameters, _ = strategy.aggregate_fit(1, client_results, [])
    aggregated = strategy.aggregate_server_fit(1, [survivor_result])
    finalized, _ = strategy.finalize_round(1, client_parameters, aggregated)

    assert np.allclose(aggregated[0], initial[0] + 1.0)
    assert np.allclose(aggregated[-1], initial[-1] + 2.0)
    assert all(
        np.array_equal(client_value, server_value)
        for client_value, server_value in zip(
            parameters_to_ndarrays(finalized), aggregated
        )
    )


def test_per_client_aggregation_still_fails_when_no_suffix_result_survives() -> None:
    model = DeepNet().eval()
    strategy = AutoSplitStrategy(
        model=model,
        sample_inputs=torch.randn(2, 4),
        aggregation_policy="splitfed",
    )
    initial = [
        np.array(value, copy=True)
        for value in strategy.backend_adapter.export_ndarrays(model)
    ]
    strategy._round_initial_client_states[1] = [np.array(v, copy=True) for v in initial]
    strategy._round_initial_server_states[1] = [np.array(v, copy=True) for v in initial]
    orphan = ServerModelFitRes(parameters=list(initial), config={"num_examples": 4})
    orphan.sid = "someone-else"

    strategy.aggregate_fit(
        1,
        [
            (
                _FakeClientProxy("device"),
                FitRes(
                    status=Status(Code.OK, ""),
                    parameters=ndarrays_to_parameters(list(initial)),
                    num_examples=4,
                    metrics={},
                ),
            )
        ],
        [],
    )

    with pytest.raises(RuntimeError, match="matching per-client suffix result"):
        strategy.aggregate_server_fit(1, [orphan])
