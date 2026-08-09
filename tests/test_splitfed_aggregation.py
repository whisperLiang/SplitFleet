"""SplitFed round bookkeeping: failed rounds, tied weights, and state lifetime."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from torch import nn

from splitfleet.backends import TorchBackendAdapter
from splitfleet.common import ServerModelFitRes
from splitfleet.server.strategy import AutoSplitStrategy
from splitfleet.server.strategy.autosplit_strategy import _reassemble_named_state


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(x)))


class TiedNet(nn.Module):
    """A prefix and a suffix layer that share one weight tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4, bias=False)
        self.decoder = nn.Linear(4, 4, bias=False)
        self.decoder.weight = self.encoder.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.relu(self.encoder(x)))


class FakeClientProxy:
    def __init__(self, cid: str) -> None:
        self.cid = cid


class FakeClientManager:
    def __init__(self, cids: list[str]) -> None:
        self.clients = [FakeClientProxy(cid) for cid in cids]

    def num_available(self) -> int:
        return len(self.clients)

    def sample(self, num_clients: int, min_num_clients: int | None = None):
        return self.clients[:num_clients]


def _splitfed_strategy(model: nn.Module) -> AutoSplitStrategy:
    return AutoSplitStrategy(
        model=model,
        sample_inputs=torch.randn(2, 4),
        aggregation_policy="splitfed",
    )


def _fit_res(state: list[np.ndarray], cid: str, num_examples: int = 4):
    return (
        FakeClientProxy(cid),
        FitRes(
            status=Status(Code.OK, ""),
            parameters=ndarrays_to_parameters(state),
            num_examples=num_examples,
            metrics={},
        ),
    )


def _server_res(state: list[np.ndarray], sid: str, num_examples: int = 4):
    result = ServerModelFitRes(parameters=state, config={"num_examples": num_examples})
    result.sid = sid
    return result


def _seed_round(strategy: AutoSplitStrategy, round_id: int, initial: list[np.ndarray]) -> None:
    strategy._round_initial_client_states[round_id] = [np.array(v, copy=True) for v in initial]
    strategy._round_initial_server_states[round_id] = [np.array(v, copy=True) for v in initial]


def test_a_round_where_every_client_failed_keeps_the_run_alive() -> None:
    model = TinyNet().eval()
    strategy = _splitfed_strategy(model)
    initial = [np.array(v, copy=True) for v in strategy.backend_adapter.export_ndarrays(model)]
    _seed_round(strategy, 1, initial)

    # Every selected client failed, so `aggregate_fit` has no results at all
    # while the suffix replicas the round created still report back.
    client_parameters, metrics = strategy.aggregate_fit(1, [], [RuntimeError("client down")])
    orphan_suffix = _server_res(list(initial), "device-a")
    aggregated = strategy.aggregate_server_fit(1, [orphan_suffix])
    finalized, server_final = strategy.finalize_round(1, client_parameters, aggregated)

    assert client_parameters is None and metrics == {}
    assert aggregated is None
    # `None` on both sides means the server keeps the previous global model.
    assert finalized is None and server_final is None
    assert strategy._pending_client_updates == {}
    assert strategy._reassembled_client_states == {}
    assert strategy._round_initial_client_states == {}


def test_out_of_order_suffix_aggregation_is_still_rejected() -> None:
    model = TinyNet().eval()
    strategy = _splitfed_strategy(model)
    initial = [np.array(v, copy=True) for v in strategy.backend_adapter.export_ndarrays(model)]
    _seed_round(strategy, 1, initial)

    with pytest.raises(RuntimeError, match="requires matching client updates first"):
        strategy.aggregate_server_fit(1, [_server_res(list(initial), "device-a")])


def test_round_state_does_not_accumulate_across_rounds() -> None:
    model = TinyNet().eval()
    strategy = _splitfed_strategy(model)
    initial = strategy.backend_adapter.export_ndarrays(model)
    manager = FakeClientManager(["a"])

    strategy.configure_fit(1, ndarrays_to_parameters(initial), manager)
    strategy.configure_server_fit(1, initial, ["a"])
    assert set(strategy._round_initial_client_states) == {1}

    # Round 1 never reaches aggregation (for example the round raised); starting
    # round 2 must not keep round 1's full model copies alive.
    strategy.configure_fit(2, ndarrays_to_parameters(initial), manager)
    strategy.configure_server_fit(2, initial, ["a"])

    assert set(strategy._round_initial_client_states) == {2}
    assert set(strategy._round_initial_server_states) == {2}


def test_torch_adapter_reports_tied_state_positions() -> None:
    adapter = TorchBackendAdapter()

    tied = adapter.tied_state_groups(TiedNet())
    untied = adapter.tied_state_groups(TinyNet())

    names = [entry.name for entry in adapter.state_manifest(TiedNet()).entries]
    assert untied == ()
    assert len(tied) == 1
    assert sorted(names[index] for index in tied[0]) == ["decoder.weight", "encoder.weight"]


def test_tied_weight_updates_from_both_stages_are_summed() -> None:
    names = ["encoder.weight", "decoder.weight", "head.bias"]
    initial = [np.zeros(2), np.zeros(2), np.zeros(2)]
    # The shared tensor is exported under both names, so a prefix update shows up
    # in both client entries and a suffix update in both server entries.
    client_state = [np.full(2, 1.0), np.full(2, 1.0), np.zeros(2)]
    server_state = [np.full(2, 3.0), np.full(2, 3.0), np.full(2, 5.0)]

    merged = _reassemble_named_state(
        names,
        initial,
        initial,
        client_state,
        server_state,
        tied_groups=((0, 1),),
        tied_weight_update_mode="additive_sgd",
    )

    assert np.allclose(merged[0], 4.0)
    assert np.allclose(merged[1], 4.0)
    assert np.allclose(merged[2], 5.0)


def test_tied_weight_updates_from_both_stages_require_explicit_sgd_mode() -> None:
    with pytest.raises(RuntimeError, match="identical stateless SGD"):
        _reassemble_named_state(
            ["encoder.weight", "decoder.weight"],
            [np.zeros(2), np.zeros(2)],
            [np.zeros(2), np.zeros(2)],
            [np.ones(2), np.ones(2)],
            [np.full(2, 2.0), np.full(2, 2.0)],
            tied_groups=((0, 1),),
        )


def test_a_skipped_finalize_cannot_publish_a_half_stale_model() -> None:
    """`aggregate_fit` alone must never yield a usable per-client model."""

    model = TinyNet().eval()
    strategy = _splitfed_strategy(model)
    initial = [np.array(v, copy=True) for v in strategy.backend_adapter.export_ndarrays(model)]
    _seed_round(strategy, 1, initial)
    client_state = [np.array(v, copy=True) for v in initial]
    client_state[0] = initial[0] + 1.0

    client_parameters, metrics = strategy.aggregate_fit(
        1, [_fit_res(client_state, "device-a")], []
    )

    assert client_parameters is None
    assert metrics == {}
    # A caller that never reaches `finalize_round` keeps the previous model
    # instead of adopting one whose suffix half is a round behind.
    assert strategy.finalize_round(1, client_parameters, None) == (None, None)


def test_shared_replica_scope_still_aggregates_in_one_call() -> None:
    model = TinyNet().eval()
    strategy = AutoSplitStrategy(model=model, sample_inputs=torch.randn(2, 4))
    initial = [np.array(v, copy=True) for v in strategy.backend_adapter.export_ndarrays(model)]
    updated = [np.array(v, copy=True) for v in initial]
    updated[0] = initial[0] + 2.0

    client_parameters, _ = strategy.aggregate_fit(1, [_fit_res(updated, "device-a")], [])
    finalized, server_parameters = strategy.finalize_round(1, client_parameters, ["unchanged"])

    assert finalized is client_parameters
    assert server_parameters == ["unchanged"]
    assert np.allclose(parameters_to_ndarrays(finalized)[0], initial[0] + 2.0)


def test_untied_parameters_still_reject_ambiguous_ownership() -> None:
    names = ["shared.weight"]
    initial = [np.zeros(2)]

    with pytest.raises(RuntimeError, match="ownership is ambiguous"):
        _reassemble_named_state(
            names, initial, initial, [np.full(2, 1.0)], [np.full(2, 3.0)]
        )


def test_splitfed_reassembly_survives_a_weight_tied_model() -> None:
    model = TiedNet().eval()
    strategy = AutoSplitStrategy(
        model=model,
        sample_inputs=torch.randn(2, 4),
        aggregation_policy="splitfed",
        tied_weight_update_mode="additive_sgd",
    )
    initial = [np.array(v, copy=True) for v in strategy.backend_adapter.export_ndarrays(model)]
    _seed_round(strategy, 1, initial)
    names = [entry.name for entry in strategy.backend_adapter.state_manifest(model).entries]
    tied = [index for index, name in enumerate(names) if "weight" in name]

    client_state = [np.array(v, copy=True) for v in initial]
    server_state = [np.array(v, copy=True) for v in initial]
    for index in tied:
        client_state[index] = initial[index] + 1.0
        server_state[index] = initial[index] + 2.0

    client_parameters, _ = strategy.aggregate_fit(1, [_fit_res(client_state, "device-a")], [])
    aggregated = strategy.aggregate_server_fit(1, [_server_res(server_state, "device-a")])
    finalized, _ = strategy.finalize_round(1, client_parameters, aggregated)

    for index in tied:
        assert np.allclose(aggregated[index], initial[index] + 3.0)
    # The client-side model of a per-client round only exists after the suffix
    # halves are reassembled, so it is produced by `finalize_round`.
    assert client_parameters is None
    assert all(
        np.allclose(client_value, server_value)
        for client_value, server_value in zip(
            parameters_to_ndarrays(finalized), aggregated
        )
    )


def test_zero_example_client_update_is_not_aggregated() -> None:
    model = TinyNet().eval()
    strategy = AutoSplitStrategy(model=model, sample_inputs=torch.randn(2, 4))
    initial = [np.array(v, copy=True) for v in strategy.backend_adapter.export_ndarrays(model)]
    updated = [np.array(v, copy=True) for v in initial]
    updated[0] += 7.0

    aggregated, _ = strategy.aggregate_fit(
        1,
        [
            _fit_res(initial, "skipped", num_examples=0),
            _fit_res(updated, "trained", num_examples=4),
        ],
        [],
    )

    values = parameters_to_ndarrays(aggregated)
    assert np.array_equal(values[0], updated[0])


def test_all_zero_example_updates_make_an_empty_splitfed_round() -> None:
    model = TinyNet().eval()
    strategy = _splitfed_strategy(model)
    initial = [np.array(v, copy=True) for v in strategy.backend_adapter.export_ndarrays(model)]
    _seed_round(strategy, 1, initial)

    client_parameters, metrics = strategy.aggregate_fit(
        1,
        [_fit_res(initial, "skipped", num_examples=0)],
        [],
    )
    suffix = _server_res(initial, "skipped", num_examples=0)

    assert client_parameters is None
    assert metrics == {}
    assert strategy.aggregate_server_fit(1, [suffix]) is None
