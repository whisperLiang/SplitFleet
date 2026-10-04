"""Arrival permutations must not change the benchmark's arithmetic order."""

from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from experiments.physical_multitask import RecordingFedAvg, RecordingSplit, _ordered_fit_results
from splitfleet.server.strategy.autosplit_strategy import AutoSplitStrategy


def updates():
    # These values expose floating-point nonassociativity in a weighted mean.
    return [(SimpleNamespace(cid=f"client-{index}"), SimpleNamespace(
        parameters=ndarrays_to_parameters([np.array([value], dtype=np.float32)]),
        num_examples=1, metrics={"client_index": index},
    )) for index, value in enumerate([1e20, -1e20, 3.0])]


def test_native_aggregation_is_identical_for_all_arrival_orders():
    strategy = RecordingFedAvg(workload=None, model=torch.nn.Linear(1, 1), device="cpu")
    outputs = []
    for arrival in permutations(updates()):
        result, _ = strategy.aggregate_fit(1, list(arrival), [])
        outputs.append(parameters_to_ndarrays(result)[0])
        assert [row["metrics"]["client_index"] for row in strategy.fit_records[-3:]] == [0, 1, 2]
    assert all(np.array_equal(output, outputs[0]) for output in outputs)


def test_split_aggregation_receives_the_same_canonical_order(monkeypatch):
    received = []
    monkeypatch.setattr(AutoSplitStrategy, "aggregate_fit",
        lambda self, round_id, results, failures: received.append(
            [result.metrics["client_index"] for _, result in results]))
    strategy = RecordingSplit.__new__(RecordingSplit)
    strategy.fit_records, strategy.fit_failures = [], []
    for arrival in permutations(updates()):
        strategy.aggregate_fit(1, list(arrival), [])
    assert received == [[0, 1, 2]] * 6


def test_unknown_or_repeated_client_indices_are_rejected():
    rows = updates()
    with pytest.raises(ValueError, match="distinct nonnegative"):
        _ordered_fit_results([rows[0], rows[0]])
    rows[0][1].metrics["client_index"] = -1
    with pytest.raises(ValueError, match="distinct nonnegative"):
        _ordered_fit_results(rows)
    del rows[0][1].metrics["client_index"]
    with pytest.raises(KeyError, match="client_index"):
        _ordered_fit_results(rows)
