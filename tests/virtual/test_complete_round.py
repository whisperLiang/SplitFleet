"""Convergence runs stop before aggregation when a selected client fails."""

from types import SimpleNamespace

import pytest

from splitfleet.server.strategy import AutoSplitStrategy
from splitfleet.virtual.server_adapter import VirtualClientFailure, VirtualStrategy


@pytest.mark.parametrize("successful_clients", [0, 1])
def test_complete_round_requirement_keeps_failed_updates_out_of_aggregation(monkeypatch, successful_clients):
    def aggregate(*args, **kwargs):
        pytest.fail("An incomplete round reached model aggregation")

    monkeypatch.setattr(AutoSplitStrategy, "aggregate_fit", aggregate)
    strategy = VirtualStrategy.__new__(VirtualStrategy)
    strategy.spec = SimpleNamespace(frozen_partitions=None, trainer=None)
    strategy.virtual_config = SimpleNamespace(workload={"require_complete_round": True})
    strategy.round_clients = {1: {"selected": ["0", "1"]}}
    strategy.client_properties = {
        str(i): {"virtual_partition_id": i, "logical_client_id": f"virtual-{i}"}
        for i in range(2)
    }
    strategy.observed_startup_ms, strategy.observed_prepare_ms = {}, {}
    strategy.actual_task_start_order = {}
    records = []
    strategy.evidence = SimpleNamespace(append=lambda name, row: records.append(row))
    results = [(SimpleNamespace(cid=str(i)), SimpleNamespace(num_examples=1, metrics={}))
               for i in range(successful_clients)]
    failures = [VirtualClientFailure(str(i), "Nonfinite loss")
                for i in range(successful_clients, 2)]
    with pytest.raises(RuntimeError, match="round is incomplete"):
        strategy.aggregate_fit(1, results, failures)
    assert len(records) == 2
    assert sum(row["status"] == "failure" for row in records) == 2 - successful_clients
    assert all(row["reason"] == "Nonfinite loss" for row in records if row["status"] == "failure")
