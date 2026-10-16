import copy
import json

import pytest
import torch

from experiments.common.training import aggregate_named_states
from experiments.virtual_sfl.workloads import (
    TinyClassifier,
    fixture_dataset,
    fixture_spec,
)
from splitfleet.virtual import VirtualConfig, run_virtual
from splitfleet.virtual.config import ClientPopulation, Resources
from splitfleet.virtual.partitions import ClientBatches, partition_dataset
from splitfleet.virtual.state import rng_scope


def read_rows(path, filename):
    return [json.loads(row) for row in (path / filename).read_text().splitlines()]


def direct_reference(config, samples, *, exclude=None):
    dataset = fixture_dataset(samples, config.seed)
    parts = partition_dataset(
        samples, config.virtual_clients.total, config=config.partition, seed=config.seed
    )
    with rng_scope(seed=config.seed):
        global_model = TinyClassifier()
    for round_id in range(1, config.rounds + 1):
        states, counts = [], []
        for pid, indices in parts.items():
            if exclude and (round_id, int(pid)) in exclude:
                continue
            model = copy.deepcopy(global_model)
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
            count = 0
            source = ClientBatches(
                dataset,
                indices,
                batch_size=config.batch_size,
                seed=config.seed,
                partition_id=int(pid),
                round_id=round_id,
                local_epochs=config.local_epochs,
            )
            for inputs, labels in source:
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(model(inputs), labels).backward()
                optimizer.step()
                count += len(labels)
            states.append(
                {k: v.detach().clone() for k, v in model.state_dict().items()}
            )
            counts.append(count)
        global_model.load_state_dict(aggregate_named_states(states, counts))
    return global_model.state_dict()


@pytest.mark.simulation
def test_real_prefix_suffix_updates_match_direct_sgd_and_worker_reuse(tmp_path):
    pytest.importorskip("ray")
    config = VirtualConfig(
        rounds=2,
        batch_size=2,
        boundary="after:head",
        virtual_clients=ClientPopulation(total=2),
        resources=Resources(max_concurrent_clients=1, worker_count_policy="pooled"),
    )
    output = tmp_path / "real-training"
    summary = run_virtual(config, fixture_spec(samples=8), output=output)
    assert summary["optimization_steps"] == 8
    actual = torch.load(output / "final_model.pt", weights_only=True)
    expected = direct_reference(config, 8)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-7)
    validation = json.loads((output / "validation.json").read_text())
    assert validation["prefix_updates"] == validation["suffix_updates"] == 4
    rows = [
        r for r in read_rows(output, "client_metrics.jsonl") if r["status"] == "success"
    ]
    assert len({r["metrics"]["worker_pid"] for r in rows}) == 1
    for pid in (0, 1):
        client_rows = sorted(
            [r for r in rows if r["partition_id"] == pid], key=lambda r: r["round_id"]
        )
        assert [r["metrics"]["model_version"] for r in client_rows] == [1, 2]
        assert [r["metrics"]["completed_batches"] for r in client_rows] == [2, 4]
        assert len({r["metrics"]["partition_hash"] for r in client_rows}) == 1
    with pytest.raises(FileExistsError):
        run_virtual(config, fixture_spec(samples=8), output=output)


@pytest.mark.simulation
def test_failed_partial_suffix_is_discarded_and_client_recovers(tmp_path):
    pytest.importorskip("ray")
    config = VirtualConfig(
        rounds=2,
        batch_size=2,
        boundary="after:head",
        virtual_clients=ClientPopulation(
            total=2, failure_rounds={"0": [1]}, fail_after_batches=1
        ),
        resources=Resources(max_concurrent_clients=1, worker_count_policy="pooled"),
    )
    output = tmp_path / "recover"
    summary = run_virtual(config, fixture_spec(samples=8), output=output)
    assert summary["training_tasks"] == 3
    rows = read_rows(output, "client_metrics.jsonl")
    assert any(
        r["status"] == "failure" and r["partition_id"] == 0 and r["round_id"] == 1
        for r in rows
    )
    assert not any(
        r["status"] == "suffix_committed"
        and r["partition_id"] == 0
        and r["round_id"] == 1
        for r in rows
    )
    actual = torch.load(output / "final_model.pt", weights_only=True)
    expected = direct_reference(config, 8, exclude={(1, 0)})
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-7)


@pytest.mark.simulation
@pytest.mark.parametrize("total", [16, 100])
def test_many_real_clients_on_four_native_workers_finish_two_rounds(tmp_path, total):
    pytest.importorskip("ray")
    config = VirtualConfig(
        rounds=2,
        batch_size=2,
        boundary="after:head",
        virtual_clients=ClientPopulation(total=total),
        resources=Resources(max_concurrent_clients=4, worker_count_policy="pooled"),
    )
    output = tmp_path / f"clients-{total}"
    summary = run_virtual(config, fixture_spec(samples=total * 2), output=output)
    assert summary["training_tasks"] == total * 2
    assert summary["optimization_steps"] == total * 2
    rounds = read_rows(output, "round_metrics.jsonl")
    assert all(
        row["successful_clients"] == total and row["failed_clients"] == 0
        for row in rounds
    )
    assert all(1 <= row["worker_processes_observed"] <= 4 for row in rounds)
    assert all(row["suffix_replica_peak"] <= 4 for row in rounds)
