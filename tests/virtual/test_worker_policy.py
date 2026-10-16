"""Worker population contracts, native admission, and real paired SGD."""

import json
from dataclasses import replace

import pytest
import torch

from experiments.virtual_sfl.workloads import fixture_spec
from splitfleet.virtual.config import ClientPopulation, Profile, Resources, VirtualConfig
from splitfleet.virtual.fedavg import run_fedavg
from splitfleet.virtual.runtime import initialize_ray, run_virtual
from tests.virtual.test_training import direct_reference, read_rows
from tests.virtual.test_worker_slots import train_fixture


def test_default_population_shares_existing_cpu_budget_and_survives_serialization(monkeypatch):
    monkeypatch.setattr("splitfleet.virtual.config.os.sched_getaffinity", lambda pid: set(range(32)))
    config = VirtualConfig(virtual_clients=ClientPopulation(total=100))
    assert config.resources.max_concurrent_clients == 100
    assert config.resources.client_cpus_per_worker == .32
    assert config.resources.worker_count_policy == "one_per_client"
    assert VirtualConfig.from_dict(config.as_dict()) == config


def test_smaller_worker_pool_requires_explicit_queueing_policy():
    with pytest.raises(ValueError, match="use worker_count_policy=pooled"):
        VirtualConfig(virtual_clients=ClientPopulation(total=100),
                      resources=Resources(max_concurrent_clients=2))
    with pytest.raises(ValueError, match="explicit max_concurrent_clients"):
        Resources(worker_count_policy="pooled")
    config = VirtualConfig(virtual_clients=ClientPopulation(total=100),
                          resources=Resources(max_concurrent_clients=2, worker_count_policy="pooled"))
    assert config.resources.max_concurrent_clients == 2


def test_native_gpu_pool_admits_all_clients_or_rejects_without_fallback(monkeypatch):
    import ray

    monkeypatch.setattr("splitfleet.virtual.config.os.sched_getaffinity", lambda pid: set(range(32)))
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(ray, "nodes", lambda: [{"Resources": {"CPU": 32., "GPU": 1.}}])

    class Devices:
        backend = "torch"

        def gpu_count(self):
            return 2

    config = VirtualConfig(virtual_clients=ClientPopulation(total=100),
        resources=Resources(client_gpus_per_worker=.01, client_gpu_pool_size=1),
        profiles={"gpu": Profile(device="cuda:0", gpu_fraction=.01)})
    options, workers = initialize_ray(config, ray, Devices())
    assert workers == 100 and options["num_cpus"] == 32 and options["num_gpus"] == 1
    undersized = replace(config, resources=replace(config.resources, client_gpus_per_worker=.5),
                         profiles={"gpu": Profile(device="cuda:0", gpu_fraction=.5)})
    with pytest.raises(ValueError, match="will not fall back to fewer workers"):
        initialize_ray(undersized, ray, Devices())


@pytest.mark.simulation
@pytest.mark.parametrize("method,workers", [("fedavg", 3), ("splitfleet", 3), ("splitfleet", 1)])
def test_private_suffix_population_matches_clients_with_real_worker_fallback(tmp_path, method, workers):
    pytest.importorskip("ray")
    config = VirtualConfig(rounds=2, batch_size=2, boundary="after:head",
                           virtual_clients=ClientPopulation(total=3),
                           resources=Resources(max_concurrent_clients=workers,
                                               worker_count_policy="one_per_client" if workers == 3 else "pooled",
                                               max_suffix_replicas=3, max_resident_suffix_replicas=workers,
                                               per_client_suffix_models=True, server_concurrency=workers))
    spec = replace(fixture_spec(samples=12), trainer=train_fixture)
    output = tmp_path / method
    result = (run_fedavg if method == "fedavg" else run_virtual)(config, spec, output=output)
    assert result["worker_limit"] == workers and result["logical_clients"] == 3
    assert result["worker_count_policy"] == ("one_per_client" if workers == 3 else "pooled")
    assert result["optimization_steps"] == 12
    actual = torch.load(output / "final_model.pt", weights_only=True)
    for name, expected in direct_reference(config, 12).items():
        torch.testing.assert_close(actual[name], expected, rtol=1e-6, atol=1e-7)
    rows = [row for row in read_rows(output, "client_metrics.jsonl") if row["status"] == "success"]
    assert len({(row["metrics"]["worker_host"], row["metrics"]["worker_pid"]) for row in rows}) == workers
    if method == "splitfleet":
        resources = json.loads((output / 'resource_metrics.json').read_text())
        assert resources['observed_suffix_model_count'] == 3
        assert resources['suffix_replicas_reused'] == 0
        assert resources['peak_suffix_replicas'] <= workers
        for round_id in (1, 2):
            assert len({row['metrics']['suffix_model_instance_id'] for row in rows
                        if row['round_id'] == round_id}) == 3
