"""Worker population contracts, native admission, and real paired SGD."""

import json
from dataclasses import replace

import pytest
import torch

from experiments.virtual_sfl.workloads import fixture_spec
from splitfleet.virtual.config import ClientPopulation, Profile, Resources, VirtualConfig
from splitfleet.virtual.fedavg import run_fedavg
from splitfleet.virtual.runtime import _admit_resources, initialize_ray, run_virtual
from tests.virtual.test_training import direct_reference, read_rows


def train_fixture(model, raw, optimizer, device, learning_rate):
    inputs, labels = (value.to(device) for value in raw)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(inputs), labels)
    loss.backward()
    optimizer.step()
    return len(labels), float(loss.detach())


def test_default_population_shares_existing_cpu_budget_and_survives_serialization(monkeypatch):
    monkeypatch.setattr("splitfleet.virtual.config.os.sched_getaffinity", lambda pid: set(range(32)))
    config = VirtualConfig(virtual_clients=ClientPopulation(total=100))
    assert config.resources.max_concurrent_clients == 100
    assert config.resources.client_cpus_per_worker == .32
    assert config.resources.worker_count_policy == "max_concurrency"
    assert config.placement == "greedy"
    assert config.resources.server_concurrency == config.resources.effective_server_concurrency == 1
    assert VirtualConfig.from_dict(config.as_dict()) == config


def test_smaller_worker_pool_is_an_explicit_max_concurrency_cap():
    default_policy = VirtualConfig(
        virtual_clients=ClientPopulation(total=100),
        resources=Resources(max_concurrent_clients=2),
    )
    assert default_policy.resources.effective_worker_capacity == 2
    assert default_policy.resources.effective_suffix_slots == 2
    maximized_default = VirtualConfig(
        virtual_clients=ClientPopulation(total=100),
        resources=Resources(worker_count_policy="max_concurrency"),
    )
    assert maximized_default.resources.worker_target_count == 100
    config = VirtualConfig(virtual_clients=ClientPopulation(total=100),
                          resources=Resources(max_concurrent_clients=2, worker_count_policy="max_concurrency"))
    assert config.resources.max_concurrent_clients == 2


def test_explicit_suffix_cap_bounds_resident_models_and_round_trips_admission():
    config = VirtualConfig(
        virtual_clients=ClientPopulation(total=100),
        resources=Resources(max_concurrent_clients=100, max_suffix_replicas=4),
    )
    assert config.resources.effective_suffix_slots == 100
    assert config.resources.resident_suffix_capacity == 4
    assert config.resources.effective_server_concurrency == 1

    config.resources.adopt_capacity(3, predicted=100)
    restored = VirtualConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.resources.effective_worker_capacity == 3
    assert restored.resources.capacity_observation["actual"] == 3

    with pytest.raises(ValueError, match="cannot exceed max_suffix_replicas"):
        VirtualConfig(
            virtual_clients=ClientPopulation(total=4),
            resources=Resources(
                max_concurrent_clients=4,
                max_suffix_replicas=1,
                max_resident_suffix_replicas=4,
            ),
        )


def test_native_gpu_pool_admits_all_clients_or_rejects_when_capacity_is_insufficient(monkeypatch):
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
    _, undersized_workers = initialize_ray(undersized, ray, Devices())
    assert undersized_workers == 2
    assert undersized.resources.effective_worker_capacity == 2


@pytest.mark.parametrize("capacity", [3, 4, 13])
def test_memory_reduction_keeps_integral_node_cpus_and_exact_native_pool(monkeypatch, capacity):
    import ray

    monkeypatch.setattr("splitfleet.virtual.config.os.sched_getaffinity", lambda pid: set(range(32)))
    model = torch.nn.Linear(4, 2)
    state_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    config = VirtualConfig(virtual_clients=ClientPopulation(total=100), resources=Resources(
        server_memory_budget_mb=state_bytes * (4 * capacity + 3.5) / 2**20))
    _admit_resources(config, model)
    assert config.resources.client_cpus_per_worker == .32
    assert config.resources.effective_worker_capacity == capacity
    initialized = []
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", lambda **kwargs: initialized.append(kwargs))
    monkeypatch.setattr(ray, "nodes", lambda: [{"Resources": {"CPU": initialized[0]["num_cpus"]}}])
    options, workers = initialize_ray(config, ray)
    assert type(options["num_cpus"]) is int
    assert workers == capacity
    assert config.resources.effective_worker_capacity == capacity
    assert VirtualConfig.from_dict(config.as_dict()) == config


@pytest.mark.parametrize("remote", [False, True])
def test_native_pool_above_admitted_limit_is_rejected_before_execution(monkeypatch, remote):
    import ray

    config = VirtualConfig(virtual_clients=ClientPopulation(total=8), resources=Resources(
        max_concurrent_clients=2, client_cpus_per_worker=1))
    if remote:
        # This exercises connection/admission directly; transport validation is
        # tested separately with actual TLS configuration.
        object.__setattr__(config.resources, "ray_address", "auto")
    initialized = []
    monkeypatch.setattr(ray, "is_initialized", lambda: not remote)
    monkeypatch.setattr(ray, "init", lambda **kwargs: initialized.append(kwargs))
    monkeypatch.setattr(ray, "nodes", lambda: [{"Resources": {"CPU": 8.}}])
    with pytest.raises(ValueError, match="Native Flower pool has 8 workers; expected 2"):
        initialize_ray(config, ray)
    assert config.resources.effective_worker_capacity == 2
    assert config.resources.capacity_observation["actual"] == 8
    assert len(initialized) == int(remote)


@pytest.mark.simulation
def test_memory_reduced_fractional_cpu_reservations_start_real_ray(monkeypatch):
    ray = pytest.importorskip("ray")
    monkeypatch.setattr("splitfleet.virtual.config.os.sched_getaffinity", lambda pid: set(range(32)))
    config = VirtualConfig(virtual_clients=ClientPopulation(total=100))
    config.resources.adopt_capacity(4, predicted=100, source="memory_admission")
    assert not ray.is_initialized()
    try:
        options, workers = initialize_ray(config, ray)
        assert options["num_cpus"] == 2 and workers == 4
        assert config.resources.client_cpus_per_worker == .5
        assert ray.cluster_resources()["CPU"] == 2
    finally:
        ray.shutdown()


@pytest.mark.simulation
@pytest.mark.parametrize("method,workers", [("fedavg", 3), ("splitfleet", 3), ("splitfleet", 1)])
def test_private_suffix_population_matches_clients_with_bounded_workers(tmp_path, method, workers):
    pytest.importorskip("ray")
    config = VirtualConfig(rounds=2, batch_size=2, boundary="after:head",
                           virtual_clients=ClientPopulation(total=3),
                           resources=Resources(max_concurrent_clients=workers,
                                               worker_count_policy="max_concurrency",
                                               max_suffix_replicas=3, max_resident_suffix_replicas=workers,
                                               per_client_suffix_models=True, server_concurrency=workers))
    spec = replace(fixture_spec(samples=12), trainer=train_fixture)
    output = tmp_path / method
    result = (run_fedavg if method == "fedavg" else run_virtual)(config, spec, output=output)
    assert result["worker_limit"] == workers and result["logical_clients"] == 3
    assert result["worker_count_policy"] == "max_concurrency"
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
