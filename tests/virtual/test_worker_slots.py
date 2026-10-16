import json
import os
from dataclasses import replace

import pytest
import torch
from flwr.app import Context, Message, RecordDict
from flwr.common import FitIns, ndarrays_to_parameters
from flwr.common.recorddict_compat import fitins_to_recorddict

from experiments.virtual_sfl.workloads import fixture_spec
from splitfleet.virtual.backend_ops import FrameworkOps
from splitfleet.virtual.client_app import make_client_app
from splitfleet.virtual.config import (
    ClientPopulation,
    Profile,
    Resources,
    VirtualConfig,
)
from splitfleet.virtual.fedavg import _FedAvgFactory, run_fedavg
from splitfleet.virtual.runtime import run_virtual
from splitfleet.virtual.worker_slots import WorkerSlots, worker_slots
from tests.virtual.test_training import direct_reference, read_rows


def test_active_capacity_failure_eviction_and_scope_isolation():
    discarded = []
    pool = WorkerSlots(scope="scope", capacity=1)
    first, reused = pool.acquire("cpu", object, discarded.append)
    assert not reused
    with pytest.raises(RuntimeError, match="active"):
        pool.acquire("cpu", object, discarded.append)
    with pytest.raises(RuntimeError, match="capacity"):
        pool.acquire("gpu", object, discarded.append)
    assert not discarded
    pool.finish("cpu", first, success=True)
    same, reused = pool.acquire("cpu", object, discarded.append)
    assert same is first and reused
    pool.finish("cpu", same, success=False)
    assert discarded == [first] and not pool.idle
    second, reused = pool.acquire("cpu", object, discarded.append)
    assert second is not first and not reused
    pool.finish("cpu", second, success=True)
    third, _ = pool.acquire("gpu", object, discarded.append)
    assert discarded == [first, second] and pool.peak_resident == 1
    pool.finish("gpu", third, success=True)
    pool.clear()
    shared = worker_slots(scope="first-test", capacity=1)
    value, _ = shared.acquire("cpu", object, discarded.append)
    with pytest.raises(RuntimeError, match="active"):
        worker_slots(scope="next-test", capacity=1)
    shared.finish("cpu", value, success=True)
    other = worker_slots(scope="next-test", capacity=1)
    assert other is not shared and value in discarded
    other.clear()


def train_fixture(model, raw, optimizer, device, learning_rate):
    inputs, labels = (value.to(device) for value in raw)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(inputs), labels)
    loss.backward()
    optimizer.step()
    return len(labels), float(loss.detach())


def test_outer_native_flower_return_validation_discards_a_completed_worker_slot(tmp_path):
    torch.set_num_threads(1)
    config = VirtualConfig(rounds=1, batch_size=2, virtual_clients=ClientPopulation(total=1),
                           resources=Resources(client_runtime_slots=1))
    spec = replace(fixture_spec(samples=4), trainer=train_fixture)
    scope = str(tmp_path / "outer-validation")
    ops = FrameworkOps.for_spec(spec)
    model = ops.build_model(spec.model_factory, "cpu")
    parameters = ops.export_arrays(model)
    base = _FedAvgFactory(config, spec, {"0": list(range(4))}, run_scope=scope)
    marker = {}

    def incorrect_result(context):
        client = base(context)
        native_fit = client.fit

        def fit(values, options):
            arrays, count, metrics = native_fit(values, options)
            marker["real_steps"] = metrics["num_batches"]
            marker["slot"] = worker_slots(scope=scope, capacity=1)
            # Force native return validation to fail after real SGD and the
            # adapter's successful slot retirement.
            return arrays, float(count), metrics

        client.fit = fit
        return client

    incorrect_result.run_scope = scope
    options = {"quality_round": 1, "virtual_partition_id": 0,
               "virtual_global_model_hash": ops.state_hash(model)}
    content = fitins_to_recorddict(FitIns(ndarrays_to_parameters(parameters), options), keep_input=True)
    message = Message(content=content, dst_node_id=11, message_type="train")
    context = Context(0, 11, {"partition-id": 0}, RecordDict(), {})
    with pytest.raises(TypeError, match="NumPyClient.fit"):
        make_client_app(incorrect_result)(message, context)
    assert marker["real_steps"] == 2
    assert not marker["slot"].active and not marker["slot"].idle


@pytest.mark.simulation
@pytest.mark.parametrize("method", ["fedavg", "splitfleet"])
def test_real_flower_serialized_tasks_reuse_private_slots_and_match_native_sgd(tmp_path, method):
    pytest.importorskip("ray")
    if 7 not in os.sched_getaffinity(0):
        pytest.skip("Dedicated CPU7 keeps the native diagnostic separate from experiment clients")
    config = VirtualConfig(rounds=2, batch_size=2, boundary="after:head",
        virtual_clients=ClientPopulation(total=4),
        resources=Resources(max_concurrent_clients=1, worker_count_policy="pooled", client_runtime_slots=1, mode="emulation"),
        profiles={"cpu": Profile(device="cpu", cpu_limit=1, affinity=[7], torch_threads=1)})
    spec = replace(fixture_spec(samples=16), trainer=train_fixture)
    output = tmp_path / method
    result = (run_fedavg if method == "fedavg" else run_virtual)(config, spec, output=output)
    assert result["training_tasks"] == 8 and result["optimization_steps"] == 16
    actual = torch.load(output / "final_model.pt", weights_only=True)
    expected = direct_reference(config, 16)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=1e-7, rtol=1e-6)
    rows = [row for row in read_rows(output, "client_metrics.jsonl") if row["status"] == "success"]
    assert len(rows) == 8
    assert len({row["metrics"]["worker_pid"] for row in rows}) == 1
    assert sum(not row["metrics"]["worker_execution_slot_reused"] for row in rows) == 1
    assert all(row["metrics"]["worker_execution_slot_capacity"] == 1 for row in rows)
    assert all(json.loads(row["metrics"]["worker_affinity"]) == [7] for row in rows)


@pytest.mark.simulation
def test_real_failed_client_slot_is_discarded_before_other_logical_clients(tmp_path):
    pytest.importorskip("ray")
    if 7 not in os.sched_getaffinity(0):
        pytest.skip("Dedicated CPU7 is unavailable")
    config = VirtualConfig(rounds=2, batch_size=2, boundary="after:head",
        virtual_clients=ClientPopulation(total=2, failure_rounds={"0": [1]}, fail_after_batches=1),
        resources=Resources(max_concurrent_clients=1, worker_count_policy="pooled", client_runtime_slots=1, mode="emulation"),
        profiles={"cpu": Profile(device="cpu", cpu_limit=1, affinity=[7], torch_threads=1)})
    output = tmp_path / "failure"
    result = run_virtual(config, fixture_spec(samples=8), output=output)
    assert result["training_tasks"] == 3
    actual = torch.load(output / "final_model.pt", weights_only=True)
    expected = direct_reference(config, 8, exclude={(1, 0)})
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=1e-7, rtol=1e-6)
    successful = [row for row in read_rows(output, "client_metrics.jsonl") if row["status"] == "success"]
    assert sum(not row["metrics"]["worker_execution_slot_reused"] for row in successful) == 1
