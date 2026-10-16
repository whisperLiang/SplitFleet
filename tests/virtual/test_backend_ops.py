"""Native state/device operations and preserved Torch virtual defaults."""

from __future__ import annotations

import os
import json
from pathlib import Path
import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from splitfleet.tasks.base import ModelInputs, TaskBatch
from splitfleet.virtual.backend_ops import FrameworkOps


def test_torch_hash_optimizer_and_checkpoint_keep_existing_runtime_contract(tmp_path):
    import torch
    from splitfleet.common.model_state import tensor_state_hash
    from splitfleet.virtual.client_adapter import OptimizerFactory

    class Model(torch.nn.Linear):
        def __init__(self):
            super().__init__(4, 2)
            self.register_buffer("counter", torch.tensor(2**60 + 3, dtype=torch.int64))

    ops = FrameworkOps()
    assert FrameworkOps("pytorch").backend == "torch"
    model = ops.build_model(Model, "cpu")
    assert ops.state_hash(model) == tensor_state_hash(model.state_dict())
    assert ops.trainable_names(model) == {"weight", "bias"}
    optimizer = OptimizerFactory("sgd", .05)(model)
    assert isinstance(optimizer, torch.optim.SGD) and optimizer.defaults["momentum"] == 0
    expected = ops.export_arrays(model)
    path = ops.save_checkpoint(model, expected, tmp_path / "final_model")
    assert path.name == "final_model.pt"
    with torch.no_grad():
        model.weight.add_(1)
    ops.load_checkpoint(model, path)
    for actual, before in zip(ops.export_arrays(model), expected, strict=True):
        np.testing.assert_array_equal(actual, before)


def test_array_host_rng_is_restored_and_does_not_claim_native_rng_state():
    ops = FrameworkOps("jax")
    previous = ops.capture_rng()
    with ops.rng_scope(seed=233):
        expected = random.random(), np.random.random()
        saved = ops.capture_rng()
        continuation = random.random(), np.random.random()
    assert ops.capture_rng() == previous
    with ops.rng_scope(seed=234, saved=saved):
        assert (random.random(), np.random.random()) == continuation
    with ops.rng_scope(seed=233):
        assert (random.random(), np.random.random()) == expected
    assert "host RNG" in ops.rng_scope_description


@pytest.mark.parametrize("backend", ["tf", "paddle", "jax"])
def test_native_array_checkpoint_counter_and_device_contract(backend, tmp_path, request):
    # Keep optional framework native libraries in independent processes.
    if os.environ.get("SPLITFLEET_BACKEND_OPS_CHILD") != backend:
        environment = dict(os.environ, SPLITFLEET_BACKEND_OPS_CHILD=backend,
                           PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", JAX_PLATFORMS="cpu",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")
        result = subprocess.run([sys.executable, "-B", "-m", "pytest", request.node.nodeid,
                                 "-o", "addopts=", "-q"],
                                cwd=Path(__file__).resolve().parents[2], env=environment,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        return
    if backend == "tf":
        tf = pytest.importorskip("tensorflow")

        class Model(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.kernel = self.add_weight(shape=(3, 3, 2, 4), dtype="float32")
                self.counter = self.add_weight(shape=(), dtype="int64", initializer="zeros", trainable=False)

            def canonical_variables(self):
                return {"conv.weight": self.kernel, "bn.num_batches_tracked": self.counter}

    elif backend == "paddle":
        paddle = pytest.importorskip("paddle")

        class Model(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.weight = self.create_parameter(shape=(4, 2), dtype="float32")
                self.register_buffer("counter", paddle.to_tensor(np.asarray(0, np.int64)))

    else:
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)

        class Model:
            def __init__(self):
                self.params = {"weight": jax.numpy.ones((4, 2), dtype=jax.numpy.float32)}
                self.counter = jax.numpy.asarray(0, dtype=jax.numpy.int64)

            def state_tensors(self):
                return {**self.params, "counter": self.counter}

            def load_state(self, values):
                self.params.update(weight=jax.numpy.asarray(values["weight"]))
                self.counter = jax.numpy.asarray(values["counter"])

            def set_params(self, params):
                self.params.update(params)

    ops = FrameworkOps(backend)
    model = ops.build_model(Model, "cpu")
    arrays = ops.export_arrays(model)
    arrays[-1] = np.asarray(2**60 + 3, np.int64)
    ops.load_arrays(model, arrays)
    assert ops.state_bytes(model) == sum(value.nbytes for value in arrays)
    before_hash = ops.state_hash(model)
    assert ops.peak_gpu_memory_mb("cpu") is None
    with pytest.raises(RuntimeError, match="native .* GPU"):
        ops.admit_device("cuda:0")
    path = ops.save_checkpoint(model, arrays, tmp_path / "final_model")
    assert path.suffix == ".npz"
    changed = [np.zeros_like(value) for value in arrays]
    ops.load_arrays(model, changed)
    assert ops.state_hash(model) != before_hash
    ops.load_checkpoint(model, path)
    assert ops.state_hash(model) == before_hash
    assert ops.export_arrays(model)[-1].item() == 2**60 + 3
    assert ops.export_arrays(model)[0].shape == ((3, 3, 2, 4) if backend == "tf" else (4, 2))
    if backend == "jax":
        from splitfleet.runtime.functional_sgd import OwnedFunctionalSGD
        optimizer = ops.optimizer(model, "sgd", .05)
        assert isinstance(optimizer, OwnedFunctionalSGD)
        inputs = jax.numpy.ones((3, 4), dtype=jax.numpy.float32)
        batch = ops.prepare_batch(model, TaskBatch(ModelInputs((None, inputs)), jax.numpy.zeros(3), 3), "cpu")
        bound = batch.inputs.args[0]
        optimizer._step({"weight": jax.numpy.ones((4, 2), dtype=jax.numpy.float32)})
        assert model.params is bound and ops.trainable_names(model) == {"weight"}
        np.testing.assert_allclose(np.asarray(bound["weight"]), .95, atol=1e-7)


@pytest.mark.parametrize("backend", ["tensorflow", "paddle", "jax", "tinygrad"])
def test_other_backends_are_rejected_before_experiment_preparation(quality_inputs, backend):
    from experiments.resnet50_quality.run import make_config
    with pytest.raises(ValueError, match=f"Unknown native backend {backend!r}"):
        make_config(method="splitfleet", hardware_profile="unused.json",
                    seed=233, protocol=quality_inputs.protocol,
                    data_root=quality_inputs.data_root, backend=backend)


@pytest.mark.parametrize("backend", ["torch", "jax", "paddle"])
def test_native_ray_processes_isolate_unused_tensorflow_before_native_import(backend):
    script = textwrap.dedent("""
        import os
        import sys
        from splitfleet.virtual.backend_ops import FrameworkOps
        from splitfleet.virtual.config import VirtualConfig, Resources, ClientPopulation
        from splitfleet.virtual.runtime import check_simulation_dependencies, initialize_ray
        backend = sys.argv[1]
        ops = FrameworkOps(backend)
        ops.prepare_process(isolate_unused_frameworks=True)
        ray, _ = check_simulation_dependencies()
        from flwr.simulation.ray_transport import utils
        assert utils.TF is None
        config = VirtualConfig(
            virtual_clients=ClientPopulation(total=2),
            resources=Resources(max_concurrent_clients=2, max_suffix_replicas=2,
                                client_gpus_per_worker=0, object_store_memory_mb=80),
            workload={'isolate_unused_frameworks': True})
        _, actual = initialize_ray(config, ray, ops)
        assert actual == 2
        @ray.remote(num_cpus=1)
        class NativeWorker:
            def probe(self):
                from flwr.simulation.ray_transport import utils
                assert utils.TF is None and sys.modules['tensorflow'] is None
                if backend == 'torch':
                    import torch
                    x = torch.tensor([1., 2., 3.], requires_grad=True)
                    loss = (x * x).sum()
                    loss.backward()
                    gradients = x.grad.tolist()
                elif backend == 'paddle':
                    import paddle
                    x = paddle.to_tensor([1., 2., 3.], stop_gradient=False)
                    loss = (x * x).sum()
                    loss.backward()
                    gradients = x.grad.numpy().tolist()
                else:
                    import jax
                    x = jax.numpy.asarray([1., 2., 3.])
                    loss, gradient = jax.value_and_grad(lambda value: (value * value).sum())(x)
                    gradients = gradient.tolist()
                assert gradients == [2., 4., 6.]
                return os.getpid(), float(loss)
        try:
            actors = [NativeWorker.remote() for _ in range(2)]
            values = ray.get([actor.probe.remote() for actor in actors], timeout=60)
            assert len({value[0] for value in values}) == 2
            assert [value[1] for value in values] == [14., 14.]
            print('two native workers passed optional-framework isolation: ' + backend)
        finally:
            ray.shutdown()
    """)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, "-u", "-B", "-c", script, backend],
                            cwd=Path(__file__).resolve().parents[2], env=environment,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_completed_round_curve_is_persisted_before_evaluation_or_run_finish(tmp_path, monkeypatch):
    import csv
    import experiments.resnet50_quality.workloads as workloads
    # State serialization is covered above; this fixture isolates the durable
    # callback contract without allocating another full model checkpoint.
    monkeypatch.setattr(workloads, "NAMES", ("weight",))
    monkeypatch.setattr(workloads, "state_hash", lambda state: "fixture")
    callback = workloads.RoundCheckpoints()
    arrays = [np.ones(2, np.float32)]
    callback(1, arrays, {"train_loss": 1.2, "num_examples": 64,
                        "successful_clients": 2, "optimization_steps": 4}, tmp_path, phase="training")
    with (tmp_path / "round_metrics.csv").open() as stream:
        row = list(csv.DictReader(stream))[0]
    assert row["examples_seen"] == "64" and row["optimization_steps"] == "4"
    assert row["val_ce"] == ""  # The completed training row survives before evaluation.
    callback(1, arrays, {"loss": 1.1, "top1": .4}, tmp_path, phase="evaluation")
    with (tmp_path / "round_metrics.csv").open() as stream:
        row = list(csv.DictReader(stream))[0]
    assert row["val_ce"] == "1.1" and row["val_top1"] == "0.4"
    assert float(row["elapsed_wall_sec"]) > 0
    with pytest.raises(RuntimeError, match="replay"):
        callback(1, arrays, {"train_loss": 1.2, "num_examples": 64,
                            "successful_clients": 2, "optimization_steps": 4}, tmp_path, phase="training")


def test_flower_queued_message_ttl_is_separate_from_native_handler_deadline():
    from flwr.common import FitIns, FitRes, Status, Code, ndarrays_to_parameters
    from flwr.common.message import DEFAULT_TTL, Message
    from flwr.server.compat.grid_client_proxy import GridClientProxy
    from flwr.common.recorddict_compat import fitres_to_recorddict

    class Grid:
        def __init__(self):
            self.messages = []

        def send_and_receive(self, messages):
            message = messages[0]
            self.messages.append(message)
            result = FitRes(Status(Code.OK, ""), ndarrays_to_parameters([]), 1, {})
            return [Message(fitres_to_recorddict(result, keep_input=True), reply_to=message)]

    grid = Grid()
    proxy = GridClientProxy(1, grid, 1)
    instruction = FitIns(ndarrays_to_parameters([]), {"virtual_handler_deadline_sec": 120.})
    proxy.fit(instruction, timeout=None, group_id=1)
    assert grid.messages[-1].metadata.ttl == DEFAULT_TTL == 43_200
    proxy.fit(instruction, timeout=5000., group_id=1)
    assert grid.messages[-1].metadata.ttl == 5000.


def test_quality_children_reject_changed_frozen_weights_and_partition_before_training(tmp_path, quality_inputs):
    from dataclasses import replace
    from experiments.resnet50_quality.run import make_config
    from experiments.resnet50_quality.workloads import quality_spec

    original = quality_inputs.protocol
    protocol = json.loads((original / "protocol.json").read_text())
    from experiments.resnet50_quality.hardware import hardware_profile
    hardware = tmp_path / "hardware.json"
    hardware.write_text(json.dumps(hardware_profile(cpus=[4, 5])))
    config = make_config(method="fedavg", seed=233, protocol=original,
                         data_root=quality_inputs.data_root, hardware_profile=hardware)
    selected = next(row for row in protocol["pairs"] if row["backend"] == "pytorch" and row["seed"] == 233)
    (tmp_path / "initial_seed_233.npz").symlink_to(original / "initial_seed_233.npz")
    partition = tmp_path / selected["partition_manifest"]
    partition.write_bytes((original / selected["partition_manifest"]).read_bytes())
    changed = replace(config, workload={**config.workload, "protocol": str(tmp_path)})
    expected = selected["initial_hash"]
    selected["initial_hash"] = "0" * 64
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Canonical initialization.*frozen"):
        quality_spec(changed)
    selected["initial_hash"] = expected
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    partition.write_bytes(partition.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Partition manifest.*frozen"):
        quality_spec(changed)
