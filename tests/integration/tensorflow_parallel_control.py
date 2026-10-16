"""Bounded real TensorFlow SGD control for independent parallel split models."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def launch(
    device,
    full=False,
    clients=None,
    unsplit=False,
    compiled_suffix=False,
    compiled_training=False,
    cold_start=False,
    full_shard_examples=None,
):
    root = Path(__file__).resolve().parents[2]
    output = Path(tempfile.mkdtemp(prefix="splitfleet_tf_parallel_"))
    snapshot = output / "runtime_snapshot"
    snapshot.mkdir()
    for name in ("splitfleet", "experiments/resnet50_quality", "tests/integration/resnet50_reference"):
        shutil.copytree(
            root / name,
            snapshot / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in ("tests/__init__.py", "tests/integration/__init__.py"):
        shutil.copyfile(root / name, snapshot / name)
    helper = snapshot / "control.py"
    shutil.copyfile(__file__, helper)
    hashes = {
        str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(snapshot.rglob("*.py"))
    }
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    for p in snapshot.rglob("*.py"):
        p.chmod(0o444)
    env = dict(
        os.environ,
        PYTHONPATH=str(snapshot),
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="0,1" if device.startswith("cuda") else "",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TF_NUM_INTRAOP_THREADS="1",
        TF_NUM_INTEROP_THREADS="1",
        TF_CPP_MIN_LOG_LEVEL="2",
        NVIDIA_TF32_OVERRIDE="0",
    )
    start = time.monotonic()
    with (output / "stdout.log").open("w") as log:
        command = [
            sys.executable,
            str(helper),
            "--worker",
            "--output",
            str(output),
            "--device",
            device,
        ]
        if full:
            command.append("--full")
        if clients is not None:
            command.extend(["--clients", str(clients)])
        if unsplit:
            command.append("--unsplit")
        if compiled_suffix:
            command.append("--compiled-suffix")
        if compiled_training:
            command.append("--compiled-training")
        if cold_start:
            command.append("--cold-start")
        if full_shard_examples is not None:
            command.extend(["--full-shard-examples", str(full_shard_examples)])
        child = subprocess.Popen(
            command,
            cwd=snapshot,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        reason = None
        while child.poll() is None:
            available = (
                int(
                    next(
                        s
                        for s in Path("/proc/meminfo").read_text().splitlines()
                        if s.startswith("MemAvailable:")
                    ).split()[1]
                )
                / 1024
            )
            if available < 8192 or time.monotonic() - start > (600 if full else 180):
                reason = (
                    "host memory reserve exhausted"
                    if available < 8192
                    else "native control deadline"
                )
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                break
            time.sleep(0.5)
    summary = output / "summary.json"
    data = json.loads(summary.read_text()) if summary.exists() else {"status": "failed"}
    data.update(
        source_identity=identity,
        elapsed_sec=time.monotonic() - start,
        formal_admission=False,
    )
    passed = child.returncode == 0 and data.get("status") == "passed" and reason is None
    if passed:
        shutil.rmtree(snapshot)
        (output / "stdout.log").unlink()
        summary.write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps({"status": "passed", "receipt": str(summary), **data}))
    else:
        tail = (output / "stdout.log").read_text(errors="replace")[-3500:]
        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason": reason or data.get("reason"),
                    "phase": data.get("phase"),
                    "completed_cases": [case["case"] for case in data.get("cases", [])],
                    "exit_code": child.returncode,
                    "source_identity": identity,
                    "native_trace": tail,
                    "temporary_output_deleted": True,
                }
            )
        )
        shutil.rmtree(output)
    return 0 if passed else 1


def worker(
    output,
    device,
    full=False,
    clients=None,
    unsplit=False,
    compiled_suffix=False,
    compiled_training=False,
    cold_start=False,
    full_shard_examples=None,
):
    import numpy as np
    import tensorflow as tf
    from tests.integration.resnet50_reference.tf_model import CanonicalBatchNorm
    from tests.integration.resnet50_reference.native import configure_backend
    from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend
    from splitfleet.runtime.tensorflow_capture_scope import tensorflow_execution_scope
    from torchlens.split.adapters.tf import _tf_gradient_source

    configure_backend("tensorflow")
    if compiled_suffix:
        # Keep the native raw-op arithmetic while testing graph execution;
        # fused BatchNorm/layout rewrites change the numerical comparison.
        tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
    scope = "/GPU:1" if device == "cuda:1" else "/CPU:0"
    clients = clients or (3 if full else 8)
    summary = {
        "status": "running",
        "phase": "setup",
        "device": device,
        "full_ResNet50": full,
        "unsplit_native_training": unsplit,
        "compiled_suffix_control": compiled_suffix,
        "compiled_training_control": compiled_training,
        "cold_parallel_first_training": cold_start,
        "client_prefix_context": "serialized native prefix calls; production clients have separate processes",
        "onednn_environment": os.environ.get("TF_ENABLE_ONEDNN_OPTS", "native_default"),
        "cases": [],
    }

    def save():
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    class Model(tf.keras.Model):
        def __init__(self):
            super().__init__(name="parallel_native_bn")
            self.training, self.defer_statistics = True, True
            self.first = tf.keras.layers.Conv2D(256, 1, use_bias=False, name="first")
            self.norm = CanonicalBatchNorm(256, name="norm")
            self.relu = tf.keras.layers.ReLU(name="relu")
            self.head = tf.keras.layers.Dense(3, name="head")
            self.pause = None

        def call(self, value, training=True):
            hidden, statistics = self.norm(
                self.first(value), training=True, defer_statistics=True
            )
            from tensorflow.python.eager import context

            if self.pause is not None and context.context().op_callbacks:
                entered, release = self.pause
                self.pause = None
                entered.set()
                if not release.wait(30):
                    raise TimeoutError("capture fixture was not released")
            return {
                "logits": self.head(tf.reduce_mean(self.relu(hidden), axis=(1, 2))),
                "batch_statistics": {"norm": statistics},
            }

        def apply_statistics(self, stats, names=None):
            if names is None or "norm" in names:
                self.norm.apply_statistics(stats["norm"])

        def __deepcopy__(self, memo):
            result = Model()
            result(tf.zeros((1, 16, 16, 3)))
            result.set_weights(self.get_weights())
            memo[id(self)] = result
            return result

    class Recorder:
        def __init__(self, model):
            self.native = tf.keras.optimizers.SGD(learning_rate=0.05)
            self.native.build(model.trainable_variables)
            self.indices = {
                id(_tf_gradient_source(v)): i
                for i, v in enumerate(model.trainable_variables)
            }
            self.grads = {}

        def apply_gradients(self, pairs):
            pairs = list(pairs)
            for grad, variable in pairs:
                index = self.indices[id(_tf_gradient_source(variable))]
                if index in self.grads:
                    raise AssertionError("duplicate native parameter update")
                self.grads[index] = grad.numpy().copy()
            return self.native.apply_gradients(pairs)

    def loss(result, labels):
        if full:
            return tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=result["logits"]
                )
            )
        return tf.reduce_mean((result["logits"] - labels) ** 2)

    random = np.random.default_rng(738)
    try:
        with tf.device(scope):
            if full:
                from experiments.resnet50_quality.data import load_cifar
                from experiments.resnet50_quality.model_contract import canonical_state
                from tests.integration.resnet50_reference.native import build_model

                data = load_cifar(
                    "/home/whisperliang/SplitFleet/data/cifar-10-batches-py"
                )
                ids = data.train_indices[:20]
                images = data.images(ids).transpose(0, 2, 3, 1).copy()
                labels = data.train_labels[ids].copy()
                del data
                base = build_model("tensorflow", canonical_state(233), device)
                base.training = True
            else:
                base = Model()
                base(tf.zeros((1, 16, 16, 3)))
                for v in base.trainable_variables:
                    v.assign(random.normal(size=v.shape).astype("float32") * 0.1)
                images = random.normal(size=(20, 16, 16, 3)).astype("float32")
                labels = random.normal(size=(20, 3)).astype("float32")
            initial = base.get_weights()
            expected_grads = []
            oracle = copy.deepcopy(base)
            opt = tf.keras.optimizers.SGD(learning_rate=0.05)
            for start, stop in ((0, 16), (16, 20)):
                with tf.GradientTape() as tape:
                    result = oracle(images[start:stop])
                    value = loss(result, labels[start:stop])
                grads = tape.gradient(value, oracle.trainable_variables)
                assert all(g is not None for g in grads)
                expected_grads.append([g.numpy().copy() for g in grads])
                opt.apply_gradients(zip(grads, oracle.trainable_variables))
                oracle.apply_statistics(result["batch_statistics"])
            expected_state = oracle.get_weights()
            backends = []
            for index in range(clients):
                model = copy.deepcopy(base)
                model.set_weights(initial)
                if unsplit:
                    from types import SimpleNamespace

                    backends.append(SimpleNamespace(model=model))
                    continue
                backend = TorchLensSplitBackend(device=device).trace(
                    model,
                    tf.constant(images[:16]),
                    batch_axes={"/args/0": 0},
                    dynamic_batch=(1, 16),
                    trainable=True,
                )
                if full:
                    from tests.integration.resnet50_reference.array_helpers import stage_boundaries

                    endpoint = stage_boundaries(backend.make_handle(), "tensorflow")[
                        f"stage{index % 3 + 1}"
                    ]
                    backend.repartition(endpoint["boundary"])
                else:
                    endpoint = next(
                        n
                        for n in backend.runtime.trace_graph.nodes
                        if n.module_path == "first" and n.op_type.lower() == "conv2d"
                    )
                    backend.repartition("after:" + endpoint.canonical_id)
                if compiled_suffix:
                    from splitfleet.runtime.tensorflow_forward import (
                        bind_tensorflow_function_suffix,
                    )

                    bind_tensorflow_function_suffix(backend.runtime)
                backends.append(backend)

        references = {}
        # Production clients own separate TensorFlow contexts in Ray processes.
        # This in-process server control must not introduce concurrent eager
        # prefix tapes into the shared server context. Suffix calls stay parallel.
        client_context = threading.RLock()

        def check(actual, expected, label):
            try:
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
            except AssertionError as error:
                raise AssertionError(f"{label}: {error}") from error

        def train(backend, barrier=None, *, reference=False, dispatched=None):
            if barrier is not None:
                barrier.wait(timeout=15)
            if dispatched is not None:
                dispatched.set()
            with tensorflow_execution_scope(), tf.device(scope):
                model = backend.model
                model.set_weights(initial)
                opt = Recorder(model)
                steps = []
                statistic_steps = []
                for step, (start, stop) in enumerate(((0, 16), (16, 20))):
                    opt.grads = {}
                    if unsplit:
                        if compiled_training:
                            from splitfleet.runtime.tensorflow_forward import (
                                native_training_gradients,
                            )

                            result, value, gradients = native_training_gradients(
                                model,
                                tf.constant(images[start:stop]),
                                tf.constant(labels[start:stop]),
                                model.trainable_variables,
                            )
                        else:
                            with tf.GradientTape() as tape:
                                if compiled_suffix:
                                    from splitfleet.runtime.tensorflow_forward import (
                                        native_training_forward,
                                    )

                                    result = native_training_forward(
                                        model, tf.constant(images[start:stop])
                                    )
                                else:
                                    result = model(tf.constant(images[start:stop]))
                                value = loss(result, tf.constant(labels[start:stop]))
                            gradients = tape.gradient(value, model.trainable_variables)
                        opt.apply_gradients(zip(gradients, model.trainable_variables))
                        native_statistics = result["batch_statistics"]
                    else:
                        with client_context:
                            boundary = backend.run_prefix(
                                tf.constant(images[start:stop]), training=True
                            )
                        _, gradients = backend.train_suffix(
                            boundary,
                            tf.constant(labels[start:stop]),
                            loss_fn=loss,
                            optimizer=opt,
                        )
                        with client_context:
                            backend.backward_prefix(boundary, gradients, optimizer=opt)
                        native_statistics = backend.train_suffix_auxiliary
                    assert set(opt.grads) == set(range(len(model.trainable_variables)))
                    gradients = [
                        opt.grads[i] for i in range(len(model.trainable_variables))
                    ]
                    # Serial and concurrent controls use identical native graph
                    # decomposition. Both serial updates also retain the full
                    # unsplit native-gradient control.
                    expected = (
                        expected_grads[step]
                        if reference
                        else references[id(backend)]["gradients"][step]
                    )
                    for i, value in enumerate(expected):
                        check(
                            gradients[i],
                            value,
                            f"step={step + 1}, parameter={i}, serial={reference}",
                        )
                    steps.append(gradients)
                    statistics = {
                        name: {key: value.numpy().copy() for key, value in row.items()}
                        for name, row in native_statistics.items()
                    }
                    if not reference:
                        expected_statistics = references[id(backend)]["statistics"][
                            step
                        ]
                        assert set(statistics) == set(expected_statistics)
                        for name, row in statistics.items():
                            for key, value in row.items():
                                check(
                                    value,
                                    expected_statistics[name][key],
                                    f"auxiliary step={step + 1}, BN={name}, field={key}",
                                )
                    statistic_steps.append(statistics)
                    model.apply_statistics(statistics)
                state = model.get_weights()
                if not reference:
                    for i, (value, expected) in enumerate(
                        zip(state, references[id(backend)]["state"], strict=True)
                    ):
                        check(value, expected, f"final state={i}")
                else:
                    for i, (value, expected) in enumerate(
                        zip(state, expected_state, strict=True)
                    ):
                        check(value, expected, f"unsplit final state={i}")
                if full:
                    counters = [
                        v
                        for name, v in model.canonical_variables().items()
                        if name.endswith(".num_batches_tracked")
                    ]
                    counts = [int(v.numpy()) for v in counters]
                    assert len(counts) == 53 and counts == [2] * 53, counts
                    assert (
                        len(model.trainable_variables) == 161
                        and len(model.weights) == 320
                    )
                else:
                    assert int(model.norm.num_batches_tracked.numpy()) == 2
                if reference:
                    references[id(backend)] = {
                        "gradients": steps,
                        "state": state,
                        "statistics": statistic_steps,
                    }
                return {
                    "parameters": len(model.trainable_variables),
                    "state": len(model.weights),
                    "BN_count": 2,
                    "BN_tensors": 53 if full else 1,
                    "batches": [16, 4],
                }

        summary["phase"] = "serial_native_reference"
        save()
        if cold_start:
            summary["phase"] = "cold_parallel_native_reference"
            save()
            barrier = threading.Barrier(clients)
            with ThreadPoolExecutor(max_workers=clients) as pool:
                jobs = [
                    pool.submit(train, backend, barrier, reference=True)
                    for backend in backends
                ]
                results = [job.result(timeout=120) for job in jobs]
        else:
            results = [train(backend, reference=True) for backend in backends]
        summary["cases"].append(
            {
                "case": "cold_parallel_native_reference"
                if cold_start
                else "serial_native_reference",
                "clients": clients,
                "all_steps_unsplit_gradient_control": True,
                "native_results": results,
            }
        )
        for case in ("serial_repeat", "prepared_parallel", "capture_overlap"):
            summary["phase"] = case
            save()
            if case == "serial_repeat":
                results = [train(backend) for backend in backends]
                summary["cases"].append(
                    {"case": case, "clients": clients, "native_results": results}
                )
                continue
            entered, release = threading.Event(), threading.Event()
            with ThreadPoolExecutor(max_workers=clients + 1) as pool:
                capture = None
                if case == "capture_overlap":

                    def capture_new():
                        with tf.device(scope):
                            model = copy.deepcopy(base)
                            if full:
                                from types import MethodType

                                original_call = model.call

                                def paused_call(self, *args, **kwargs):
                                    result = original_call(*args, **kwargs)
                                    from tensorflow.python.eager import context

                                    if (
                                        context.context().op_callbacks
                                        and not entered.is_set()
                                    ):
                                        entered.set()
                                        if not release.wait(30):
                                            raise TimeoutError(
                                                "full native capture fixture was not released"
                                            )
                                    return result

                                model.call = MethodType(paused_call, model)
                            else:
                                model.pause = (entered, release)
                            return TorchLensSplitBackend(device=device).trace(
                                model,
                                tf.constant(images[:16]),
                                batch_axes={"/args/0": 0},
                                dynamic_batch=(1, 16),
                                trainable=True,
                            )

                    capture = pool.submit(capture_new)
                    assert entered.wait(30), "capture fixture never entered"
                barrier = threading.Barrier(clients)
                dispatched = [threading.Event() for _ in backends]
                jobs = [
                    pool.submit(train, b, barrier, dispatched=event)
                    for b, event in zip(backends, dispatched, strict=True)
                ]
                try:
                    if capture is not None:
                        assert all(event.wait(15) for event in dispatched)
                        assert all(not job.done() for job in jobs), (
                            "native training overlapped live capture hooks"
                        )
                        release.set()
                    results = [job.result(timeout=45) for job in jobs]
                finally:
                    release.set()
                if capture is not None:
                    capture.result(timeout=30)
                summary["cases"].append(
                    {"case": case, "clients": clients, "native_results": results}
                )
        if full_shard_examples is not None:
            from tests.integration.resnet50_reference.native import (
                export_state,
                native_batch,
                optimizer,
                train_step,
            )

            summary["phase"] = "full_shard_native_training"
            save()
            data = load_cifar("/home/whisperliang/SplitFleet/data/cifar-10-batches-py")
            ids = data.train_indices[:full_shard_examples]
            shard_images, shard_labels = data.images(ids), data.train_labels[ids]
            del data

            def train_shard(backend):
                model = backend.model
                model.set_weights(initial)
                native_optimizer = optimizer("tensorflow", model, .05)
                losses = []
                for start in range(0, len(ids), 16):
                    batch = native_batch("tensorflow", shard_images[start:start + 16],
                                         shard_labels[start:start + 16], device)
                    result = train_step("tensorflow", model, *batch,
                                        native_optimizer=native_optimizer)
                    assert np.isfinite(result["loss"]), result
                    losses.append(result["loss"])
                state = export_state("tensorflow", model)
                assert all(np.isfinite(value).all() for value in state.values())
                counters = [int(value) for name, value in state.items()
                            if name.endswith(".num_batches_tracked")]
                assert len(counters) == 53 and counters == [len(losses)] * 53
                return {"examples": len(ids), "batches": len(losses),
                        "tail_batch": len(ids) % 16 or 16,
                        "BN_tensors": len(counters), "BN_count": len(losses),
                        "losses": losses, "state": state}

            with ThreadPoolExecutor(max_workers=clients) as pool:
                shards = list(pool.map(train_shard, backends))
            for actual in shards[1:]:
                check(actual["losses"], shards[0]["losses"], "full shard losses")
                for name, value in actual["state"].items():
                    check(value, shards[0]["state"][name], f"full shard state={name}")
            for shard in shards:
                del shard["state"]
            summary["full_shard_training"] = {"clients": clients, "results": shards,
                                              "private_models_agree": True}
        summary["status"] = "passed"
        save()
        return 0
    except BaseException as error:
        import traceback

        traceback.print_exc()
        summary.update(status="failed", reason=str(error)[:700])
        save()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--clients", type=int, choices=range(1, 9))
    parser.add_argument("--unsplit", action="store_true")
    parser.add_argument("--compiled-suffix", action="store_true")
    parser.add_argument("--compiled-training", action="store_true")
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument("--full-shard-examples", type=int)
    args = parser.parse_args()
    if args.compiled_training and not args.unsplit:
        parser.error("The complete native training function control requires --unsplit")
    if args.full_shard_examples is not None and (
            not (args.full and args.unsplit and args.compiled_training)
            or not 1 <= args.full_shard_examples <= 45_000):
        parser.error("Full-shard control requires full compiled unsplit training and 1..45000 examples")
    raise SystemExit(
        worker(
            args.output,
            args.device,
            args.full,
            args.clients,
            args.unsplit,
            args.compiled_suffix,
            args.compiled_training,
            args.cold_start,
            args.full_shard_examples,
        )
        if args.worker
        else launch(
            args.device,
            args.full,
            args.clients,
            args.unsplit,
            args.compiled_suffix,
            args.compiled_training,
            args.cold_start,
            args.full_shard_examples,
        )
    )
