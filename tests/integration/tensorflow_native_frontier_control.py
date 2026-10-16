"""Bounded native CPU evidence for a privately bound TensorFlow frontier.

This independently checks native graph contracts, gradients and model state.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def prepare_control(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "runtime_snapshot"
    paths = sorted(
        path
        for folder in ("experiments", "splitfleet", "tests/integration/resnet50_reference")
        for path in (ROOT / folder).rglob("*.py")
    )
    paths += [Path(__file__), ROOT / "tests/integration/test_native_frontier.py",
              ROOT / "tests/__init__.py", ROOT / "tests/integration/__init__.py"]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    for path in paths:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha256(path) != hashes[str(path.relative_to(ROOT))] or sha256(
            target
        ) != sha256(path):
            raise RuntimeError(f"Source changed while freezing: {path}")
    if hashes != {str(path.relative_to(ROOT)): sha256(path) for path in paths}:
        raise RuntimeError("Source changed during the full control snapshot")
    native_files = {}
    for name in ("planner", "pipeline", "api", "runtime", "training"):
        import importlib

        module = importlib.import_module("torchlens.split." + name)
        native_files[module.__file__] = sha256(module.__file__)
    from torchlens.split.training import _train_suffix_tf

    original_suffix = inspect.getsource(_train_suffix_tf)
    write_json(
        output / "native_suffix_gradient_source.json",
        {
            "file": inspect.getsourcefile(_train_suffix_tf),
            "file_sha256": sha256(inspect.getsourcefile(_train_suffix_tf)),
            "function_signature": str(inspect.signature(_train_suffix_tf)),
            "function_sha256": hashlib.sha256(original_suffix.encode()).hexdigest(),
            "function_source": original_suffix,
            "gradient_calls": [
                row.strip()
                for row in original_suffix.splitlines()
                if "tape.gradient(" in row
            ],
            "first_call_sources": "root_values",
            "second_call_sources": "suffix_sources",
            "second_call_condition": "optimizer is not None and suffix_vars",
            "optimization_implemented": False,
            "measured_speedup": None,
        },
    )
    source_identity = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()
    receipt = {
        "status": "prepared",
        "snapshot": str(snapshot),
        "source_hashes": hashes,
        "source_identity": source_identity,
        "native_source_hashes": native_files,
        "executable": sys.executable,
        "hard_timeout_seconds": 120,
        "cpu_affinity": [6],
        "threads": 1,
        "cuda_visible_devices": "",
        "full_model_batch_sizes": [16, 4],
        "small_model_batch_sizes": [2, 1],
        "automatic_retries": 0,
        "scientific_gate_admission": False,
    }
    write_json(output / "plan.json", receipt)
    return receipt


def _native_handle(model, images):
    from splitfleet.autosplit import torchlens_runtime
    from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend
    from splitfleet.runtime.native_frontier import (
        _clone,
        prepare_native_frontier_runtime,
    )

    # Retain the entire production facade and pinned preparation body. Bind
    # this control instance privately; do not change shared module globals.
    private_prepare = _clone(
        torchlens_runtime.prepare_split_runtime, prepare=prepare_native_frontier_runtime
    )
    backend = TorchLensSplitBackend(device="cpu")
    backend.trace = MethodType(
        _clone(TorchLensSplitBackend.trace, prepare_split_runtime=private_prepare),
        backend,
    )
    backend.trace(
        model,
        images,
        trainable=True,
        batch_axes={"/args/0": 0},
        dynamic_batch=(1, 16),
        boundary="50%",
    )
    # Exercise the original native keyword API as well as recut/materialize.
    backend.runtime = backend.runtime.with_placement(plan=backend.runtime.placement)
    return backend.make_handle()



def portable(value):
    """Serialize native metadata through ordinary APIs, without object addresses."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, dict):
        return {str(key): portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable(item) for item in value]
    if dataclasses.is_dataclass(value):
        return {field.name: portable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if hasattr(value, "as_dict"):
        return portable(value.as_dict())
    # TF dtype values and shape descriptors are public metadata.
    if type(value).__name__ == "DType":
        return {"dtype": value.name}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"native_type": type(value).__module__ + "." + type(value).__qualname__,
                "shape": list(value.shape), "dtype": str(value.dtype),
                "device": str(getattr(value, "device", ""))}
    return {"native_type": type(value).__module__ + "." + type(value).__qualname__, "text": str(value)}



def _plan_proof(handle):
    import numpy as np
    from torchlens.split.runtime import SplitRuntime

    from splitfleet.runtime.native_frontier import frontier_proof
    from splitfleet.split_engine import graph_contract_for_runtime_handle
    from splitfleet.split_engine.contracts import (
        ContractValidationError,
        validate_contract,
    )

    runtime = handle.runtime
    before = portable(runtime.graph_ir.as_dict())
    with runtime._temporary_segment_state_snapshots() as snapshots:
        original = SplitRuntime._analyze_point(
            runtime, runtime.request.point, state_snapshots=snapshots
        )
    if (
        original.plan.prefix_node_ids != runtime.plan.prefix_node_ids
        or original.plan.suffix_node_ids != runtime.plan.suffix_node_ids
    ):
        raise AssertionError("The private frontier changed the native partition")
    old_ir, new_ir = portable(original.graph_ir.as_dict()), before
    if {key: value for key, value in old_ir.items() if key != "boundary_schema"} != {
        key: value for key, value in new_ir.items() if key != "boundary_schema"
    }:
        raise AssertionError("The private frontier changed non-boundary native IR")
    old_runtime = copy.copy(runtime)
    old_runtime.plan, old_runtime.graph_ir = original.plan, original.graph_ir
    old_runtime.capability_report = original.capability_report
    old_handle = SimpleNamespace(
        runtime=old_runtime, model=handle.model, plan=handle.plan
    )
    old_contract = graph_contract_for_runtime_handle(old_handle)
    new_contract = graph_contract_for_runtime_handle(handle)
    try:
        validate_contract(old_contract, new_contract)
    except ContractValidationError as error:
        if error.mismatch.field != "boundary_schema_hash":
            raise
    else:
        raise AssertionError(
            "An old boundary contract incorrectly admitted the new ABI"
        )

    proof = frontier_proof(runtime.trace_graph, runtime.plan)

    def fields(plan, batch):
        rows = []
        for key, schema in plan.boundary_spec.items():
            shape = schema.shape.as_tuple()
            resolved = [batch if value == "B" else int(value) for value in shape]
            dtype = schema.dtype.replace("<dtype: '", "").replace("'>", "")
            rows.append(
                {
                    "field_id": key,
                    "shape": resolved,
                    "dtype": dtype,
                    "tensor_bytes": int(np.prod(resolved, dtype=np.int64))
                    * np.dtype(dtype).itemsize,
                }
            )
        return rows

    proof.update(
        boundary=runtime.request.boundary,
        original_contract=old_contract.to_dict(),
        new_contract=new_contract.to_dict(),
        native_partition_unchanged=True,
        native_non_boundary_ir_unchanged=True,
        batch_sizes={
            str(batch): {
                "original_fields": fields(original.plan, batch),
                "new_fields": fields(runtime.plan, batch),
            }
            for batch in (1, 4, 16)
        },
        original_ir=old_ir,
        new_ir=new_ir,
    )
    return proof


class _RecordingOptimizer:
    def __init__(self, model):
        from tests.integration.resnet50_reference.native import native_parameters, optimizer

        self.parameters = native_parameters("tensorflow", model)
        self.native = optimizer("tensorflow", model, 0.05)
        self.names = {
            id(candidate): name
            for name, variable in self.parameters.items()
            for candidate in (variable, variable.value, variable.handle)
        }
        self.gradients, self.steps = {}, 0

    def apply_gradients(self, pairs):
        pairs = list(pairs)
        for gradient, variable in pairs:
            name = self.names[id(variable)]
            if name in self.gradients:
                raise AssertionError(f"Duplicate SGD update: {name}")
            self.gradients[name] = gradient.numpy().copy()
        self.steps += 1
        self.native.apply_gradients(pairs)


def _step(handle, images, labels):
    from tests.integration.resnet50_reference.array_helpers import RecordingLoss
    from splitfleet.autosplit.state_ownership import state_ownership
    from splitfleet.backends.utils import adapter_for
    from splitfleet.runtime.auxiliary_state import commit_auxiliary_state
    from splitfleet.split_engine import graph_contract_for_runtime_handle
    from splitfleet.transport import encode_boundary
    from splitfleet.transport.split_wire import boundary_to_envelope

    prefix, suffix = (
        _RecordingOptimizer(handle.model),
        _RecordingOptimizer(handle.model),
    )
    recording_loss = RecordingLoss()
    boundary = handle.backend.run_prefix(images, training=True)
    contract = graph_contract_for_runtime_handle(handle)
    envelope = boundary_to_envelope(
        boundary,
        round_id=1,
        client_id="cpu-control",
        step_id="1",
        plan_id=handle.plan.plan_id,
        split_id=handle.runtime.split_id,
        canonical_graph_hash=contract.canonical_graph_hash,
        boundary_schema_hash=contract.boundary_schema_hash,
        model_version=1,
    )
    actual_payload = len(encode_boundary(envelope))
    loss, gradients = handle.backend.train_suffix(
        boundary, labels, loss_fn=recording_loss, optimizer=suffix
    )
    handle.backend.backward_prefix(boundary, gradients, optimizer=prefix)
    statistics = handle.backend.train_suffix_auxiliary
    commit_auxiliary_state(handle.model, statistics)
    if len(recording_loss.calls) != 1 or prefix.steps != 1 or suffix.steps != 1:
        raise AssertionError(
            "Each native stage must perform exactly one real forward/SGD"
        )
    if prefix.gradients.keys() & suffix.gradients.keys():
        raise AssertionError("Native trainable state has overlapping updates")
    ownership = state_ownership(
        handle,
        adapter_for(handle.model, (images,)).state_manifest(handle.model).schema_hash,
    )
    owners = dict(zip(ownership["names"], ownership["owners"], strict=True))
    for role, recorder in (("prefix", prefix), ("suffix", suffix)):
        expected = {name for name in recorder.parameters if owners[name] == role}
        if set(recorder.gradients) != expected:
            raise AssertionError(
                f"Actual {role} SGD gradient names differ from proven native ownership"
            )
    result = dict(
        recording_loss.calls[0], gradients={**prefix.gradients, **suffix.gradients}
    )
    return result, {
        "prefix_gradients": len(prefix.gradients),
        "suffix_gradients": len(suffix.gradients),
        "bn_statistics": len(statistics),
        "boundary_fields": len(boundary.tensors),
        "actual_wire_bytes": actual_payload,
        "actual_tensor_bytes": sum(
            tensor.numpy().nbytes for tensor in boundary.tensors.values()
        ),
        "boundary_gradient_fields": list(gradients),
        "loss": float(loss.numpy()),
    }


def _small_model(initial=None):
    import tensorflow as tf

    from tests.integration.resnet50_reference.tf_model import CanonicalBatchNorm

    class Model(tf.keras.Model):
        def __init__(self):
            super().__init__(name="frontier_small", dtype="float32")
            self.device, self.training, self.defer_statistics = "/CPU:0", True, True
            self.conv1 = tf.keras.layers.Conv2D(
                4, 3, padding="same", use_bias=False, name="conv1"
            )
            self.bn1 = CanonicalBatchNorm(4, name="bn1")
            self.relu = tf.keras.layers.ReLU(name="relu")
            self.avgpool = tf.keras.layers.GlobalAveragePooling2D(name="avgpool")
            self.fc = tf.keras.layers.Dense(10, name="fc")

        def call(self, images, training=None):
            training = self.training if training is None else training
            hidden, stats = self.bn1(
                self.conv1(images), training=training, defer_statistics=True
            )
            return {
                "logits": self.fc(self.avgpool(self.relu(hidden))),
                "batch_statistics": {"bn1": stats},
            }

        def canonical_variables(self):
            return {
                "conv1.weight": self.conv1.kernel,
                "bn1.weight": self.bn1.gamma,
                "bn1.bias": self.bn1.beta,
                "bn1.running_mean": self.bn1.running_mean,
                "bn1.running_var": self.bn1.running_var,
                "bn1.num_batches_tracked": self.bn1.num_batches_tracked,
                "fc.weight": self.fc.kernel,
                "fc.bias": self.fc.bias,
            }

        def state_associations(self):
            return {
                name: ("bn1.weight", "bn1.bias")
                for name in self.canonical_variables()
                if name.startswith("bn1.") and name not in {"bn1.weight", "bn1.bias"}
            }

        def apply_statistics(self, statistics, names=None):
            if names is not None and set(names) != {"bn1"}:
                raise AssertionError("Small control received incorrect BN owners")
            self.bn1.apply_statistics(statistics["bn1"])

        def __deepcopy__(self, memo):
            # The native probe verifies exact layer TYPES as well as independent
            # variables. Reuse this declared class, never a new local subclass.
            with tf.device("/CPU:0"):
                clone = type(self)()
                clone(tf.zeros((2, 8, 8, 3), tf.float32))
                for name, variable in clone.canonical_variables().items():
                    variable.assign(self.canonical_variables()[name])
            memo[id(self)] = clone
            clone.training = self.training
            return clone

    with tf.device("/CPU:0"):
        model = Model()
        model(tf.zeros((2, 8, 8, 3), tf.float32))
        if initial is not None:
            for name, variable in model.canonical_variables().items():
                variable.assign(initial[name])
    return model


def _comparison(actual, expected, state, expected_state, *, full):
    from tests.integration.resnet50_reference.array_helpers import canonical_array
    from experiments.resnet50_quality.gate_torch import compare_tensor

    actual_gradients, expected_gradients = actual["gradients"], expected["gradients"]
    if set(actual_gradients) != set(expected_gradients):
        raise AssertionError(
            "Native split training gradient names differ from full-model CE"
        )
    checks = [compare_tensor(actual["logits"], expected["logits"], name="logits")]
    for name in expected_gradients:
        left, right = actual_gradients[name], expected_gradients[name]
        if full:
            left, right = (
                canonical_array("tensorflow", name, left),
                canonical_array("tensorflow", name, right),
            )
        checks.append(compare_tensor(left, right, name="gradient:" + name))
    checks.extend(
        compare_tensor(state[name], expected_state[name], name="state:" + name)
        for name in expected_state
    )
    if abs(actual["loss"] - expected["loss"]) > 1e-5 + 1e-4 * abs(expected["loss"]):
        raise AssertionError("Real native cross entropy differs from full-model CE")
    failed = [row for row in checks if not row["passed"]]
    if failed:
        raise AssertionError(f"Native frontier numerical mismatches: {failed}")
    return checks


def run_control(output, data_root, *, small_only=False):
    output = Path(output).resolve()
    plan = json.loads((output / "plan.json").read_text())
    snapshot = Path(plan["snapshot"])
    if (
        Path(__file__).resolve()
        != snapshot / "tests/integration/tensorflow_native_frontier_control.py"
    ):
        raise RuntimeError("Execute only the frozen control helper")
    for name, digest in plan["source_hashes"].items():
        if sha256(snapshot / name) != digest:
            raise RuntimeError(f"Frozen source changed: {name}")
    for name, digest in plan["native_source_hashes"].items():
        if sha256(name) != digest:
            raise RuntimeError(f"Pinned native source changed: {name}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Frontier control must run without visible GPUs")
    os.sched_setaffinity(0, {6})
    receipt = {
        "status": "running",
        "source_identity": plan["source_identity"],
        "automatic_retries": 0,
        "pid": os.getpid(),
        "linux_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "TF_NUM_INTRAOP_THREADS",
                "TF_NUM_INTEROP_THREADS",
            )
        },
        "scientific_gate_admission": False,
        "cuda_jobs_launched": 0,
    }
    # Fresh output is claimed once; failed native controls are never rerun or
    # overwritten. This is separate from any production experiment admission.
    with (output / "execution_attempt.json").open("x") as stream:
        json.dump(
            {
                "pid": os.getpid(),
                "boot_id": receipt["linux_boot_id"],
                "source_identity": plan["source_identity"],
            },
            stream,
            sort_keys=True,
        )
        stream.flush()
        os.fsync(stream.fileno())
    started = time.monotonic()
    try:
        sys.path.insert(0, str(snapshot))
        import numpy as np
        import tensorflow as tf
        from torchlens.split import planner

        from experiments.resnet50_quality.data import load_cifar
        from tests.integration.resnet50_reference.array_helpers import stage_boundaries
        from experiments.resnet50_quality.model_contract import (
            NAMES,
            TRAINABLE_PARAMETERS,
            canonical_state,
            state_hash,
        )
        from tests.integration.resnet50_reference.native import (
            build_model,
            configure_backend,
            export_state,
            native_batch,
            native_parameters,
            train_step,
            training_mode,
        )
        from splitfleet.autosplit.runtime import AutoSplitSession
        from splitfleet.autosplit.state_ownership import state_ownership
        from splitfleet.backends.utils import adapter_for, tensorflow_model_device
        from splitfleet.runtime.native_frontier import _PINNED_FUNCTIONS

        configure_backend("tensorflow")
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        receipt["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("torchlens", "tensorflow", "numpy")
        }
        receipt["native_functions"] = {
            f"{module}.{name}": {"sha256": source, "signature": signature}
            for (module, name), (source, signature) in _PINNED_FUNCTIONS.items()
        }
        receipt["native_frontier_source"] = {
            "file": inspect.getsourcefile(planner),
            "frontier_sha256": hashlib.sha256(
                inspect.getsource(planner._frontier_node_ids).encode()
            ).hexdigest(),
        }
        native_frontier = planner._frontier_node_ids
        rng = np.random.default_rng(233)
        model = _small_model()
        initial = {
            name: v.numpy().copy() for name, v in model.canonical_variables().items()
        }
        oracle = _small_model(initial)
        with tf.device("/CPU:0"):
            images = tf.convert_to_tensor(
                rng.normal(size=(2, 8, 8, 3)).astype(np.float32)
            )
        session = AutoSplitSession(device="cpu")
        handle = _native_handle(model, images)
        node = [
            row
            for row in handle.runtime.trace_graph.nodes
            if row.module_path == "relu" and not row.is_output
        ][-1]
        handle = session.repartition_runtime(handle, "after:" + node.canonical_id)
        small_proof = _plan_proof(handle)
        write_json(output / "small_frontier_proof.json", small_proof)
        receipt["small_model_steps"] = []
        for step, batch in enumerate((2, 1), 1):
            with tf.device("/CPU:0"):
                x = tf.identity(images[:batch])
                labels = tf.convert_to_tensor(np.arange(batch, dtype=np.int64))
            expected = train_step("tensorflow", oracle, x, labels, capture=True)
            actual, step_proof = _step(handle, x, labels)
            checks = _comparison(
                actual,
                expected,
                {n: v.numpy() for n, v in model.canonical_variables().items()},
                {n: v.numpy() for n, v in oracle.canonical_variables().items()},
                full=False,
            )
            ownership = state_ownership(
                handle, adapter_for(model, (x,)).state_manifest(model).schema_hash
            )
            if (
                len(actual["gradients"]) != 5
                or int(model.bn1.num_batches_tracked.numpy()) != step
            ):
                raise AssertionError(
                    "Small native step omitted a parameter or BN counter"
                )
            receipt["small_model_steps"].append(
                dict(step_proof, batch_size=batch, checks=checks, ownership=ownership)
            )
        if not small_only:
            data = load_cifar(data_root)
            ids = data.train_indices[:20]
            host_images = data.images(ids)
            host_labels = data.train_labels[ids]
            images, labels = native_batch("tensorflow", host_images, host_labels, "cpu")
            initial = canonical_state(233)
            receipt["canonical_initial_hash"] = state_hash(initial)
            receipt["real_data"] = {
                "sample_ids": ids.tolist(),
                "image_sha256": hashlib.sha256(host_images.tobytes()).hexdigest(),
                "label_sha256": hashlib.sha256(host_labels.tobytes()).hexdigest(),
            }
            model = build_model("tensorflow", initial, "cpu")
            training_mode("tensorflow", model)
            oracle = build_model("tensorflow", initial, "cpu")
            if (
                len(native_parameters("tensorflow", model)) != 161
                or tuple(export_state("tensorflow", model)) != NAMES
            ):
                raise AssertionError("Full real ResNet50 lost its canonical state")
            if (
                sum(
                    np.prod(v.shape)
                    for v in native_parameters("tensorflow", model).values()
                )
                != TRAINABLE_PARAMETERS
            ):
                raise AssertionError("Full real ResNet50 parameter budget differs")
            handle = _native_handle(model, images[:16])
            endpoints = stage_boundaries(handle, "tensorflow")
            receipt["full_model_steps"] = []
            for step, (stage, start, stop) in enumerate(((1, 0, 16), (3, 16, 20)), 1):
                handle = session.repartition_runtime(
                    handle, endpoints[f"stage{stage}"]["boundary"]
                )
                proof = _plan_proof(handle)
                write_json(output / f"full_stage{stage}_frontier_proof.json", proof)
                expected = train_step(
                    "tensorflow",
                    oracle,
                    images[start:stop],
                    labels[start:stop],
                    capture=True,
                )
                actual, step_proof = _step(
                    handle, images[start:stop], labels[start:stop]
                )
                state, expected_state = (
                    export_state("tensorflow", model),
                    export_state("tensorflow", oracle),
                )
                checks = _comparison(actual, expected, state, expected_state, full=True)
                ownership = state_ownership(
                    handle,
                    adapter_for(model, (images[start:stop],))
                    .state_manifest(model)
                    .schema_hash,
                )
                counters = [
                    int(state[n]) for n in NAMES if n.endswith("num_batches_tracked")
                ]
                if (
                    len(actual["gradients"]) != 161
                    or len(state) != 320
                    or counters != [step] * 53
                ):
                    raise AssertionError(
                        "Full native step omitted state/gradients/BN updates"
                    )
                if tensorflow_model_device(model) != "cpu" or any(
                    not value.device.endswith("CPU:0")
                    for value in model.canonical_variables().values()
                    for value in (value.value,)
                ):
                    raise AssertionError("Real control state left CPU")
                receipt["full_model_steps"].append(
                    dict(
                        step_proof,
                        batch_size=stop - start,
                        stage=stage,
                        checked_trainable_gradients=161,
                        checked_canonical_state_tensors=320,
                        checked_bn_counters=53,
                        counter_values=sorted(set(counters)),
                        checks=checks,
                        ownership=ownership,
                    )
                )
        if planner._frontier_node_ids is not native_frontier:
            raise AssertionError("The installed SDK was modified by private binding")
        receipt.update(
            status="passed",
            native_sdk_unmodified=True,
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            full_model_executed=not small_only,
        )
    except BaseException as error:  # noqa: BLE001 - preserve interrupted native control evidence
        receipt.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        traceback.print_exc()
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "summary.json", receipt)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in ("status", "elapsed_seconds", "source_identity")
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--small-only", action="store_true")
    parser.add_argument("--data-root", default=str(ROOT / "data/cifar-10-batches-py"))
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare_control(args.output), sort_keys=True))
        return 0
    return (
        0
        if run_control(args.output, args.data_root, small_only=args.small_only)[
            "status"
        ]
        == "passed"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
