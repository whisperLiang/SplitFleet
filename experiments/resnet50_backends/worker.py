"""Native admission and physical Flower/gRPC server/client entry points."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

from .common import (BACKENDS, Images, client_identities, file_hash, learning_rate, save_json,
                     training_batch_sizes, validate_client_budget, validate_round_budget)


@contextmanager
def execution(backend, device):
    from splitfleet.backends.utils import native_device_scope
    with native_device_scope(backend, device):
        yield


def initialize(args):
    # Flower optionally imports TensorFlow; its LLVM symbols conflict with Paddle.
    if args.backend != "tensorflow":
        sys.modules["tensorflow"] = None
    from tests.integration.resnet50_reference.native import (
        build_model, configure_backend, native_batch, training_mode,
    )
    from experiments.resnet50_quality.model_contract import NAMES, state_hash
    from splitfleet.virtual.backend_ops import FrameworkOps

    configure_backend(args.backend)
    config = json.loads((args.bundle / "protocol.json").read_text())
    if args.identity and "logical_clients" in config:
        expected = config["logical_clients"][args.index]
        if (args.identity, args.device, args.physical_host) != (
                expected["id"], expected["device"], expected["host_id"]):
            raise RuntimeError("Worker launch disagrees with its frozen logical client specification")
    for name in ("initial.npz", f"client_{args.index}.npz" if args.command != "server" else "validation.npz"):
        if file_hash(args.bundle / name) != config["files"][name]:
            raise RuntimeError(f"Frozen input bytes changed: {name}")
    with np.load(args.bundle / "initial.npz") as record:
        initial = {name: record[name].copy() for name in NAMES}
    if state_hash(initial) != config["initial_hash"]:
        raise RuntimeError("Canonical initialization hash mismatch")
    model = build_model(args.backend, initial, args.device)
    training_mode(args.backend, model)
    FrameworkOps(args.backend).assert_model_device(model, args.device)
    sample = Images(args.bundle / ("validation.npz" if args.command == "server" else
                                  f"client_{getattr(args, 'index', 0)}.npz"), config)
    size = getattr(args, "admission_batch_size", 2) if args.command == "admit" else 2
    if args.command == "admit" and not 1 <= size <= min(config["batch_size"], len(sample.labels)):
        raise ValueError("Admission batch size must fit the configured batch size and client dataset")
    images, labels = native_batch(args.backend, sample.images(list(range(size))), sample.labels[:size], args.device)
    return config, initial, model, images, labels


def receipt(backend, model, device):
    from splitfleet.virtual.backend_ops import FrameworkOps
    versions = {}
    for name in ("torch", "torchlens", "flwr", "grpcio", "numpy", "tensorflow", "jax", "jaxlib",
                 "paddlepaddle", "paddlepaddle-gpu", "pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {**FrameworkOps(backend).environment_receipt(model, device),
            "host": socket.gethostname(), "machine": platform.machine(), "pid": os.getpid(),
            "python": sys.version, "executable": sys.executable, "packages": versions,
            "logical_client_id": os.environ.get("SPLITFLEET_CLIENT_ID", "server"),
            "physical_host": os.environ.get("SPLITFLEET_PHYSICAL_HOST", "coordinator"),
            "JAX_DISABLE_JIT": os.environ.get("JAX_DISABLE_JIT"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "boot_id": Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            "jax_training_program_cache": bool(getattr(model, '_splitfleet_jax_training_program_cache', False))}


def endpoints(handle, backend):
    from tests.integration.resnet50_reference.array_helpers import stage_boundaries
    return stage_boundaries(handle, backend)


def calibration_batch(args, model, config):
    """Measure a real training minibatch while retaining the B=2 capture sample."""
    from tests.integration.resnet50_reference.native import call_inputs, native_batch
    from splitfleet.tasks import ModelInputs, TaskBatch

    # The coordinator retains every frozen training partition. Its first
    # client supplies a representative batch even for a tiny validation set.
    index = args.index if args.command == "client" else 0
    name = f"client_{index}.npz"
    if file_hash(args.bundle / name) != config["files"][name]:
        raise RuntimeError(f"Frozen input bytes changed: {name}")
    data = Images(args.bundle / name, config)
    ids, raw, targets = next(data.batches(client=index, round_id=1))
    images, labels = native_batch(args.backend, raw, targets, args.device)
    return TaskBatch(ModelInputs(args=call_inputs(args.backend, model, images)),
                     labels, num_examples=len(ids))


def admit(args):
    from .common import ATOL, RTOL, compare_tensor
    from experiments.resnet50_quality.model_contract import NAMES, SCHEMA, TRAINABLE_NAMES, state_hash
    from tests.integration.resnet50_reference.native import (
        build_model, call_inputs, export_state, numpy_value, optimizer, train_step, training_mode,
    )
    from tests.integration.resnet50_reference.array_helpers import (
        RecordingLoss, RecordingOptimizer, canonical_array,
    )
    from splitfleet.autosplit import AutoSplitSession
    from splitfleet.autosplit.state_ownership import state_ownership
    from splitfleet.backends.utils import adapter_for
    from splitfleet.tasks.losses import extract_logits, sparse_cross_entropy
    from splitfleet.transport import encode_boundary, decode_boundary, encode_gradients, decode_gradients
    from splitfleet.transport.split_wire import (
        boundary_to_envelope, envelope_to_boundary, gradients_to_envelope, envelope_to_gradients,
    )

    config, initial, reference, images, labels = initialize(args)
    started = time.perf_counter()
    with execution(args.backend, args.device):
        expected = train_step(args.backend, reference, images, labels, capture=True)
        expected_state = export_state(args.backend, reference)
    del reference
    gc.collect()
    record = {"status": "running", "backend": args.backend, "atol": ATOL, "rtol": RTOL,
              "initial_hash": config["initial_hash"], "batch_size": int(labels.shape[0]),
              "cuts": [], "environment": None}
    save_json(args.output, record)
    model = build_model(args.backend, initial, args.device)
    training_mode(args.backend, model)
    with execution(args.backend, args.device):
        inputs = call_inputs(args.backend, model, images)
        handle = AutoSplitSession(device=args.device).prepare_runtime(
            model, inputs, trainable=True, batch_axes={"/args/1" if args.backend == "jax" else "/args/0": 0},
            dynamic_batch=(1, config["batch_size"]))
        all_endpoints = endpoints(handle, args.backend)
        adapter = adapter_for(model, inputs)
        native_initial = adapter.export_ndarrays(model)
    record["environment"] = receipt(args.backend, model, args.device)
    save_json(args.output, record)
    requested = getattr(args, "verify_boundary", None)
    choices = [(None, {"boundary": requested})] if requested else [
        (stage, all_endpoints[f"stage{stage}"]) for stage in (1, 2, 3)]
    for stage, endpoint in choices:
        with execution(args.backend, args.device):
            adapter.load_ndarrays(model, native_initial)
            inputs = call_inputs(args.backend, model, images)
            handle = handle.backend.repartition(endpoint["boundary"])
            ownership = state_ownership(handle, adapter.state_manifest(model).schema_hash)
            owners = dict(zip(ownership["names"], ownership["owners"], strict=True))
            expected_owners = ({row.name: "prefix" if row.stage <= stage else "suffix" for row in SCHEMA}
                               if stage is not None else owners)
            if (owners != expected_owners or set(owners.values()) != {"prefix", "suffix"}
                    or state_hash(export_state(args.backend, model)) != config["initial_hash"]):
                raise RuntimeError("Capture changed initial state or native stage ownership")
            if args.backend == "jax":
                prefix_opt = suffix_opt = None
            else:
                # As in physical execution, each stage has its own optimizer.
                prefix_opt = RecordingOptimizer(args.backend, model)
                suffix_opt = RecordingOptimizer(args.backend, model)
                for owner, recorder in (("prefix", prefix_opt), ("suffix", suffix_opt)):
                    recorder.parameters = {name: value for name, value in recorder.parameters.items()
                                           if owners[name] == owner}
                    if args.backend == "paddle":
                        import paddle
                        recorder.native = paddle.optimizer.SGD(
                            learning_rate=.05, parameters=list(recorder.parameters.values()))
            local = handle.backend.run_prefix(*inputs, training=True)
            from splitfleet.split_engine import graph_contract_for_runtime_handle
            contract = graph_contract_for_runtime_handle(handle)
            envelope = boundary_to_envelope(local, round_id=1, client_id="admission", step_id="1",
                plan_id=handle.plan.plan_id, split_id=contract.split_id,
                canonical_graph_hash=contract.canonical_graph_hash,
                boundary_schema_hash=contract.boundary_schema_hash, model_version=1)
            wire = encode_boundary(envelope)
            decoded = decode_boundary(wire)
            remote = envelope_to_boundary(decoded, handle.runtime, args.device)
            observed_loss = RecordingLoss(args.backend)
            loss, gradients = handle.backend.train_suffix(remote, labels,
                loss_fn=observed_loss, optimizer=suffix_opt)
            gradient_wire = encode_gradients(gradients_to_envelope(decoded, gradients))
            returned = envelope_to_gradients(decode_gradients(gradient_wire), args.device)
            update = handle.backend.backward_prefix(local, returned, optimizer=prefix_opt)
            if args.backend == "jax":
                native_gradients = update["inputs"][0]
                grads = {name: numpy_value(native_gradients[name]) for name in TRAINABLE_NAMES}
                model.set_params({name: value - np.float32(.05) * native_gradients[name]
                                  for name, value in model.params.items()})
            else:
                if set(prefix_opt.gradients) & set(suffix_opt.gradients):
                    raise RuntimeError("Prefix and suffix both updated the same parameter")
                grads = {**prefix_opt.gradients, **suffix_opt.gradients}
                if set(grads) != set(TRAINABLE_NAMES):
                    raise RuntimeError("Split SGD omitted a native parameter gradient")
            statistics = getattr(handle.backend, "train_suffix_auxiliary", {})
            if statistics:
                model.apply_statistics(statistics)
            actual = export_state(args.backend, model)
            if len(observed_loss.calls) != 1:
                raise RuntimeError("Admission must observe the single differentiated suffix forward")
            checks = [compare_tensor(np.asarray(numpy_value(loss), np.float32),
                                     np.asarray(expected["loss"], np.float32), name="loss"),
                      compare_tensor(observed_loss.calls[0]["logits"], expected["logits"], name="logits")]
            for name in TRAINABLE_NAMES:
                checks.append(compare_tensor(grads[name], canonical_array(args.backend, name,
                    expected["gradients"][name]), name="gradient:" + name))
            checks.extend(compare_tensor(actual[name], expected_state[name], name="state:" + name) for name in NAMES)
            checks.extend(compare_tensor(actual[name] - initial[name], expected_state[name] - initial[name],
                                         name="sgd_update:" + name) for name in TRAINABLE_NAMES)
            counters = {name: int(actual[name]) for name in NAMES if name.endswith("num_batches_tracked")}
            if len(counters) != 53 or set(counters.values()) != {1}:
                raise RuntimeError("BatchNorm state was not committed exactly once")
            passed = all(row["passed"] for row in checks)
            row = {"stage": stage, **endpoint, "status": "passed" if passed else "failed",
                   "checks": checks, "ownership": ownership, "activation_wire_bytes": len(wire),
                   "gradient_wire_bytes": len(gradient_wire), "batch_validation": handle.runtime.batch_validation,
                   "gradient_tensor_count": len(grads), "state_tensor_count": len(actual),
                   "bn_counters": counters}
            record["cuts"].append(row)
            record["environment"] = receipt(args.backend, model, args.device)
            save_json(args.output, record)
            print(json.dumps({"event": "admission_cut", "stage": stage, "status": row["status"],
                              "backend": args.backend, "failed_checks": [v["name"] for v in checks if not v["passed"]]}), flush=True)
            if not passed:
                raise RuntimeError(f"Native/split equivalence failed at stage{stage}")
        del prefix_opt, suffix_opt, local, remote, gradients, returned, update, actual, grads
        gc.collect()
    record.update(status="passed", duration_sec=time.perf_counter() - started)
    save_json(args.output, record)


def admit_selected_boundaries(args, boundaries, config):
    """Strictly verify every selected cut at all actual training batch shapes."""
    directory = args.output.parent / "selected_admission"
    directory.mkdir(exist_ok=True)
    for boundary in sorted(set(boundaries)):
        identity = args.identity or "server"
        digest = hashlib.sha256(boundary.encode()).hexdigest()[:16]
        for size in training_batch_sizes(config):
            output = directory / f"{identity}_{digest}_b{size}.json"
            if output.exists():
                previous = json.loads(output.read_text())
                if (previous.get("status") == "passed" and previous.get("batch_size") == size
                        and len(previous["cuts"]) == 1
                        and previous["cuts"][0]["boundary"] == boundary):
                    continue
                raise RuntimeError("A selected-cut admission failed previously; retain its receipt")
            verification = SimpleNamespace(**{**vars(args), "command": "admit", "output": output,
                "verify_boundary": boundary, "admission_batch_size": size})
            admit(verification)
            gc.collect()


class Batches:
    def __init__(self, args, model, config):
        self.args, self.model, self.config = args, model, config
        self.data = Images(args.bundle / f"client_{args.index}.npz", config)
        self.round_id, self.executed, self.digest = 0, 0, ""

    def __len__(self):
        return math.ceil(len(self.data.labels) / self.config["batch_size"])

    def __iter__(self):
        from tests.integration.resnet50_reference.native import call_inputs, native_batch
        from splitfleet.tasks import ModelInputs, TaskBatch
        digest = hashlib.sha256()
        self.executed = 0
        started = time.perf_counter()
        total_batches = (len(self.data.labels) + self.config["batch_size"] - 1) // self.config["batch_size"]
        for ids, raw, targets in self.data.batches(client=self.args.index, round_id=self.round_id):
            images, labels = native_batch(self.args.backend, raw, targets, self.args.device)
            digest.update(ids.astype("<i8").tobytes())
            digest.update(raw.tobytes())
            digest.update(targets.tobytes())
            yield TaskBatch(ModelInputs(args=call_inputs(self.args.backend, self.model, images)),
                            labels, num_examples=len(ids))
            self.executed += 1
            self.digest = digest.hexdigest()
            if self.executed % 25 == 0 or self.executed == total_batches:
                print(json.dumps({"event": "client_batches", "identity": self.args.identity,
                                  "round": self.round_id, "completed_batches": self.executed,
                                  "total_batches": total_batches,
                                  "elapsed_sec": time.perf_counter() - started}), flush=True)


def client(args):
    config, _, model, images, labels = initialize(args)
    from tests.integration.resnet50_reference.native import call_inputs, optimizer, train_step
    from splitfleet.backends.utils import adapter_for
    from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
    from splitfleet.tasks import ImageClassificationTask
    from splitfleet.client.app import start_client
    from flwr.client import NumPyClient, start_client as start_full_client

    batches = Batches(args, model, config)
    records = []
    adapter = adapter_for(model, call_inputs(args.backend, model, images))
    current_lr = [config["learning_rate"]]

    def make_optimizer(value):
        return optimizer(args.backend, value, current_lr[0])

    def record(round_id, values, examples, metrics, started):
        metrics.update(client_index=args.index, data_batch_hash=batches.digest,
                       device=args.device, host=socket.gethostname(), machine=platform.machine(),
                       pid=os.getpid(), logical_client_id=args.identity, physical_host=args.physical_host)
        validate_client_budget({"client_index": args.index, "num_examples": examples, "metrics": metrics}, config)
        if len(records) != round_id - 1:
            raise RuntimeError("Client round identity was repeated or skipped")
        records.append({"round": round_id, "num_examples": examples, "metrics": metrics,
                        "duration_sec": time.perf_counter() - started})
        save_json(args.output, {"status": "running", "backend": args.backend, "method": args.method,
                  "environment": receipt(args.backend, model, args.device), "rounds": records})
        print(json.dumps({"event": "client_round", "round": round_id, "identity": args.identity,
                          "loss": metrics["loss"], "num_examples": examples}), flush=True)
        return values, examples, metrics

    class SplitClient(AutoSplitSplitLearningClient):
        def get_properties(self, config):
            properties = super().get_properties(config)
            if "framework_backend" in properties:
                properties.update(logical_client_id=args.identity, physical_host=args.physical_host,
                                  batch_size=training_batch_sizes(batches.config)[0], num_batches=len(batches))
            return properties

        def fit(self, parameters, round_config):
            from splitfleet.common.constants import AUTOSPLIT_BOUNDARY_CONFIG_KEY
            batches.round_id = int(round_config["physical_round"])
            current_lr[0] = learning_rate(batches.round_id, config)
            verification_started = time.perf_counter()
            if args.placement == "greedy":
                with execution(args.backend, args.device):
                    admit_selected_boundaries(args, [round_config[AUTOSPLIT_BOUNDARY_CONFIG_KEY]], config)
            verification_seconds = time.perf_counter() - verification_started
            started = time.perf_counter()
            with execution(args.backend, args.device):
                result = super().fit(parameters, round_config)
            result[2].update(placement_policy=args.placement,
                             selected_boundary=round_config[AUTOSPLIT_BOUNDARY_CONFIG_KEY],
                             selected_admission_sec=verification_seconds)
            # Keep numerical admission out of the measured transport residual.
            result[2]["client_fit_handler_ms"] += verification_seconds * 1000
            return record(batches.round_id, *result, started)

    class FullClient(NumPyClient):
        def get_parameters(self, config):
            return adapter.export_ndarrays(model)

        def fit(self, parameters, round_config):
            batches.round_id = int(round_config["physical_round"])
            current_lr[0] = learning_rate(batches.round_id, config)
            adapter.load_ndarrays(model, parameters)
            opt = make_optimizer(model)
            started, total, loss = time.perf_counter(), 0, 0.
            with execution(args.backend, args.device):
                for batch in batches:
                    result = train_step(args.backend, model, batch.inputs.args[-1], batch.targets,
                                        learning_rate=current_lr[0], native_optimizer=opt)
                    loss += batch.num_examples * result["loss"]
                    total += batch.num_examples
            return record(batches.round_id, adapter.export_ndarrays(model), total,
                          {"loss": loss / total, "num_batches": batches.executed}, started)

        def evaluate(self, parameters, config):
            return 0., 0, {}

    if args.method == "splitfleet":
        admission = json.loads(args.admission.read_text())
        boundary = next(row["boundary"] for row in admission["cuts"] if row["stage"] == 2)
        instance = SplitClient(model=model, sample_inputs=call_inputs(args.backend, model, images),
            train_data=batches, task=ImageClassificationTask(), device=args.device,
            batch_axes={"/args/1" if args.backend == "jax" else "/args/0": 0},
            optimizer_fn=make_optimizer, max_cached_runtimes=1,
            calibration_batch=(calibration_batch(args, model, config) if args.placement == "greedy" else None))
        with execution(args.backend, args.device):
            instance.prewarm_runtime(boundary=boundary, dynamic_batch=(1, config["batch_size"]))
        start_client(server_address=args.address, client=instance.to_client(),
                     max_retries=3, max_wait_time=120, reconnect_after_instruction=False)
    else:
        start_full_client(server_address=args.address, client=FullClient().to_client(),
                          max_retries=3, max_wait_time=120)
    if len(records) != config["rounds"]:
        raise RuntimeError("Client exited before completing the frozen budget")
    result = json.loads(args.output.read_text())
    result["status"] = "completed"
    save_json(args.output, result)


def evaluate(model, data, backend, device, *, batch_size=128):
    from tests.integration.resnet50_reference.native import call_inputs, native_batch, numpy_value, training_mode
    from splitfleet.tasks.losses import extract_logits
    from splitfleet.backends.utils import inference_context
    training_mode(backend, model, False)
    total, correct = 0., 0
    with inference_context("tf" if backend == "tensorflow" else backend):
        for start in range(0, len(data.labels), batch_size):
            positions = np.arange(start, min(start + batch_size, len(data.labels)))
            images, _ = native_batch(backend, data.images(positions), data.labels[positions], device)
            logits = numpy_value(extract_logits(model(*call_inputs(backend, model, images))))
            if not np.isfinite(logits).all():
                raise FloatingPointError("Nonfinite validation logits")
            shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
            total += float((np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(positions)), data.labels[positions]]).sum())
            correct += int((logits.argmax(axis=1) == data.labels[positions]).sum())
    training_mode(backend, model)
    return {"loss": total / len(data.labels), "top1": correct / len(data.labels), "samples": len(data.labels)}


def _publish_server_round(output, record):
    """Commit metrics before publishing the durable checkpoint or pruning it."""
    save_json(output, record)
    row = record["rounds"][-1]
    round_id = row["round"]
    if not round_id:
        return
    save_json(output.parent / "latest_checkpoint.json", {"round": round_id,
              "path": f"checkpoints/round_{round_id:03d}.npz", "state_hash": row["checkpoint_state_hash"]})
    if record["protocol"].get("checkpoint_retention") == "latest_and_best" and round_id > 1:
        (output.parent / "checkpoints" / f"round_{round_id - 1:03d}.npz").unlink(missing_ok=True)


def server(args):
    started = time.perf_counter()
    config, _, model, images, _ = initialize(args)
    from tests.integration.resnet50_reference.native import call_inputs, export_state, optimizer
    from splitfleet.server.strategy import AutoSplitStrategy
    from splitfleet.autosplit import PlacementConstraint
    from splitfleet.tasks import ImageClassificationTask
    from splitfleet.backends.utils import adapter_for
    from splitfleet.autosplit.state_ownership import aggregate_owned
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerConfig, start_server as start_full_server
    from flwr.server.strategy import FedAvg
    from splitfleet.server.app import start_server

    adapter = adapter_for(model, call_inputs(args.backend, model, images))
    validation = Images(args.bundle / "validation.npz", config)
    fits, suffixes, rounds = {}, {}, []
    current_lr, best = [config["learning_rate"]], [-1.]

    def configure(round_id):
        current_lr[0] = learning_rate(round_id, config)
        return {"physical_round": round_id, "learning_rate": current_lr[0]}

    def record_fits(round_id, results, failures):
        if failures:
            raise RuntimeError(f"Incomplete physical round {round_id}: {failures}")
        results.sort(key=lambda item: int(item[1].metrics["client_index"]))
        rows = [{"client_index": int(r.metrics["client_index"]), "num_examples": r.num_examples,
                 "metrics": dict(r.metrics)} for _, r in results]
        validate_round_budget(rows, config)
        if {row["metrics"]["logical_client_id"] for row in rows} != set(client_identities(config)):
            raise RuntimeError("Round did not use every declared CPU/GPU logical client")
        fits[round_id] = rows
        return results

    def assessment(round_id, arrays):
        adapter.load_ndarrays(model, arrays)
        metrics = evaluate(model, validation, args.backend, args.device)
        state = export_state(args.backend, model)
        row = {"round": round_id, "val_ce": metrics["loss"], "val_top1": metrics["top1"],
               "elapsed_wall_sec": time.perf_counter() - started, "clients": fits.get(round_id, []),
               "suffixes": suffixes.get(round_id, [])}
        if round_id:
            budget = validate_round_budget(row["clients"], config)
            row.update(**budget, train_online_ce=sum(v["num_examples"] * v["metrics"]["loss"]
                       for v in row["clients"]) / config["train_samples"], learning_rate=current_lr[0])
            checkpoint = args.output.parent / "checkpoints"
            checkpoint.mkdir(exist_ok=True)
            temporary = checkpoint / f"round_{round_id:03d}.tmp.npz"
            np.savez(temporary, **state)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.replace(checkpoint / f"round_{round_id:03d}.npz")
            checkpoint_directory = os.open(checkpoint, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(checkpoint_directory)
            finally:
                os.close(checkpoint_directory)
            from experiments.resnet50_quality.model_contract import state_hash
            row["checkpoint_state_hash"] = state_hash(state)
            if metrics["top1"] > best[0]:
                best_temporary = args.output.parent / "best_validation.tmp.npz"
                np.savez(best_temporary, **state)
                with best_temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                best_temporary.replace(args.output.parent / "best_validation.npz")
                best[0] = metrics["top1"]
        rounds.append(row)
        _publish_server_round(args.output, {"status": "running", "backend": args.backend, "method": args.method,
                              "protocol": config, "environment": receipt(args.backend, model, args.device), "rounds": rounds})
        print(json.dumps({"event": "server_round", "backend": args.backend, "method": args.method,
                          "round": round_id, **metrics, "elapsed_sec": row["elapsed_wall_sec"]}), flush=True)
        return metrics["loss"], {"top1": metrics["top1"]}

    class SplitStrategy(AutoSplitStrategy):
        def configure_fit(self, server_round, parameters, client_manager):
            instructions = super().configure_fit(server_round, parameters, client_manager)
            if self.placement_policy is not None:
                from splitfleet.common.constants import AUTOSPLIT_BOUNDARY_CONFIG_KEY
                policy = self.placement_policy
                assigned = {str(client.cid): ins.config[AUTOSPLIT_BOUNDARY_CONFIG_KEY]
                            for client, ins in instructions}
                diagnostics = policy.round_diagnostics[server_round]
                if len(assigned) != config["clients"] or assigned != diagnostics["assignment"]:
                    raise RuntimeError("Every physical client must receive its own Greedy decision")
                with execution(args.backend, args.device):
                    admit_selected_boundaries(args, assigned.values(), config)
                save_json(args.output.parent / f"placement_round_{server_round:03d}.json",
                    {"round": server_round, "policy": args.placement, "diagnostics": diagnostics,
                     "decisions": [row for row in policy.decision_receipts if row["round_id"] == server_round]})
                save_json(args.output.parent / "calibration.json", policy.telemetry_provider.receipt)
                print(json.dumps({"event": "client_placements", "round": server_round,
                    "decisions": [row for row in policy.decision_receipts if row["round_id"] == server_round]}), flush=True)
            return instructions

        def aggregate_fit(self, server_round, results, failures):
            return super().aggregate_fit(server_round, record_fits(server_round, results, failures), failures)

        def aggregate_server_fit(self, server_round, results):
            if len(results) != config["clients"] or sum(int(r.config["num_examples"]) for r in results) != config["train_samples"]:
                raise RuntimeError("Physical suffix result budget is incomplete")
            suffixes[server_round] = [{"sid": str(r.sid), "num_examples": r.config["num_examples"],
                                      "metrics": dict(r.config)} for r in results]
            return super().aggregate_server_fit(server_round, results)

        def evaluate(self, server_round, client_parameters, server_parameters):
            return assessment(server_round, parameters_to_ndarrays(client_parameters))

    class FullStrategy(FedAvg):
        def __init__(self, **kwargs):
            self.initial = adapter.export_ndarrays(model)
            super().__init__(initial_parameters=ndarrays_to_parameters(self.initial),
                             on_fit_config_fn=configure, accept_failures=False, **kwargs)

        def aggregate_fit(self, server_round, results, failures):
            results = record_fits(server_round, results, failures)
            ownership = {"names": [v.name for v in adapter.state_manifest(model).entries],
                         "owners": ["prefix"] * len(self.initial)}
            arrays = aggregate_owned(self.initial, ((ownership, parameters_to_ndarrays(r.parameters),
                                     [], r.num_examples) for _, r in results), integer_policy="maximum")
            self.initial = arrays
            return ndarrays_to_parameters(arrays), {}

        def evaluate(self, server_round, parameters):
            return assessment(server_round, parameters_to_ndarrays(parameters))

    options = dict(fraction_fit=1., fraction_evaluate=0., min_fit_clients=config["clients"],
                   min_evaluate_clients=0, min_available_clients=config["clients"])
    if args.method == "splitfleet":
        admission = json.loads(args.admission.read_text())
        boundary = next(row["boundary"] for row in admission["cuts"] if row["stage"] == 2)
        policy = None
        if args.placement == "greedy":
            from splitfleet.server.placement import GreedyConfig, GreedyPlacementPolicy, TorchLensCandidateProvider
            provider = TorchLensCandidateProvider(model=model,
                sample_inputs=call_inputs(args.backend, model, images),
                batch_axes={"/args/1" if args.backend == "jax" else "/args/0": 0},
                dynamic_batch=(1, config["batch_size"]), require_trainable_prefix=True,
                resident_parameter_transport=args.backend == "jax")
            policy = GreedyPlacementPolicy(candidate_provider=provider, config=GreedyConfig(
                server_concurrency=1, seed=config["seed"], warm_start=False,
                state_path=str(args.output.parent / "placement_state.json")))
        strategy = SplitStrategy(model=model, sample_inputs=call_inputs(args.backend, model, images),
            task=ImageClassificationTask(), boundary=boundary,
            placement_policy=policy,
            constraints=PlacementConstraint(max_payload_bytes=config["placement_max_payload_bytes"]),
            batch_axes={"/args/1" if args.backend == "jax" else "/args/0": 0},
            dynamic_batch=(1, config["batch_size"]), runtime_device=args.device,
            aggregation_policy="splitfed", owned_state_exchange=True, integer_aggregation_policy="maximum",
            config_client_fit_fn=configure, optimizer_fn=lambda m: optimizer(args.backend, m, current_lr[0]), **options)
        strategy.server_concurrency = 1
        if policy is not None:
            sample = calibration_batch(args, model, config)
            with execution(args.backend, args.device):
                catalog = policy.candidate_provider.get_candidates(training=True)
                save_json(args.output.parent / "candidate_catalog.json", {
                    "policy": args.placement, "candidates": [asdict(row) for row in catalog],
                    "diagnostics": policy.candidate_provider.catalog_diagnostics[True]})
                policy.prepare_calibration(sample.inputs, sample.targets, loss_fn=strategy.loss_fn,
                    device=args.device, optimizer_fn=strategy.optimizer_fn)
        start_server(server_address=args.address, config=ServerConfig(num_rounds=config["rounds"]), strategy=strategy)
    else:
        strategy = FullStrategy(**options)
        start_full_server(server_address=args.address, config=ServerConfig(num_rounds=config["rounds"]), strategy=strategy)
    if [row["round"] for row in rounds if row["round"]] != list(range(1, config["rounds"] + 1)):
        raise RuntimeError("The physical server did not finish every round")
    if file_hash(args.bundle / "test.npz") != config["files"]["test.npz"]:
        raise RuntimeError("Final test data bytes changed")
    test = Images(args.bundle / "test.npz", config)
    final_test = evaluate(model, test, args.backend, args.device)
    from tests.integration.resnet50_reference.native import build_model
    with np.load(args.output.parent / "best_validation.npz") as saved:
        best_model = build_model(args.backend, dict(saved), args.device)
    best_test = evaluate(best_model, test, args.backend, args.device)
    result = json.loads(args.output.read_text())
    result.update(status="completed", duration_sec=time.perf_counter() - started,
                  duration_scope="server initialization, training, per-round validation/checkpoints and two final test evaluations",
                  test={"final": final_test, "best_validation": best_test})
    save_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("admit", "server", "client"))
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("splitfleet", "fedavg"), default="splitfleet")
    parser.add_argument("--placement", choices=("greedy", "fixed_stage2"), default="greedy")
    parser.add_argument("--verify-boundary", help="Native numerical admission for one canonical operation cut")
    parser.add_argument("--admission-batch-size", type=int, default=2)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--address")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--identity", default="")
    parser.add_argument("--physical-host", default="")
    args = parser.parse_args()
    if args.admission_batch_size < 1:
        parser.error("Admission batch size must be positive")
    if args.backend != "tensorflow":
        sys.modules["tensorflow"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.with_suffix(".pid").exists():
        raise FileExistsError("Preserve the prior attempt and choose a new output path")
    args.output.with_suffix(".pid").write_text(str(os.getpid()))
    try:
        globals()[args.command](args)
    except BaseException as error:
        traceback.print_exc()
        record = json.loads(args.output.read_text()) if args.output.exists() else {}
        record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        save_json(args.output, record)
        raise


if __name__ == "__main__":
    main()
