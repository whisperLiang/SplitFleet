"""Real CIFAR-10 training through SplitFleet's Flower and gRPC round loop.

The prepare/server/client/admit commands also work in isolated remote checkouts.
Experiments keep the semantic cut after flatten, not a backend operation ratio.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np

from experiments.multibackend_cifar import (
    BACKENDS, arrays_hash, batch, build, numpy, optimizer, prepare, save_json,
)


def context(backend, device):
    if backend == "tf":
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        return tf.device(device)
    return nullcontext()


def initialize(args):
    bundle = Path(args.bundle)
    config = json.loads((bundle / "config.json").read_text())
    weights = dict(np.load(bundle / "initial_weights.npz"))
    if arrays_hash(weights) != config["initial_weights_sha256"]:
        raise RuntimeError("Initial model weights differ from the frozen bundle")
    model = build(args.backend, weights, args.device)
    sample = np.load(bundle / "sample.npz")
    call = batch(args.backend, model, sample["images"], sample["labels"], args.device)
    return config, model, call


def environment(backend, model, device):
    from splitfleet.backends import BACKEND_ADAPTERS
    adapter = BACKEND_ADAPTERS.create(backend)
    if backend == "torch":
        import torch
        observed = sorted({str(p.device) for p in model.parameters()})
        accelerator = torch.cuda.get_device_name() if "cuda" in device else None
    elif backend == "tf":
        observed = sorted({str(getattr(p, "device", getattr(getattr(p, "value", None), "device", "unknown")))
                           for p in model.weights})
        accelerator = None
    elif backend == "jax":
        import jax
        observed = [str(d) for d in jax.devices()]
        accelerator = None
    elif backend == "paddle":
        import paddle
        observed = [paddle.device.get_device()]
        accelerator = None
    else:
        observed = sorted({str(p.device) for p in adapter._state(model).values()})
        accelerator = None
    packages = {}
    for name in ("torchlens", "torch", "tensorflow", "jax", "jaxlib", "paddlepaddle", "tinygrad", "flwr"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"hostname": socket.gethostname(), "machine": platform.machine(), "python": sys.version,
            "pid": os.getpid(), "requested_device": device, "observed_devices": observed,
            "accelerator": accelerator, "packages": packages}


def resolve_cut(model, call, config):
    from splitfleet.autosplit import prepare_torchlens_runtime
    seed = prepare_torchlens_runtime(model, call.inputs.args, boundary="50%", trainable=True,
                                    batch_axes={}, dynamic_batch=(config["batch_size"], config["batch_size"]))
    candidates = [node for node in seed.runtime.trace_graph.nodes
                  if not (node.is_input or node.is_output or node.is_param_source or node.is_buffer)
                  and tuple(node.output_shape or ()) == (config["batch_size"], 1024)]
    if not candidates:
        raise RuntimeError("Captured graph exposes no NCHW flatten activation")
    # Select the activation produced by flatten/reshape, before either Dense.
    # Several low-level aliases can share the shape. Analysis identifies an
    # executable frontier with trainable convolution and dense parameters.
    failures = []
    for node in candidates:
        boundary = f"after:{node.canonical_id}"
        try:
            handle = seed.backend.repartition(boundary)
            payload = handle.backend.run_prefix(*call.inputs.args)
            logits = handle.backend.run_suffix(payload)
            np.testing.assert_allclose(numpy(logits), numpy(model(*call.inputs.args)), rtol=2e-3, atol=2e-4)
            return handle
        except Exception as exc:
            failures.append(f"{boundary}: {type(exc).__name__}: {exc}")
    raise RuntimeError("No executable flatten frontier: " + "; ".join(failures))


def native_step(backend, model, call, adapter, task, lr):
    if backend == "jax":
        import jax
        parameters, images = call.inputs.args
        loss, grads = jax.value_and_grad(lambda p: task.loss(model(p, images), call.targets))(parameters)
        adapter.bind_external_params(jax.tree_util.tree_map(lambda p, g: p - lr * g, parameters, grads))
        return float(loss)
    opt = optimizer(backend, model, lr)
    if backend == "tf":
        import tensorflow as tf
        with tf.GradientTape() as tape:
            loss = task.loss(model(*call.inputs.args), call.targets)
        gradients = tape.gradient(loss, model.trainable_variables)
        value = adapter.scalar_value(loss)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
    else:
        clear = opt.clear_grad if backend == "paddle" else opt.zero_grad
        clear()
        loss = task.loss(model(*call.inputs.args), call.targets)
        value = adapter.scalar_value(loss)
        loss.backward()
        opt.step()
    return value


def admit(args):
    started = time.perf_counter()
    config, model, call = initialize(args)
    from splitfleet.backends.utils import adapter_for
    from splitfleet.tasks import ImageClassificationTask
    from splitfleet.split_engine import graph_contract_for_runtime_handle
    from splitfleet.transport import encode_boundary, decode_boundary, encode_gradients, decode_gradients
    from splitfleet.transport.split_wire import (
        boundary_to_envelope, envelope_to_boundary, gradients_to_envelope, envelope_to_gradients,
    )
    adapter = adapter_for(model, call.inputs.args)
    replica = adapter.clone_model(model)
    if adapter.state_manifest(replica).schema_hash != adapter.state_manifest(model).schema_hash:
        raise RuntimeError("Native model replicas do not preserve the parameter schema")
    for original, copied in zip(adapter.export_ndarrays(model), adapter.export_ndarrays(replica), strict=True):
        np.testing.assert_array_equal(original, copied)
    del replica
    adapter.set_training(model, True)
    task = ImageClassificationTask()
    initial = adapter.export_ndarrays(model)
    expected_output = numpy(model(*call.inputs.args)).copy()
    expected_loss = native_step(args.backend, model, call, adapter, task, config["learning_rate"])
    expected_state = adapter.export_ndarrays(model)
    # Use independent native/split models. In a lazy backend, assigning old
    # weights into a model that already trained retains assignment UOps.
    config, model, call = initialize(args)
    adapter = adapter_for(model, call.inputs.args)
    adapter.set_training(model, True)
    handle = resolve_cut(model, call, config)
    local = handle.backend.run_prefix(*call.inputs.args, training=True)
    contract = graph_contract_for_runtime_handle(handle)
    envelope = boundary_to_envelope(local, round_id=1, client_id="admission", step_id="0",
                                   plan_id=handle.plan.plan_id, split_id=contract.split_id,
                                   canonical_graph_hash=contract.canonical_graph_hash,
                                   boundary_schema_hash=contract.boundary_schema_hash, model_version=1)
    wire = encode_boundary(envelope)
    decoded = decode_boundary(wire)
    remote = envelope_to_boundary(decoded, handle.runtime, args.device)
    actual_output = numpy(handle.backend.run_suffix(remote))
    np.testing.assert_allclose(actual_output, expected_output, rtol=2e-3, atol=2e-4)
    opt = None if args.backend == "jax" else optimizer(args.backend, model, config["learning_rate"])
    measured = []

    def loss_fn(logits, targets):
        loss = task.loss(logits, targets)
        if args.backend != "jax":
            measured.append(adapter.scalar_value(loss))
        return loss

    loss, gradients = handle.backend.train_suffix(remote, call.targets, loss_fn=loss_fn, optimizer=opt)
    if args.backend == "jax":
        measured.append(adapter.scalar_value(loss))
    if not gradients:
        raise RuntimeError("Flatten cut produced no boundary gradients")
    gradient_wire = encode_gradients(gradients_to_envelope(decoded, gradients))
    returned = envelope_to_gradients(decode_gradients(gradient_wire), args.device)
    update = handle.backend.backward_prefix(local, returned, optimizer=opt)
    if args.backend == "jax":
        import jax
        adapter.bind_external_params(jax.tree_util.tree_map(lambda p, g: p - config["learning_rate"] * g,
                                                          adapter.external_params, update["inputs"][0]))
    actual_state = adapter.export_ndarrays(model)
    errors = []
    for actual, reference in zip(actual_state, expected_state, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=3e-3, atol=3e-5)
        errors.append(float(np.max(np.abs(actual - reference))))
    np.testing.assert_allclose(measured[0], expected_loss, rtol=2e-3, atol=2e-4)
    record = {"status": "passed", "backend": args.backend, "environment": environment(args.backend, model, args.device),
              "boundary": handle.plan.boundary, "prefix_nodes": handle.plan.prefix_node_count,
              "suffix_nodes": handle.plan.suffix_node_count, "activation_bytes": len(wire),
              "gradient_bytes": len(gradient_wire), "loss": measured[0],
              "output_max_abs_error": float(np.max(np.abs(actual_output - expected_output))),
              "one_sgd_step_max_abs_error": max(errors), "duration_sec": time.perf_counter() - started,
              "parameter_tensors": len(actual_state), "parameter_tensors_changed": sum(not np.array_equal(a, b) for a, b in zip(initial, actual_state))}
    save_json(args.output, record)
    print(json.dumps(record), flush=True)


class Batches:
    def __init__(self, path, backend, model, device, config, index):
        data = np.load(path)
        self.images, self.labels = data["images"], data["labels"]
        self.backend, self.model, self.device, self.config, self.index = backend, model, device, config, index
        self.round_id = 0

    def __iter__(self):
        order = np.random.default_rng(self.config["seed"] + self.index * 1000 + self.round_id).permutation(len(self.labels))
        for offset in range(0, len(order), self.config["batch_size"]):
            positions = order[offset:offset + self.config["batch_size"]]
            yield batch(self.backend, self.model, self.images[positions], self.labels[positions], self.device)


def client(args):
    config, model, call = initialize(args)
    from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
    from splitfleet.client.app import start_client
    from splitfleet.tasks import ImageClassificationTask
    loader = Batches(Path(args.bundle) / f"client_{args.index}.npz", args.backend, model, args.device, config, args.index)
    records = []

    class RecordingClient(AutoSplitSplitLearningClient):
        def fit(self, parameters, round_config):
            round_id = int(round_config["physical_round"])
            loader.round_id = round_id
            start = time.perf_counter()
            values, examples, metrics = super().fit(parameters, round_config)
            if examples != len(loader.labels) or metrics.get("skipped_examples", 0):
                raise RuntimeError("A physical client did not train its full partition")
            metrics["physical_client_index"] = args.index
            records.append({"round": round_id, "num_examples": examples,
                            "duration_sec": time.perf_counter() - start, "metrics": metrics})
            save_json(args.output, {"backend": args.backend, "client_index": args.index,
                                   "environment": environment(args.backend, self.model, args.device), "rounds": records})
            print(f"CLIENT_ROUND backend={args.backend} client={args.index} round={round_id} loss={metrics['loss']:.6f}", flush=True)
            return values, examples, metrics

    functional_update = None
    if args.backend == "jax":
        import jax
        functional_update = lambda params, result: jax.tree_util.tree_map(
            lambda p, g: p - config["learning_rate"] * g, params, result["inputs"][0])
    instance = RecordingClient(model=model, sample_inputs=call, train_data=loader,
                               task=ImageClassificationTask(), batch_axes={}, device=args.device,
                               optimizer_fn=None if args.backend == "jax" else lambda m: optimizer(args.backend, m, config["learning_rate"]),
                               functional_update_fn=functional_update)
    admission = json.loads(Path(args.admission).read_text())
    instance.prewarm_runtime(boundary=admission["boundary"], dynamic_batch=(config["batch_size"], config["batch_size"]))
    start_client(server_address=args.address, client=instance.to_client(), max_retries=3,
                 max_wait_time=30, reconnect_after_instruction=False)
    if len(records) != config["rounds"]:
        raise RuntimeError("Client exited before completing all configured rounds")


def server(args):
    started = time.perf_counter()
    config, model, call = initialize(args)
    from flwr.common import parameters_to_ndarrays
    from flwr.server import ServerConfig
    from splitfleet.backends.utils import bind_model_inputs, inference_context
    from splitfleet.server.strategy import AutoSplitStrategy
    from splitfleet.server.app import start_server
    from splitfleet.tasks import ImageClassificationTask
    admission = json.loads(Path(args.admission).read_text())
    evaluation = np.load(Path(args.bundle) / "evaluation.npz")
    records, fits, suffixes, downloads = [], {}, {}, {}

    class RecordingStrategy(AutoSplitStrategy):
        def configure_fit(self, server_round, parameters, client_manager):
            instructions = super().configure_fit(server_round, parameters, client_manager)
            downloads[server_round] = [{"round_id": int(server_round), "cid": str(proxy.cid),
                "model_state_download_bytes": sum(len(value) for value in ins.parameters.tensors)}
                for proxy, ins in instructions]
            return instructions

        def aggregate_fit(self, server_round, results, failures):
            if failures or len(results) != config["clients"]:
                raise RuntimeError(f"Round {server_round}: expected every physical client; failures={failures}")
            results = sorted(results, key=lambda item: int(item[1].metrics["physical_client_index"]))
            fits[server_round] = [{"client_index": int(r.metrics["physical_client_index"]),
                                  "cid": str(proxy.cid), "num_examples": r.num_examples,
                                  "model_state_upload_bytes": sum(len(value) for value in r.parameters.tensors),
                                  "metrics": r.metrics} for proxy, r in results]
            return super().aggregate_fit(server_round, results, failures)

        def aggregate_server_fit(self, server_round, results):
            if len(results) != config["clients"]:
                raise RuntimeError("Missing physical suffix replica result")
            suffixes[server_round] = [{"num_examples": r.config["num_examples"], "metrics": r.config} for r in results]
            return super().aggregate_server_fit(server_round, results)

        def evaluate(self, server_round, client_parameters, server_parameters):
            self.backend_adapter.load_ndarrays(self.model, parameters_to_ndarrays(client_parameters))
            self.backend_adapter.set_training(self.model, False)
            correct, loss_total = 0, 0.
            with inference_context(args.backend):
                for offset in range(0, len(evaluation["labels"]), config["batch_size"]):
                    value = batch(args.backend, self.model, evaluation["images"][offset:offset + config["batch_size"]],
                                  evaluation["labels"][offset:offset + config["batch_size"]], args.device)
                    logits = self.model(*bind_model_inputs(value.inputs.args, self.backend_adapter))
                    loss = self.backend_adapter.scalar_value(self.task.loss(logits, value.targets))
                    if not np.isfinite(loss):
                        raise RuntimeError("Nonfinite evaluation loss")
                    loss_total += loss * value.num_examples
                    correct += int(np.count_nonzero(numpy(logits).argmax(axis=1) == numpy(value.targets)))
            self.backend_adapter.set_training(self.model, True)
            entry = {"round": server_round, "test_loss": loss_total / config["test_samples"],
                     "accuracy": correct / config["test_samples"], "elapsed_sec": time.perf_counter() - started,
                     "clients": sorted(fits.get(server_round, []), key=lambda v: v["client_index"]),
                     "state_download_records": downloads.get(server_round, []),
                     "suffixes": suffixes.get(server_round, [])}
            if server_round:
                if sum(r["num_examples"] for r in entry["clients"]) != config["train_samples"]:
                    raise RuntimeError("Round omitted real training examples")
                entry["train_loss"] = sum(r["metrics"]["loss"] * r["num_examples"] for r in entry["clients"]) / config["train_samples"]
            records.append(entry)
            record = {"backend": args.backend, "source": "real", "config": config,
                      "environment": environment(args.backend, self.model, args.device), "rounds": records,
                      "boundary": admission["boundary"], "transport": "SplitFleet Flower bidi + server-model gRPC",
                      "communication_accounting": "training_client_application_buffers_v1",
                      "method": "SplitFed with fixed semantic cut; per-client suffix replicas"}
            save_json(args.output, record)
            print(f"SERVER_ROUND backend={args.backend} round={server_round} loss={entry['test_loss']:.6f} accuracy={entry['accuracy']:.4f}", flush=True)
            return entry["test_loss"], {"accuracy": entry["accuracy"]}

    strategy = RecordingStrategy(model=model, sample_inputs=call, task=ImageClassificationTask(),
                                 boundary=admission["boundary"], batch_axes={},
                                 dynamic_batch=(config["batch_size"], config["batch_size"]),
                                 aggregation_policy="splitfed", runtime_device=args.device,
                                 optimizer_fn=None if args.backend == "jax" else lambda m: optimizer(args.backend, m, config["learning_rate"]),
                                 config_client_fit_fn=lambda r: {"physical_round": int(r)},
                                 fraction_fit=1., fraction_evaluate=0., min_fit_clients=config["clients"],
                                 min_evaluate_clients=0, min_available_clients=config["clients"])
    from flwr.common import ndarrays_to_parameters
    initial = strategy.backend_adapter.export_ndarrays(strategy.model)
    strategy.evaluate(0, ndarrays_to_parameters(initial), initial)
    history = start_server(server_address=args.address, config=ServerConfig(num_rounds=config["rounds"]), strategy=strategy)
    if len(records) != config["rounds"] + 1:
        raise RuntimeError("Real training did not complete all configured rounds")
    final = strategy.backend_adapter.export_ndarrays(strategy.model)
    if not all(np.isfinite(value).all() for value in final):
        raise RuntimeError("Final model contains nonfinite parameters")
    saved = getattr(args, "save_model", False)
    if saved:
        np.savez(Path(args.output).with_suffix(".weights.npz"), **{str(i): value for i, value in enumerate(final)})
    record = json.loads(Path(args.output).read_text())
    record.update(status="completed", duration_sec=time.perf_counter() - started,
                  final_state_sha256=arrays_hash({str(i): value for i, value in enumerate(final)}),
                  final_state_verification={"finite": True, "weights_saved": saved,
                                            "basis": "in_memory_before_result_write"})
    save_json(args.output, record)


def communication_records(server_result):
    """Expose actual Flower parameter buffers to the shared accounting audit.

    Legacy receipts remain incomplete: no .npy sizes are reconstructed from
    their raw tensor counters. Suffix replicas live on the coordinator and do
    not count as client network transfers.
    """
    config = server_result["config"]
    fits, downloads, suffixes = [], [], []
    for row in server_result["rounds"]:
        if not row["round"]:
            continue
        for client in row["clients"]:
            metrics = dict(client["metrics"])
            if "model_state_upload_bytes" in client:
                metrics["model_state_upload_bytes"] = client["model_state_upload_bytes"]
            fits.append({"round_id": row["round"],
                         "cid": client.get("cid", str(client["client_index"])), "metrics": metrics})
        downloads.extend(row.get("state_download_records", []))
        suffixes.extend({"metrics": value["metrics"]} for value in row["suffixes"])
    return {"communication_accounting": server_result.get("communication_accounting"),
            "method": "splitfed_fixed", "rounds": config["rounds"], "expected_clients": config["clients"],
            "fit_records": fits, "state_download_records": downloads,
            "server_fit_records": suffixes, "fit_failures": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--data-root", default="data")
    prepare_parser.add_argument("--clients", type=int, default=3)
    prepare_parser.add_argument("--train-samples", type=int, default=5000)
    prepare_parser.add_argument("--test-samples", type=int, default=1000)
    prepare_parser.add_argument("--batch-size", type=int, default=50)
    prepare_parser.add_argument("--rounds", type=int, default=10)
    prepare_parser.add_argument("--seed", type=int, default=20261006)
    prepare_parser.add_argument("--learning-rate", type=float, default=.05)
    for name in ("admit", "server", "client"):
        p = sub.add_parser(name)
        p.add_argument("--backend", choices=BACKENDS, required=True)
        p.add_argument("--bundle", type=Path, required=True)
        p.add_argument("--device", required=True)
        p.add_argument("--output", type=Path, required=True)
        if name != "admit":
            p.add_argument("--admission", type=Path, required=True)
            p.add_argument("--address", required=True)
        if name == "server":
            p.add_argument("--save-model", action="store_true", help="Retain full final weights")
        if name == "client":
            p.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.output, args.data_root, clients=args.clients, train_samples=args.train_samples,
                                 test_samples=args.test_samples, batch_size=args.batch_size, rounds=args.rounds,
                                 seed=args.seed, lr=args.learning_rate)), flush=True)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".pid").write_text(str(os.getpid()))
    with context(args.backend, args.device):
        globals()[args.command](args)


if __name__ == "__main__":
    main()
