"""Bounded native/serialized split profiles with a common ResNet50 protocol.

This is a single-process diagnostic, not a distributed timing experiment.
Frameworks run in separate processes, using the same canonical initialization,
CIFAR images, batch shapes, FP32 policy and native stage2 cut.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np


def profile(args):
    if args.backend != "tensorflow":
        sys.modules["tensorflow"] = None
    from .common import Images, save_json
    from .worker import execution, initialize, endpoints, receipt
    from experiments.resnet50_quality.model_contract import NAMES, state_hash
    from tests.integration.resnet50_reference.native import (
        build_model, call_inputs, export_state, native_batch, numpy_value,
        optimizer, train_step, training_mode,
    )
    from splitfleet.autosplit import AutoSplitSession
    from splitfleet.backends.utils import adapter_for
    from splitfleet.split_engine import graph_contract_for_runtime_handle
    from splitfleet.tasks.losses import extract_logits, sparse_cross_entropy
    from splitfleet.transport import encode_boundary, decode_boundary, encode_gradients, decode_gradients
    from splitfleet.transport.split_wire import (
        boundary_to_envelope, envelope_to_boundary, gradients_to_envelope, envelope_to_gradients,
    )

    setup = SimpleNamespace(backend=args.backend, bundle=args.bundle, device=args.device,
                            index=0, identity=None, command="server")
    config, initial, model, _, _ = initialize(setup)
    if args.backend == "tensorflow":
        model._splitfleet_tensorflow_function_prefix = args.tensorflow_prefix_policy == "compiled"
    data = Images(args.bundle / "validation.npz", config)
    batches = {size: native_batch(args.backend, data.images(list(range(size))),
                                 data.labels[:size], args.device) for size in (16, 12)}
    adapter = adapter_for(model, call_inputs(args.backend, model, batches[16][0]))
    def synchronize():
        if not args.device.startswith("cuda"):
            return
        if args.backend == "pytorch":
            import torch
            torch.cuda.synchronize(args.device)
        elif args.backend == "paddle":
            import paddle
            paddle.device.cuda.synchronize(int(args.device.split(":")[-1]))
        else:
            from tensorflow.python.eager import context
            context.context().sync_executors()
    report = {"status": "running", "backend": args.backend, "device": args.device,
              "environment": receipt(args.backend, model, args.device),
              "initial_hash": config["initial_hash"], "batch_sizes": [16, 12],
              "warmup_cycles": args.warmup, "measured_cycles": args.cycles,
              "tensorflow_prefix_policy": args.tensorflow_prefix_policy if args.backend == "tensorflow" else None,
              "synchronization": "explicit framework CUDA synchronization after every timed device phase",
              "scope": "single process, native full SGD and serialized native split SGD; no LAN or six-client contention",
              "source_files": {}, "full": [], "split": []}
    from .common import file_hash
    for name in ("splitfleet/runtime/tensorflow_prefix_training.py", "splitfleet/runtime/tensorflow_forward.py",
                 "splitfleet/runtime/native_frontier.py", "splitfleet/runtime/auxiliary_state.py"):
        path = Path(__file__).resolve().parents[2] / name
        report["source_files"][name] = {"path": str(path), "sha256": file_hash(path)}
    save_json(args.output, report)
    with execution(args.backend, args.device):
        opt = optimizer(args.backend, model, .05)
        for cycle in range(args.warmup + args.cycles):
            for size, (images, labels) in batches.items():
                synchronize()
                started = time.perf_counter()
                result = train_step(args.backend, model, images, labels, native_optimizer=opt)
                synchronize()
                report["full"].append({"cycle": cycle, "batch_size": size,
                    "measured": cycle >= args.warmup, "seconds": time.perf_counter() - started,
                    "loss": result["loss"]})
        del model, opt
        gc.collect()
        model = build_model(args.backend, initial, args.device)
        if args.backend == "tensorflow":
            model._splitfleet_tensorflow_function_prefix = args.tensorflow_prefix_policy == "compiled"
        training_mode(args.backend, model)
        inputs = call_inputs(args.backend, model, batches[16][0][:2])
        started = time.perf_counter()
        handle = AutoSplitSession(device=args.device).prepare_runtime(
            model, inputs, trainable=True, batch_axes={"/args/0": 0}, dynamic_batch=(1, 16))
        handle = handle.backend.repartition(endpoints(handle, args.backend)["stage2"]["boundary"])
        contract = graph_contract_for_runtime_handle(handle)
        report["capture_seconds"] = time.perf_counter() - started
        prefix_opt = optimizer(args.backend, model, .05)
        suffix_opt = optimizer(args.backend, model, .05)
        objective = lambda output, targets: sparse_cross_entropy(extract_logits(output), targets)
        for cycle in range(args.warmup + args.cycles):
            for size, (images, labels) in batches.items():
                for opt in (prefix_opt, suffix_opt):
                    if args.backend == "pytorch":
                        opt.zero_grad(set_to_none=True)
                    elif args.backend == "paddle":
                        opt.clear_grad()
                row = {"cycle": cycle, "batch_size": size, "measured": cycle >= args.warmup}
                synchronize()
                total = started = time.perf_counter()
                boundary = handle.backend.run_prefix(*call_inputs(args.backend, model, images), training=True)
                synchronize()
                row["prefix_forward_seconds"] = time.perf_counter() - started
                started = time.perf_counter()
                envelope = boundary_to_envelope(boundary, round_id=1, client_id="matched-profile",
                    step_id=f"{cycle}-{size}", plan_id=handle.plan.plan_id, split_id=contract.split_id,
                    canonical_graph_hash=contract.canonical_graph_hash,
                    boundary_schema_hash=contract.boundary_schema_hash, model_version=1)
                wire = encode_boundary(envelope)
                decoded = decode_boundary(wire)
                remote = envelope_to_boundary(decoded, handle.runtime, args.device)
                synchronize()
                row.update(boundary_encode_decode_seconds=time.perf_counter() - started,
                           activation_wire_bytes=len(wire), activation_fields=len(envelope.tensors))
                started = time.perf_counter()
                loss, gradients = handle.backend.train_suffix(remote, labels, loss_fn=objective, optimizer=suffix_opt)
                value = float(numpy_value(loss))
                synchronize()
                row["suffix_train_seconds"] = time.perf_counter() - started
                started = time.perf_counter()
                grad_wire = encode_gradients(gradients_to_envelope(decoded, gradients))
                returned = envelope_to_gradients(decode_gradients(grad_wire), args.device)
                synchronize()
                row.update(gradient_encode_decode_seconds=time.perf_counter() - started,
                           gradient_wire_bytes=len(grad_wire))
                started = time.perf_counter()
                handle.backend.backward_prefix(boundary, returned, optimizer=prefix_opt)
                synchronize()
                row["prefix_backward_seconds"] = time.perf_counter() - started
                started = time.perf_counter()
                statistics = getattr(handle.backend, "train_suffix_auxiliary", None)
                if statistics:
                    model.apply_statistics(statistics)
                synchronize()
                row.update(auxiliary_commit_seconds=time.perf_counter() - started,
                           total_seconds=time.perf_counter() - total, loss=value)
                report["split"].append(row)
                save_json(args.output, report)
                print(json.dumps({"event": "profile_step", "backend": args.backend, **row}), flush=True)
                del boundary, envelope, wire, decoded, remote, gradients, grad_wire, returned
        final = export_state(args.backend, model)
        counters = {name: int(final[name]) for name in NAMES if name.endswith("num_batches_tracked")}
        if len(counters) != 53 or set(counters.values()) != {2 * (args.warmup + args.cycles)}:
            raise RuntimeError("Diagnostic SGD did not commit BatchNorm once per step")
        if not all(np.isfinite(value).all() for value in final.values()):
            raise RuntimeError("Nonfinite diagnostic model state")
        report.update(status="completed", final_state_hash=state_hash(final), bn_counters=counters)
    save_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pytorch", "tensorflow", "paddle"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--tensorflow-prefix-policy", choices=("native_eager", "compiled"), default="compiled")
    args = parser.parse_args()
    if args.output.exists() or args.warmup < 0 or args.cycles < 1:
        parser.error("Choose a new output, nonnegative warmup and positive cycles")
    profile(args)


if __name__ == "__main__":
    main()
