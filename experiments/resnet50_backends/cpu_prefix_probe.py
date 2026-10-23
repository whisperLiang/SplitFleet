"""Bounded CPU prefix timing; run against an explicitly selected source snapshot.

No SGD, RPC, or formal-client state is changed. Each process receives its thread
policy before importing a framework. The cotangent is fixed to compare native
stage2 forward/backward work without involving a suffix or network.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pytorch", "paddle"), required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--client-index", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--operator-profile", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1 or args.steps < 1:
        parser.error("Choose a new output and positive thread/step counts")
    sys.modules["tensorflow"] = None
    from experiments.resnet50_backends.common import Images, save_json
    from experiments.resnet50_quality.model_contract import SCHEMA, NAMES, state_hash
    from tests.integration.resnet50_reference import native

    native.configure_backend(args.backend)
    if args.backend == "pytorch":
        import torch
        torch.set_num_threads(args.threads)
        actual_threads = torch.get_num_threads()
    else:
        import paddle
        actual_threads = paddle.get_flags(["FLAGS_paddle_num_threads"])["FLAGS_paddle_num_threads"]
        if actual_threads != args.threads:
            raise RuntimeError("Paddle thread flag must be configured before process startup")
    config = json.loads((args.bundle / "protocol.json").read_text())
    with np.load(args.bundle / "initial.npz") as z:
        initial = {name: z[name] for name in NAMES}
    model = native.build_model(args.backend, initial, "cpu")
    native.training_mode(args.backend, model, True)
    source = Images(args.bundle / f"client_{args.client_index}.npz", config)
    ids, raw, labels = next(source.batches(client=args.client_index, round_id=1))
    images, _ = native.native_batch(args.backend, raw, labels, "cpu")
    parameters = native.native_parameters(args.backend, model)
    selected = {row.name for row in SCHEMA if row.trainable and row.stage <= 2}
    cotangent = np.random.default_rng(412).normal(0, .001, (len(ids), 512, 8, 8)).astype(np.float32)
    gradient, _ = native.native_batch(args.backend, cotangent, labels, "cpu")
    norms = ({name: layer for name, layer in model.named_sublayers()
              if hasattr(layer, "_statistics_sink")} if args.backend == "paddle" else {})
    output = {"status": "running", "backend": args.backend, "threads": actual_threads,
        "host": platform.node(), "machine": platform.machine(), "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)), "batch_size": len(ids),
        "source_native_helper": native.__file__, "initial_state_hash": state_hash(initial),
        "input_sha256": hashlib.sha256(raw.tobytes() + cotangent.tobytes()).hexdigest(),
        "environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "FLAGS_paddle_num_threads")},
        "scope": "native CPU stage2 prefix, fixed cotangent, no SGD or network; concurrent formal experiment remains active",
        "steps": []}
    save_json(args.output, output)

    def step(measured):
        for value in parameters.values():
            if args.backend == "pytorch":
                value.grad = None
            else:
                value.clear_gradient()
        statistics_sink = {}
        for layer in norms.values():
            layer._statistics_sink = statistics_sink
        try:
            started = time.perf_counter()
            x = images
            block_times = {}
            for name, function in (
                ("stem", lambda value: model.maxpool(model.relu(model.bn1(model.conv1(value))))),
                ("stage1", model.layer1), ("stage2", model.layer2),
            ):
                before = time.perf_counter()
                x = function(x)
                block_times[name] = time.perf_counter() - before
            forward = time.perf_counter() - started
            before = time.perf_counter()
            if args.backend == "pytorch":
                x.backward(gradient)
            else:
                paddle.autograd.backward([x], [gradient])
            backward = time.perf_counter() - before
        finally:
            for layer in norms.values():
                layer._statistics_sink = None
        row = {"measured": measured, "forward_seconds": forward, "backward_seconds": backward,
               "total_seconds": forward + backward, "forward_blocks_seconds": block_times}
        output["steps"].append(row)
        save_json(args.output, output)
        print(json.dumps({"event": "cpu_prefix_step", "backend": args.backend,
                          "threads": actual_threads, **row}), flush=True)
        return x

    last = step(False)
    for _ in range(args.steps):
        last = step(True)
    gradients = {name: native.numpy_value(value.grad) for name, value in parameters.items()
                 if value.grad is not None}
    if set(gradients) != selected:
        raise RuntimeError("Native prefix did not differentiate exactly its owned parameters")
    np.savez(args.output.with_suffix(".npz"), output=native.numpy_value(last), **gradients)
    if args.operator_profile:
        if args.backend == "paddle":
            import paddle.profiler as profiler
            with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU],
                                   on_trace_ready=None) as prof:
                step(False)
            prof.export_chrome_tracing(str(args.output.with_suffix(".trace.json")))
        else:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                step(False)
            prof.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
    measured = [row for row in output["steps"] if row["measured"]]
    output.update(status="completed", prefix_trainable_tensors=len(gradients),
        median_forward_seconds=statistics.median(row["forward_seconds"] for row in measured),
        median_backward_seconds=statistics.median(row["backward_seconds"] for row in measured),
        median_total_seconds=statistics.median(row["total_seconds"] for row in measured))
    save_json(args.output, output)


if __name__ == "__main__":
    main()
