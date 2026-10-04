"""Calibrate operation costs once, using synthetic inputs and a frozen model."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.utils.data import DataLoader
from experiments.physical_multitask import load_bundle, _trace_item, _batch_axes, ItemDataset, _configure_primary_math
from experiments.common.workload_training import _batch
from splitfleet.autosplit import AutoSplitSession
from splitfleet.autosplit.state_ownership import state_ownership
from splitfleet.backends.utils import adapter_for


def profile(bundle_path, device):
    started = time.perf_counter()
    workload, bundle = load_bundle(bundle_path)
    math_policy = _configure_primary_math(bundle)
    torch.set_num_threads(1)
    torch.manual_seed(int(bundle["seed"]))
    model = workload.model_factory().to(device).train()
    model.load_state_dict(bundle["initial_model_state"])
    item = _trace_item(workload.train_dataset[0], bundle["task"])
    if bundle["task"] == "object_detection":
        item = (item[0], {"boxes": torch.tensor([[.2, .2, .6, .6]]),
                         "labels": torch.tensor([0], dtype=torch.long)})
    loader = DataLoader(ItemDataset([item] * bundle["batch_size"], bundle["task"]),
                        batch_size=bundle["batch_size"], collate_fn=workload.collate_fn)
    call, targets, _ = _batch(workload, next(iter(loader)), torch.device(device))
    session = AutoSplitSession(device=device)
    handle = session.prepare_runtime(model, call, boundary=bundle["fixed_cut_resolution"]["50%"],
                                     batch_axes=_batch_axes(bundle["task"]),
                                     dynamic_batch=(1, bundle["batch_size"]), trainable=True)
    runtime = handle.runtime
    ownership = state_ownership(handle, adapter_for(model, call).state_manifest(model).schema_hash)
    parameters = dict(model.named_parameters())
    optimizers = {owner: torch.optim.Adam([parameters[name] for name, selected in
        zip(ownership["names"], ownership["owners"]) if selected == owner and name in parameters
        and parameters[name].requires_grad], lr=1e-5) for owner in ("prefix", "suffix")}
    adapter = workload.task.make_adapter()
    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
    def step():
        model.zero_grad(set_to_none=True)
        sync(); begin = time.perf_counter()
        boundary = handle.backend.run_prefix(*call.args, input_kwargs=dict(call.kwargs), training=True)
        sync(); forward = time.perf_counter()
        loss, gradients = handle.backend.train_suffix(boundary, targets,
                            loss_fn=adapter.loss, optimizer=optimizers["suffix"])
        sync(); tail = time.perf_counter()
        handle.backend.backward_prefix(boundary, boundary_grads=gradients, optimizer=optimizers["prefix"])
        sync(); end = time.perf_counter()
        return ((forward-begin)*1000, (end-tail)*1000, (tail-forward)*1000, float(loss.detach()))
    step()
    measured = np.median([step() for _ in range(3)], axis=0)
    for segment in (runtime.segments.training_prefix, runtime.segments.suffix):
        original = segment._execute_func
        def marked(self, node, args, kwargs, _original=original, **options):
            with torch.autograd.profiler.record_function("splitfleet-node:" + node.canonical_id):
                return _original(node, args, kwargs, **options)
        segment._execute_func = MethodType(marked, segment)
    with torch.autograd.profiler.profile(use_cuda=str(device).startswith("cuda")) as trace:
        step()
    events = trace.function_events
    sequence_owner = {}
    forward, backward = {}, {}
    gpu = str(device).startswith("cuda")
    def cost(event):
        return max(float(event.device_time_total if gpu else event.cpu_time_total) / 1000, 0)
    for event in events:
        if event.name.startswith("splitfleet-node:"):
            node = event.name.split(":", 1)[1]
            forward[node] = forward.get(node, 0) + cost(event)
        if event.sequence_nr >= 0 and not event.name.startswith("autograd::"):
            parent = event.cpu_parent
            while parent is not None and not parent.name.startswith("splitfleet-node:"):
                parent = parent.cpu_parent
            if parent is not None:
                sequence_owner[event.sequence_nr] = parent.name.split(":", 1)[1]
    for event in events:
        if event.name.startswith("autograd::engine::evaluate_function:"):
            node = sequence_owner.get(event.sequence_nr)
            if node is not None:
                backward[node] = backward.get(node, 0) + cost(event)
    if not forward or not any(forward.values()):
        raise RuntimeError(f"Profiler produced no forward costs: events={len(events)} "
                           f"markers={[ (e.name, e.cpu_time_total, e.device_time_total) for e in events if e.name.startswith('splitfleet-node:')][:5]}")
    nodes = list(runtime.trace_graph.nodes)
    prefix = set(runtime.plan.prefix_node_ids)
    suffix = set(runtime.plan.suffix_node_ids)
    def scale(costs, selected, target):
        total = sum(costs.get(node.canonical_id, 0) for node in nodes if node.canonical_id in selected)
        if total <= 0:
            raise RuntimeError(f"Profiler produced no operation costs: selected={list(selected)[:5]} "
                               f"costs={list(costs.items())[:5]} markers={len(costs)}")
        return target / total
    fscale = scale(forward, prefix, measured[0])
    bscale = scale(backward, prefix, measured[1])
    # Scale the suffix separately to include its loss and autograd overhead.
    tail_sum = sum(forward.get(n, 0) * fscale + backward.get(n, 0) * bscale for n in suffix)
    tail_scale = measured[2] / tail_sum
    costs = []
    for node in nodes:
        multiplier = 1 if node.canonical_id in prefix else tail_scale
        costs.append({"node": node.canonical_id, "forward_ms": forward.get(node.canonical_id, 0)*fscale*multiplier,
                      "backward_ms": backward.get(node.canonical_id, 0)*bscale*multiplier})
    return {"schema": "splitfleet.operation-costs.v1", "device": device,
            "task": bundle["task"], "model_id": bundle.get("model_id", bundle.get("image_model")),
            "batch_size": bundle["batch_size"], "optimizer": "adam", "learning_rate": 1e-5,
            "suffix_loss_and_optimizer_included": True,
            "math_policy": math_policy,
            "torch": torch.__version__, "graph_signature": handle.plan.graph_signature,
            "initial_model_hash": bundle["initial_model_hash"],
            "synthetic_inputs": True, "num_threads": 1,
            "calibration_boundary": handle.plan.boundary,
            "calibration_ms": {"forward": measured[0], "backward": measured[1], "tail": measured[2]},
            "loss": measured[3], "nodes": costs,
            "ownership_counts": {s: ownership["owners"].count(s) for s in ("prefix", "suffix", "initial")},
            "elapsed_sec": time.perf_counter()-started}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = profile(args.bundle, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "nodes"}))
