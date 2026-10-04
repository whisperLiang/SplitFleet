"""Real-data admission checks, separate from algorithm performance evidence."""

from __future__ import annotations

import argparse
import copy
import json
import platform
from pathlib import Path
import resource
import time

import torch
from torch.utils.data import DataLoader

from experiments.physical_multitask import _batch_axes, load_bundle, _configure_primary_math
from experiments.common.workload_training import _batch, _forward
from splitfleet.autosplit import AutoSplitSession
from splitfleet.autosplit.state_ownership import state_ownership
from splitfleet.backends.utils import adapter_for
from splitfleet.split_engine import graph_contract_for_runtime_handle
from splitfleet.transport import encode_boundary, decode_boundary, encode_gradients, decode_gradients
from splitfleet.transport.split_wire import (
    boundary_to_envelope, envelope_to_boundary, gradients_to_envelope, envelope_to_gradients,
)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generator_state(device):
    return torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()


def restore_generator(state, device):
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.set_rng_state(state)


def first_example(value):
    if isinstance(value, torch.Tensor):
        return value[:1]
    if isinstance(value, dict):
        return {key: first_example(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(first_example(item) for item in value)
    if isinstance(value, list):
        return [first_example(item) for item in value]
    return value


def check_close(actual, expected, *, label, rtol=2e-4, atol=2e-6):
    if actual.is_floating_point():
        error = float((actual.detach() - expected.detach()).abs().max()) if actual.numel() else 0.0
    else:
        error = float(torch.count_nonzero(actual != expected))
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol,
                                msg=lambda message: f"{label}: {message}")
    return error


def verify_cut(workload, bundle, model, raw, fraction, device):
    native, prefix, suffix = (copy.deepcopy(model).to(device).train() for _ in range(3))
    call, targets, size = _batch(workload, raw, device)
    cut = bundle["fixed_cut_resolution"][fraction]
    kwargs = dict(boundary=cut, trainable=True, dynamic_batch=(1, bundle["batch_size"]),
                  batch_axes=_batch_axes(bundle["task"]))
    first = AutoSplitSession(device=device).prepare_runtime(prefix, call, **kwargs)
    last = AutoSplitSession(device=device).prepare_runtime(suffix, call, **kwargs)
    owners = state_ownership(first, adapter_for(prefix, call).state_manifest(prefix).schema_hash)
    for replica in (native, prefix, suffix):
        replica.load_state_dict(bundle["initial_model_state"])
        replica.train()
    optimizers = [torch.optim.Adam(replica.parameters(), lr=1e-5)
                  for replica in (native, prefix, suffix)]
    adapter = workload.task.make_adapter()
    contract = graph_contract_for_runtime_handle(first)
    records = []
    for index, partial in enumerate((False, True)):
        if partial:
            call = call.map_values(first_example)
            targets = targets[:1] if isinstance(targets, (torch.Tensor, list)) else targets
            size = 1
        rng = generator_state(device).clone()
        optimizers[0].zero_grad(set_to_none=True)
        reference_loss = adapter.loss(_forward(native, call), targets)
        reference_loss.backward()
        optimizers[0].step()
        advanced = generator_state(device).clone()
        restore_generator(rng, device)
        boundary = first.backend.run_prefix(*call.args, input_kwargs=dict(call.kwargs), training=True)
        envelope = boundary_to_envelope(boundary, round_id=1, client_id="edge-admission",
            step_id=str(index), plan_id=first.plan.plan_id, split_id=contract.split_id,
            canonical_graph_hash=contract.canonical_graph_hash,
            boundary_schema_hash=contract.boundary_schema_hash, model_version=1)
        remote = envelope_to_boundary(decode_boundary(encode_boundary(envelope)), last.runtime, device)
        loss, gradients = last.backend.train_suffix(remote, targets, loss_fn=adapter.loss,
                                                     optimizer=optimizers[2])
        returned = envelope_to_gradients(decode_gradients(encode_gradients(
            gradients_to_envelope(envelope, gradients))), device)
        first.backend.backward_prefix(boundary, boundary_grads=returned, optimizer=optimizers[1])
        record = {"batch_size": size, "loss_abs_error": check_close(loss, reference_loss, label="loss"),
                  "state_max_abs_error": 0.0, "gradient_max_abs_error": 0.0,
                  "optimizer_max_abs_error": 0.0,
                  "rng_advance_matches": torch.equal(advanced, generator_state(device))}
        if not record["rng_advance_matches"]:
            raise AssertionError("Split and native execution consumed different random streams")
        replicas = {"prefix": (prefix, optimizers[1]), "suffix": (suffix, optimizers[2])}
        reference_parameters = dict(native.named_parameters())
        for name, owner in zip(owners["names"], owners["owners"]):
            actual = bundle["initial_model_state"][name].to(device) if owner == "initial" \
                else replicas[owner][0].state_dict()[name]
            record["state_max_abs_error"] = max(record["state_max_abs_error"],
                check_close(actual, native.state_dict()[name], label=name))
            if owner == "initial" or name not in reference_parameters:
                continue
            parameter = dict(replicas[owner][0].named_parameters())[name]
            reference = reference_parameters[name]
            if reference.grad is None:
                assert parameter.grad is None
                continue
            record["gradient_max_abs_error"] = max(record["gradient_max_abs_error"],
                check_close(parameter.grad, reference.grad, label=name + " gradient"))
            optimizer = replicas[owner][1]
            for key in ("step", "exp_avg", "exp_avg_sq"):
                record["optimizer_max_abs_error"] = max(record["optimizer_max_abs_error"],
                    check_close(optimizer.state[parameter][key], optimizers[0].state[reference][key],
                                label=name + " " + key))
        records.append(record)
    return {"fraction": fraction, "cut": cut, "batches": records, "status": "passed"}


def admit(bundle_path, device, *, equivalence=False, cuda_memory_fraction=None):
    started = time.perf_counter()
    torch.set_num_threads(1)
    workload, bundle = load_bundle(bundle_path)
    math_policy = _configure_primary_math(bundle)
    if bundle["role"] != "client":
        raise ValueError("Admission must consume a private real-data client bundle")
    device = torch.device(device)
    if cuda_memory_fraction is not None and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction, device)
    torch.manual_seed(bundle["seed"])
    model = workload.model_factory().to(device).train()
    model.load_state_dict(bundle["initial_model_state"])
    raw = next(iter(DataLoader(workload.train_dataset, batch_size=bundle["batch_size"],
                               collate_fn=workload.collate_fn)))
    call, targets, examples = _batch(workload, raw, device)
    report = {"schema": "splitfleet.edge-model-admission.v1", "model_id": bundle["model_id"],
        "task": bundle["task"], "device": str(device), "python": platform.python_version(),
        "torch": torch.__version__, "num_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "batch_size": examples, "initial_model_hash": bundle["initial_model_hash"],
        "pretrain_checkpoint_sha256": bundle["pretrain_checkpoint_sha256"],
        "model_metadata": bundle["model_metadata"], "equivalence": [],
        "math_policy": math_policy,
        "cuda_allocator_fraction": cuda_memory_fraction,
        "scope": "Model/resource admission; no FL/SFL performance or convergence comparison"}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    adapter = workload.task.make_adapter()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    losses, timings = [], []
    for _ in range(4):
        synchronize(device)
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = adapter.loss(_forward(model, call), targets)
        if not torch.isfinite(loss):
            raise AssertionError("Native real-data training loss is not finite")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise AssertionError("Native gradients are not finite")
        optimizer.step()
        synchronize(device)
        timings.append(time.perf_counter() - begin)
        losses.append(float(loss.detach()))
    report.update(native_step_seconds=timings[1:], native_losses=losses,
                  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None)
    del optimizer
    # The native resource check is complete. Its gradients and final loss must
    # not inflate capture's memory budget or survive into independent replicas.
    model.zero_grad(set_to_none=True)
    del loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.load_state_dict(bundle["initial_model_state"])
    if equivalence:
        for fraction in ("25%", "50%", "75%"):
            report["equivalence"].append(verify_cut(workload, bundle, model, raw, fraction, device))
    else:
        # Execute one split backward even when only checking resource admission.
        handle = AutoSplitSession(device=device).prepare_runtime(model, call,
            boundary=bundle["fixed_cut_resolution"]["50%"], trainable=True,
            batch_axes=_batch_axes(bundle["task"]), dynamic_batch=(1, bundle["batch_size"]))
        boundary = handle.backend.run_prefix(*call.args, input_kwargs=dict(call.kwargs), training=True)
        split_loss, gradients = handle.backend.train_suffix(boundary, targets,
                                                            loss_fn=adapter.loss, optimizer=None)
        handle.backend.backward_prefix(boundary, boundary_grads=gradients, optimizer=None)
        if not torch.isfinite(split_loss):
            raise AssertionError("Split training loss is not finite")
        report.update(split_smoke_loss=float(split_loss.detach()),
                      resolved_split_boundary=handle.plan.boundary,
                      split_graph_signature=handle.plan.graph_signature)
    report.update(status="passed", elapsed_sec=time.perf_counter() - started,
                  split_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  split_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                  process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--equivalence", action="store_true")
    parser.add_argument("--cuda-memory-fraction", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = admit(args.bundle, args.device, equivalence=args.equivalence,
                        cuda_memory_fraction=args.cuda_memory_fraction)
    except Exception as exc:
        args.output.write_text(json.dumps({"status": "failed", "device": args.device,
            "bundle": str(args.bundle), "exception_type": type(exc).__name__, "error": str(exc)}, indent=2) + "\n")
        raise
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("model_metadata", "equivalence")}))


if __name__ == "__main__":
    main()
