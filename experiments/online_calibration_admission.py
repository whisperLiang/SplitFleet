"""Execute all bootstrap anchors and verify restoration on an actual worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import torch
from torch.utils.data import DataLoader

from splitfleet.server.placement.cosplit_ucb.calibration import calibrate_split
from experiments.common.workload_training import _batch
from experiments.physical_multitask import (
    _batch_axes, _configure_primary_math, _resolve_training_cut_domains, _seed_training_stream, load_bundle,
)
from splitfleet.autosplit import AutoSplitSession


def admit(bundle_path, device, *, cuda_memory_fraction=None):
    device = torch.device(device)
    torch.set_num_threads(1)
    if cuda_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction, device)
    workload, bundle = load_bundle(bundle_path)
    _configure_primary_math(bundle)
    torch.manual_seed(int(bundle["seed"]))
    model = workload.model_factory().to(device)
    model.load_state_dict(bundle["initial_model_state"])
    _seed_training_stream(bundle, bundle.get("client_index", 999))
    raw = next(iter(DataLoader(workload.train_dataset, batch_size=bundle["batch_size"], collate_fn=workload.collate_fn)))
    call, targets, _ = _batch(workload, raw, device)
    anchors = bundle.get("online_calibration_boundaries")
    if anchors is None:
        _, anchors = _resolve_training_cut_domains(model=model, sample_inputs=call, bundle=bundle)
    session = AutoSplitSession(device=device)
    handle = session.prepare_runtime(model, call, boundary=bundle["fixed_cut_resolution"]["50%"],
        trainable=True, batch_axes=_batch_axes(bundle["task"]), dynamic_batch=(1, bundle["batch_size"]))
    receipt = calibrate_split(model, call, targets, boundaries=anchors,
        make_handle=lambda cut: session.repartition_runtime(handle, cut),
        loss_fn=workload.task.make_adapter().loss, device=device,
        source="server_shape_matched_current_deployment" if bundle["role"] == "server" else "client_private_sample_current_deployment")
    if receipt["model_hash_before"] != bundle["initial_model_hash"]:
        raise ValueError("Admission capture changed the frozen initial model")
    return dict(status="passed", receipt=receipt,
        rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-memory-fraction", type=float)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Admission receipts are immutable; use a fresh path")
    started = time.perf_counter()
    try:
        report = admit(args.bundle, args.device, cuda_memory_fraction=args.cuda_memory_fraction)
    except Exception as exc:
        args.output.write_text(json.dumps(dict(status="failed", error=f"{type(exc).__name__}: {exc}"), indent=2)+"\n")
        raise
    report["elapsed_sec"] = time.perf_counter()-started
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": "passed", "output": str(args.output), "elapsed_sec": report["elapsed_sec"]}), flush=True)


if __name__ == "__main__":
    main()
