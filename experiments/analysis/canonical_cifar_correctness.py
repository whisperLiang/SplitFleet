"""Six-check, all-cut numerical validation of the actual five-backend CIFAR CNN.

The representative input is taken from a frozen real-data bundle. This checks
one SGD update, not model convergence or canonical ResNet18 equivalence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiments.analysis.canonical_backend_correctness import (
    BACKENDS, CHECKS, CanonicalTraining, _compare, _numpy, validate_training,
)
from experiments.multibackend_cifar import ARCHITECTURE, arrays_hash, batch, build


class CanonicalCifarTraining(CanonicalTraining):
    """Reuse the numerical checker with native CNN weights and cross entropy."""

    def __init__(self, backend, bundle, *, batch_size=2, learning_rate=.05):
        self.backend, self.lr = backend, learning_rate
        bundle = Path(bundle)
        self.config = json.loads((bundle / "config.json").read_text())
        for name, digest in self.config["files"].items():
            if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Frozen bundle changed: {name}")
        weights = dict(np.load(bundle / "initial_weights.npz"))
        if arrays_hash(weights) != self.config["initial_weights_sha256"]:
            raise ValueError("Initial parameter contents differ from the frozen bundle")
        device = "/CPU:0" if backend == "tf" else "CPU" if backend == "tinygrad" else "cpu"
        self.model = build(backend, weights, device)
        sample = np.load(bundle / "sample.npz")
        if not 1 <= batch_size <= len(sample["labels"]):
            raise ValueError("Representative batch must fit the recorded real sample")
        call = batch(backend, self.model, sample["images"][:batch_size], sample["labels"][:batch_size], device)
        self.x, self.targets = call.inputs.args[-1], call.targets
        if backend == "jax":
            self.parameters = dict(self.model.initial_params)
        elif backend in ("torch", "paddle"):
            self.parameters = {
                name + suffix: getattr(getattr(self.model, name), attribute)
                for name in ("conv1", "conv2", "fc1", "fc2")
                for suffix, attribute in (("_w", "weight"), ("_b", "bias"))
            }
        elif backend == "tf":
            self.parameters = {
                name + suffix: getattr(self.model.get_layer(name), attribute)
                for name in ("conv1", "conv2", "fc1", "fc2")
                for suffix, attribute in (("_w", "kernel"), ("_b", "bias"))
            }
        else:
            self.parameters = {name: getattr(self.model, name) for name in weights}
        # Native layouts are used for restoration and updates. Only evidence
        # arrays are transposed to common OIHW convolution and IO dense layouts.
        self.initial = {name: _numpy(value) for name, value in self.parameters.items()}
        from splitfleet.backends.utils import adapter_for
        from splitfleet.tasks import ImageClassificationTask
        adapter_for(self.model, call.inputs.args).set_training(self.model, True)
        self.loss = ImageClassificationTask().loss

    def canonical_layout(self, values):
        result = {}
        for name, value in values.items():
            value = _numpy(value)
            if value is not None and name.endswith("_w"):
                if self.backend == "tf" and name.startswith("conv"):
                    value = value.transpose(3, 2, 0, 1)
                elif self.backend == "torch" and name.startswith("fc"):
                    value = value.T
            result[name] = value
        return result

    def snapshot(self):
        return self.canonical_layout(self.parameters)

    def full_step(self):
        result = super().full_step()
        result["parameter_gradients"] = self.canonical_layout(result["parameter_gradients"])
        return result

    def split_step(self, handle):
        result, gradients, received = super().split_step(handle)
        result["parameter_gradients"] = self.canonical_layout(result["parameter_gradients"])
        return result, gradients, received


def validate_cifar_backend(backend, bundle, *, batch_size=2, rtol=2e-4, atol=2e-6,
                          on_progress=None):
    module = "tensorflow" if backend == "tf" else backend
    if importlib.util.find_spec(module) is None:
        return {"backend": backend, "status": "unsupported",
                "reason": f"Missing optional dependency: {module}", "nodes": []}
    native = CanonicalCifarTraining(backend, bundle, batch_size=batch_size)
    result = validate_training(native, model_name="canonical_cifar_cnn", batch_size=batch_size,
                               rtol=rtol, atol=atol, on_progress=on_progress)
    result["parameter_count"] = sum(value.size for value in native.initial.values())
    result["representative_examples"] = batch_size
    return result


def run(output, bundle, *, backends=BACKENDS, batch_size=2, timeout=1800,
        rtol=2e-4, atol=2e-6):
    from experiments.common.evidence import begin_run, finish_run, write_csv
    from experiments.www2027_study import save_json

    if not backends or len(set(backends)) != len(backends) or not set(backends) <= set(BACKENDS):
        raise ValueError("Declare a nonempty set of distinct supported backends")
    output, bundle = Path(output).resolve(), Path(bundle).resolve()
    config = {"backends": list(backends), "bundle": str(bundle),
              "bundle_config_sha256": hashlib.sha256((bundle / "config.json").read_bytes()).hexdigest(),
              "architecture": ARCHITECTURE, "batch_size": batch_size, "learning_rate": .05,
              "rtol": rtol, "atol": atol, "timeout_sec": timeout, "device": "cpu",
              "selection": "all enumerated before/after boundaries",
              "scope": "real CIFAR representative input, numerical SGD parity; no convergence or ResNet18 claim"}
    manifest = begin_run(output, config)
    rows = []
    for backend in backends:
        receipt = output / (backend + ".json")
        command = [sys.executable, "-u", "-m", "experiments.analysis.canonical_cifar_correctness",
                   "--child", backend, "--bundle", str(bundle), "--output", str(receipt),
                   "--batch-size", str(batch_size), "--rtol", str(rtol), "--atol", str(atol)]
        env = {**os.environ, "PYTHONPATH": str(output / "runtime_snapshot"),
               "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "DEV": "CPU", "DEBUG": "0"}
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
            env[key] = "1"
        entry = {"backend": backend, "command": command, "timeout_sec": timeout}
        try:
            if hashlib.sha256((bundle / "config.json").read_bytes()).hexdigest() != config["bundle_config_sha256"]:
                raise ValueError("Bundle configuration changed after freeze")
            with (output / (backend + ".log")).open("x") as log:
                process = subprocess.run(command, cwd=output / "runtime_snapshot", env=env,
                                         stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            entry["exit_code"] = process.returncode
            if process.returncode or not receipt.exists():
                raise RuntimeError(f"Backend exited {process.returncode}; see {backend}.log")
            if hashlib.sha256((bundle / "config.json").read_bytes()).hexdigest() != config["bundle_config_sha256"]:
                raise ValueError("Bundle configuration changed during backend validation")
            rows.append(json.loads(receipt.read_text()))
        except Exception as exc:
            rows.append({"backend": backend, "status": "failed", "nodes": [],
                         "reason": f"{type(exc).__name__}: {exc}"})
        save_json(output / (backend + ".execution.json"), entry)
        print(f"CIFAR_CONTRACT backend={backend} status={rows[-1]['status']} counts={rows[-1].get('counts')}", flush=True)
    reference = next((row["native_reference"] for row in rows if row.get("native_reference")), None)
    for row in rows:
        row["native_cross_backend_equivalence"] = None
        if reference is not None and row.get("native_reference"):
            try:
                row["native_cross_backend_max_abs_error"] = _compare(reference, row["native_reference"], rtol=rtol, atol=atol)
                row["native_cross_backend_equivalence"] = True
            except AssertionError as exc:
                row.update(status="failed", native_cross_backend_equivalence=False, reason=str(exc))
    status = "failed" if any(row["status"] == "failed" for row in rows) else (
        "unsupported" if any(row["status"] == "unsupported" for row in rows) else "passed")
    result = {"schema": "splitfleet.canonical-cifar-correctness.v1", "status": status,
              "config": config, "source_identity": manifest["source_identity"], "rows": rows,
              "checks": list(CHECKS), "unsupported_is_success": False}
    save_json(output / "backend_correctness.json", result)
    flat = [{"backend": row["backend"], "boundary": node.get("boundary"), "status": node["status"],
             "reason": node.get("reason"), **{key: node.get("checks", {}).get(key) for key in CHECKS}}
            for row in rows for node in (row["nodes"] or [{"status": row["status"], "reason": row.get("reason")}])]
    write_csv(output / "backend_correctness.csv", flat)
    lines = ["# Canonical CIFAR CNN: six-check numerical correctness", "", config["scope"], "",
             "| Backend | Status | Passed | Failed | Unsupported | Native parity |",
             "|---|---|---:|---:|---:|---|"]
    for row in rows:
        counts = row.get("counts", {})
        lines.append(f"| {row['backend']} | {row['status']} | {counts.get('passed', 0)} | {counts.get('failed', 0)} | {counts.get('unsupported', 0)} | {row['native_cross_backend_equivalence']} |")
    lines += ["", "Same 34,522-parameter model, real images, labels, initial weights and SGD rate across backends. Native parameter arrays are compared in OIHW/IO layout. Unsupported cuts retain reasons and count separately.", ""]
    (output / "backend_correctness.md").write_text("\n".join(lines))
    finish_run(output, result, validation={"status": status, "all_six_checks_required": True,
               "real_representative_inputs": True, "refusals_retained": True}, metrics=flat)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--child", choices=BACKENDS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--atol", type=float, default=2e-6)
    args = parser.parse_args()
    if args.child:
        from experiments.www2027_study import save_json
        progress = args.output.with_suffix(".progress.json")
        def record(nodes):
            save_json(progress, {"backend": args.child, "cuts_checked": len(nodes),
                                 "counts": dict(Counter(row["status"] for row in nodes)), "nodes": nodes})
        try:
            result = validate_cifar_backend(args.child, args.bundle, batch_size=args.batch_size,
                                           rtol=args.rtol, atol=args.atol, on_progress=record)
        except Exception as exc:
            result = {"backend": args.child, "status": "failed", "nodes": [],
                      "reason": f"{type(exc).__name__}: {exc}"}
        save_json(args.output, result)
        return
    result = run(args.output, args.bundle, backends=args.backends, batch_size=args.batch_size,
                 timeout=args.timeout, rtol=args.rtol, atol=args.atol)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
