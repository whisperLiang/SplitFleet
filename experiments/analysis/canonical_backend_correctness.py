"""RQ1: one identical residual MLP, weights and inputs across five backends.

This small numerical contract experiment is download-free. It is distinct from
the gated backend-specific ResNet18 architecture checks and from task accuracy.
Every enumerated before/after cut retains passed/failed/unsupported outcomes.
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


BACKENDS = ("torch", "tf", "jax", "paddle", "tinygrad")
PARAMETERS = ("w1", "b1", "w2", "b2", "w3", "b3")
CHECKS = ("output", "loss", "parameter_gradients", "one_step_update",
          "boundary_roundtrip", "gradient_roundtrip")


def canonical_arrays(seed=2027):
    rng = np.random.default_rng(seed)
    shapes = ((4, 4), (4,), (4, 4), (4,), (4, 3), (3,))
    state = {name: rng.uniform(.1, .3, shape).astype("float32")
             for name, shape in zip(PARAMETERS, shapes)}
    return state, rng.uniform(.1, .9, (2, 4)).astype("float32"), rng.uniform(0, .5, (2, 3)).astype("float32")


def _numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value).copy()


def _compare(expected, actual, *, rtol, atol, path="value"):
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError("Parameter/gradient keys differ")
        return max((_compare(expected[key], actual[key], rtol=rtol, atol=atol,
                             path=f"{path}.{key}") for key in expected), default=0.)
    if expected is None or actual is None:
        if expected is not None or actual is not None:
            raise AssertionError(f"Missing parameter gradient: {path}")
        return 0.
    left, right = np.asarray(expected), np.asarray(actual)
    if left.shape != right.shape:
        raise AssertionError(f"Shape mismatch: {left.shape} versus {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise AssertionError(f"Nonfinite numerical values: {path}")
    np.testing.assert_allclose(right, left, rtol=rtol, atol=atol)
    # Boolean frontiers are legitimate native graph values. NumPy disallows
    # boolean subtraction; equality was checked above, so use numeric copies
    # solely for the error statistic.
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.


class CanonicalTraining:
    """Backend math only for the same (affine, ReLU, residual, affine) fixture."""

    def __init__(self, backend, *, seed, learning_rate):
        self.backend, self.lr = backend, learning_rate
        state, inputs, targets = canonical_arrays(seed)
        if backend == "torch":
            import torch
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    for name, value in state.items():
                        setattr(self, name, torch.nn.Parameter(torch.tensor(value)))
                def forward(self, x):
                    y = (x @ self.w1 + self.b1).relu()
                    y = (y @ self.w2 + self.b2 + x).relu()
                    return y @ self.w3 + self.b3
            self.model = Model().train()
            self.x, self.targets = torch.tensor(inputs, requires_grad=True), torch.tensor(targets)
            self.parameters = dict(self.model.named_parameters())
        elif backend == "tf":
            import tensorflow as tf
            class Model(tf.keras.Model):
                def __init__(self):
                    super().__init__(name="canonical_residual_mlp")
                    for name, value in state.items():
                        setattr(self, name, self.add_weight(name=name, shape=value.shape,
                            initializer=tf.keras.initializers.Constant(value)))
                def call(self, x, training=None):
                    y = tf.nn.relu(x @ self.w1 + self.b1)
                    y = tf.nn.relu(y @ self.w2 + self.b2 + x)
                    return y @ self.w3 + self.b3
            self.model = Model()
            self.x, self.targets = tf.constant(inputs), tf.constant(targets)
            self.model(self.x)
            self.parameters = {name: getattr(self.model, name) for name in PARAMETERS}
        elif backend == "paddle":
            # Load Paddle before SplitFleet/TorchLens discovers native frameworks.
            import paddle
            paddle.set_device("cpu")
            class Model(paddle.nn.Layer):
                def __init__(self):
                    super().__init__()
                    for name, value in state.items():
                        parameter = self.create_parameter(value.shape,
                            default_initializer=paddle.nn.initializer.Assign(value))
                        self.add_parameter(name, parameter)
                def forward(self, x):
                    y = paddle.nn.functional.relu(x @ self.w1 + self.b1)
                    y = paddle.nn.functional.relu(y @ self.w2 + self.b2 + x)
                    return y @ self.w3 + self.b3
            self.model = Model().train()
            self.x, self.targets = paddle.to_tensor(inputs, stop_gradient=False), paddle.to_tensor(targets)
            self.parameters = dict(self.model.named_parameters())
        elif backend == "jax":
            import jax.numpy as jnp
            def model(p, x):
                y = jnp.maximum(x @ p["w1"] + p["b1"], 0)
                y = jnp.maximum(y @ p["w2"] + p["b2"] + x, 0)
                return y @ p["w3"] + p["b3"]
            self.model = model
            self.parameters = {name: jnp.asarray(value) for name, value in state.items()}
            self.x, self.targets = jnp.asarray(inputs), jnp.asarray(targets)
        else:
            from tinygrad import Tensor
            Tensor.training = True
            class Model:
                def __init__(self):
                    for name, value in state.items():
                        parameter = Tensor(value).realize()
                        parameter.requires_grad = True
                        setattr(self, name, parameter)
                def __call__(self, x):
                    y = (x @ self.w1 + self.b1).relu()
                    y = (y @ self.w2 + self.b2 + x).relu()
                    return y @ self.w3 + self.b3
            self.model = Model()
            self.x, self.targets = Tensor(inputs).realize(), Tensor(targets).realize()
            self.x.requires_grad = True
            self.parameters = {name: getattr(self.model, name) for name in PARAMETERS}
        self.initial = state

    @property
    def args(self):
        return (self.parameters, self.x) if self.backend == "jax" else (self.x,)

    def snapshot(self):
        return {name: _numpy(value) for name, value in self.parameters.items()}

    def restore(self):
        for name, value in self.parameters.items():
            original = self.initial[name]
            if self.backend == "torch":
                import torch
                with torch.no_grad(): value.copy_(torch.tensor(original))
                value.grad = None
            elif self.backend == "tf": value.assign(original)
            elif self.backend == "paddle":
                value.set_value(original)
                value.clear_gradient()
            elif self.backend == "jax":
                import jax.numpy as jnp
                self.parameters[name] = jnp.asarray(original)
            else:
                from tinygrad import Tensor
                value.assign(Tensor(original)).realize()
                value.grad = None

    def loss(self, output, targets):
        value = (output - targets) ** 2
        if self.backend == "tf":
            import tensorflow as tf
            return tf.reduce_mean(value)
        return value.mean()

    def _update(self, gradients):
        for name, gradient in gradients.items():
            if gradient is None:
                continue
            parameter = self.parameters[name]
            if self.backend == "torch":
                import torch
                with torch.no_grad(): parameter.add_(torch.tensor(gradient), alpha=-self.lr)
            elif self.backend == "tf": parameter.assign_sub(self.lr * gradient)
            elif self.backend == "paddle": parameter.set_value(_numpy(parameter) - self.lr * gradient)
            elif self.backend == "jax": self.parameters[name] = parameter - self.lr * gradient
            else:
                from tinygrad import Tensor
                parameter.assign(Tensor(_numpy(parameter) - self.lr * gradient)).realize()

    def full_step(self):
        output = self.model(*self.args)
        if self.backend == "tf":
            import tensorflow as tf
            with tf.GradientTape() as tape:
                output = self.model(*self.args)
                loss = self.loss(output, self.targets)
            gradients = dict(zip(self.parameters, tape.gradient(loss, list(self.parameters.values()))))
        elif self.backend == "jax":
            import jax
            loss, gradients = jax.value_and_grad(lambda p: self.loss(self.model(p, self.x), self.targets))(self.parameters)
        else:
            loss = self.loss(output, self.targets)
            loss.backward()
            gradients = {name: value.grad for name, value in self.parameters.items()}
        out, loss_value = _numpy(output), _numpy(loss)
        gradients = {name: _numpy(value) for name, value in gradients.items()}
        self._update(gradients)
        return {"output": out, "loss": loss_value, "parameter_gradients": gradients,
                "one_step_update": self.snapshot()}

    def split_step(self, handle):
        from splitfleet.validation.matrix import _wire_boundary
        from splitfleet.transport import encode_gradients, decode_gradients
        from splitfleet.transport.split_wire import gradients_to_envelope, envelope_to_gradients

        training = self
        class GradientCollector:
            def __init__(self): self.gradients = {name: None for name in training.parameters}
            def apply_gradients(self, pairs):
                for gradient, variable in pairs:
                    names = [name for name, value in training.parameters.items()
                             if variable is value or variable is getattr(value, "value", None)]
                    if len(names) != 1:
                        raise AssertionError("Cannot bind TensorFlow gradient to canonical parameter")
                    name = names[0]
                    value = _numpy(gradient)
                    old = self.gradients[name]
                    self.gradients[name] = value if old is None else old + value

        collector = GradientCollector() if self.backend == "tf" else None
        local = handle.backend.run_prefix(*self.args, training=True)
        remote, wire = _wire_boundary(handle, local)
        loss, boundary_gradients = handle.backend.train_suffix(remote, self.targets,
            loss_fn=self.loss, optimizer=collector)
        received = envelope_to_gradients(decode_gradients(encode_gradients(
            gradients_to_envelope(wire, boundary_gradients))), None)
        prefix_result = handle.backend.backward_prefix(local, received, optimizer=collector)
        if self.backend == "tf": gradients = collector.gradients
        elif self.backend == "jax": gradients = prefix_result["inputs"][0]
        else: gradients = {name: value.grad for name, value in self.parameters.items()}
        gradients = {name: _numpy(value) for name, value in gradients.items()}
        loss_value = _numpy(loss)
        self._update(gradients)
        return {"loss": loss_value, "parameter_gradients": gradients, "one_step_update": self.snapshot()}, boundary_gradients, received


def validate_backend(backend, *, seed=2027, learning_rate=.01, rtol=2e-4, atol=2e-6):
    module = "tensorflow" if backend == "tf" else backend
    if importlib.util.find_spec(module) is None:
        return {"backend": backend, "status": "unsupported", "reason": f"Missing optional dependency: {module}", "nodes": []}
    native = CanonicalTraining(backend, seed=seed, learning_rate=learning_rate)
    return validate_training(native, model_name="canonical_residual_mlp", batch_size=2,
                             rtol=rtol, atol=atol)


def validate_training(native, *, model_name, batch_size, rtol=2e-4, atol=2e-6,
                      on_progress=None):
    """Apply the same six checks to every enumerated cut of a native model."""
    backend = native.backend
    from splitfleet.autosplit import prepare_torchlens_runtime
    from torchlens.split.errors import SplitUnsupportedError, SplitRequestError
    from dataclasses import replace
    import copy

    native.restore()
    expected = native.full_step()
    native.restore()
    handle = prepare_torchlens_runtime(native.model, native.args, boundary="50%", trainable=True,
        batch_axes={}, dynamic_batch=(batch_size, batch_size), model_name=model_name, model_family=model_name)
    view = copy.copy(handle.runtime)
    view.request = replace(view.request, validation="permissive")
    rows = []
    for site in view.split_points(diagnose=False).candidates:
        row = {"boundary": f"{site.kind}:{site.node_id}", "status": "failed", "checks": {}}
        rows.append(row)
        try:
            analysis = view.analyze(site.point)
            report = analysis.capability_report
            if report is not None and not (report.preflight.supported and report.replay.supported and report.training.supported):
                row.update(status="unsupported", reason="; ".join(report.unsupported_reasons) or "Native training refusal")
                continue
            nodes = {node.canonical_id: node for node in view.trace_graph.nodes}
            if not any(not nodes[node].is_output for node in analysis.plan.suffix_node_ids):
                row.update(status="unsupported", reason="suffix_not_trainable: terminal boundary")
                continue
            native.restore()
            runtime = handle.backend.repartition(row["boundary"])
            from splitfleet.validation.matrix import _wire_boundary
            payload = runtime.backend.run_prefix(*native.args)
            remote, _ = _wire_boundary(runtime, payload)
            row["checks"]["boundary_roundtrip"] = _compare(
                {name: _numpy(value) for name, value in payload.tensors.items()},
                {name: _numpy(value) for name, value in remote.tensors.items()}, rtol=rtol, atol=atol)
            row["checks"]["output"] = _compare(expected["output"], _numpy(runtime.backend.run_suffix(remote)), rtol=rtol, atol=atol)
            native.restore()
            actual, gradients, received = native.split_step(runtime)
            row["checks"]["gradient_roundtrip"] = _compare(
                {name: _numpy(value) for name, value in gradients.items()},
                {name: _numpy(value) for name, value in received.items()}, rtol=rtol, atol=atol)
            for key in ("loss", "parameter_gradients", "one_step_update"):
                row["checks"][key] = _compare(expected[key], actual[key], rtol=rtol, atol=atol)
            row.update(status="passed", split_id=runtime.plan.split_id, graph_signature=runtime.plan.graph_signature,
                       feature_abi_id=runtime.plan.feature_abi_id)
        except (SplitUnsupportedError, SplitRequestError) as exc:
            row.update(status="unsupported", reason=f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        finally:
            if on_progress is not None:
                on_progress(rows)
    native.restore()
    counts = dict(Counter(row["status"] for row in rows))
    status = "failed" if counts.get("failed") else "passed" if counts.get("passed") else "unsupported"
    serialize = lambda value: {k: serialize(v) for k, v in value.items()} if isinstance(value, dict) else (
        None if value is None else np.asarray(value).tolist())
    return {"backend": backend, "status": status, "counts": {key: counts.get(key, 0) for key in ("passed", "failed", "unsupported")},
            "all_enumerated_boundaries_passed": bool(rows) and counts.get("passed", 0) == len(rows),
            "nodes": rows, "native_reference": serialize(expected)}


def run(output: Path, *, backends=BACKENDS, seed=2027, timeout=600, rtol=2e-4, atol=2e-6):
    from experiments.common.evidence import begin_run, finish_run, write_csv
    from experiments.www2027_study import save_json

    output = output.resolve()
    arrays, x, target = canonical_arrays(seed)
    input_hash = hashlib.sha256(b"".join(value.tobytes() for value in [*arrays.values(), x, target])).hexdigest()
    config = {"backends": list(backends), "seed": seed, "model": "canonical_residual_mlp_4_4_3",
        "learning_rate": .01, "rtol": rtol, "atol": atol, "input_state_sha256": input_hash,
        "selection": "all enumerated before/after boundaries", "device": "cpu", "timeout_sec": timeout,
        "scope": "Numerical correctness on synthetic inputs; no task accuracy or ResNet18 claim"}
    manifest = begin_run(output, config)
    results = []
    for backend in backends:
        receipt = output / f"{backend}.json"
        env = {**os.environ, "PYTHONPATH": str(output / "runtime_snapshot"), "CUDA_VISIBLE_DEVICES": "",
               "JAX_PLATFORMS": "cpu", "DEV": "CPU", "DEBUG": "0"}
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
            env[key] = "1"
        command = [sys.executable, "-m", "experiments.analysis.canonical_backend_correctness", "--child", backend,
                   "--output", str(receipt), "--seed", str(seed), "--rtol", str(rtol), "--atol", str(atol)]
        with (output / "stdout.log").open("a") as log:
            log.write(f"Backend {backend}: {command}\n"); log.flush()
            try:
                child = subprocess.run(command, cwd=output / "runtime_snapshot", env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                if child.returncode or not receipt.exists():
                    raise RuntimeError(f"Backend process exited {child.returncode}; see stdout.log")
                results.append(json.loads(receipt.read_text()))
            except Exception as exc:
                results.append({"backend": backend, "status": "failed", "reason": f"{type(exc).__name__}: {exc}", "nodes": []})
        print(f"canonical backend={backend} status={results[-1]['status']}", flush=True)
    reference = next((row.get("native_reference") for row in results if row.get("native_reference")), None)
    for row in results:
        row["native_cross_backend_equivalence"] = None
        if reference is not None and row.get("native_reference"):
            try:
                row["native_cross_backend_max_abs_error"] = _compare(reference, row["native_reference"], rtol=rtol, atol=atol)
                row["native_cross_backend_equivalence"] = True
            except AssertionError as exc:
                row.update(status="failed", native_cross_backend_equivalence=False, reason=str(exc))
    result = {"schema": "splitfleet.canonical-backend-correctness.v1", "config": config,
        "status": "failed" if any(row["status"] == "failed" for row in results) else
                  "unsupported" if any(row["status"] == "unsupported" for row in results) else "passed",
        "source_identity": manifest["source_identity"], "rows": results,
        "checks": list(CHECKS), "unsupported_is_success": False}
    save_json(output / "backend_correctness.json", result)
    flat = [{"backend": row["backend"], "boundary": node.get("boundary"), "status": node["status"],
        "reason": node.get("reason"), **{key: node.get("checks", {}).get(key) for key in CHECKS}}
        for row in results for node in (row["nodes"] or [{"status": row["status"], "reason": row.get("reason")}])]
    write_csv(output / "backend_correctness.csv", flat)
    lines = ["# Canonical five-backend numerical correctness", "", config["scope"], "",
        "All backends use the same affine/ReLU residual MLP, input, target and initial parameter arrays.", "",
        "| Backend | Status | Passed | Failed | Unsupported | Native parity |", "|---|---|---:|---:|---:|---|"]
    for row in results:
        counts = row.get("counts", {})
        lines.append(f"| {row['backend']} | {row['status']} | {counts.get('passed', 0)} | {counts.get('failed', 0)} | {counts.get('unsupported', 0)} | {row['native_cross_backend_equivalence']} |")
    lines += ["", "Unsupported boundaries retain refusal reasons and do not count as passed training cuts. Backend passed means its supported checked cuts passed; see per-cut rows for coverage. All six checks must pass together.", ""]
    (output / "backend_correctness.md").write_text("\n".join(lines))
    finish_run(output, result, validation={"status": result["status"], "all_six_checks_required": True,
        "source_frozen": True, "refusals_retained": True}, metrics=flat)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--child", choices=BACKENDS)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--atol", type=float, default=2e-6)
    args = parser.parse_args()
    if args.child:
        try:
            result = validate_backend(args.child, seed=args.seed, rtol=args.rtol, atol=args.atol)
        except Exception as exc:
            result = {"backend": args.child, "status": "failed", "reason": f"{type(exc).__name__}: {exc}", "nodes": []}
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return
    result = run(args.output, backends=args.backends, seed=args.seed, timeout=args.timeout, rtol=args.rtol, atol=args.atol)
    if result["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__":
    main()
