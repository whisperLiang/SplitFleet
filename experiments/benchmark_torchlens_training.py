"""Measure split training with matched inputs, state, and optimizer settings.

Run once before changing the installed wheel and once afterwards. Each result
records the installed wheel hash and raw timings for a reproducible comparison.
Timings include prefix forward, suffix loss/backward, prefix backward, and one
SGD update; capture, correctness checks, and warmup are reported separately.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time
from urllib.parse import unquote, urlparse

import torch
from torch import nn

from splitfleet.autosplit import prepare_torchlens_runtime


def _workload(name: str, device: torch.device):
    if name in ("resnet18", "resnet50"):
        from torchvision import models

        model = getattr(models, name)(weights=None, num_classes=10).to(device).train()
        inputs = torch.randn(4, 3, 96, 96, device=device)
        targets = torch.randint(0, 10, (4,), device=device)
        return model, inputs, targets, nn.CrossEntropyLoss()
    if name == "rfdetr_nano":
        from experiments.rfdetr_nano_physical import RFDETRDetectionTask, RFDETRNanoDetector

        model = RFDETRNanoDetector().to(device).train()
        inputs = torch.rand(1, 3, 384, 384, device=device)
        targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device),
                    "labels": torch.tensor([5], device=device)}]
        return model, inputs, targets, RFDETRDetectionTask().loss
    raise ValueError(f"Unknown workload: {name}")


def _installed_wheel() -> dict:
    distribution = importlib.metadata.distribution("torchlens")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    url = direct.get("url", "")
    path = Path(unquote(urlparse(url).path)) if url.startswith("file:") else None
    return {
        "version": distribution.version,
        "url": url,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None,
        "provenance": distribution.read_text("SPLITFLEET_PATCHES"),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_hash(value) -> str:
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            raise TypeError(f"Unsupported benchmark input: {type(item)}")

    visit(value)
    return digest.hexdigest()


def _restore(model, state, seed: int) -> None:
    model.load_state_dict(state)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_step(handle, inputs, targets, loss_fn, optimizer):
    optimizer.zero_grad(set_to_none=True)
    boundary = handle.backend.run_prefix(inputs, training=True)
    loss, gradients = handle.backend.train_suffix(boundary, targets, loss_fn=loss_fn)
    handle.backend.backward_prefix(boundary, boundary_grads=gradients)
    optimizer.step()
    return loss.detach()


def _native_step(model, inputs, targets, loss_fn, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


def _measure(step, *, device, warmup, steps):
    for _ in range(warmup):
        step()
    _synchronize(device)
    timings = []
    loss = None
    for _ in range(steps):
        started = time.perf_counter_ns()
        loss = step()
        _synchronize(device)
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    if not torch.isfinite(loss).all():
        raise AssertionError("Measured training produced a non-finite loss")
    return timings, float(loss)


def benchmark(name: str, device: torch.device, args) -> dict:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, inputs, targets, loss_fn = _workload(name, device)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    input_sha256, target_sha256 = _tensor_hash(inputs), _tensor_hash(targets)
    reference = copy.deepcopy(model)
    _synchronize(device)
    started = time.perf_counter()
    handle = prepare_torchlens_runtime(
        model, inputs, boundary=args.boundary, trainable=True,
        batch_axes={"/args/0": 0}, dynamic_batch=(inputs.shape[0], inputs.shape[0]),
    )
    _synchronize(device)
    capture_sec = time.perf_counter() - started
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    native_optimizer = torch.optim.SGD(reference.parameters(), lr=args.lr)

    # Capture can update BatchNorm buffers. Restore both models before checking
    # the same complete update, including every gradient and mutable buffer.
    _restore(reference, state, args.seed)
    native_loss = _native_step(reference, inputs, targets, loss_fn, native_optimizer)
    _restore(model, state, args.seed)
    split_loss = _split_step(handle, inputs, targets, loss_fn, optimizer)
    torch.testing.assert_close(split_loss, native_loss, rtol=2e-4, atol=2e-5)
    reference_parameters = dict(reference.named_parameters())
    max_gradient_difference = 0.0
    for key, parameter in model.named_parameters():
        expected = reference_parameters[key].grad
        if (parameter.grad is None) != (expected is None):
            raise AssertionError(f"Gradient ownership mismatch: {key}")
        if expected is not None:
            torch.testing.assert_close(parameter.grad, expected, rtol=2e-4, atol=2e-5,
                                       msg=lambda msg: f"{key}: {msg}")
            max_gradient_difference = max(
                max_gradient_difference, float((parameter.grad - expected).abs().max()),
            )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=2e-4, atol=2e-5,
                                   msg=lambda msg: f"{key}: {msg}")

    split_timings, native_timings, split_losses, native_losses = [], [], [], []
    for repeat in range(args.repeats):
        # Alternate measurement order to reduce cache/temperature order bias.
        for method in (("split", "native") if repeat % 2 == 0 else ("native", "split")):
            selected = model if method == "split" else reference
            _restore(selected, state, args.seed)
            if method == "split":
                step = lambda: _split_step(handle, inputs, targets, loss_fn, optimizer)
            else:
                step = lambda: _native_step(reference, inputs, targets, loss_fn, native_optimizer)
            measured, loss = _measure(step, device=device, warmup=args.warmup, steps=args.steps)
            (split_timings if method == "split" else native_timings).append(measured)
            (split_losses if method == "split" else native_losses).append(loss)
    split_medians = [statistics.median(times) for times in split_timings]
    native_medians = [statistics.median(times) for times in native_timings]
    result = {
        "model": name, "device": str(device), "input_shape": list(inputs.shape),
        "boundary": handle.plan.boundary, "graph_signature": handle.plan.graph_signature,
        "input_sha256": input_sha256, "target_sha256": target_sha256,
        "initial_state_sha256": _tensor_hash(state),
        "capture_sec": capture_sec, "correctness": {
            "loss": float(split_loss), "gradient_max_abs_difference": max_gradient_difference,
            "loss_gradients_and_updated_state_match_native": True,
            "rtol": 2e-4, "atol": 2e-5,
        },
        "split_median_ms": statistics.median(split_medians),
        "native_median_ms": statistics.median(native_medians),
        "split_repeat_medians_ms": split_medians, "native_repeat_medians_ms": native_medians,
        "split_steps_ms": split_timings, "native_steps_ms": native_timings,
        "final_split_losses": split_losses, "final_native_losses": native_losses,
        "final_loss_max_abs_difference": max(abs(a - b) for a, b in zip(split_losses, native_losses)),
    }
    del handle, reference, reference_parameters, model, state, optimizer, native_optimizer, step
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("resnet18", "resnet50", "rfdetr_nano"),
                        default=["resnet18", "resnet50", "rfdetr_nano"])
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--boundary", default="50%")
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.threads) < 1 or args.warmup < 0:
        parser.error("steps, repeats, and threads must be positive; warmup must be nonnegative")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {
        "schema": "splitfleet.torchlens-training-benchmark.v1", "label": args.label,
        "torchlens": _installed_wheel(), "torch": torch.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "settings": {key: value for key, value in vars(args).items() if key not in ("output", "label")},
        "numeric_settings": {"cudnn_benchmark": False, "cudnn_deterministic": True,
                             "cudnn_allow_tf32": False, "matmul_allow_tf32": False},
        "scope": "Local synthetic-input training; no data loading, network, or federated aggregation",
        "measurements": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for device in args.devices:
        for name in args.models:
            measurement = benchmark(name, torch.device(device), args)
            result["measurements"].append(measurement)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({key: value for key, value in measurement.items()
                              if key in ("model", "device", "split_median_ms", "native_median_ms",
                                         "capture_sec", "correctness")}), flush=True)
    result["complete"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
