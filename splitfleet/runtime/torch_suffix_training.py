"""Torch suffix training compatible with in-place residual operations."""

from __future__ import annotations

import time
from typing import Any

import torch
from torchlens.split import ReplayBoundary


def _timed_phase(fn, device: torch.device | None) -> tuple[Any, float]:
    if device is not None and device.type == "cuda":
        # gRPC computation threads need not inherit the model's CUDA device.
        # Record both events on that device's stream and wait on the actual
        # end event; synchronizing another device cannot complete these events.
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream.synchronize()
            start.record(stream)
            value = fn()
            end.record(stream)
            end.synchronize()
            return value, float(start.elapsed_time(end))
    started = time.perf_counter_ns()
    value = fn()
    return value, (time.perf_counter_ns() - started) / 1_000_000.0


def train_torch_suffix(
    runtime,
    boundary: ReplayBoundary,
    targets: Any,
    *,
    loss_fn=None,
    optimizer=None,
    measurements: dict[str, float] | None = None,
):
    """Train suffix while replaying non-leaf views of leaf gradient roots.

    TorchLens 2.34.1 replays its leaf roots directly. Models such as timm ResNet
    use in-place residual adds, which PyTorch correctly rejects on leaf tensors.
    A zero-add view remains connected to the root while being safe for replay.
    """
    if loss_fn is None:
        raise ValueError("Suffix training requires an explicit loss_fn.")
    service_started_ns = time.perf_counter_ns()
    runtime.validate_boundary(boundary)
    roots: dict[str, torch.Tensor] = {}
    replay: dict[str, Any] = {}
    for key, value in boundary.tensors.items():
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            root = value.detach().clone().requires_grad_(True)
            roots[key] = root
            replay[key] = root + torch.zeros((), dtype=root.dtype, device=root.device)
        else:
            replay[key] = value
    replay_boundary = ReplayBoundary(
        backend=boundary.backend,
        tensors=replay,
        spec=boundary.spec,
        metadata={**boundary.metadata, "suffix_training_roots": tuple(roots)},
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    device = next(
        (value.device for value in replay.values() if isinstance(value, torch.Tensor)),
        None,
    )
    output, forward_ms = _timed_phase(lambda: runtime.run_suffix(replay_boundary), device)
    loss, loss_ms = _timed_phase(lambda: loss_fn(output, targets), device)
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Split suffix loss is nonfinite; refusing the optimizer update")
    def backward_and_step():
        loss.backward()
        gradients = {
            key: root.grad.detach().clone()
            for key, root in roots.items()
            if root.grad is not None
        }
        if optimizer is not None:
            optimizer.step()
        return gradients

    gradients, backward_ms = _timed_phase(backward_and_step, device)
    if measurements is not None:
        # Detection matching can spend substantial time on CPU between GPU
        # kernels. The service wall span includes validation, boundary roots,
        # loss/matching, backward and the optimizer; F+B alone understates it.
        measurements["server_forward_ms"] = forward_ms
        measurements["server_loss_ms"] = loss_ms
        measurements["server_backward_ms"] = backward_ms
        measurements["server_total_ms"] = (time.perf_counter_ns() - service_started_ns) / 1_000_000.0
    return loss, gradients
