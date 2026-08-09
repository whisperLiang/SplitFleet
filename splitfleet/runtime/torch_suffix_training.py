"""Torch suffix training compatible with in-place residual operations."""

from __future__ import annotations

import time
from typing import Any

import torch
from torchlens.split import ReplayBoundary


def _timed_phase(fn, device: torch.device | None) -> tuple[Any, float]:
    if device is not None and device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        value = fn()
        end.record()
        torch.cuda.synchronize(device)
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

    TorchLens 2.31 replays its leaf roots directly. Models such as timm ResNet
    use in-place residual adds, which PyTorch correctly rejects on leaf tensors.
    A zero-add view remains connected to the root while being safe for replay.
    """
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
    if loss_fn is not None:
        loss = loss_fn(output, targets)
    elif targets is not None and isinstance(output, torch.Tensor) and isinstance(targets, torch.Tensor):
        loss = torch.nn.functional.mse_loss(output, targets)
    else:
        raise ValueError("Torch suffix training requires loss_fn for non-tensor outputs")
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
        # `server_total_ms` is the key every backend fills; the per-phase split
        # is only available where the suffix is executed phase by phase.
        measurements["server_forward_ms"] = forward_ms
        measurements["server_backward_ms"] = backward_ms
        measurements["server_total_ms"] = forward_ms + backward_ms
    return loss, gradients
