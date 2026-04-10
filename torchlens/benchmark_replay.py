"""Benchmark helpers for direct forward, replay forward, and partitioned replay."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch

from .replay_engine import _build_model_call_roots, replay_forward, replay_partitioned
from .replay_plan import ExecutionPlan, FrontierSplit
from .replay_utils import resolve_device, tree_to_device


def benchmark_replay(
    plan: ExecutionPlan,
    inputs: Any,
    *,
    input_kwargs: Optional[Dict[str, Any]] = None,
    split: Optional[FrontierSplit] = None,
    device: Any = "auto",
    iterations: int = 10,
    warmup: int = 2,
    measure_peak_memory: bool = False,
) -> Dict[str, Any]:
    """Benchmark direct forward, full replay, and optional partitioned replay.

    Args:
        plan: Compiled execution plan.
        inputs: Raw benchmark inputs.
        input_kwargs: Optional benchmark keyword inputs.
        split: Optional split used for partitioned replay timing.
        device: Target benchmark device selector.
        iterations: Timed iterations per benchmark.
        warmup: Untimed warmup iterations.
        measure_peak_memory: Whether to capture CUDA peak memory metrics.

    Returns:
        Dict of benchmark measurements in seconds and optional bytes.
    """

    model = plan.model
    if model is None:
        raise ValueError(
            "Benchmarking requires the execution plan to retain a live model reference."
        )

    runtime_device = resolve_device(device, model=model, inputs=inputs, default=plan.default_device)
    model = model.to(runtime_device)
    inputs = tree_to_device(inputs, runtime_device)
    input_kwargs = tree_to_device(input_kwargs or {}, runtime_device)

    direct_args, direct_kwargs = _build_model_call_roots(plan, inputs, input_kwargs, runtime_device)

    results = {
        "device": str(runtime_device),
        "iterations": iterations,
        "warmup": warmup,
        "direct_forward_s": _time_callable(
            lambda: model(*direct_args, **direct_kwargs),
            iterations=iterations,
            warmup=warmup,
            measure_peak_memory=measure_peak_memory,
            device=runtime_device,
        ),
        "replay_forward_s": _time_callable(
            lambda: replay_forward(plan, inputs, input_kwargs=input_kwargs, device=runtime_device),
            iterations=iterations,
            warmup=warmup,
            measure_peak_memory=measure_peak_memory,
            device=runtime_device,
        ),
    }

    if split is not None:
        results["replay_partitioned_s"] = _time_callable(
            lambda: replay_partitioned(
                plan,
                inputs,
                split=split,
                input_kwargs=input_kwargs,
                input_mode="raw",
                device=runtime_device,
            ),
            iterations=iterations,
            warmup=warmup,
            measure_peak_memory=measure_peak_memory,
            device=runtime_device,
        )

    return results


def _time_callable(
    fn,
    *,
    iterations: int,
    warmup: int,
    measure_peak_memory: bool,
    device: torch.device,
) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if measure_peak_memory and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    result = {"avg_seconds": elapsed / iterations, "total_seconds": elapsed}
    if measure_peak_memory and device.type == "cuda":
        result["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    return result
