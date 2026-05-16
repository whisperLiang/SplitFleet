"""Thin SplitFleet adapter over the public Ariadne split runtime."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from ariadne import BoundaryPayload, SplitSpec, prepare_split


@dataclass
class AriadneSplitPlan:
    plan_id: str
    split_id: str
    graph_signature: str
    boundary: str
    mode: str
    trainable: bool
    dynamic_batch: tuple[int, int] | None
    trace_batch_mode: str
    boundary_bytes: int
    prefix_node_count: int
    suffix_node_count: int
    trainable_suffix: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AriadneRuntimeHandle:
    model: torch.nn.Module
    runtime: Any
    plan: AriadneSplitPlan


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def infer_trace_batch_mode(example_inputs: Sequence[Any]) -> str:
    """Infer Ariadne's trace batch mode from the first batched tensor."""

    for tensor in _iter_tensors(example_inputs):
        if tensor.ndim == 0:
            continue
        batch_size = int(tensor.shape[0])
        if batch_size == 1:
            return "batch_1"
        if batch_size > 1:
            return "batch_gt1"
    raise ValueError(
        "Ariadne tracing requires at least one batched tensor in sample inputs."
    )


def normalize_example_inputs(sample_inputs: Any) -> tuple[Any, ...]:
    """Normalize SplitFleet sample inputs for Ariadne's positional API."""

    if isinstance(sample_inputs, tuple):
        return sample_inputs
    if isinstance(sample_inputs, list):
        return tuple(sample_inputs)
    return (sample_inputs,)


def _make_plan_id(graph_signature: str, split_id: str, boundary: str, mode: str) -> str:
    digest = hashlib.sha1(
        "|".join([graph_signature, split_id, boundary, mode]).encode("utf-8")
    ).hexdigest()
    return f"ariadne_{digest[:12]}"


def _safe_int(value: Any) -> int:
    return int(value or 0)


def prepare_ariadne_runtime(
    model: torch.nn.Module,
    sample_inputs: Any,
    *,
    boundary: str = "50%",
    mode: str = "generated_eager",
    trainable: bool = True,
    dynamic_batch: tuple[int, int] | None = None,
    trace_batch_mode: str | None = None,
    objective: Any = None,
    compile_options: Any = None,
) -> AriadneRuntimeHandle:
    """Prepare an Ariadne SplitRuntime and summarize the selected split."""

    example_inputs = normalize_example_inputs(sample_inputs)
    inferred_trace_batch_mode = trace_batch_mode or infer_trace_batch_mode(example_inputs)
    runtime = prepare_split(
        model,
        example_inputs=example_inputs,
        split=SplitSpec(
            boundary=boundary,
            trainable=trainable,
            trace_batch_mode=inferred_trace_batch_mode,
            dynamic_batch=dynamic_batch,
        ),
        mode=mode,
        objective=objective,
        compile_options=compile_options,
    )
    candidate = runtime.candidate
    cost = candidate.cost
    graph_signature = str(runtime.graph_signature)
    split_id = str(runtime.split_id)
    plan = AriadneSplitPlan(
        plan_id=_make_plan_id(graph_signature, split_id, boundary, mode),
        split_id=split_id,
        graph_signature=graph_signature,
        boundary=boundary,
        mode=mode,
        trainable=trainable,
        dynamic_batch=dynamic_batch,
        trace_batch_mode=inferred_trace_batch_mode,
        boundary_bytes=_safe_int(getattr(cost, "boundary_bytes", 0)),
        prefix_node_count=_safe_int(getattr(cost, "prefix_node_count", 0)),
        suffix_node_count=_safe_int(getattr(cost, "suffix_node_count", 0)),
        trainable_suffix=bool(getattr(candidate, "trainable_suffix", False)),
        metadata={
            "boundary_nodes": tuple(getattr(candidate, "boundary_nodes", ())),
            "boundary_after": getattr(candidate, "boundary_after", None),
            "passthrough_inputs": tuple(getattr(candidate, "passthrough_inputs", ())),
            "trace_node_count": len(getattr(runtime.trace_plan, "nodes", ())),
            "_example_inputs": example_inputs,
            "_objective": objective,
            "_compile_options": compile_options,
        },
    )
    return AriadneRuntimeHandle(model=model, runtime=runtime, plan=plan)


def run_prefix(handle: AriadneRuntimeHandle, *inputs: Any) -> BoundaryPayload:
    return handle.runtime.run_prefix(*inputs)


def run_training_prefix(handle: AriadneRuntimeHandle, *inputs: Any) -> BoundaryPayload:
    return handle.runtime.run_training_prefix(*inputs)


def run_suffix(handle: AriadneRuntimeHandle, boundary: BoundaryPayload) -> Any:
    return handle.runtime.run_suffix(boundary)


def train_suffix(
    handle: AriadneRuntimeHandle,
    boundary: BoundaryPayload,
    targets: Any,
    *,
    loss_fn=None,
    optimizer=None,
):
    return handle.runtime.train_suffix(
        boundary,
        targets,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )


def backward_prefix(
    handle: AriadneRuntimeHandle,
    boundary: BoundaryPayload,
    boundary_grads: Any,
    *,
    optimizer=None,
) -> None:
    handle.runtime.backward_prefix(
        boundary,
        boundary_grads=boundary_grads,
        optimizer=optimizer,
    )

