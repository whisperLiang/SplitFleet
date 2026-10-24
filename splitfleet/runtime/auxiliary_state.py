"""Auxiliary state from the same native split forward used for differentiation.

Models opt in by exposing ``apply_statistics`` and enabling ``defer_statistics``
(an inherently functional model can omit the latter).  Capture and replay stay
pure; the caller commits stage-owned BatchNorm prefixes after the real batch.
All pinned-runtime extensions bind private function/instance copies, leaving
installed TorchLens and framework globals untouched.
"""

from __future__ import annotations

import hashlib
import importlib
import types
from collections.abc import Mapping
from dataclasses import replace
from typing import Any


def supports_auxiliary_state(model: Any) -> bool:
    return callable(getattr(model, "apply_statistics", None)) and bool(
        getattr(model, "defer_statistics", True)
    )


def _clone(function, **bindings):
    local = types.FunctionType(function.__code__, dict(function.__globals__, **bindings),
                               function.__name__, function.__defaults__, function.__closure__)
    local.__kwdefaults__ = function.__kwdefaults__
    return local


def restore_auxiliary_output(graph: Any, output: Any) -> Mapping[str, Any]:
    """Recover native flattened leaves using captured paths, never leaf indexes."""
    if isinstance(output, Mapping):
        restored = output
    else:
        from torchlens.ir.container import rebuild_container_from_spec, reorder_container_leaves

        nodes = [graph.node_by_id[node_id] for node_id in graph.output_node_ids]
        values = list(output) if isinstance(output, (tuple, list)) else [output]
        if len(nodes) != len(values):
            raise RuntimeError("Native auxiliary output leaf count differs from captured metadata")
        specifications = [node.output_container_spec for node in nodes
                          if node.output_container_spec is not None]
        if not specifications or any(spec != specifications[0] for spec in specifications):
            raise RuntimeError("Native auxiliary output has missing or inconsistent container metadata")
        leaves = []
        for node, value in zip(nodes, values, strict=True):
            paths = node.output_container_paths or (node.output_container_path,)
            if not all(paths):
                raise RuntimeError("Native auxiliary output has an unaddressed leaf")
            leaves.extend((path, value) for path in paths)
        restored = rebuild_container_from_spec(
            specifications[0], reorder_container_leaves(specifications[0], leaves),
        )
    if not isinstance(restored, Mapping) or set(restored) != {"logits", "batch_statistics"}:
        raise RuntimeError("Auxiliary model must return logits and batch_statistics")
    statistics = restored["batch_statistics"]
    if not isinstance(statistics, Mapping) or any(
        not isinstance(name, str) or not isinstance(row, Mapping)
        or set(row) != {"mean", "variance", "count"}
        for name, row in statistics.items()
    ):
        raise RuntimeError("Auxiliary BatchNorm statistics do not match the declared protocol")
    return restored


def commit_auxiliary_state(model: Any, statistics: Mapping[str, Any], *, names=None) -> None:
    """Commit one actual batch for the explicitly selected canonical BN prefixes."""
    if not supports_auxiliary_state(model):
        raise TypeError("Model does not enable the auxiliary statistics protocol")
    selected = set(statistics) if names is None else ({names} if isinstance(names, str) else set(names))
    if selected - set(statistics):
        raise ValueError("Selected BatchNorm prefixes have no statistics for this batch")
    model.apply_statistics(statistics, names=selected)


def _container_metadata(output: Any, is_tensor):
    """Describe the actual native output tree, including dotted dictionary keys."""
    from torchlens.ir.container import ContainerSpec, DictKey, TupleIndex

    paths = []

    def describe(value, path=()):
        if is_tensor(value):
            paths.append((path, value))
            return None
        if isinstance(value, dict):
            children = []
            for key, item in value.items():
                child = describe(item, (*path, key))
                if child is not None:
                    children.append((DictKey(key), child))
            return ContainerSpec("dict", length=len(value), keys=tuple(value), child_specs=tuple(children))
        if isinstance(value, (tuple, list)):
            children = []
            for index, item in enumerate(value):
                child = describe(item, (*path, index))
                if child is not None:
                    children.append((TupleIndex(index), child))
            return ContainerSpec("tuple" if isinstance(value, tuple) else "list",
                                 length=len(value), child_specs=tuple(children))
        raise TypeError("Auxiliary outputs must be native tensors in builtin containers")

    specification = describe(output)
    if specification is None:
        raise RuntimeError("Auxiliary model returned a tensor without its declared container")
    return specification, paths


def _amend_output_metadata(trace, bindings, specification):
    from torchlens.ir.op_record import amend_preview_output_parent_rebind

    by_label = {}
    for label, path in bindings:
        if label is None:
            raise RuntimeError("Auxiliary output has no native producer label")
        by_label.setdefault(label, []).append(path)
    for label, paths in by_label.items():
        event = trace.capture_events.op_event_by_label_raw.get(label)
        if event is None:
            raise RuntimeError("Auxiliary output producer is absent from the native capture")
        # A producer's own output path addresses its function's return value.
        # Final model paths are separate aliases and must not alter raw replay.
        updated = replace(event.output, container_spec=specification)
        trace.capture_events.append_amendment(amend_preview_output_parent_rebind(
            event.seq, label, is_output_parent=True, output=updated,
        ))
    trace._splitfleet_auxiliary_output_paths = {label: tuple(paths) for label, paths in by_label.items()}


def _capture_auxiliary_model(model, inputs, request, *, input_kwargs=None,
                             output_observer=None, random_seed=None, shape_witness=False):
    """Attach final-container metadata to an independent native Paddle capture."""
    if str(request.backend) != "paddle":
        raise ValueError("Auxiliary container capture is only needed for Paddle")
    from torchlens.backends.paddle.backend import PaddleBackend

    options = dict(input_kwargs=input_kwargs, layers_to_save="all", keep_orphans=True,
                   output_transform=output_observer)
    if random_seed is not None:
        options["random_seed"] = random_seed
    backend = PaddleBackend()
    mark_outputs = backend._mark_outputs

    def mark(trace, output):
        specification, leaves = _container_metadata(output, backend.is_tensor)
        bindings = [(backend.tensor_store.get_label(value), path) for path, value in leaves]
        mark_outputs(trace, output)
        _amend_output_metadata(trace, bindings, specification)

    backend._mark_outputs = mark
    return backend.capture_trace(model, inputs, **options)


def prepare_auxiliary_runtime(model, inputs, request, *, input_kwargs=None):
    """Preserve formal output paths while retaining native planning and probes."""
    from splitfleet.autosplit.torchlens_runtime import require_torchlens_version

    require_torchlens_version()
    if not supports_auxiliary_state(model):
        raise TypeError("Auxiliary preparation requires an explicitly opted-in model")
    api = importlib.import_module("torchlens.split.api")
    pipeline = importlib.import_module("torchlens.split.pipeline")
    planner_bindings = {}
    if request.backend == "paddle":
        from splitfleet.runtime.native_frontier import _native_bindings

        # Prefix BN statistics are already final output leaves. Carry those
        # leaves, together with every actual suffix operand, without uploading
        # their unused transpose/square activation parents on every batch.
        # Preparation, numeric batch probes and later recuts must share this
        # same source-pinned planner and the resulting boundary ABI.
        private_plan, _ = _native_bindings()
        planner_bindings["plan_split"] = private_plan

    def graph_from_capture(capture):
        graph = pipeline.split_graph_from_trace(capture)
        paths = capture._splitfleet_auxiliary_output_paths
        nodes = tuple(replace(node, output_container_paths=paths[node.raw_label])
                      if node.is_output and node.raw_label in paths else node
                      for node in graph.nodes)
        metadata = tuple((node.canonical_id, node.output_container_paths, node.output_container_spec)
                         for node in nodes if node.is_output)
        signature = hashlib.sha256(
            f"{graph.graph_shape_hash or ''}:{metadata!r}".encode("utf-8"),
        ).hexdigest()
        return replace(graph, nodes=nodes, graph_shape_hash=signature)

    native_execute_probe = _clone(pipeline._execute_batch_probe, **planner_bindings)

    def execute_probe(adapter, graph, *args, **kwargs):
        result = native_execute_probe(adapter, graph, *args, **kwargs)
        return restore_auxiliary_output(graph, result)

    canonical = _clone(pipeline.capture_canonical_model, capture_model=_capture_auxiliary_model)
    probe = _clone(pipeline._probe_batch_replay, capture_model=_capture_auxiliary_model,
                   split_graph_from_trace=graph_from_capture, _execute_batch_probe=execute_probe)
    normalize = _clone(pipeline.normalize_to_split_ir, split_graph_from_trace=graph_from_capture,
                       _probe_batch_replay=probe, **planner_bindings)

    prepare = _clone(api.prepare, capture_canonical_model=canonical, normalize_to_split_ir=normalize,
                     **planner_bindings)
    runtime = prepare(model, inputs, request, input_kwargs=input_kwargs)
    if request.backend == "paddle":
        from splitfleet.runtime.native_frontier import bind_native_frontier

        runtime = bind_native_frontier(runtime)
    return runtime
