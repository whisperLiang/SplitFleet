"""Rebind a compact TorchLens graph to an independent Torch model replica.

The server passes copies of the same model. Only live registered state is
rebound; captured constants and portable graph metadata remain shared. Other
backends use their native preparation path. Incompatible Torch replicas are
rejected.
"""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace
from types import MethodType
from weakref import ref

import torch
from torchlens.split import SplitRuntime
from torchlens.split._torch_compact import _BufferHandle, _ParamHandle, compact_torch_graph


class _UnsupportedReplica(ValueError):
    pass


def _state_bindings(source, target):
    source_modules = dict(source.named_modules(remove_duplicate=False))
    target_modules = dict(target.named_modules(remove_duplicate=False))
    if source_modules.keys() != target_modules.keys():
        raise _UnsupportedReplica("different module tree")
    owners = {}
    owner_sources = {}
    tensors = {}
    reverse = {}
    for path, module in source_modules.items():
        other = target_modules[path]
        if type(module) is not type(other) or module.training != other.training:
            raise _UnsupportedReplica("different module type or mode")
        if id(module) in owners and owners[id(module)] is not other:
            raise _UnsupportedReplica("untied module replica")
        if id(other) in owner_sources and owner_sources[id(other)] is not module:
            raise _UnsupportedReplica("newly tied module replica")
        owners[id(module)] = other
        owner_sources[id(other)] = module
        # Scalar attributes (dropout, branch selectors, etc.) are baked into
        # replay templates. A replica with changed configuration needs tracing.
        for name, value in vars(module).items():
            if name == "training" or name.startswith("_tl"):
                continue
            if isinstance(value, torch.Tensor):
                raise _UnsupportedReplica("unregistered tensor attribute")
            if type(value) in (str, int, float, bool, type(None)):
                if vars(other).get(name) != value:
                    raise _UnsupportedReplica("different scalar configuration")
    for getter in ("named_parameters", "named_buffers"):
        left = dict(getattr(source, getter)(remove_duplicate=False))
        right = dict(getattr(target, getter)(remove_duplicate=False))
        if left.keys() != right.keys():
            raise _UnsupportedReplica("different state schema")
        for name, value in left.items():
            other = right[name]
            if (type(value), value.shape, value.dtype, value.device, value.layout, value.requires_grad) != (
                type(other), other.shape, other.dtype, other.device, other.layout, other.requires_grad
            ):
                raise _UnsupportedReplica("different tensor schema or device")
            if id(value) in tensors and tensors[id(value)] is not other:
                raise _UnsupportedReplica("untied replica state")
            if id(other) in reverse and reverse[id(other)] is not value:
                raise _UnsupportedReplica("newly tied replica state")
            tensors[id(value)] = other
            reverse[id(other)] = value
    return tensors, owners


def rebind_torch_runtime(runtime: SplitRuntime, model) -> SplitRuntime | None:
    """Return a replica runtime without capture, or None for unsupported cases.

    Callers must supply a copy of the captured architecture. This deliberately
    uses the pinned compact Torch graph representation and requires the same device.
    New segment state bindings prevent replicas from sharing optimizer state,
    parameters, mutable buffers or previously resolved replay state.
    """
    if (
        not isinstance(model, torch.nn.Module)
        or not isinstance(runtime.model, torch.nn.Module)
        or runtime.trace_graph.backend != "torch"
        or not runtime.request.trainable
        or runtime.request.use_live_param_sources is False
    ):
        return None
    try:
        tensors, owners = _state_bindings(runtime.model, model)
        rebound = {}

        def tensor(value):
            return tensors.get(id(value), value)

        def buffer(handle):
            if not isinstance(handle, _BufferHandle):
                raise _UnsupportedReplica("noncompact buffer handle")
            key = id(handle)
            if key not in rebound:
                owner = None if handle._owner_ref is None else handle._owner_ref()
                if owner is None or id(owner) not in owners:
                    raise _UnsupportedReplica("unregistered buffer")
                new_owner = owners[id(owner)]
                value = new_owner._buffers[handle._name]
                rebound[key] = _BufferHandle(value, id(value), ref(new_owner), handle._name)
            return rebound[key]

        def parameter(handle):
            if not isinstance(handle, _ParamHandle) or id(handle.handle) not in tensors:
                raise _UnsupportedReplica("noncompact or unregistered parameter")
            key = id(handle)
            if key not in rebound:
                module = handle.module
                if module is not None:
                    module = replace(module, buffers={name: buffer(item) for name, item in module.buffers.items()})
                rebound[key] = replace(handle, handle=tensor(handle.handle), module=module)
            return rebound[key]

        def template(value):
            if isinstance(value, torch.Tensor):
                return tensor(value)
            if is_dataclass(value) and not isinstance(value, type):
                return replace(value, **{field.name: template(getattr(value, field.name)) for field in fields(value)})
            if isinstance(value, tuple):
                return tuple(template(item) for item in value)
            if isinstance(value, list):
                return [template(item) for item in value]
            if isinstance(value, dict):
                return {key: template(item) for key, item in value.items()}
            return value

        source_graph = compact_torch_graph(runtime.trace_graph) if runtime.retains_trace else runtime.trace_graph
        nodes = []
        for node in source_graph.nodes:
            if isinstance(node.target, MethodType) and isinstance(node.target.__self__, torch.nn.Module):
                raise _UnsupportedReplica("module-bound target")
            nodes.append(replace(
                node,
                param_refs=tuple(parameter(item) for item in node.param_refs),
                buffer_refs=tuple(buffer(item) for item in node.buffer_refs),
                args_template=template(node.args_template),
                kwargs_template=template(node.kwargs_template),
                op=replace(node.op, out=template(node.op.out)),
            ))
        graph = replace(source_graph, nodes=tuple(nodes))
        # Retain the original portable identity and capability proof: compaction
        # removes diagnostic provenance, while the executing nodes and schemas
        # are unchanged. Torch execution consults the rebound trace graph only.
        segments = runtime.adapter.build_segments(graph, runtime.plan, runtime.request)
        return SplitRuntime(
            model=model, trace=None, trace_graph=graph, request=runtime.request,
            plan=runtime.plan, adapter=runtime.adapter, segments=segments,
            capability_report=runtime.capability_report,
            prefix_program=runtime.prefix_program, suffix_program=runtime.suffix_program,
            graph_ir=runtime.graph_ir, model_profile=runtime.model_profile,
            prepared_input_kwargs=dict(runtime.prepared_input_kwargs), batch_spec=runtime.batch_spec,
        )
    except _UnsupportedReplica:
        return None


def clone_runtime_handle(base, model):
    """Copy facade bookkeeping while binding execution to independent state."""
    runtime = rebind_torch_runtime(base.runtime, model)
    if runtime is None:
        return None
    backend = copy.copy(base.backend)
    backend.model = model
    backend.runtime = runtime
    handle = backend.make_handle()
    handle.plan.metadata["_reused_capture"] = True
    return handle
