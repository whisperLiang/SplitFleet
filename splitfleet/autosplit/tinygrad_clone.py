"""Rebind native Tinygrad graph recipes to independent opted-in model state."""

from __future__ import annotations

import copy
from dataclasses import replace

from .replica_metadata import Operation, StateReference


def clone_tinygrad_runtime_handle(base, model):
    from torchlens.backends.tinygrad.backend import TinygradUOpCapture
    from torchlens.split import SplitRuntime

    runtime, source = base.runtime, base.runtime.model
    contract = getattr(source, '_splitfleet_replica_contract', None)
    if (runtime.trace_graph.backend != 'tinygrad' or type(source) is not type(model)
            or not callable(contract) or not callable(getattr(model, '_splitfleet_replica_contract', None))
            or contract() != model._splitfleet_replica_contract()):
        return None
    original, replica = source.state_tensors(), model.state_tensors()
    if original.keys() != replica.keys():
        return None
    bindings, reverse = {}, {}
    for name, value in original.items():
        other = replica[name]
        if (value.shape, value.dtype, value.device, getattr(value, 'is_param', None),
                getattr(value, 'requires_grad', None)) != (
                other.shape, other.dtype, other.device, getattr(other, 'is_param', None),
                getattr(other, 'requires_grad', None)):
            return None
        if ((id(value) in bindings and bindings[id(value)] is not other)
                or (id(other) in reverse and reverse[id(other)] is not value)):
            return None
        bindings[id(value)], reverse[id(other)] = other, value

    references = {}

    def parameter(value):
        if id(value) not in references:
            handle = getattr(value, '_param_ref', None)
            if handle is None:
                handle = getattr(value, 'handle', None)
            if id(handle) not in bindings:
                raise ValueError('Unregistered Tinygrad state reference')
            references[id(value)] = StateReference(value, bindings[id(handle)])
        return references[id(value)]

    nodes = []
    try:
        for node in runtime.trace_graph.nodes:
            capture = node.target
            if isinstance(capture, TinygradUOpCapture):
                # Raw UOps and captured constants remain immutable. Native
                # BUFFER/view replay resolves these fresh live state handles.
                capture = replace(capture, live_tensor=bindings.get(id(capture.live_tensor), capture.live_tensor))
            elif capture is not None:
                return None
            output = bindings.get(id(node.op.out), node.op.out)
            nodes.append(replace(node, target=capture,
                param_refs=tuple(parameter(value) for value in node.param_refs),
                op=Operation(node.op, output)))
    except ValueError:
        return None
    graph = replace(runtime.trace_graph, nodes=tuple(nodes))
    segments = runtime.adapter.build_segments(graph, runtime.plan, runtime.request)
    rebound = SplitRuntime(model=model, trace=None, trace_graph=graph,
        request=runtime.request, plan=runtime.plan, adapter=runtime.adapter, segments=segments,
        capability_report=runtime.capability_report, prefix_program=runtime.prefix_program,
        suffix_program=runtime.suffix_program, graph_ir=runtime.graph_ir,
        model_profile=runtime.model_profile, prepared_input_kwargs=dict(runtime.prepared_input_kwargs),
        batch_spec=runtime.batch_spec)
    backend = copy.copy(base.backend)
    backend.model, backend.runtime = model, rebound
    handle = backend.make_handle()
    handle.plan.metadata['_reused_capture'] = True
    return handle
