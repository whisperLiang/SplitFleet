"""Bind an opted-in TensorFlow replica to immutable native capture recipes."""

from __future__ import annotations

import copy
from dataclasses import replace
from threading import RLock

from .replica_metadata import Operation, StateReference

_FUNCTION_POOL_LOCK = RLock()


def clone_tensorflow_runtime_handle(base, model):
    """Reuse capture only for an explicitly declared, identical model contract.

    Mutable variables, resource handles, segment bindings and runtime caches
    belong to the new model. Unknown resource tensors reject reuse.
    """
    import tensorflow as tf
    from torchlens.backends.tf.op_callback_capture import TFOpCapture
    from torchlens.split import SplitRuntime

    runtime, source = base.runtime, base.runtime.model
    contract = getattr(source, '_splitfleet_replica_contract', None)
    if (runtime.trace_graph.backend != 'tf' or type(model) is not type(source)
            or not callable(contract) or not callable(getattr(model, '_splitfleet_replica_contract', None))
            or contract() != model._splitfleet_replica_contract()):
        return None
    left, right = source.canonical_variables(), model.canonical_variables()
    if left.keys() != right.keys():
        return None
    bindings, reverse = {}, {}

    def bind(value, other):
        if id(value) in bindings and bindings[id(value)] is not other:
            raise ValueError('Untied TensorFlow replica state')
        if id(other) in reverse and reverse[id(other)] is not value:
            raise ValueError('Newly tied TensorFlow replica state')
        bindings[id(value)], reverse[id(other)] = other, value

    for name, value in left.items():
        other = right[name]
        original = getattr(value, 'value', value)
        target = getattr(other, 'value', other)
        if callable(original):
            original, target = value, other
        if (value.shape, value.dtype, value.trainable, original.device) != (
                other.shape, other.dtype, other.trainable, target.device):
            return None
        bind(value, other)
        bind(original, target)
        bind(original.handle, target.handle)

    def tensor(value):
        result = bindings.get(id(value))
        if result is not None:
            return result
        if isinstance(value, tf.Variable) or (tf.is_tensor(value) and value.dtype == tf.resource):
            raise ValueError('Unregistered TensorFlow resource in native capture')
        return value

    references = {}

    def parameter(value):
        if id(value) not in references:
            original = getattr(value, '_param_ref', None)
            if original is None:
                original = getattr(value, 'handle', None)
            if id(original) not in bindings:
                raise ValueError('Unregistered TensorFlow parameter reference')
            references[id(value)] = StateReference(value, bindings[id(original)])
        return references[id(value)]

    nodes = []
    try:
        for node in runtime.trace_graph.nodes:
            capture = node.target
            if isinstance(capture, TFOpCapture):
                capture = replace(capture,
                    inputs=tuple(replace(item, tensor=tensor(item.tensor)) for item in capture.inputs),
                    output_tensor=tensor(capture.output_tensor))
            elif capture is not None:
                return None
            nodes.append(replace(node, target=capture,
                param_refs=tuple(parameter(value) for value in node.param_refs),
                op=Operation(node.op, tensor(node.op.out))))
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
    backend._bind_native_tensorflow_execution()
    if getattr(model, '_splitfleet_tensorflow_function_forward', False):
        from splitfleet.runtime.tensorflow_forward import TensorFlowSuffixFunctionPool
        with _FUNCTION_POOL_LOCK:
            pool = getattr(runtime, '_splitfleet_shared_suffix_functions', None)
            if pool is None:
                pool = runtime._splitfleet_shared_suffix_functions = TensorFlowSuffixFunctionPool(runtime)
        rebound._splitfleet_shared_suffix_functions = pool
    handle = backend.make_handle()
    handle.plan.metadata['_reused_capture'] = True
    return handle
