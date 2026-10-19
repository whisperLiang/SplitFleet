"""Auxiliary state from the same native split forward used for differentiation.

Models opt in by exposing ``apply_statistics`` and enabling ``defer_statistics``
(an inherently functional model can omit the latter).  Capture and replay stay
pure; the caller commits stage-owned BatchNorm prefixes after the real batch.
All pinned-runtime extensions bind private function/instance copies, leaving
installed TorchLens and framework globals untouched.
"""

from __future__ import annotations

import builtins
import ast
import hashlib
import importlib
import inspect
import textwrap
import types
import weakref
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


def snapshot_auxiliary_state(statistics: Mapping[str, Any], *, backend: str):
    """Materialize lazy state outputs before the native optimizer mutates weights."""
    if backend != "tinygrad":
        return statistics
    from tinygrad import Tensor
    result = {name: {key: value.detach().clone() for key, value in row.items()}
              for name, row in statistics.items()}
    values = [value for row in result.values() for value in row.values()]
    if values:
        Tensor.realize(*values)
    return result


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


def _checked_tinygrad_capture_functions():
    """Refuse an SDK whose private native extension seams have changed."""
    from importlib.metadata import version
    from torchlens.backends.tinygrad.backend import TinygradBackend, _observe_tensor_ops
    from torchlens.backends.tinygrad._uop_graph import _is_materializable_uop

    if version("torchlens") != "2.34.1" or version("tinygrad") != "0.13.0":
        raise RuntimeError("Tiny auxiliary capture requires pinned TorchLens 2.34.1/tinygrad 0.13.0")
    capture_parameters = (
        "self", "model", "input_args", "input_kwargs", "layers_to_save", "keep_orphans",
        "output_device", "activation_transform", "save_raw_activations", "detach_saved_activations",
        "save_grads", "random_seed", "num_context_lines", "save_arg_values", "save_code_context",
        "save_rng_states", "recurrence_detection", "compute_input_output_distances", "verbose",
        "backward_ready", "name", "module_filter", "transform", "raw_input", "save_raw_input",
        "batch_render", "output_transform", "save_raw_output", "layer_visualizers",
        "save_visualizations", "lookback", "lookback_payload_policy", "module_identity_mode",
        "grad_options", "kwargs",
    )
    seams = (
        (TinygradBackend.capture_trace, "6b44840bebd429708f12842ab1270b303203dbffb48429651a680cb8ab69ec34", capture_parameters),
        (_observe_tensor_ops.__init__, "4b7d2301511de30d862ae69ec6a7938909dc69790d4dfdd7113e4774ea61a4e4",
         ("self", "observed_ops", "observed_module_stacks", "observed_tensors")),
        (_observe_tensor_ops.__enter__, "ffd77a5c977b111adfb4fd426e70e3a3f6024cabfc3adcf311ad55de95db9bee", ("self",)),
        (_observe_tensor_ops.__exit__, "b2e78c259ea53a7474d58823ee3880ee38cb6cdeda0fc3a81ec17474216cfc5f",
         ("self", "exc_type", "exc", "tb")),
        (_is_materializable_uop, "a85b2076d0825e3aff5aaefef0eaf4b174d1e73b7dc59999e92b7d1d1f00c425", ("uop",)),
    )
    for function, source_hash, parameters in seams:
        if (tuple(inspect.signature(function).parameters) != parameters
                or hashlib.sha256(inspect.getsource(function).encode()).hexdigest() != source_hash):
            raise RuntimeError(f"Pinned Tiny native capture seam changed: {function.__qualname__}")
    return TinygradBackend.capture_trace, _observe_tensor_ops, _is_materializable_uop


def _retain_tinygrad_capture_observations(backend):
    """Use the native retention argument so observed UOp ids stay live."""
    from tinygrad import Tensor

    native_capture, native_observer, _ = _checked_tinygrad_capture_functions()

    def capture(*args, **kwargs):
        retention = {}
        scopes = []

        class RetainedObservation(native_observer):
            def __init__(self, *values, **options):
                if options.get("observed_tensors") is not None:
                    raise RuntimeError("Native forward capture unexpectedly supplied a retention map")
                super().__init__(*values, **dict(options, observed_tensors=retention))

            def __exit__(self, *exception):
                try:
                    if set(retention) != set(self.observed_ops):
                        raise RuntimeError("Native observed names and retained Tensor keys differ")
                    links = []
                    for key, values in retention.items():
                        names = self.observed_ops[key]
                        if len(values) != len(names):
                            raise RuntimeError("Native observed names and retained Tensor counts differ")
                        representative = values[0].uop
                        for value in values:
                            if (not isinstance(value, Tensor) or id(value.uop) != key
                                    or value.uop is not representative):
                                raise RuntimeError("Native observed Tensor has a recycled UOp identity")
                            links.append((key, id(value), id(value.uop)))
                    scopes.append({"uop_keys": len(retention), "returned_tensors": len(links),
                                   "identity_links": tuple(links)})
                finally:
                    super().__exit__(*exception)

        actual_capture = _clone(native_capture, _observe_tensor_ops=RetainedObservation)
        try:
            trace = actual_capture(backend, *args, **kwargs)
        finally:
            retention.clear()
        trace._splitfleet_tinygrad_observation_retention = {
            "scopes": tuple(scopes), "released": True, "remaining_keys": len(retention),
        }
        return trace

    backend.capture_trace = capture


def _flat_tinygrad_materializability_probe():
    """Change only the ignored host-conversion shape in the pinned guard."""
    _, _, native_guard = _checked_tinygrad_capture_functions()
    tree = ast.parse(textwrap.dedent(inspect.getsource(native_guard)))
    replacements = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "tolist" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tensor" and not node.args and not node.keywords):
            node.func.value = ast.Call(
                func=ast.Attribute(value=ast.Name(id="tensor", ctx=ast.Load()),
                                   attr="flatten", ctx=ast.Load()), args=[], keywords=[],
            )
            replacements += 1
    if replacements != 1:
        raise RuntimeError("Pinned Tiny materializability conversion differs from its proved source")
    namespace = dict(native_guard.__globals__)
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    return namespace[native_guard.__name__]


def _batch_tinygrad_capture_payloads(backend):
    """Batch genuine native payload copies on one private capture backend.

    The pinned emitter checks each UOp with ``Tensor.tolist`` and then realizes
    a second copy from its entire ancestor graph.  Realize the detached copies
    together, retaining the original emitter, every actual converted value,
    all raw UOps, and native topology/hash logic. Only the discarded boolean
    probe's nested host list is flattened; emitted shapes stay original.
    """
    from tinygrad import Device, Tensor
    from torchlens.backends.tinygrad.backend import TinygradBackend
    from torchlens.backends.tinygrad._uop_graph import _unique_uops

    native_guard = _flat_tinygrad_materializability_probe()
    native_copy = backend._realized_copy
    payloads = {}

    def realized_copy(value):
        payload = payloads.get(id(value.uop))
        if payload is None:
            return native_copy(value)
        if (payload.shape, payload.dtype, payload.device) != (value.shape, value.dtype, value.device):
            raise RuntimeError("Native batched capture payload differs in shape, dtype or device")
        return payload

    def emit(trace, outputs, labels, observed_ops, observed_modules, module_tree,
             live_tensors_by_uop=None, live_buffer_tensors_by_uop=None):
        uops = _unique_uops(outputs)
        retained = {id(value): value for value in (
            *outputs, *(live_tensors_by_uop or {}).values(),
            *(live_buffer_tensors_by_uop or {}).values(),
        )}
        original = [(value, value.uop) for value in retained.values()]
        candidates, by_raw = [], {}
        for uop in uops:
            if id(uop) in labels:
                continue
            try:
                value = Tensor(uop)
                tuple(value.shape)
                if getattr(value.dtype.base, "fmt", None) is None:
                    continue
            except Exception:
                continue
            candidates.append((uop, value.uop, value.detach().clone()))
        try:
            for start in range(0, len(candidates), 128):
                chunk = candidates[start:start + 128]
                Tensor.realize(*(payload for _, _, payload in chunk))
                # Realization can queue native work. Complete the copies before
                # another chunk rebinds live tensors or their UOps are restored.
                devices = set()
                for _, _, payload in chunk:
                    devices.update((payload.device,) if isinstance(payload.device, str) else payload.device)
                for device in devices:
                    Device[device].synchronize()
                for raw, converted, payload in chunk:
                    by_raw[id(raw)] = payload
                    payloads[id(converted)] = payload
            # Native realization updates live Tensor handles. The capture
            # describes the actual lazy forward's original immutable UOps.
            for value, uop in original:
                value.uop = uop

            def tensor_lookup(uop):
                return by_raw[id(uop)] if id(uop) in by_raw else Tensor(uop)

            def guard_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "tinygrad" and "Tensor" in fromlist:
                    return types.SimpleNamespace(Tensor=tensor_lookup)
                return builtins.__import__(name, globals, locals, fromlist, level)

            guard = _clone(native_guard,
                           __builtins__=dict(vars(builtins), __import__=guard_import))
            native_emit = _clone(TinygradBackend._emit_uop_graph,
                                 _unique_uops=lambda _: uops, _is_materializable_uop=guard)
            return native_emit(backend, trace, outputs, labels, observed_ops, observed_modules,
                               module_tree, live_tensors_by_uop, live_buffer_tensors_by_uop)
        finally:
            for value, uop in original:
                value.uop = uop
            payloads.clear()

    backend._realized_copy = realized_copy
    backend._emit_uop_graph = emit


def _capture_auxiliary_model(model, inputs, request, *, input_kwargs=None,
                             output_observer=None, random_seed=None, shape_witness=False):
    """Attach final-container metadata to an independent native capture backend."""
    name = str(request.backend)
    options = dict(input_kwargs=input_kwargs, layers_to_save="all", keep_orphans=True,
                   output_transform=output_observer)
    if random_seed is not None:
        options["random_seed"] = random_seed
    if name == "paddle":
        from torchlens.backends.paddle.backend import PaddleBackend

        backend = PaddleBackend()
        mark_outputs = backend._mark_outputs

        def mark(trace, output):
            specification, leaves = _container_metadata(output, backend.is_tensor)
            bindings = [(backend.tensor_store.get_label(value), path) for path, value in leaves]
            mark_outputs(trace, output)
            _amend_output_metadata(trace, bindings, specification)

        backend._mark_outputs = mark
        return backend.capture_trace(model, inputs, **options)
    if name != "tinygrad":
        raise ValueError("Auxiliary container capture is only needed for Paddle and tinygrad")
    from torchlens.backends.tinygrad.backend import TinygradBackend, _uop_signature, _fallback_output_label
    from torchlens.split._tinygrad_state import tinygrad_capture_state

    backend = TinygradBackend()
    _retain_tinygrad_capture_observations(backend)
    _batch_tinygrad_capture_payloads(backend)
    metadata = {}

    def observe(output):
        specification, leaves = _container_metadata(output, backend.is_tensor)
        metadata.update(specification=specification, leaves=leaves)
        if output_observer is not None:
            output_observer(output)

    fallback_bindings = {}

    def native_fallback(output, captures, used_labels):
        # Observe the label assigned by the native marker to this very Tensor.
        # This records its existing binding, rather than inferring a leaf index.
        label = _fallback_output_label(output, captures, used_labels)
        fallback_bindings[id(output)] = label
        return label

    mark_outputs = types.MethodType(_clone(backend._mark_output_events.__func__,
                                           _fallback_output_label=native_fallback), backend)

    def mark(trace, outputs, uop_labels, captures=()):
        mark_outputs(trace, outputs, uop_labels, captures)
        memo = {}
        by_signature = {_uop_signature(capture.uop, memo): capture.label_raw for capture in captures}
        bindings = []
        for path, value in metadata["leaves"]:
            label = uop_labels.get(id(value.uop))
            if label is None:
                label = by_signature.get(_uop_signature(value.uop, memo))
            if label is None:
                label = fallback_bindings.get(id(value))
            bindings.append((label, path))
        _amend_output_metadata(trace, bindings, metadata["specification"])

    backend._mark_output_events = mark
    options["output_transform"] = observe
    with tinygrad_capture_state(model):
        return backend.capture_trace(model, inputs, **options)


def _tinygrad_native_shape_bindings():
    """Bind the proved pinned native batch semantics to a private runtime.

    A transitive parameter reference is not proof of fixed-state provenance.
    The native shape compiler must preserve actual public-input ancestry, and
    native replay must keep audited constant-only lazy broadcasts at ALU ops.
    No native shape rule, unresolved guard, graph or backward is replaced.
    """
    from importlib.metadata import version
    from torchlens.split import shape_program
    from torchlens.split.adapters import tinygrad as adapter_module
    replay_base = adapter_module._TinygradGeneratedSegmentBase

    if version("torchlens") != "2.34.1" or version("tinygrad") != "0.13.0":
        raise RuntimeError("Tiny native shapes require pinned TorchLens 2.34.1/tinygrad 0.13.0")
    native_execute = adapter_module._TinygradGeneratedSegmentBase._execute_uop
    seams = (
        (shape_program._is_parameter_lineage, "982f5be098dc7e5a7d590d2f9c3d5c669864c74ff8ae942b9952da5bac2c3f45",
         ("graph", "node", "seen")),
        (shape_program._propagate_shapes, "ee4789b14278faf393581ef68bc3e747c0b956ab0e636877998002d4cf3d8dcc",
         ("graph", "input_axes", "batch_symbol", "traced_batch_size", "shape_witnesses")),
        (shape_program._unresolved_dynamic_nodes, "4ab5096fd0ca52796014a479cd0970f370756d198b1d53717c476e6490c19f9e",
         ("graph", "value_shapes", "batch_symbol", "traced_batch_size", "shape_witnesses")),
        (shape_program.compile_shape_program, "9b20458e0287f42927c64a512474a29c4f78b8b9221aee9a4bb970992c347065",
         ("graph", "inputs", "input_kwargs", "request", "adapter", "batch_axes", "inference_mode", "shape_witnesses")),
        (native_execute, "5a9625cfc6583ba2487aba55ec45d929ea053b00bdefe5e396e81a4bf221d46c",
         ("self", "node", "overlay", "preserve_autograd", "target_device")),
        (replay_base._execute_nodes, "d3f92e3c1069de61f4ff76a12739ae132343cf49dc34d6a1153f8864cdcf5ed2",
         ("self", "overlay", "preserve_autograd", "target_device")),
        (adapter_module.TinygradSplitAdapter.build_segments, "bffd56674bdbc7d85fcb9c76941525ce3866655971c03d5a44fb5d67303ad4eb",
         ("self", "graph", "plan", "spec")),
        (replay_base._is_parameter_lineage, "0ff87c2598a22de7d69985b21d940b159fd2892dab4161e35bfd10544a777d4e",
         ("self", "node", "seen")),
    )
    for function, digest, parameters in seams:
        if (hashlib.sha256(inspect.getsource(function).encode()).hexdigest() != digest
                or tuple(inspect.signature(function).parameters) != parameters):
            raise RuntimeError(f"Pinned Tiny native shape seam changed: {function.__qualname__}")

    ancestry_caches = {}

    def input_ancestry(graph, node):
        key = id(graph)
        entry = ancestry_caches.get(key)
        if entry is None or entry[0]() is not graph:
            entry = ancestry_caches[key] = (
                weakref.ref(graph, lambda _reference, identity=key: ancestry_caches.pop(identity, None)), {},
            )
        cache = entry[1]

        def visit(current):
            identity = current.canonical_id
            if identity in cache:
                return cache[identity]
            if current.is_input:
                cache[identity] = True
                return True
            parents = [graph.node_for_label(label) for label in current.parents]
            if any(parent is None for parent in parents):
                raise RuntimeError(f"Unresolved native Tiny graph parent at {identity}")
            cache[identity] = any(visit(parent) for parent in parents)
            return cache[identity]
        return visit(node)

    lineage_tree = ast.parse(textwrap.dedent(inspect.getsource(shape_program._is_parameter_lineage)))
    lineage_tree.body[0].body.insert(1, ast.parse(
        "if _input_ancestry(graph, node):\n    return False").body[0])
    namespace = dict(shape_program._is_parameter_lineage.__globals__, _input_ancestry=input_ancestry)
    exec(compile(ast.fix_missing_locations(lineage_tree), __file__, "exec"), namespace)
    lineage = namespace[shape_program._is_parameter_lineage.__name__]
    propagate = _clone(shape_program._propagate_shapes, _is_parameter_lineage=lineage)
    unresolved = _clone(shape_program._unresolved_dynamic_nodes, _is_parameter_lineage=lineage)
    compiler = _clone(shape_program.compile_shape_program,
                      _propagate_shapes=propagate, _unresolved_dynamic_nodes=unresolved)

    execute_tree = ast.parse(textwrap.dedent(inspect.getsource(native_execute)))
    replacements = 0
    lazy_intermediates = 0
    device_memos = 0
    for node in ast.walk(execute_tree):
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == 'device_rewrites' and isinstance(node.value, ast.Dict)
                and not node.value.keys):
            node.value = ast.parse('self._device_rewrite_memo(target_device)', mode='eval').body
            device_memos += 1
        if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp):
            for index, value in enumerate(tuple(node.test.values)):
                if ast.unparse(value) == "capture.uop.op not in GroupOp.ALU":
                    del node.test.values[index]
                    replacements += 1
        if (isinstance(node, ast.Call)
                and ast.unparse(node.func) == 'self._backend._realized_copy'
                and len(node.args) == 1 and isinstance(node.args[0], ast.Name)
                and node.args[0].id == 'value'):
            # Preserve Tinygrad's native lazy computation between read-only
            # nodes. Eager copies of every expanded convolution operand retain
            # tens of GiB for one batch. Boundary copies, source-state copies
            # and the training/autograd branch retain their original behavior.
            node.func = ast.Attribute(value=node.args[0], attr='detach', ctx=ast.Load())
            node.args = []
            lazy_intermediates += 1
    if replacements != 1:
        raise RuntimeError("Pinned Tiny constant-only replay predicate differs from proved source")
    if lazy_intermediates != 1:
        raise RuntimeError('Pinned Tiny intermediate materialization seam changed')
    if device_memos != 1:
        raise RuntimeError('Pinned Tiny device rewrite memo seam changed')
    execute_namespace = dict(native_execute.__globals__)
    exec(compile(ast.fix_missing_locations(execute_tree), __file__, "exec"), execute_namespace)
    execute = execute_namespace[native_execute.__name__]

    class InputLineage:
        def _execute_nodes(self, overlay, *, preserve_autograd=False, target_device=None):
            # The native device transformer is pure metadata. Share its own
            # memo across this DAG traversal instead of revisiting every
            # ancestor for every node. It expires before the next execution;
            # live tensors are still inserted separately by native replay.
            target = target_device or self._overlay_device(overlay)
            previous = getattr(self, '_device_rewrite_scope', None)
            self._device_rewrite_scope = (target, {})
            try:
                return replay_base._execute_nodes(self, overlay,
                    preserve_autograd=preserve_autograd, target_device=target)
            finally:
                self._device_rewrite_scope = previous

        def _device_rewrite_memo(self, target):
            scope = getattr(self, '_device_rewrite_scope', None)
            return scope[1] if scope is not None and scope[0] == target else {}

        def _is_parameter_lineage(self, node, seen=None):
            # The replay shape rewriter needs the same provenance as the
            # shape compiler: a convolution's parameter references do not
            # make its input-dependent activation a fixed parameter branch.
            if input_ancestry(self.graph, node):
                return False
            return replay_base._is_parameter_lineage(self, node, seen)

    class Prefix(InputLineage, adapter_module.TinygradGeneratedPrefix):
        _execute_uop = execute

    class Suffix(InputLineage, adapter_module.TinygradGeneratedSuffix):
        _execute_uop = execute

    def bind_adapter(adapter):
        adapter.build_segments = types.MethodType(_clone(adapter.build_segments.__func__,
            TinygradGeneratedPrefix=Prefix, TinygradGeneratedSuffix=Suffix), adapter)
        return adapter

    return compiler, bind_adapter


def prepare_auxiliary_runtime(model, inputs, request, *, input_kwargs=None, initial_boundary_selector=None):
    """Preserve formal output paths while retaining native planning and probes."""
    from splitfleet.autosplit.torchlens_runtime import require_torchlens_version

    require_torchlens_version()
    if not supports_auxiliary_state(model):
        raise TypeError("Auxiliary preparation requires an explicitly opted-in model")
    api = importlib.import_module("torchlens.split.api")
    pipeline = importlib.import_module("torchlens.split.pipeline")
    if request.backend == 'tinygrad':
        from torchlens.split.batching import resolve_batch_spec
        from splitfleet.runtime.tinygrad_capture import actual_batch_runtime_factory

        positional = inputs if isinstance(inputs, tuple) else (inputs,)
        batch = resolve_batch_spec(positional, input_kwargs,
            adapter=api.resolve_split_adapter('tinygrad'), explicit_axes=request.features.batch_axes)
        if batch.axes and batch.user_batch_size is not None and batch.user_batch_size > 1:
            prepare = actual_batch_runtime_factory(memoize_shape_rewrites=True,
                                                   initial_boundary_selector=initial_boundary_selector)
            return prepare(model, positional, request, input_kwargs=input_kwargs,
                expected_batch=batch.user_batch_size, capture_batch_size=2, probe_batch=3)
    compiler_bindings = {}
    def bind_tiny_adapter(adapter):
        return adapter
    if request.backend == "tinygrad":
        compiler, bind_tiny_adapter = _tinygrad_native_shape_bindings()
        compiler_bindings["compile_shape_program"] = compiler

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

    def execute_probe(adapter, graph, *args, **kwargs):
        result = pipeline._execute_batch_probe(adapter, graph, *args, **kwargs)
        return restore_auxiliary_output(graph, result)

    canonical = _clone(pipeline.capture_canonical_model, capture_model=_capture_auxiliary_model)
    probe = _clone(pipeline._probe_batch_replay, capture_model=_capture_auxiliary_model,
                   split_graph_from_trace=graph_from_capture, _execute_batch_probe=execute_probe,
                   **compiler_bindings)
    normalize = _clone(pipeline.normalize_to_split_ir, split_graph_from_trace=graph_from_capture,
                       _probe_batch_replay=probe, **compiler_bindings)

    def private_adapter(specification):
        # The pinned resolver constructs a fresh native adapter. Its supported
        # probe hook selects a real independent batch after a genuine B=1
        # capture failure; all native witness/replay/numeric checks still run.
        adapter = api.resolve_split_adapter(specification)
        if adapter.name == "tinygrad":
            adapter = bind_tiny_adapter(adapter)
            native_probes = getattr(adapter, "batch_probe_sizes", pipeline.witness_probe_sizes)
            adapter.batch_probe_sizes = lambda batch: (3,) if batch == 2 else native_probes(batch)
        return adapter

    prepare = _clone(api.prepare, capture_canonical_model=canonical, normalize_to_split_ir=normalize,
                     resolve_split_adapter=private_adapter)
    return prepare(model, inputs, request, input_kwargs=input_kwargs)


def train_jax_suffix_with_auxiliary(runtime, boundary, targets, *, loss_fn, optimizer=None):
    """Run the pinned native JAX trainer with a private has_aux transform binding."""
    import jax
    from splitfleet.autosplit.torchlens_runtime import require_torchlens_version

    require_torchlens_version()
    training = importlib.import_module("torchlens.split.training")
    captured = []

    def auxiliary_loss(output, target):
        restored = restore_auxiliary_output(runtime.trace_graph, output)
        return loss_fn(restored, target), restored["batch_statistics"]

    class JaxWithAuxiliary:
        def __getattr__(self, name):
            return getattr(jax, name)

        @staticmethod
        def value_and_grad(function, *args, **kwargs):
            if kwargs.get("has_aux"):
                raise RuntimeError("Pinned native JAX trainer unexpectedly already uses has_aux")
            differentiated = jax.value_and_grad(function, *args, **dict(kwargs, has_aux=True))

            def evaluate(*values, **options):
                (loss, auxiliary), gradients = differentiated(*values, **options)
                captured.append(auxiliary)
                return loss, gradients

            return evaluate

    native_import = builtins.__import__
    facade = JaxWithAuxiliary()

    def scoped_import(name, *args, **kwargs):
        return facade if name == "jax" else native_import(name, *args, **kwargs)

    native = _clone(training._train_suffix_jax,
                    __builtins__=dict(vars(builtins), __import__=scoped_import))
    loss, gradients, _ = native(runtime, boundary, targets, loss_fn=auxiliary_loss, optimizer=optimizer)
    if captured:
        if len(captured) != 1:
            raise RuntimeError("Native JAX suffix differentiation evaluated multiple forwards")
        auxiliary = captured[0]
    else:
        # The native trainer directly calls suffix_loss when it has no
        # differentiable roots. That result still comes from its sole replay.
        loss, auxiliary = loss
    return (loss, gradients), auxiliary
