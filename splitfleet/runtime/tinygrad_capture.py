"""Private native Tinygrad capture with an actual B=2 capture and B=3 probe.

The original pinned public preparation, strict guards and native replay remain
in force. Rewrite memoization retains only immutable UOp shape metadata.
"""
from __future__ import annotations

import builtins
import ast
from collections import OrderedDict
from functools import wraps
import hashlib
import inspect
import time
import types
from dataclasses import replace


def _memoize_native_shape_rewrites(adapter, stats):
    """Memoize the pinned pure UOp shape rewrite, never runtime tensors.

    The original pure UOp-rewrite bodies execute on every cache miss.
    Recursive entries expire before the next node. Completed rewrites are
    reusable only for the same node, source UOps, shape program and complete
    symbol/input-shape binding. Four contexts bound retained metadata. The
    native batch/boundary guards and live parent insertion still run on every
    call; no tensor value, validation result or gradient is cached.
    """
    from torchlens.split.adapters.tinygrad import _TinygradGeneratedSegmentBase as Base
    from splitfleet.runtime.auxiliary_state import _clone

    for name, expected in (
        ('_rewrite_shape_uop_tree', '78d4e747db8d25ed72fb21b9c05fac9a60c22385bfc7d4430b6b7263ce765085'),
        ('_rewrite_parameter_expand_tree', '6b8dc13b291b12f75f26170a20426b9ee25d97eee0150de3af578542bfd5248f'),
        ('_rewrite_dynamic_uop_src', 'ecfa7807e88082617627d99a800f611b7e12032bf89e6d2c8e2f9d9f2f0414e6'),
    ):
        if hashlib.sha256(inspect.getsource(getattr(Base, name)).encode()).hexdigest() != expected:
            raise RuntimeError('Pinned native Tiny rewrite changed: ' + name)

    def memoized(native):
        @wraps(native)
        def rewrite(self, node, uop):
            stats['calls'] += 1
            cache = getattr(self, '_control_shape_memo', None)
            if cache is None:
                return native(self, node, uop)
            key = (native.__name__, id(uop))
            if key in cache:
                original, result = cache[key]
                assert original is uop
                stats['cache_hits'] += 1
                return result
            result = native(self, node, uop)
            cache[key] = (uop, result)
            stats['native_calls'] += 1
            return result
        return rewrite

    class Memo:
        _rewrite_shape_uop_tree = memoized(Base._rewrite_shape_uop_tree)
        _rewrite_parameter_expand_tree = memoized(Base._rewrite_parameter_expand_tree)

        def _rewrite_dynamic_uop_src(self, node, src, overlay):
            binding, program = self._shape_binding, self.graph.shape_program
            cache, key = None, None
            if binding is not None and program is not None:
                contexts = getattr(self, '_control_shape_contexts', None)
                if contexts is None:
                    contexts = self._control_shape_contexts = OrderedDict()
                context_key = (id(program), tuple(sorted(binding.symbols.items())),
                               tuple(sorted(binding.input_shapes.items())))
                context = contexts.get(context_key)
                if context is None:
                    context = contexts[context_key] = (program, {})
                    if len(contexts) > 4:
                        contexts.popitem(last=False)
                contexts.move_to_end(context_key)
                assert context[0] is program
                cache = context[1]
                key = (node.canonical_id, tuple(src))
                if key in cache:
                    stats['context_cache_hits'] += 1
                    # Replay inserts current live parents into its own list.
                    return list(cache[key])
                stats['context_cache_misses'] += 1
            previous = getattr(self, '_control_shape_memo', None)
            self._control_shape_memo = {}
            stats['scopes'] += 1
            try:
                result = Base._rewrite_dynamic_uop_src(self, node, src, overlay)
                if cache is not None:
                    cache[key] = tuple(result)
                return result
            finally:
                stats['max_entries'] = max(stats['max_entries'], len(self._control_shape_memo))
                self._control_shape_memo = previous

    native = adapter.build_segments.__func__
    bindings = {}
    for name in ('TinygradGeneratedPrefix', 'TinygradGeneratedSuffix'):
        bindings[name] = type('Memoized' + name, (Memo, native.__globals__[name]), {})
    adapter.build_segments = types.MethodType(_clone(native, **bindings), adapter)
    return adapter


def actual_batch_runtime_factory(*, stage_observer=None, memoize_shape_rewrites=False,
                                 probe_observer=None, initial_boundary_selector=None):
    from importlib.metadata import version
    from torchlens import user_funcs
    from torchlens.backends import get_backend_spec, resolve_backend_spec, default_specs
    from torchlens.backends.tinygrad.backend import TinygradBackend, _fallback_output_label, _uop_signature
    from torchlens.split import api, pipeline
    from torchlens.split.batching import BatchSpec, rebatch_inputs, resolve_batch_spec
    from splitfleet.runtime import auxiliary_state as auxiliary

    if version('torchlens') != '2.34.1' or version('tinygrad') != '0.13.0':
        raise RuntimeError('Actual-batch Tiny control requires pinned TorchLens2.34.1/tinygrad0.13.0')
    seams = (
        (api.prepare, 'd0940f8a4eb47ceff4541964610d7585f14828050a279063c732050591fabfd6'),
        (pipeline.capture_model, '080bf88043ab3b1e165a66b37829324103bd8d5e45be20a6e08316530933cde3'),
        (pipeline.normalize_to_split_ir, 'b55812419c706739568472894f5598c1683b72e9c7c9626af04fe40df1f9faf9'),
        (pipeline._probe_batch_replay, '9ce652f16f838a1869d9f51922bdef50422d1c60a56fb79d98acb6bbfbecc7ad'),
        (user_funcs.trace, 'e26cf0a54ee835880ec6bfbec5f3e1667e640a5962525f6a24e3ab5b1a474b27'),
        (default_specs._tinygrad_capture_trace, 'af0f982192bf5c40a43f5bf078b36b210304b7d83c5e746f8db1a2e324824fcc'),
        (pipeline._probe_shape_alignment, 'e0197c3181b8eb3593a42bc3495ac380d6d70e8cfefb2efc2aab52accc670174'),
        (pipeline._isolated_probe_model, 'acb73e952b07b16ba843e7579057593891445a0fb339ec48d685f9362e2719c5'),
        (pipeline._execute_batch_probe, '63ca99052a5accb68f2df7846c8a8170273bb7ef98f0ee67b69b89236aebdaf8'),
        (pipeline._snapshot_probe_output, 'bb2df4063d59877a4dae46d30f41b51d238dccaa5cf9605a6bde24b2a74e6e79'),
        (pipeline._probe_output_mismatch, '21bfb83a0f824bae1d983635c04b957d5df35516ca5263799158bc44a4534652'),
    )
    for function, expected in seams:
        if hashlib.sha256(inspect.getsource(function).encode()).hexdigest() != expected:
            raise RuntimeError('Pinned actual-batch native seam changed: ' + function.__qualname__)
    if tuple(inspect.signature(api.prepare).parameters) != ('model','inputs','request','input_kwargs'):
        raise RuntimeError('Pinned public native prepare signature changed')
    clone = auxiliary._clone
    original_spec = get_backend_spec('tinygrad')
    registry_ids = {name: id(get_backend_spec(name)) for name in ('tinygrad', 'torch', 'jax', 'tf', 'paddle')}
    sdk_ids = {function.__qualname__: id(function) for function in
               (pipeline.capture_model, pipeline.normalize_to_split_ir,
                pipeline.compile_shape_program, user_funcs.trace,
                default_specs._tinygrad_capture_trace)}
    compiler, bind_adapter = auxiliary._tinygrad_native_shape_bindings()
    native_import = builtins.__import__
    stage_sequence = 0
    memo_stats = dict(calls=0, cache_hits=0, native_calls=0, scopes=0, max_entries=0,
                     context_cache_hits=0, context_cache_misses=0)

    def observed(name, function):
        # Default preparation retains the exact previously proved callables.
        # No tracing/profiling hooks or Tensor reads are installed here.
        if stage_observer is None:
            return function

        @wraps(function)
        def invoke(*args, **kwargs):
            nonlocal stage_sequence
            stage_sequence += 1
            sequence = stage_sequence
            started = time.monotonic_ns()
            stage_observer({'stage': name, 'event': 'start', 'sequence': sequence})
            try:
                result = function(*args, **kwargs)
            except BaseException as error:
                stage_observer({'stage': name, 'event': 'failed', 'sequence': sequence,
                                'elapsed_seconds': (time.monotonic_ns() - started) / 1e9,
                                'exception_type': type(error).__name__})
                raise
            stage_observer({'stage': name, 'event': 'complete', 'sequence': sequence,
                            'elapsed_seconds': (time.monotonic_ns() - started) / 1e9})
            return result

        return invoke

    def capture_backend():
        # The public trace dispatcher still runs every original option/capability
        # check. Its private factory constructs one ordinary native backend.
        backend = TinygradBackend()
        auxiliary._retain_tinygrad_capture_observations(backend)
        auxiliary._batch_tinygrad_capture_payloads(backend)
        metadata, fallback_bindings = {}, {}

        def fallback(value, captures, used_labels):
            label = _fallback_output_label(value, captures, used_labels)
            fallback_bindings[id(value)] = label
            return label

        native_mark = types.MethodType(clone(backend._mark_output_events.__func__,
                                            _fallback_output_label=fallback), backend)

        def mark(trace, outputs, labels, captures=()):
            native_mark(trace, outputs, labels, captures)
            memo = {}
            by_signature = {_uop_signature(item.uop, memo): item.label_raw for item in captures}
            bindings = []
            for path, value in metadata['leaves']:
                label = labels.get(id(value.uop))
                if label is None:
                    label = by_signature.get(_uop_signature(value.uop, memo))
                if label is None:
                    label = fallback_bindings.get(id(value))
                bindings.append((label, path))
            auxiliary._amend_output_metadata(trace, bindings, metadata['specification'])

        backend._mark_output_events = observed('capture.output_metadata', mark)
        # Each backend is fresh and private. These existing bound functions
        # still execute once with the original arguments/results/exceptions.
        for attribute, label in (
            ('_emit_input_sources', 'capture.input_sources'),
            ('_emit_uop_graph', 'capture.native_payload_realization_and_emit'),
            ('_finish_trace', 'capture.native_finalize'),
        ):
            setattr(backend, attribute, observed(label, getattr(backend, attribute)))
        native_capture = backend.capture_trace

        def capture(*args, **options):
            observer = options.get('output_transform')

            def observe(value):
                specification, leaves = auxiliary._container_metadata(value, backend.is_tensor)
                metadata.update(specification=specification, leaves=leaves)
                return observed('capture.output_transform', observer)(value) if callable(observer) else None

            options['output_transform'] = observe
            return observed('capture.native_backend', native_capture)(*args, **options)

        backend.capture_trace = capture
        return backend

    def dispatch_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'tinygrad' and level == 1 and 'TinygradBackend' in fromlist:
            return types.SimpleNamespace(TinygradBackend=capture_backend)
        return native_import(name, globals, locals, fromlist, level)

    native_dispatch = clone(default_specs._tinygrad_capture_trace,
                            __builtins__=dict(vars(builtins), __import__=dispatch_import))
    private_spec = replace(original_spec, capture_trace=native_dispatch)

    def private_get(name):
        return private_spec if name == 'tinygrad' else get_backend_spec(name)

    def private_resolve(*args, **kwargs):
        resolved = resolve_backend_spec(*args, **kwargs)
        return private_spec if resolved is original_spec else resolved

    public_trace = clone(user_funcs.trace, get_backend_spec=private_get, resolve_backend_spec=private_resolve)

    def capture_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'user_funcs' and level == 2 and 'trace' in fromlist:
            return types.SimpleNamespace(trace=public_trace)
        return native_import(name, globals, locals, fromlist, level)

    capture_model = clone(pipeline.capture_model,
                          __builtins__=dict(vars(builtins), __import__=capture_import))
    capture_model = observed('capture.public_native_pipeline', capture_model)

    def graph_from_capture(capture):
        graph = pipeline.split_graph_from_trace(capture)
        paths = capture._splitfleet_auxiliary_output_paths
        nodes = tuple(replace(node, output_container_paths=paths[node.raw_label])
                      if node.is_output and node.raw_label in paths else node for node in graph.nodes)
        metadata = tuple((node.canonical_id, node.output_container_paths, node.output_container_spec)
                         for node in nodes if node.is_output)
        signature = hashlib.sha256(f'{graph.graph_shape_hash or ""}:{metadata!r}'.encode()).hexdigest()
        return replace(graph, nodes=nodes, graph_shape_hash=signature)

    def execute_probe(adapter, graph, *args, **kwargs):
        return auxiliary.restore_auxiliary_output(
            graph, observed('probe.native_replay', pipeline._execute_batch_probe)(adapter, graph, *args, **kwargs))

    graph_from_capture = observed('normalize.graph_from_capture', graph_from_capture)
    compiler = observed('normalize.compile_shape_program', compiler)

    def compare_probe(adapter, expected, actual, path='output'):
        result = pipeline._probe_output_mismatch(adapter, expected, actual, path)
        if probe_observer is not None:
            probe_observer(adapter, expected, actual, result)
        return result

    probe = clone(pipeline._probe_batch_replay, capture_model=capture_model,
                  split_graph_from_trace=graph_from_capture, _execute_batch_probe=execute_probe,
                  compile_shape_program=compiler,
                  _isolated_probe_model=observed('probe.copy_model', pipeline._isolated_probe_model),
                  rebatch_inputs=observed('probe.rebatch_inputs', pipeline.rebatch_inputs),
                  _snapshot_probe_output=observed('probe.native_output_snapshot', pipeline._snapshot_probe_output),
                  _probe_shape_alignment=observed('probe.native_shape_alignment', pipeline._probe_shape_alignment),
                  _probe_output_mismatch=observed('probe.native_host_output_compare', compare_probe))
    probe = observed('probe.complete_native_policy', probe)
    normalize = clone(pipeline.normalize_to_split_ir, split_graph_from_trace=graph_from_capture,
                      _probe_batch_replay=probe, compile_shape_program=compiler)
    normalize = observed('normalize.complete_native_pipeline', normalize)

    def prepare(model, inputs, request, *, probe_batch, capture=None, expected_batch=16,
                input_kwargs=None, phase_observer=None, capture_batch_size=None):
        if request.backend != 'tinygrad' or request.validation != 'strict':
            raise ValueError('Actual-batch control requires strict native Tiny preparation')
        inputs = inputs if isinstance(inputs, tuple) else (inputs,)
        canonical_batch = expected_batch if capture_batch_size is None else capture_batch_size
        if not isinstance(canonical_batch, int) or not 1 < canonical_batch <= expected_batch:
            raise ValueError('Actual native capture batch must be a declared non-singleton input subset')

        def actual_capture(model, positional, specification, *, input_kwargs=None, adapter):
            nonlocal capture
            resolved = resolve_batch_spec(positional, input_kwargs, adapter=adapter,
                                          explicit_axes=specification.features.batch_axes)
            actual_batch = resolved.user_batch_size
            if not resolved.axes or actual_batch != expected_batch or actual_batch <= 1:
                raise ValueError('Actual capture inputs must match the declared non-singleton batch')
            identity = (id(model), tuple((id(value), adapter.shape(value)) for value in positional),
                        tuple(sorted(resolved.axes.items())), actual_batch, canonical_batch)
            if capture is None:
                if canonical_batch == actual_batch:
                    canonical_inputs, canonical_kwargs = positional, dict(input_kwargs or {})
                else:
                    canonical_inputs, canonical_kwargs = rebatch_inputs(
                        positional, input_kwargs, axes=resolved.axes,
                        batch_size=canonical_batch, adapter=adapter)
                if len(canonical_inputs) != 1 or set(resolved.axes) != {'/args/0'}:
                    raise ValueError('The native model control requires one explicitly batched tensor')
                if adapter.shape(canonical_inputs[0])[resolved.axes['/args/0']] != canonical_batch:
                    raise RuntimeError('Public rebatching did not produce the declared actual capture batch')
                if phase_observer is not None:
                    phase_observer('actual_native_B%d_capture_start' % canonical_batch)
                capture = capture_model(model, canonical_inputs, specification, input_kwargs=canonical_kwargs)
                capture._splitfleet_actual_batch_control_identity = identity
                capture._splitfleet_actual_capture_inputs = canonical_inputs
                capture._splitfleet_actual_capture_kwargs = canonical_kwargs
                if phase_observer is not None:
                    phase_observer('actual_native_B%d_capture_complete' % canonical_batch)
            elif getattr(capture, '_splitfleet_actual_batch_control_identity', None) != identity:
                raise RuntimeError('Reused actual native capture has different model/input identities or batch')
            return capture, capture._splitfleet_actual_capture_inputs, capture._splitfleet_actual_capture_kwargs, BatchSpec(
                axes=resolved.axes, inference=resolved.inference,
                user_batch_size=actual_batch, canonical_batch_size=canonical_batch)

        def resolver(specification):
            adapter = bind_adapter(api.resolve_split_adapter(specification))
            if memoize_shape_rewrites:
                adapter = _memoize_native_shape_rewrites(adapter, memo_stats)
            if adapter.name != 'tinygrad':
                raise RuntimeError('Actual native control resolved a different backend')
            native_probes = pipeline.witness_probe_sizes
            adapter.batch_probe_sizes = lambda traced: (probe_batch,) if traced == canonical_batch else native_probes(traced)
            return adapter

        native_prepare = clone(api.prepare, capture_canonical_model=actual_capture,
                               normalize_to_split_ir=normalize, resolve_split_adapter=resolver,
                               plan_split=observed('prepare.native_plan', api.plan_split),
                               lower_split_program=observed('prepare.native_lower', api.lower_split_program),
                               analyze_split_capabilities=observed('prepare.native_capabilities', api.analyze_split_capabilities),
                               ensure_capability_report_supported=observed('prepare.strict_capability_guard', api.ensure_capability_report_supported),
                               execute_split_runtime=observed('prepare.native_segments', api.execute_split_runtime))
        if initial_boundary_selector is not None:
            def resolve_initial_boundary(captured, specification):
                from torchlens.split import after
                return replace(specification, point=after(initial_boundary_selector(graph_from_capture(captured))))

            # An automatic caller may choose a genuine model endpoint after
            # capture. All subsequent native normalization, probing, planning,
            # lowering and capability guards receive that same explicit point.
            tree = ast.parse(inspect.getsource(api.prepare))
            body = tree.body[0].body
            locations = [index for index, statement in enumerate(body)
                         if isinstance(statement, ast.Assign)
                         and isinstance(statement.value, ast.Call)
                         and isinstance(statement.value.func, ast.Name)
                         and statement.value.func.id == 'capture_canonical_model']
            if len(locations) != 1:
                raise RuntimeError('Pinned native capture-to-normalization seam changed')
            body.insert(locations[0] + 1, ast.parse('request = _resolve_initial_boundary(capture, request)').body[0])
            namespace = dict(native_prepare.__globals__, _resolve_initial_boundary=resolve_initial_boundary)
            exec(compile(ast.fix_missing_locations(tree), '<splitfleet-native-tiny-initial-point>', 'exec'), namespace)
            native_prepare = namespace[api.prepare.__name__]
        runtime = observed('prepare.complete_native_api', native_prepare)(model, inputs, request, input_kwargs=input_kwargs)
        if not all(id(get_backend_spec(name)) == value for name, value in registry_ids.items()):
            raise RuntimeError('Native control mutated the backend registry')
        actual_ids = {function.__qualname__: id(function) for function in
                      (pipeline.capture_model, pipeline.normalize_to_split_ir,
                       pipeline.compile_shape_program, user_funcs.trace,
                       default_specs._tinygrad_capture_trace)}
        if actual_ids != sdk_ids:
            raise RuntimeError('Native control mutated an SDK global function')
        runtime._splitfleet_public_control_sources = {
            function.__qualname__: expected for function, expected in seams}
        return runtime

    prepare.shape_rewrite_memo_stats = memo_stats
    return prepare
