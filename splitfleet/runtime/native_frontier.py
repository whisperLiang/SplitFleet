"""Faithful output frontiers through private TorchLens 2.34.1 bindings.

The pinned planner redundantly carries the parents of prefix-owned final
outputs. Those outputs are already present in the boundary. This extension
retains every suffix operand and final output, and leaves capture, capability,
shape, ownership and wire validation to the unchanged native implementations.
"""
from __future__ import annotations
import __future__

import ast
import importlib
import inspect
from hashlib import sha256
from importlib.metadata import version
from types import FunctionType, MethodType

NATIVE_VERSION = "2.34.1"
# Each privately rebound function must retain the exact approved native body.
# Only _frontier_node_ids gets the one documented conditional AST change.
_PINNED_FUNCTIONS = {
    ("planner", "_frontier_node_ids"): (
        "0a98199b4ed27528fc498f255cb3e6f7c3253932b011bf65d0ffcdcaa7f4ea0d",
        "(graph: 'SplitTraceGraph', *, prefix_node_ids: 'frozenset[str]', suffix_node_ids: 'frozenset[str]') -> 'tuple[str, ...]'"),
    ("planner", "plan_split"): (
        "0f023f49ffdeb0b5b9f17d551b3b28a36053f55519102287bd8c00477573a18e",
        "(graph: 'SplitTraceGraph', spec: 'SplitRequest') -> 'SplitPlan'"),
    ("pipeline", "_execute_batch_probe"): (
        "63ca99052a5accb68f2df7846c8a8170273bb7ef98f0ee67b69b89236aebdaf8",
        "(adapter: 'Any', graph: 'SplitTraceGraph', request: 'SplitRequest', inputs: 'tuple[Any, ...]', input_kwargs: 'dict[str, Any]') -> 'Any'"),
    ("pipeline", "_probe_batch_replay"): (
        "9ce652f16f838a1869d9f51922bdef50422d1c60a56fb79d98acb6bbfbecc7ad",
        "(model: 'Any', inputs: 'tuple[Any, ...]', input_kwargs: 'dict[str, Any] | None', request: 'SplitRequest', graph: 'SplitTraceGraph', shape_program: 'ShapeProgram', adapter: 'Any', *, random_seed: 'int | None' = None) -> 'ShapeProgram'"),
    ("pipeline", "normalize_to_split_ir"): (
        "b55812419c706739568472894f5598c1683b72e9c7c9626af04fe40df1f9faf9",
        "(capture: 'Any', spec: 'SplitRequest', *, inputs: 'tuple[Any, ...]' = (), input_kwargs: 'dict[str, Any] | None' = None, adapter: 'Any | None' = None, plan: 'SplitPlan | None' = None, model_profile: 'SplitModelProfile | None' = None, model: 'Any | None' = None, batch_spec: 'BatchSpec | None' = None, _release_capture: 'bool' = False) -> 'tuple[SplitTraceGraph, SplitGraphIR]'"),
    ("api", "prepare"): (
        "d0940f8a4eb47ceff4541964610d7585f14828050a279063c732050591fabfd6",
        "(model: 'Any', inputs: 'Any', request: 'SplitRequest', *, input_kwargs: 'dict[str, Any] | None' = None) -> 'SplitRuntime'"),
    ("runtime", "SplitRuntime._analyze_point"): (
        "3c75716df894a557c71b205e12f71425ff5d735fd52bbe2ae505f4d594e947c3",
        "(self, point: 'SplitPoint', *, state_snapshots: 'dict[str, dict[str, Any]]') -> 'SplitPointAnalysis'"),
    ("runtime", "SplitRuntime.materialize"): (
        "86ffe44dc74d24256525da454f037ac0d94a69b92b7b3e50a0e13594538121a4",
        "(self, analysis: 'SplitPointAnalysis') -> 'SplitRuntime'"),
    ("runtime", "SplitRuntime.with_placement"): (
        "1c8c8bc352fd4ce8143e51ac7a48a407f0a7263d5344020e0a4161e024a72c3e",
        "(self, plan: 'PlacementPlan') -> 'SplitRuntime'"),
}


def _clone(function, **bindings):
    result = FunctionType(function.__code__, dict(function.__globals__, **bindings),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    result.__module__ = function.__module__
    result.__qualname__ = function.__qualname__
    return result


def _verify_native_function(function, module, name):
    expected_hash, expected_signature = _PINNED_FUNCTIONS[module, name]
    source = inspect.getsource(function)
    if (sha256(source.encode("utf-8")).hexdigest() != expected_hash
            or str(inspect.signature(function)) != expected_signature):
        raise RuntimeError(f"Unsupported native source/signature for torchlens.split.{module}.{name}")
    return source


def _private_native_frontier(function):
    """Copy the exact native frontier body, changing only its parent condition."""
    source = _verify_native_function(function, "planner", "_frontier_node_ids")
    tree = ast.parse(source)
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "parent"
             and isinstance(node.iter, ast.Attribute)
             and isinstance(node.iter.value, ast.Name) and node.iter.value.id == "node"
             and node.iter.attr == "parents"]
    if len(loops) != 1:
        raise RuntimeError("Pinned native frontier has no unique parent loop")
    # Carry prefix-owned output VALUES as the native body already does;
    # carry their parents only when that output is itself in the suffix.
    loops[0].iter = ast.IfExp(
        test=ast.Compare(left=ast.Attribute(value=ast.Name(id="node", ctx=ast.Load()),
                                           attr="canonical_id", ctx=ast.Load()),
                         ops=[ast.In()], comparators=[ast.Name(id="suffix_node_ids", ctx=ast.Load())]),
        body=loops[0].iter, orelse=ast.Tuple(elts=[], ctx=ast.Load()),
    )
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<splitfleet-private-native-frontier>", "exec",  # noqa: S102 - source-hash-pinned SDK AST
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), namespace)
    result = namespace[function.__name__]
    result.__module__ = function.__module__
    return result


def minimal_frontier_node_ids(graph, *, prefix_node_ids, suffix_node_ids):
    """Independent dependency audit; this is never used as a native planner."""
    from torchlens.split.errors import SplitRequestError

    ids = tuple(node.canonical_id for node in graph.nodes)
    prefix, suffix = set(prefix_node_ids), set(suffix_node_ids)
    if len(set(ids)) != len(ids) or prefix & suffix or prefix | suffix != set(ids):
        raise SplitRequestError("Frontier requires a unique, complete native graph partition")
    required = set()
    for node in graph.nodes:
        if node.is_output and node.canonical_id in prefix:
            required.add(node.canonical_id)
        if node.canonical_id not in suffix:
            continue
        parents = set()
        for label in node.parents:
            parent = graph.node_for_label(label)
            if parent is None:
                raise SplitRequestError(f"Unresolved native suffix operand {label!r}")
            parents.add(parent.canonical_id)
            if parent.canonical_id in prefix:
                required.add(parent.canonical_id)
        # TF replay uses these native capture inputs directly. Require the
        # portable graph edges to represent every labeled native operand.
        for item in getattr(node.target, "inputs", ()):
            label = getattr(item, "producer_label_raw", None) or getattr(item, "source_label_raw", None)
            if label is not None:
                parent = graph.node_for_label(label)
                if parent is None or parent.canonical_id not in parents:
                    raise SplitRequestError(f"Native TF operand is absent from graph edges at {node.canonical_id!r}: {label!r}")
    required.update(node_id for node_id in graph.input_node_ids if node_id in suffix)
    if not required and not suffix and not graph.output_node_ids:
        terminals = [node for node in graph.nodes if node.canonical_id in prefix
                     and not (node.is_input or node.is_buffer or node.is_buffer_only_source)]
        if terminals:
            required.add(terminals[-1].canonical_id)
    return tuple(sorted(required, key=graph.order_by_id.__getitem__))


def frontier_proof(graph, plan):
    """Audit all consumers/outputs and report native surplus fields explicitly."""
    from torchlens.split.planner import _frontier_node_ids
    expected = minimal_frontier_node_ids(graph, prefix_node_ids=plan.prefix_node_ids,
                                        suffix_node_ids=plan.suffix_node_ids)
    if tuple(plan.boundary_node_ids) != expected:
        raise ValueError("Executable native plan does not contain its complete minimal frontier")
    original = _frontier_node_ids(graph, prefix_node_ids=plan.prefix_node_ids,
                                 suffix_node_ids=plan.suffix_node_ids)
    return {
        "policy": "native_suffix_dependencies_and_prefix_outputs_v1",
        "original_native_field_ids": list(original), "required_field_ids": list(expected),
        "removed_field_ids": [node_id for node_id in original if node_id not in expected],
        "prefix_output_ids": [node.canonical_id for node in graph.nodes
                              if node.is_output and node.canonical_id in plan.prefix_node_ids],
        "suffix_dependencies": {node.canonical_id: [graph.node_for_label(label).canonical_id
            for label in node.parents if graph.node_for_label(label).canonical_id in plan.prefix_node_ids]
            for node in graph.nodes if node.canonical_id in plan.suffix_node_ids},
    }


def _native_bindings():
    if version("torchlens") != NATIVE_VERSION:
        raise RuntimeError("Private native frontier binding requires torchlens==2.34.1")
    for (module, name) in _PINNED_FUNCTIONS:
        function = importlib.import_module("torchlens.split." + module)
        for part in name.split("."):
            function = getattr(function, part)
        _verify_native_function(function, module, name)
    planner = importlib.import_module("torchlens.split.planner")
    pipeline = importlib.import_module("torchlens.split.pipeline")
    api = importlib.import_module("torchlens.split.api")
    plan = _clone(planner.plan_split, _frontier_node_ids=_private_native_frontier(planner._frontier_node_ids))
    execute_probe = _clone(pipeline._execute_batch_probe, plan_split=plan)
    probe = _clone(pipeline._probe_batch_replay, _execute_batch_probe=execute_probe)
    normalize = _clone(pipeline.normalize_to_split_ir, plan_split=plan, _probe_batch_replay=probe)
    prepare = _clone(api.prepare, plan_split=plan, normalize_to_split_ir=normalize)
    return plan, prepare


def bind_native_frontier(runtime):
    """Keep native analysis/recut/materialization on this instance's policy."""
    if runtime.adapter.name not in {"tf", "tensorflow", "paddle"}:
        raise ValueError("Native minimal frontier requires TensorFlow or Paddle")
    if getattr(runtime, "_splitfleet_minimal_frontier", False):
        return runtime
    private_plan, _ = _native_bindings()
    native_type = type(runtime)
    if native_type is not importlib.import_module("torchlens.split.runtime").SplitRuntime:
        raise RuntimeError("Private frontier binding requires the exact pinned native runtime type")
    runtime._analyze_point = MethodType(_clone(native_type._analyze_point, plan_split=private_plan), runtime)
    native_materialize, native_placement = native_type.materialize, native_type.with_placement

    def materialize(self, analysis):
        result = native_materialize(self, analysis)
        frontier_proof(result.trace_graph, result.plan)
        return bind_native_frontier(result)

    def with_placement(self, plan):
        result = native_placement(self, plan)
        frontier_proof(result.trace_graph, result.plan)
        return bind_native_frontier(result)

    runtime.materialize = MethodType(materialize, runtime)
    runtime.with_placement = MethodType(with_placement, runtime)
    runtime._splitfleet_minimal_frontier = True
    frontier_proof(runtime.trace_graph, runtime.plan)
    return runtime


def prepare_native_frontier_runtime(model, inputs, request, *, input_kwargs=None):
    """Prepare unchanged native capture/probes with a faithful new ABI.

    Caller retains its actual native device scope. No framework, SDK module,
    class, registry or captured graph object is modified by this function.
    """
    if str(request.backend) not in {"tf", "tensorflow", "paddle"}:
        raise ValueError("Native minimal frontier requires TensorFlow or Paddle")
    _, prepare = _native_bindings()
    return bind_native_frontier(prepare(model, inputs, request, input_kwargs=input_kwargs))
