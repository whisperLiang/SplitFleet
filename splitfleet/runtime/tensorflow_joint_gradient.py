"""One native TensorFlow suffix gradient query through pinned private bindings.

The TorchLens trainer, its native tape, replay, boundary checks, gradient
filters and SGD update order are retained. Only the two gradient-query
assignments change: roots and conditionally needed variables share one query.
No SDK module, class, framework or engine registry is modified.
"""
from __future__ import annotations
import __future__

import ast
import builtins
import importlib
import inspect
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from types import FunctionType, MethodType, SimpleNamespace

NATIVE_VERSION = "2.34.1"
_PINNED = {
    ("training", "_train_suffix_tf"): (
        "a501b6271cb4df0b5ca31d4a8a2110d2f597cc934fe90029ff58956eef3be3df",
        "(runtime: 'Any', boundary: 'ReplayBoundary', targets: 'Any', loss_fn: 'Callable[[Any, Any], Any] | None' = None, optimizer: 'Any | None' = None) -> 'tuple[Any, BoundaryGradients, bool]'"),
    ("training", "TensorFlowTrainingEngine.train_suffix"): (
        "19a627351de91ae14613dc1c275dd4330e8aec429b38b055d03e99da1550b1a3",
        "(self, runtime: 'Any', boundary: 'ReplayBoundary', targets: 'Any', loss_fn: 'Callable[[Any, Any], Any] | None' = None, optimizer: 'Any | None' = None) -> 'TrainingStepResult'"),
    ("training", "training_engine_for"): (
        "b369b8030ad1ae5001a55f53893dee27de114d4636fc1bb0c924ad1c332f8586",
        "(backend: 'str') -> 'BackendTrainingEngine'"),
    ("training", "train_suffix_result"): (
        "a7493c8ed9b839d46136aeb13743aa3fde018e3fb3da3305a3a853da2afbbd7f",
        "(runtime: 'Any', boundary: 'ReplayBoundary', targets: 'Any', loss_fn: 'Callable[[Any, Any], Any] | None' = None, optimizer: 'Any | None' = None, *, microbatch_size: 'int | None' = None, microbatch_reduction: 'str' = 'mean', target_slicer: 'Callable[[Any, int, int, int], Any] | None' = None) -> 'TrainingStepResult'"),
    ("training", "train_suffix"): (
        "ca269b9a5ab48b7631fba24262fa9f6e02b5b099d17faf056e337ad718ea6fd7",
        "(runtime: 'Any', boundary: 'ReplayBoundary', targets: 'Any', loss_fn: 'Callable[[Any, Any], Any] | None' = None, optimizer: 'Any | None' = None, *, microbatch_size: 'int | None' = None, microbatch_reduction: 'str' = 'mean', target_slicer: 'Callable[[Any, int, int, int], Any] | None' = None) -> 'tuple[Any, BoundaryGradients]'"),
    ("runtime", "SplitRuntime.train_suffix"): (
        "8c248429e0ee0f866b79fd83a53cdc2d28397d3f7b4e23db4f9e1cbaa30ab638",
        "(self, boundary: 'ReplayBoundary', targets: 'Any', loss_fn: 'Any' = None, optimizer: 'Any' = None, *, microbatch_size: 'int | None' = None, microbatch_reduction: 'str' = 'mean', target_slicer: 'Callable[[Any, int, int, int], Any] | None' = None) -> 'tuple[Any, dict[str, Any]]'"),
    ("runtime", "SplitRuntime.train_suffix_result"): (
        "50f83e3645869546b0c5828d5e7ec1c6f4b0fb63606bc5987c1e7027639034cc",
        "(self, boundary: 'ReplayBoundary', targets: 'Any', loss_fn: 'Any' = None, optimizer: 'Any' = None, *, microbatch_size: 'int | None' = None, microbatch_reduction: 'str' = 'mean', target_slicer: 'Callable[[Any, int, int, int], Any] | None' = None) -> 'Any'"),
}


def _clone(function, **bindings):
    result = FunctionType(function.__code__, dict(function.__globals__, **bindings),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    result.__module__ = function.__module__
    result.__qualname__ = function.__qualname__
    return result


def _verify_function(function, module, name):
    source = inspect.getsource(function)
    expected_hash, expected_signature = _PINNED[module, name]
    if (sha256(source.encode()).hexdigest() != expected_hash
            or str(inspect.signature(function)) != expected_signature):
        raise RuntimeError(f"Unsupported native source/signature for torchlens.split.{module}.{name}")
    return source


def _verified_functions():
    if version("torchlens") != NATIVE_VERSION:
        raise RuntimeError("Private TensorFlow joint gradient requires torchlens==2.34.1")
    functions = []
    for module, name in _PINNED:
        function = importlib.import_module("torchlens.split." + module)
        for part in name.split("."):
            function = getattr(function, part)
        _verify_function(function, module, name)
        functions.append(function)
    return tuple(functions)


def _private_joint_suffix(function):
    """Compile the pinned native body with one joint native tape query."""
    source = _verify_function(function, "training", "_train_suffix_tf")
    tree = ast.parse(source)
    root_assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == "root_grads"]
    variable_assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                            and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                            and node.targets[0].id == "var_grads"]
    if len(root_assignments) != 1 or len(variable_assignments) != 1:
        raise RuntimeError("Pinned native TensorFlow gradient assignments are not unique")
    root_assignment, variable_assignment = root_assignments[0], variable_assignments[0]
    queries = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == "gradient"
               and isinstance(node.func.value, ast.Name) and node.func.value.id == "tape"]
    if len(queries) != 2:
        raise RuntimeError("Pinned native TensorFlow trainer does not have two tape queries")
    # Do not compute variable gradients when native SGD does not need them;
    # preserve the original empty-source behavior, watched sources and filters.
    replacement = ast.parse("""
joint_sources = root_values + (suffix_sources if optimizer is not None and suffix_vars else [])
joint_grads = tape.gradient(loss, joint_sources) if joint_sources else []
root_grads = joint_grads[:len(root_values)]
""").body
    body = tree.body[0].body
    offset = body.index(root_assignment)
    body[offset:offset + 1] = replacement
    variable_assignment.value = ast.parse("joint_grads[len(root_values):]", mode="eval").body
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<splitfleet-private-native-tf-joint-gradient>", "exec",  # noqa: S102
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), namespace)
    result = namespace[function.__name__]
    result.__module__ = function.__module__
    return result


def _training_functions(suffix_function):
    """Bind unchanged native dispatch and engine bodies to a private trainer."""
    training = importlib.import_module("torchlens.split.training")
    engine = training.TensorFlowTrainingEngine()
    engine.train_suffix = MethodType(_clone(training.TensorFlowTrainingEngine.train_suffix,
                                           _train_suffix_tf=suffix_function), engine)
    engines = dict(training._TRAINING_ENGINES, tf=engine, tensorflow=engine)
    engine_for = _clone(training.training_engine_for, _TRAINING_ENGINES=engines)
    structured = _clone(training.train_suffix_result, training_engine_for=engine_for)
    compact = _clone(training.train_suffix, train_suffix_result=structured)
    return compact, structured


@lru_cache(maxsize=1)
def _cached_bindings(functions):
    # This cache retains only immutable compiled functions and a stateless
    # native engine, with no models, arrays, tapes, optimizers or client state.
    return _training_functions(_private_joint_suffix(functions[0]))


def _bindings():
    # Recheck native source/version/signatures before returning a cached recipe.
    return _cached_bindings(_verified_functions())


def _require_tensorflow(runtime):
    if runtime.adapter.name not in {"tf", "tensorflow"}:
        raise ValueError("Private joint suffix gradients are validated only for TensorFlow")


def train_tensorflow_suffix_joint(runtime, boundary, targets, loss_fn=None, optimizer=None,
                                  *, microbatch_size=None, microbatch_reduction="mean", target_slicer=None):
    """Call native compact training with the joint query and all native guards."""
    _require_tensorflow(runtime)
    compact, _ = _bindings()
    return compact(runtime, boundary, targets, loss_fn=loss_fn, optimizer=optimizer,
                   microbatch_size=microbatch_size, microbatch_reduction=microbatch_reduction,
                   target_slicer=target_slicer)


def _bind_runtime_functions(runtime, functions):
    """Bind unchanged native methods; dedicated controls may observe the tape."""
    _require_tensorflow(runtime)
    runtime_type = importlib.import_module("torchlens.split.runtime").SplitRuntime
    if type(runtime) is not runtime_type:
        raise RuntimeError("Private joint gradients require the exact pinned native runtime type")
    compact, structured = functions
    imports = SimpleNamespace(train_suffix=compact, train_suffix_result=structured)

    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name == "training" and fromlist in (("train_suffix",), ("train_suffix_result",)):
            return imports
        return builtins.__import__(name, globals, locals, fromlist, level)

    for name in ("train_suffix", "train_suffix_result"):
        native_method = getattr(runtime_type, name)
        local = _clone(native_method, __builtins__=dict(vars(builtins), __import__=private_import))
        setattr(runtime, name, MethodType(local, runtime))
    return runtime


def bind_tensorflow_joint_gradient(runtime):
    """Bind both native result APIs privately, retaining the caller's device scope."""
    return _bind_runtime_functions(runtime, _bindings())
