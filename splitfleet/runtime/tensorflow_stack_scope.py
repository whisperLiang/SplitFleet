"""Keep native TF function stack captures outside their small inline storage.

The installed TF 2.21/CPython 3.11 combination loses Python code references
when a short native StackTrace is moved into its cache. A retained native
trace can then outlive that code object. The same ToStackFrames path has
crashed during function graph partitioning in the full-shard control.

Seventeen ordinary Python call frames avoid the observed inline-vector path.
The native function still executes once, with identical arguments and return
values. No TensorFlow/TorchLens function, registry, graph or tensor is changed.
"""

TENSORFLOW_FUNCTION_STACK_FLOOR = 17


def _invoke(depth, function, args, kwargs):
    if depth:
        return _invoke(depth - 1, function, args, kwargs)
    return function(*args, **kwargs)


def call_tensorflow_function(function, *args, **kwargs):
    """Preserve native function execution while retaining safe trace lifetimes."""
    return _invoke(TENSORFLOW_FUNCTION_STACK_FLOOR, function, args, kwargs)
