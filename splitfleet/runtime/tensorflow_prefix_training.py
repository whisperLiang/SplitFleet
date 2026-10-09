"""Keep TensorFlow resource reads connected to the prefix's live variables."""

from typing import Any


def run_tensorflow_training_prefix(runtime: Any, *inputs: Any, input_kwargs=None):
    """Record native replay with explicit watches for resource-backed weights.

    TorchLens can capture custom Keras weights as raw ReadVariableOp nodes
    without parameter references. Its prefix tape then misses those variables,
    although native prefix backward resolves them from their resource handles.
    An enclosing tape covers the same native replay and supplies that backward
    with the complete recording. Parameter ownership and updates remain native.
    """
    import tensorflow as tf

    tape = tf.GradientTape(persistent=True)
    with tape:
        for variable in getattr(runtime.model, "trainable_variables", ()):
            source = variable if isinstance(variable, tf.Variable) else variable.value
            tape.watch(source)
        boundary = runtime.run_training_prefix(*inputs, input_kwargs=input_kwargs)
    boundary.metadata["tf_tape"] = tape
    return boundary
