"""Keep native TensorFlow prefix replay connected to its live variables."""

from dataclasses import replace
from threading import RLock
from typing import Any


class _CompiledTrainingPrefix:
    """Compile an instance's original native replay, retaining its boundary ABI.

    Only tensor results cross the TF function. The actual enclosing tape stays
    with this minibatch at the client, and differentiates the function's live
    resource reads. No tape or model variable is reused between minibatches.
    """

    def __init__(self, native):
        self.native = native
        self.functions = {}
        self.lock = RLock()

    def __call__(self, *inputs, input_kwargs=None, detach_boundary=False):
        from torchlens.split.boundary import ReplayBoundary
        from splitfleet.runtime.tensorflow_forward import _input_signature, _require_policy
        from splitfleet.runtime.tensorflow_stack_scope import call_tensorflow_function

        tf = _require_policy()
        kwargs = dict(input_kwargs or {})
        key = _input_signature(inputs, kwargs)
        if detach_boundary or key is None:
            return self.native(*inputs, input_kwargs=kwargs, detach_boundary=detach_boundary)
        with self.lock:
            if key not in self.functions:
                template = {}

                @tf.function(autograph=False)
                def replay(values, keywords):
                    boundary = self.native(*values, input_kwargs=keywords, detach_boundary=False)
                    template.update(backend=boundary.backend, spec=boundary.spec,
                        metadata={name: value for name, value in boundary.metadata.items()
                                  if name not in {"tf_tape", "prefix_boundary_tensors"}})
                    return boundary.tensors

                self.functions[key] = replay, template
            function, template = self.functions[key]
        tensors = call_tensorflow_function(function, inputs, kwargs)
        metadata = dict(template["metadata"], prefix_boundary_tensors=tensors)
        return ReplayBoundary(template["backend"], tensors, template["spec"], metadata)


def _bind_compiled_training_prefix(runtime):
    from importlib.metadata import version
    from torchlens.split.adapters.tf import TfGeneratedPrefix

    native = runtime.segments.training_prefix
    if isinstance(native, _CompiledTrainingPrefix):
        return
    if version("torchlens") != "2.34.1" or type(native) is not TfGeneratedPrefix:
        raise RuntimeError("Compiled TensorFlow prefix requires the pinned native generated prefix")
    runtime.segments = replace(runtime.segments, training_prefix=_CompiledTrainingPrefix(native))


def run_tensorflow_training_prefix(runtime: Any, *inputs: Any, input_kwargs=None):
    """Record native replay with explicit watches for resource-backed weights.

    TorchLens can capture custom Keras weights as raw ReadVariableOp nodes
    without parameter references. Its prefix tape then misses those variables,
    although native prefix backward resolves them from their resource handles.
    An enclosing tape covers the same native replay and supplies that backward
    with the complete recording. Parameter ownership and updates remain native.
    """
    import tensorflow as tf

    if getattr(runtime.model, "_splitfleet_tensorflow_function_prefix",
               getattr(runtime.model, "_splitfleet_tensorflow_function_forward", False)):
        _bind_compiled_training_prefix(runtime)
    tape = tf.GradientTape(persistent=True)
    with tape:
        for variable in getattr(runtime.model, "trainable_variables", ()):
            source = variable if isinstance(variable, tf.Variable) else variable.value
            tape.watch(source)
        boundary = runtime.run_training_prefix(*inputs, input_kwargs=input_kwargs)
    boundary.metadata["tf_tape"] = tape
    return boundary
