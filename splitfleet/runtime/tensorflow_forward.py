"""Native TF function execution with unchanged TorchLens split operands.

Each private suffix owns its live resource bindings and optimizer state.
Verified replicas can share immutable native function programs.
The original native tape/gradient recipe runs inside the function. Boundary
validation, owned optimizer updates and auxiliary commits remain outside it.
Graph rewrites are disabled to retain the native raw-op arithmetic.
"""

from __future__ import annotations

from dataclasses import replace
from threading import RLock
import time

from splitfleet.runtime.tensorflow_stack_scope import call_tensorflow_function


def _loss_identity(function):
    owner, method = getattr(function, '__self__', None), getattr(function, '__func__', None)
    return (id(owner), method) if method is not None else (id(function), None)


def _input_signature(tensors, targets):
    import tensorflow as tf

    values = tf.nest.flatten((tensors, targets))
    if any(not isinstance(value, tf.Tensor) for value in values):
        return None
    structure = tf.nest.map_structure(
        lambda value: ('Tensor', value.dtype.name, tuple(value.shape)), (tensors, targets))
    return repr(structure)


def _replica_bindings(source, target):
    """Map the three public representations of each registered variable."""
    left, right = source.canonical_variables(), target.canonical_variables()
    if left.keys() != right.keys():
        raise ValueError('TensorFlow function replica state names changed')
    bindings, reverse = {}, {}
    for name, value in left.items():
        other = right[name]
        original, replacement = getattr(value, 'value', value), getattr(other, 'value', other)
        if callable(original):
            original, replacement = value, other
        if (value.shape, value.dtype, value.trainable, original.device) != (
                other.shape, other.dtype, other.trainable, replacement.device):
            raise ValueError('TensorFlow function replica variable contract changed')
        for before, after in ((value, other), (original, replacement),
                              (original.handle, replacement.handle)):
            if id(before) in bindings and bindings[id(before)] is not after:
                raise ValueError('TensorFlow function replica state alias changed')
            if id(after) in reverse and reverse[id(after)] is not before:
                raise ValueError('TensorFlow function replica state alias changed')
            bindings[id(before)] = after
            reverse[id(after)] = before
    return bindings


class _ReboundSuffixTraining:
    """Invoke an unchanged native function with this replica's resource handles."""

    def __init__(self, template, concrete, source, target):
        import tensorflow as tf

        bindings = _replica_bindings(source, target)
        self.concrete = concrete
        self.captures = []
        for capture in concrete.captured_inputs:
            if capture.dtype == tf.resource:
                if id(capture) not in bindings:
                    raise ValueError('Unregistered resource in TensorFlow training function')
                capture = bindings[id(capture)]
            self.captures.append(capture)
        if any(id(variable) not in bindings for variable in template.variables):
            raise ValueError('Unregistered native suffix optimizer variable')
        self.variables = tuple(bindings[id(variable)] for variable in template.variables)
        # A raw PartitionedCall has its own rewrite configuration; the eager
        # tf.function policy is not inherited by an empty config_proto.
        config = tf.compat.v1.ConfigProto()
        config.graph_options.rewrite_options.disable_meta_optimizer = True
        self.config_proto = config.SerializeToString()
        outputs = tf.nest.flatten(concrete.structured_outputs)
        if len(outputs) != len(concrete.outputs) or any(not tf.is_tensor(value) for value in outputs):
            raise ValueError('Unsupported native suffix function output structure')
        self.input_specs = concrete.inputs[:len(concrete.inputs) - len(self.captures)]

    def __call__(self, tensors, targets, optimizer):
        import tensorflow as tf

        inputs = tf.nest.flatten((tensors, targets))
        if len(inputs) != len(self.input_specs) or any(
                value.dtype != spec.dtype or not spec.shape.is_compatible_with(value.shape)
                for value, spec in zip(inputs, self.input_specs, strict=True)):
            raise ValueError('Rebound TensorFlow suffix input contract changed')
        values = call_tensorflow_function(tf.raw_ops.StatefulPartitionedCall,
            args=[*inputs, *self.captures], Tout=[value.dtype for value in self.concrete.outputs],
            f=self.concrete.name, config_proto=self.config_proto)
        loss, roots, gradients, statistics = tf.nest.pack_sequence_as(
            self.concrete.structured_outputs, values)
        if optimizer is not None and self.variables:
            optimizer.apply_gradients(zip(gradients, self.variables, strict=True))
        return (loss, roots), statistics


class TensorFlowSuffixFunctionPool:
    """Share immutable native gradient programs for explicitly verified clones.

    Templates use the already-resident placement model. Private client models,
    their optimizer states and their native resource handles never enter the pool.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.functions = {}
        self.rejected = set()
        self.lock = RLock()
        self.compile_count = 0
        self.compile_sec = 0.

    def bind(self, runtime, boundary, targets, loss_fn, *, update, auxiliary):
        key = (boundary.metadata['runtime_batch_size'], _loss_identity(loss_fn), update, auxiliary,
               _input_signature(boundary.tensors, targets))
        with self.lock:
            if key in self.rejected:
                return None
            if key not in self.functions:
                began = time.perf_counter()
                template = _CompiledSuffixTraining(self.runtime, boundary, loss_fn,
                    update=update, auxiliary=auxiliary)
                concrete = call_tensorflow_function(template.execute.get_concrete_function,
                    boundary.tensors, targets)
                self.compile_sec += time.perf_counter() - began
                try:
                    bound = _ReboundSuffixTraining(template, concrete, self.runtime.model, runtime.model)
                except ValueError:
                    # Refused resource captures must not keep a private loss
                    # variable or an otherwise unregistered state alive here.
                    self.rejected.add(key)
                    return None
                self.functions[key] = template, concrete
                self.compile_count += 1
                return bound
            template, concrete = self.functions[key]
            return _ReboundSuffixTraining(template, concrete, self.runtime.model, runtime.model)


def _require_policy():
    import tensorflow as tf

    if (
        tf.config.optimizer.get_experimental_options().get("disable_meta_optimizer")
        is not True
    ):
        raise RuntimeError(
            "Native TensorFlow function execution requires disabled graph rewrites"
        )
    return tf


def native_training_forward(model, images):
    """Give the full-model FedAvg reference the same native function policy."""
    tf = _require_policy()
    function = getattr(model, "_splitfleet_native_training_function", None)
    if function is None:

        @tf.function(autograph=False)
        def function(inputs):
            return model(inputs, training=True)

        model._splitfleet_native_training_function = function
    return call_tensorflow_function(function, images)


def native_training_gradients(model, images, labels, parameters):
    """Keep full-model loss and native reverse-mode in the same TF function.

    Optimizer updates and deferred BN commits remain outside the function.
    A persistent worker can reload this model's variables between clients;
    the function always reads those live variables in the declared order.
    """
    from splitfleet.tasks.losses import sparse_cross_entropy

    tf = _require_policy()
    sources = tuple(parameters)
    identity = tuple(id(variable) for variable in sources)
    cached = getattr(model, "_splitfleet_native_gradient_function", None)
    if cached is None:

        @tf.function(autograph=False)
        def function(inputs, targets):
            with tf.GradientTape() as tape:
                output = model(inputs, training=True)
                objective = sparse_cross_entropy(output["logits"], targets)
            return output, objective, tuple(tape.gradient(objective, sources))

        cached = (identity, function)
        model._splitfleet_native_gradient_function = cached
    if cached[0] != identity:
        raise RuntimeError("Native TensorFlow gradient variable bindings changed")
    return call_tensorflow_function(cached[1], images, labels)


class _CompiledSuffix:
    def __init__(self, native):
        self.native = native
        self.functions = {}
        self.lock = RLock()

    def __call__(self, boundary):
        from torchlens.split.boundary import ReplayBoundary

        tf = _require_policy()
        batch = boundary.metadata["runtime_batch_size"]
        with self.lock:
            if batch not in self.functions:
                # Prefix tapes stay at the client; every tensor operand comes
                # from this actual already-validated transported boundary.
                metadata = {
                    key: value
                    for key, value in boundary.metadata.items()
                    if key not in {"tf_tape", "prefix_boundary_tensors"}
                }
                backend, spec = boundary.backend, boundary.spec

                @tf.function(autograph=False)
                def replay(tensors):
                    return self.native(ReplayBoundary(backend, tensors, spec, metadata))

                self.functions[batch] = replay
            function = self.functions[batch]
        return call_tensorflow_function(function, boundary.tensors)


def bind_tensorflow_function_suffix(runtime):
    """Wrap this instance's original native suffix; preserve graph and ABI."""
    from importlib.metadata import version
    from torchlens.split.adapters.tf import TfGeneratedSuffix

    if version("torchlens") != "2.34.1" or runtime.adapter.name not in {
        "tf",
        "tensorflow",
    }:
        raise RuntimeError(
            "Native TensorFlow function binding requires TorchLens 2.34.1 TF replay"
        )
    if isinstance(runtime.segments.suffix, _CompiledSuffix):
        return runtime
    if type(runtime.segments.suffix) is not TfGeneratedSuffix:
        raise RuntimeError(
            "Native TensorFlow function binding requires the original generated suffix"
        )
    _require_policy()
    runtime.segments = replace(
        runtime.segments, suffix=_CompiledSuffix(runtime.segments.suffix)
    )
    return runtime


class _CompiledSuffixTraining:
    def __init__(self, runtime, boundary, loss_fn, *, update, auxiliary):
        from torchlens.split.boundary import ReplayBoundary
        from splitfleet.runtime.auxiliary_state import restore_auxiliary_output
        from splitfleet.runtime.tensorflow_joint_gradient import _bindings

        tf = _require_policy()
        native_train, _ = _bindings()
        self.variables = None
        self.loss_fn = loss_fn
        metadata = {
            key: value
            for key, value in boundary.metadata.items()
            if key not in {"tf_tape", "prefix_boundary_tensors"}
        }
        backend, spec = boundary.backend, boundary.spec

        @tf.function(autograph=False)
        def execute(tensors, targets):
            # The pinned native trainer still selects and watches every root
            # and suffix source. Collect its exact gradient pairs so the real
            # owned SGD optimizer runs once, after graph execution returns.
            class Gradients:
                pairs = ()

                def apply_gradients(self, pairs):
                    self.pairs = tuple(pairs)

            recorder = Gradients() if update else None
            statistics = []

            def objective(output, labels):
                if auxiliary:
                    output = restore_auxiliary_output(runtime.trace_graph, output)
                    statistics.append(output["batch_statistics"])
                return loss_fn(output, labels)

            replay = ReplayBoundary(backend, tensors, spec, metadata)
            loss, roots = native_train(
                runtime, replay, targets, loss_fn=objective, optimizer=recorder
            )
            pairs = recorder.pairs if recorder is not None else ()
            variables = tuple(variable for _, variable in pairs)
            if self.variables is not None and tuple(map(id, variables)) != tuple(
                map(id, self.variables)
            ):
                raise RuntimeError("Native suffix gradient variable bindings changed")
            self.variables = variables
            if auxiliary and len(statistics) != 1:
                raise RuntimeError(
                    "Native suffix function must expose one auxiliary forward"
                )
            return (
                loss,
                roots,
                tuple(gradient for gradient, _ in pairs),
                (statistics[0] if auxiliary else {}),
            )

        self.execute = execute

    def __call__(self, tensors, targets, optimizer):
        loss, roots, gradients, statistics = call_tensorflow_function(self.execute, tensors, targets)
        if optimizer is not None and self.variables:
            optimizer.apply_gradients(zip(gradients, self.variables, strict=True))
        return (loss, roots), statistics


def train_tensorflow_function_suffix(
    runtime, boundary, targets, *, loss_fn, optimizer, auxiliary
):
    """Execute the pinned native suffix trainer with its joint tape in TF graph mode."""
    from splitfleet.runtime.tensorflow_joint_gradient import _bindings

    _require_policy()
    _bindings()  # Recheck source/version pins on every actual training request.
    runtime.validate_boundary(boundary)
    cache = getattr(runtime, "_splitfleet_native_training_functions", None)
    if cache is None:
        cache = runtime._splitfleet_native_training_functions = {}
    key = (
        boundary.metadata["runtime_batch_size"],
        _loss_identity(loss_fn),
        optimizer is not None,
        auxiliary,
        _input_signature(boundary.tensors, targets),
    )
    compiled = cache.get(key)
    if compiled is None:
        pool = getattr(runtime, '_splitfleet_shared_suffix_functions', None)
        if pool is not None and key[-1] is not None:
            try:
                compiled = pool.bind(runtime, boundary, targets, loss_fn,
                    update=optimizer is not None, auxiliary=auxiliary)
            except ValueError:
                # Models can expose extra loss resources outside their declared
                # replica state. Keep their original private native function.
                compiled = None
        if compiled is None:
            compiled = _CompiledSuffixTraining(
                runtime,
                boundary,
                loss_fn,
                update=optimizer is not None,
                auxiliary=auxiliary,
            )
        cache[key] = compiled
    return compiled(boundary.tensors, targets, optimizer)
