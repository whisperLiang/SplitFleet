"""Batch native optimizer dispatch without changing its updates or variables."""

from __future__ import annotations

from splitfleet.runtime.tensorflow_stack_scope import call_tensorflow_function


class CompiledNativeOptimizer:
    """One native optimizer and one private update function per variable tuple."""

    def __init__(self, optimizer):
        self.native = optimizer
        self._compiled = None

    def __getattr__(self, name):
        return getattr(self.native, name)

    @property
    def learning_rate(self):
        return self.native.learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.native.learning_rate = value

    def apply_gradients(self, pairs):
        import tensorflow as tf

        pairs = tuple(pairs)
        if not pairs:
            return self.native.apply_gradients(pairs)
        variables = tuple(variable for _, variable in pairs)
        identity = tuple(map(id, variables))
        if self._compiled is None:
            # Keras builds its ordinary optimizer state once. Its original
            # apply_gradients body still owns clipping, SGD and step counters.
            if not self.native.built:
                self.native.build(variables)

            @tf.function(autograph=False)
            def update(gradients):
                return self.native.apply_gradients(zip(gradients, variables, strict=True))

            self._compiled = identity, update
        if self._compiled[0] != identity:
            raise RuntimeError('Compiled native optimizer variable bindings changed')
        return call_tensorflow_function(self._compiled[1], tuple(gradient for gradient, _ in pairs))


def native_optimizer(optimizer, model):
    if getattr(model, '_splitfleet_tensorflow_compiled_updates', False):
        return CompiledNativeOptimizer(optimizer)
    return optimizer
