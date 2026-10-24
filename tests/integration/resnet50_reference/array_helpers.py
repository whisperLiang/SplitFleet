"""Native gradient, state-layout and boundary observers for regression tests."""

from __future__ import annotations

import numpy as np

from experiments.resnet50_quality.gate_torch import LEARNING_RATE
from experiments.resnet50_quality.model_contract import NAMES, SCHEMA, TRAINABLE_NAMES

_ROWS = {row.name: row for row in SCHEMA}


def canonical_array(backend, name, value):
    """Convert one declared native state/gradient layout without dtype coercion."""
    value = np.asarray(value)
    row = _ROWS[name]
    if backend == "tensorflow":
        if len(row.shape) == 4:
            value = value.transpose(3, 2, 0, 1)
        elif name == "fc.weight":
            value = value.T
    if value.shape != row.shape or str(value.dtype) != row.dtype:
        raise RuntimeError(f"Native {backend} tensor has an incorrect canonical schema at {name}")
    return np.array(value, copy=True, order="C")


class RecordingOptimizer:
    """Record native gradients immediately before delegating actual native SGD."""

    def __init__(self, backend, model):
        from .native import native_parameters, optimizer
        self.backend, self.model = backend, model
        self.native = optimizer(backend, model, LEARNING_RATE)
        self.parameters = native_parameters(backend, model)
        self.gradients, self.recorded_steps = {}, 0
        self.names_by_identity = {}
        for name, variable in self.parameters.items():
            for candidate in (variable, getattr(variable, "value", None), getattr(variable, "handle", None)):
                if candidate is not None and not callable(candidate):
                    self.names_by_identity[id(candidate)] = name

    def __getattr__(self, name):
        return getattr(self.native, name)

    def apply_gradients(self, pairs):
        from .native import numpy_value
        pairs = list(pairs)
        for gradient, variable in pairs:
            name = self.names_by_identity.get(id(variable))
            if name is None:
                raise RuntimeError("Native TensorFlow SGD gradient has no declared parameter binding")
            if name in self.gradients:
                raise RuntimeError(f"Parameter received more than one native SGD update: {name}")
            self.gradients[name] = canonical_array(self.backend, name, numpy_value(gradient))
        self.recorded_steps += 1
        return self.native.apply_gradients(pairs)

    def step(self):
        from .native import numpy_value
        for name, variable in self.parameters.items():
            if variable.grad is not None:
                if name in self.gradients:
                    raise RuntimeError(f"Parameter received more than one native SGD update: {name}")
                self.gradients[name] = canonical_array(self.backend, name, numpy_value(variable.grad))
        self.recorded_steps += 1
        return self.native.step()


class RecordingLoss:
    def __init__(self, backend="tensorflow"):
        self.backend = backend
        self.calls = []

    def _record(self, logits, loss):
        from .native import numpy_value
        self.calls.append({"logits": numpy_value(logits), "loss": float(numpy_value(loss))})

    def __call__(self, output, labels):
        from splitfleet.tasks.losses import extract_logits, sparse_cross_entropy
        logits = extract_logits(output)
        loss = sparse_cross_entropy(logits, labels)
        if self.backend == "tensorflow":
            import tensorflow as tf
            if tf.inside_function():
                # Observe the actual differentiated invocation, rather than
                # attempting NumPy conversion while tf.function is tracing.
                # The token orders recording without entering the loss gradient.
                def record(logits, value):
                    self._record(logits, value)
                    return np.int64(0)

                token = tf.py_function(record, (logits, loss), Tout=tf.int64)
                with tf.control_dependencies([token]):
                    loss = tf.identity(loss)
            else:
                self._record(logits, loss)
        else:
            self._record(logits, loss)
        return loss


def stage_boundaries(handle, backend):
    """Select the actual residual endpoint by native address and shape."""
    endpoints = {}
    for stage, channels, spatial in ((1, 256, 16), (2, 512, 8), (3, 1024, 4)):
        expected = (spatial, spatial, channels) if backend == "tensorflow" else (channels, spatial, spatial)
        operations = {"relu"}
        matches = [node for node in handle.runtime.trace_graph.nodes
                   if str(node.module_path or "").startswith(f"layer{stage}.")
                   and str(node.op_type).split(".")[-1].lower() in operations
                   and node.output_shape is not None and tuple(node.output_shape[1:]) == expected]
        if not matches:
            raise RuntimeError(f"Native {backend} capture has no terminal stage{stage} ReLU with {expected}")
        node = matches[-1]
        blocks = (3, 4, 6)[stage - 1]
        module = str(node.module_path)
        final_block = f"layer{stage}.{blocks - 1}"
        if module != final_block and not module.startswith(final_block + "."):
            raise RuntimeError(f"stage{stage} endpoint does not belong to its final Bottleneck")
        endpoints[f"stage{stage}"] = {
            "boundary": "after:" + node.canonical_id, "canonical_node": node.canonical_id,
            "label": node.label, "module": node.module_path, "captured_shape": list(node.output_shape),
        }
    return endpoints


def owned_snapshot(backend, client, tail, ownership, initial, prefix_optimizer):
    from .native import export_state
    owners = dict(zip(ownership["names"], ownership["owners"], strict=True))
    if tuple(owners) != NAMES:
        raise RuntimeError("Native ownership must preserve the complete canonical name order")
    models = {"prefix": client.model, "suffix": tail.model}
    states = {owner: export_state(backend, model) for owner, model in models.items()}
    optimizers = {"prefix": prefix_optimizer, "suffix": tail.optimizer}
    state, gradients = {}, {}
    updates = {owner: {"trainable_tensors": 0, "updated_tensors": 0, "updated_names": []}
               for owner in models}
    for name in NAMES:
        owner = owners[name]
        state[name] = initial[name].copy() if owner == "initial" else states[owner][name]
        if name in TRAINABLE_NAMES:
            if owner not in models or name not in optimizers[owner].gradients:
                raise RuntimeError(f"Missing real {owner} SGD gradient for {name}")
            gradients[name] = optimizers[owner].gradients[name]
            updates[owner]["trainable_tensors"] += 1
            if np.any(state[name] != initial[name]):
                updates[owner]["updated_tensors"] += 1
                updates[owner]["updated_names"].append(name)
    for owner, opt in optimizers.items():
        expected = {name for name in TRAINABLE_NAMES if owners[name] == owner}
        if set(opt.gradients) != expected or opt.recorded_steps != 1:
            raise RuntimeError(f"Native {owner} SGD did not execute exactly its declared parameter ownership")
    return state, gradients, updates
