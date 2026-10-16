"""Native ResNet50 autograd reference with the RPC experiment's device placement.

This executes the real factory layers directly, with native autograd or composed
native VJPs at a residual-stage endpoint. It never captures or replays a graph,
uses the wire, or supplies reference gradients. A same-device control must also
agree with the original complete-model native reference at the original tolerance.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from experiments.resnet50_quality.model_contract import BLOCKS, NAMES, SCHEMA, TRAINABLE_NAMES
from .native import (
    build_model,
    export_state,
    native_parameters,
    numpy_value,
    optimizer,
    training_mode,
)


def stage_owner(name, stage):
    if name.startswith("layer"):
        return "prefix" if int(name[5]) <= stage else "suffix"
    return "suffix" if name.startswith("fc.") else "prefix"


def _context(backend, device):
    if backend == "tensorflow":
        import tensorflow as tf
        return tf.device("/GPU:" + device.split(":")[-1] if device.startswith("cuda") else "/CPU:0")
    if backend == "paddle":
        from splitfleet.backends.utils import native_device_scope
        return native_device_scope(backend, device)
    return nullcontext()


def _jax_forward(model, params, images, stages, *, stem, head):
    import jax
    import jax.numpy as jnp

    from .jax_model import _conv, _relu

    statistics = {}
    value = images
    if stem:
        value = _conv(value, params["conv1.weight"], stride=2, padding=3)
        value = _relu(model._batch_norm(value, params, "bn1", statistics, training=True))
        value = jax.lax.reduce_window(value, np.float32(-np.inf), jax.lax.max,
            window_dimensions=(1, 1, 3, 3), window_strides=(1, 1, 2, 2),
            padding=((0, 0), (0, 0), (1, 1), (1, 1)))
    for stage in stages:
        for index in range(BLOCKS[stage - 1]):
            name = f"layer{stage}.{index}"
            stride = 2 if stage > 1 and index == 0 else 1
            identity = value
            hidden = _relu(model._batch_norm(_conv(value, params[name + ".conv1.weight"]),
                params, name + ".bn1", statistics, training=True))
            hidden = _relu(model._batch_norm(_conv(hidden, params[name + ".conv2.weight"], stride=stride, padding=1),
                params, name + ".bn2", statistics, training=True))
            hidden = model._batch_norm(_conv(hidden, params[name + ".conv3.weight"]),
                params, name + ".bn3", statistics, training=True)
            if name + ".downsample.0.weight" in params:
                identity = model._batch_norm(_conv(identity, params[name + ".downsample.0.weight"], stride=stride),
                    params, name + ".downsample.1", statistics, training=True)
            value = _relu(hidden + identity)
    if head:
        pooled = jnp.mean(value, axis=(2, 3), dtype=jnp.float32)
        value = jax.lax.dot_general(pooled, params["fc.weight"],
            dimension_numbers=(((1,), (1,)), ((), ())), precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32) + params["fc.bias"]
    return value, statistics


def _forward(backend, model, images, stages, *, stem, head, params=None):
    if backend == "jax":
        return _jax_forward(model, params, images, stages, stem=stem, head=head)
    statistics = {}
    value = images
    if backend == "tensorflow":
        if stem:
            value, statistics["bn1"] = model.bn1(model.conv1(model.pad1(value)), training=True, defer_statistics=True)
            value = model.maxpool(model.relu(value))
        for stage in stages:
            value, current = getattr(model, f"layer{stage}")(value, training=True, defer_statistics=True)
            statistics.update({f"layer{stage}.{name}": row for name, row in current.items()})
        return (model.fc(model.avgpool(value)) if head else value), statistics
    norms = []
    if backend == "paddle":
        from .paddle_model import CanonicalBatchNorm2D
        norms = [layer for layer in model.sublayers() if isinstance(layer, CanonicalBatchNorm2D)]
        for layer in norms:
            layer._statistics_sink = statistics
    try:
        if stem:
            value = model.maxpool(model.relu(model.bn1(model.conv1(value))))
        for stage in stages:
            value = getattr(model, f"layer{stage}")(value)
        if head:
            value = model.avgpool(value)
            if backend == "pytorch":
                import torch
                value = torch.flatten(value, 1)
            else:
                import paddle
                value = paddle.flatten(value, start_axis=1)
            value = model.fc(value)
        return value, statistics
    finally:
        for layer in norms:
            layer._statistics_sink = None


def _copy(backend, value, device):
    if backend == "pytorch":
        return value.to(device)
    if backend == "paddle":
        return value.to(device=device.replace("cuda:", "gpu:"))
    if backend == "tensorflow":
        import tensorflow as tf
        return tf.identity(value)
    import jax
    native = jax.devices("gpu" if device.startswith("cuda") else "cpu")[int(device.split(":")[-1]) if ":" in device else 0]
    return jax.device_put(value, native)


def train_staged(backend, initial, images, labels, *, stage, client_device, server_device, learning_rate):
    """One full real native CE/autograd/SGD batch through both native devices."""
    if stage not in (1, 2, 3) or backend not in {"pytorch", "tensorflow", "paddle", "jax"}:
        raise ValueError("Native reference supports the three real residual-stage endpoints")
    from splitfleet.tasks.losses import sparse_cross_entropy
    from splitfleet.virtual.backend_ops import FrameworkOps

    models = {"prefix": build_model(backend, initial, client_device),
              "suffix": build_model(backend, initial, server_device)}
    devices = {"prefix": client_device, "suffix": server_device}
    paddle_composition = backend == "paddle" and client_device != server_device
    native_composition = paddle_composition or backend == "tensorflow"
    native_boundary = {}
    parameters = {owner: native_parameters(backend, model) for owner, model in models.items()}
    for owner, model in models.items():
        training_mode(backend, model)
        FrameworkOps({"tensorflow": "tf", "pytorch": "torch"}.get(backend, backend)).assert_model_device(model, devices[owner])

    def forward(prefix_params=None, suffix_params=None):
        with _context(backend, client_device):
            activation, prefix_statistics = _forward(backend, models["prefix"], images, range(1, stage + 1),
                stem=True, head=False, params=prefix_params)
        with _context(backend, server_device):
            if paddle_composition:
                native_boundary["prefix"] = activation
            activation = _copy(backend, activation, server_device)
            if paddle_composition:
                # Paddle's public cross-place Tensor.to does not join the two
                # autograd graphs. Compose their native VJPs, using the suffix
                # gradient produced by real CE differentiation as the seed.
                activation = activation.detach()
                activation.stop_gradient = False
                activation.retain_grads()
                native_boundary["suffix"] = activation
            targets = _copy(backend, labels, server_device)
            logits, suffix_statistics = _forward(backend, models["suffix"], activation, range(stage + 1, 5),
                stem=False, head=True, params=suffix_params)
            loss = sparse_cross_entropy(logits, targets)
        return loss, (logits, prefix_statistics, suffix_statistics)

    optimizers = {}
    if backend == "jax":
        import jax
        (loss, auxiliary), native_gradients = jax.value_and_grad(forward, argnums=(0, 1), has_aux=True)(
            models["prefix"].params, models["suffix"].params)
        gradients_by_owner = dict(zip(("prefix", "suffix"), native_gradients, strict=True))
    elif backend == "tensorflow":
        import tensorflow as tf
        # Eager TF gradient kernels use the current device scope. Separate
        # native tapes preserve both stages' actual backward placement too.
        # The prefix seed is the real suffix CE activation VJP, never a saved
        # reference gradient or an independently generated tangent.
        owned = {owner: {name: value for name, value in parameters[owner].items()
                         if stage_owner(name, stage) == owner} for owner in models}
        with _context(backend, client_device), tf.GradientTape() as prefix_tape:
            activation, prefix_statistics = _forward(backend, models["prefix"], images, range(1, stage + 1),
                stem=True, head=False)
        with _context(backend, server_device):
            boundary = tf.stop_gradient(tf.identity(activation))
            targets = tf.identity(labels)
            with tf.GradientTape() as suffix_tape:
                suffix_tape.watch(boundary)
                logits, suffix_statistics = _forward(backend, models["suffix"], boundary, range(stage + 1, 5),
                    stem=False, head=True)
                loss = sparse_cross_entropy(logits, targets)
            suffix_values = suffix_tape.gradient(loss, [boundary, *owned["suffix"].values()])
        if suffix_values[0] is None:
            raise RuntimeError("Native TensorFlow suffix CE produced no activation VJP")
        with _context(backend, client_device):
            activation_gradient = tf.identity(suffix_values[0])
            prefix_values = prefix_tape.gradient(activation, list(owned["prefix"].values()),
                                                 output_gradients=activation_gradient)
        gradients_by_owner = {
            "prefix": dict(zip(owned["prefix"], prefix_values, strict=True)),
            "suffix": dict(zip(owned["suffix"], suffix_values[1:], strict=True)),
        }
        auxiliary = (logits, prefix_statistics, suffix_statistics)
        from splitfleet.backends.utils import tensorflow_tensor_device
        if any(value is not None and tensorflow_tensor_device(value) != devices[owner]
               for owner, values in gradients_by_owner.items() for value in values.values()):
            raise RuntimeError("Native TensorFlow stage gradients are outside their actual backward device")
    else:
        for owner, model in models.items():
            with _context(backend, devices[owner]):
                optimizers[owner] = optimizer(backend, model, learning_rate)
                optimizers[owner].zero_grad() if backend == "pytorch" else optimizers[owner].clear_grad()
        loss, auxiliary = forward()
        with _context(backend, server_device):
            loss.backward()
        if paddle_composition:
            derivative = native_boundary["suffix"].grad
            if derivative is None:
                raise RuntimeError("Native suffix CE produced no activation VJP")
            with _context(backend, client_device):
                native_boundary["prefix"].backward(_copy(backend, derivative, client_device))
        gradients_by_owner = {owner: {name: value.grad for name, value in parameters[owner].items()
                                     if stage_owner(name, stage) == owner} for owner in models}
    logits, prefix_statistics, suffix_statistics = auxiliary
    gradients = {name: gradients_by_owner[stage_owner(name, stage)].get(name) for name in TRAINABLE_NAMES}
    if any(value is None for value in gradients.values()):
        raise RuntimeError(f"Native staged autograd did not differentiate parameters: {[name for name, value in gradients.items() if value is None]}")
    gradient_arrays = {name: numpy_value(value) for name, value in gradients.items()}
    if backend == "tensorflow":
        rows = {row.name: row for row in SCHEMA}
        gradient_arrays = {name: value.transpose(3, 2, 0, 1) if len(rows[name].shape) == 4 else
                           value.T if name == "fc.weight" else value for name, value in gradient_arrays.items()}
    loss_value, logits_array = float(numpy_value(loss)), numpy_value(logits)
    for owner, model in models.items():
        with _context(backend, devices[owner]):
            if backend == "jax":
                model.set_params({name: value - np.float32(learning_rate) * gradients_by_owner[owner][name]
                                  if stage_owner(name, stage) == owner else value for name, value in model.params.items()})
            elif backend == "tensorflow":
                opt = optimizer(backend, model, learning_rate)
                opt.apply_gradients([(gradients_by_owner[owner][name], value) for name, value in parameters[owner].items()
                                     if stage_owner(name, stage) == owner])
            else:
                optimizers[owner].step()
            if backend != "pytorch":
                model.apply_statistics(prefix_statistics if owner == "prefix" else suffix_statistics)
    native_state = {owner: export_state(backend, model) for owner, model in models.items()}
    state = {name: native_state[stage_owner(name, stage)][name] for name in NAMES}
    counters = [name for name in NAMES if name.endswith("num_batches_tracked")]
    if len(counters) != 53 or any(int(state[name]) != int(initial[name]) + 1 for name in counters):
        raise RuntimeError("Native staged BN counters did not advance exactly once")
    return {"loss": loss_value, "logits": logits_array, "gradients": gradient_arrays, "state": state,
            "reference_kind": "native_staged_same_compute_placement",
            "native_devices": devices, "native_autograd_device_copy": not native_composition,
            "native_vjp_composition": native_composition, "batch_size": int(images.shape[0]),
            "executed_trainable_tensors": len(gradient_arrays), "canonical_state_tensors": len(state)}


def placement_reference(backend, initial, images, labels, *, stage, client_device, server_device,
                        learning_rate, full_reference, ownership):
    """Validate direct native stage execution, then change only its device copy."""
    from experiments.resnet50_quality.gate_torch import compare_tensor

    expected_owners = {name: stage_owner(name, stage) for name in NAMES}
    actual_owners = dict(zip(ownership["names"], ownership["owners"], strict=True))
    if actual_owners != expected_owners:
        raise RuntimeError("Captured split ownership differs from the independent native stage partition")
    control = train_staged(backend, initial, images, labels, stage=stage, client_device=client_device,
                           server_device=client_device, learning_rate=learning_rate)
    comparisons = {
        "logits": compare_tensor(control["logits"], full_reference["logits"], name="logits"),
        "ce": compare_tensor(np.asarray(control["loss"], np.float32), np.asarray(full_reference["loss"], np.float32), name="ce"),
        "gradients": {name: compare_tensor(control["gradients"][name], full_reference["gradients"][name], name=name)
                      for name in TRAINABLE_NAMES},
        "state": {name: compare_tensor(control["state"][name], full_reference["state"][name], name=name) for name in NAMES},
    }
    checks = [comparisons["logits"], comparisons["ce"], *comparisons["gradients"].values(), *comparisons["state"].values()]
    if not all(row["passed"] for row in checks):
        failed = [row["name"] for row in checks if not row["passed"]]
        raise RuntimeError(f"Same-device native stage control differs from the original full model: {failed}")
    reference = control if client_device == server_device else train_staged(backend, initial, images, labels,
        stage=stage, client_device=client_device, server_device=server_device, learning_rate=learning_rate)
    reference["same_device_native_control"] = {"status": "passed", "comparisons": comparisons}
    reference["full_client_device_reference_differences"] = {
        "scope": "Hardware-placement differences only; does not admit comparable final accuracy",
        "ce": compare_tensor(np.asarray(reference["loss"], np.float32), np.asarray(full_reference["loss"], np.float32), name="ce"),
        "gradients": {name: compare_tensor(reference["gradients"][name], full_reference["gradients"][name], name=name)
                      for name in TRAINABLE_NAMES},
    }
    return reference


def aggregate_native(updates):
    """Independent sample-weighted FP32 sum, with integer BN-counter maximum."""
    from experiments.resnet50_quality.model_contract import NAMES, validate_state
    total = sum(count for _, count in updates)
    if total <= 0:
        raise ValueError("Native reference needs all positive-weight local updates")
    result = {}
    for name in NAMES:
        values = [state[name] for state, _ in updates]
        if values[0].dtype.kind not in "fc":
            result[name] = np.maximum.reduce(values)
        else:
            weighted = [value * np.float32(count) for value, (_, count) in zip(values, updates, strict=True)]
            result[name] = np.add.reduce(weighted) / np.float32(total)
    validate_state(result)
    return result
