"""Native full-model SGD controls for framework regression tests."""

from __future__ import annotations

import importlib
import os

import numpy as np

from splitfleet.tasks.losses import extract_logits, sparse_cross_entropy

from experiments.resnet50_quality.model_contract import TRAINABLE_NAMES

BACKENDS = {name: None for name in ("pytorch", "tensorflow", "paddle", "jax")}
MODULES = {"tensorflow": "tf", "pytorch": "torch"}


def configure_jax_environment(environment=None):
    """Freeze the admitted JAX flags before any native device initialization."""
    environment = os.environ if environment is None else environment
    flags = environment.get("XLA_FLAGS", "").split()
    for option in ("xla_gpu_deterministic_ops", "xla_gpu_exclude_nondeterministic_ops"):
        expected = f"--{option}=true"
        present = [value for value in flags if value.startswith(f"--{option}=")]
        if present and present != [expected]:
            raise ValueError(f"The admitted JAX numerical gate requires {expected}")
        if not present:
            flags.append(expected)
    environment["XLA_FLAGS"] = " ".join(flags)
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return environment


def configure_backend(backend):
    if backend == "pytorch":
        import torch
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    elif backend == "tensorflow":
        import tensorflow as tf
        tf.config.set_soft_device_placement(False)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.random.set_seed(20261014)
        tf.config.experimental.enable_op_determinism()
        tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
        for device in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(device, True)
    elif backend == "jax":
        configure_jax_environment()
        import jax
        jax.config.update("jax_default_matmul_precision", "highest")
    elif backend == "paddle":
        import faulthandler
        import paddle
        # Paddle's C++ signal handler replaced PYTHONFAULTHANDLER and emitted
        # no Python stack in a full-shard worker failure. Keep the crashing
        # native process's Python frames available before failed-output cleanup.
        faulthandler.disable()
        paddle.disable_signal_handler()
        faulthandler.enable(file=2, all_threads=True)
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({"FLAGS_cudnn_deterministic": True,
                              "FLAGS_enable_cublas_tensor_op_math": False,
                              "FLAGS_cudnn_exhaustive_search": False})


def model_module(backend):
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported experiment backend: {backend}")
    return importlib.import_module(f"tests.integration.resnet50_reference.{MODULES.get(backend, backend)}_model")


def build_model(backend, state, device):
    model = model_module(backend).build_model(state, device)
    if backend in {"tensorflow", "paddle"}:
        model.defer_statistics = True
    if backend == 'tensorflow':
        model._splitfleet_tensorflow_function_forward = True
    return model


def export_state(backend, model):
    return model_module(backend).export_state(model)


def numpy_value(value):
    if type(value).__module__.startswith("paddle"):
        from splitfleet.backends.utils import native_device_scope, paddle_tensor_device
        with native_device_scope("paddle", paddle_tensor_device(value)):
            return np.asarray(value.detach().cpu().numpy()).copy()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value).copy()


def training_mode(backend, model, training=True):
    if backend == "tensorflow":
        model.training = training
    else:
        model.train(training)


def native_batch(backend, images, labels, device):
    if backend == "pytorch":
        import torch
        return torch.as_tensor(images, device=device), torch.as_tensor(labels, device=device)
    if backend == "tensorflow":
        import tensorflow as tf
        native_device = "/GPU:" + device.split(":")[-1] if device.startswith("cuda") else "/CPU:0"
        with tf.device(native_device):
            return tf.convert_to_tensor(images.transpose(0, 2, 3, 1)), tf.convert_to_tensor(labels)
    if backend == "paddle":
        import paddle
        place = paddle.CUDAPlace(int(device.split(":")[-1])) if device.startswith("cuda") else paddle.CPUPlace()
        return paddle.to_tensor(images, place=place), paddle.to_tensor(labels, place=place)
    if backend == "jax":
        import jax
        resolved = jax.devices("gpu" if device.startswith("cuda") else "cpu")[int(device.split(":")[-1]) if ":" in device else 0]
        return jax.device_put(images, resolved), jax.device_put(labels, resolved)
    raise ValueError(backend)


def call_inputs(backend, model, images):
    return (model.params, images) if backend == "jax" else (images,)


def native_parameters(backend, model):
    if backend == "pytorch":
        return dict(model.named_parameters())
    if backend == "tensorflow":
        return {name: value for name, value in model.canonical_variables().items() if name in TRAINABLE_NAMES}
    if backend == "paddle":
        return dict(model.named_parameters())
    if backend == "jax":
        return dict(model.params)
    raise ValueError(backend)


def optimizer(backend, model, learning_rate):
    if backend == "pytorch":
        import torch
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0, weight_decay=0)
    if backend == "tensorflow":
        import tensorflow as tf
        from splitfleet.runtime.tensorflow_updates import native_optimizer

        from splitfleet.backends.utils import (
            native_device_scope,
            tensorflow_model_device,
        )
        with native_device_scope("tf", tensorflow_model_device(model)):
            value = native_optimizer(tf.keras.optimizers.SGD(learning_rate=learning_rate), model)
            value.build(model.trainable_variables)
        return value
    if backend == "paddle":
        import paddle
        return paddle.optimizer.SGD(learning_rate=learning_rate, parameters=model.parameters())
    if backend == "jax":
        from splitfleet.runtime.functional_sgd import OwnedFunctionalSGD
        return OwnedFunctionalSGD(model, learning_rate)
    raise ValueError(backend)


def train_step(backend, model, images, labels, *, learning_rate=.05, native_optimizer=None, capture=False):
    from splitfleet.backends.utils import (
        native_device_scope,
        paddle_model_device,
        tensorflow_model_device,
    )
    device = (paddle_model_device(model) if backend == "paddle" else
              tensorflow_model_device(model) if backend == "tensorflow" else "cpu")
    with native_device_scope(backend, device):
        return _train_step(backend, model, images, labels, learning_rate=learning_rate,
                           native_optimizer=native_optimizer, capture=capture)


def _train_step(backend, model, images, labels, *, learning_rate=.05, native_optimizer=None, capture=False):
    """One actual CE/SGD step; optional snapshots are only for numerical gates."""
    training_mode(backend, model)
    parameters = native_parameters(backend, model)
    opt = native_optimizer if native_optimizer is not None else optimizer(backend, model, learning_rate)
    if backend == "tensorflow":
        import tensorflow as tf
        if getattr(model, '_splitfleet_tensorflow_function_forward', False):
            from splitfleet.runtime.tensorflow_forward import native_training_gradients
            output, loss, values = native_training_gradients(model, images, labels, parameters.values())
        else:
            with tf.GradientTape() as tape:
                output = model(images, training=True)
                loss = sparse_cross_entropy(output["logits"], labels)
            values = tape.gradient(loss, list(parameters.values()))
        grads = dict(zip(parameters, values, strict=True))
    elif backend == "jax":
        import jax
        def objective(params):
            output = model(params, images)
            return sparse_cross_entropy(output["logits"], labels), output
        (loss, output), grads = jax.value_and_grad(objective, has_aux=True)(model.params)
    else:
        opt.zero_grad() if backend == "pytorch" else opt.clear_grad()
        output = model(images)
        loss = sparse_cross_entropy(extract_logits(output), labels)
        loss.backward()
        grads = {name: value.grad for name, value in parameters.items()}
    if any(value is None for value in grads.values()):
        raise RuntimeError(f"Native full-model training is missing gradients: {[name for name, value in grads.items() if value is None]}")
    loss_value = float(numpy_value(loss))
    snapshots = {name: numpy_value(value) for name, value in grads.items()} if capture else None
    logits = numpy_value(extract_logits(output)) if capture else None
    if backend == "jax":
        model.set_params({name: value - np.float32(learning_rate) * grads[name] for name, value in model.params.items()})
    elif backend == "tensorflow":
        opt.apply_gradients(zip(grads.values(), parameters.values(), strict=True))
    else:
        opt.step()
    if isinstance(output, dict) and "batch_statistics" in output:
        model.apply_statistics(output["batch_statistics"])
    return {"loss": loss_value, "logits": logits, "gradients": snapshots}
