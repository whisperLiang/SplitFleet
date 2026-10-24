from __future__ import annotations

import numpy as np
import pytest

from splitfleet.backends import BACKEND_ADAPTERS
from splitfleet.backends.utils import detect_torchlens_backend, move_value


@pytest.mark.parametrize(
    "name,canonical",
    [("tf", "tf"), ("tensorflow", "tf"), ("paddle", "paddle")],
)
def test_optional_backend_adapters_are_registered_lazily(name: str, canonical: str) -> None:
    assert BACKEND_ADAPTERS.create(name).backend_name == canonical


@pytest.mark.parametrize(
    "module,expected",
    [("tensorflow.python", "tf"), ("paddle.nn", "paddle")],
)
def test_torchlens_backend_detection_does_not_import_optional_frameworks(module: str, expected: str) -> None:
    model_type = type("Model", (), {"__module__": module})
    assert detect_torchlens_backend(model_type(), ()) == expected


def test_backend_detection_uses_custom_model_base_class() -> None:
    torch = pytest.importorskip("torch")

    class CustomModel(torch.nn.Module):
        pass

    assert detect_torchlens_backend(CustomModel(), ()) == "torch"


def test_move_numpy_value_to_torch_tensor() -> None:
    torch = pytest.importorskip("torch")
    adapter = BACKEND_ADAPTERS.create("torch")

    result = move_value(np.ones((2, 3), dtype=np.float32), adapter, "cpu")

    assert isinstance(result, torch.Tensor)
    assert result.device.type == "cpu"
    assert tuple(result.shape) == (2, 3)


def test_tensorflow_state_preserves_duplicate_leaf_names() -> None:
    tf = pytest.importorskip("tensorflow")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(4),
        tf.keras.layers.Dense(2),
    ])
    adapter = BACKEND_ADAPTERS.create("tf")

    state = adapter.export_ndarrays(model)

    assert len(state) == len(model.weights) == 4
    replacement = [np.full_like(value, index + 1) for index, value in enumerate(state)]
    adapter.load_ndarrays(model, replacement)
    for actual, expected in zip(adapter.export_ndarrays(model), replacement):
        np.testing.assert_array_equal(actual, expected)


def test_tensorflow_training_mode_reaches_keras_layers() -> None:
    tf = pytest.importorskip("tensorflow")
    tf.random.set_seed(7)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64,)),
        tf.keras.layers.Dropout(0.5),
    ])
    adapter = BACKEND_ADAPTERS.create("tf")
    inputs = tf.ones((8, 64))

    adapter.set_training(model, False)
    evaluation = model(inputs)
    adapter.set_training(model, True)
    training = model(inputs)

    np.testing.assert_array_equal(evaluation.numpy(), np.ones((8, 64), dtype=np.float32))
    assert not np.array_equal(training.numpy(), evaluation.numpy())
