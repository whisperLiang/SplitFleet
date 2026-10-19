"""Stateless Paddle replay follows actual inputs through CPU-default facades."""

import os

import numpy as np
import pytest

from tests.integration.test_torchlens_optional_frameworks import _run_isolated


@pytest.mark.integration
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stateless_paddle_models_clone_load_and_replay_on_input_device(request, device):
    options = {"CUDA_VISIBLE_DEVICES": ""} if device == "cpu" else {}
    if _run_isolated(request, extra_env=options, python=os.environ.get("SPLITFLEET_PADDLE_PYTHON")):
        return
    paddle = pytest.importorskip("paddle")
    if device.startswith("cuda") and (
        not paddle.is_compiled_with_cuda() or not paddle.device.cuda.device_count()
    ):
        pytest.skip("A native Paddle GPU is unavailable")

    from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend
    from splitfleet.backends import BACKEND_ADAPTERS
    from splitfleet.backends.utils import (
        native_device_scope,
        paddle_model_device,
        paddle_tensor_device,
    )

    paddle.set_device("cpu")
    with native_device_scope("paddle", device):
        inputs = paddle.arange(12, dtype="float32").reshape([3, 4]) - 4

    class Stateless(paddle.nn.Layer):
        def forward(self, value):
            return paddle.nn.functional.relu(value * 2) + 1

    adapter = BACKEND_ADAPTERS.create("paddle")
    model = adapter.clone_model(Stateless())
    assert adapter.export_ndarrays(model) == []
    adapter.load_ndarrays(model, [])
    assert paddle_model_device(model) == "cpu"
    for model in (model, lambda value: paddle.nn.functional.relu(value * 2) + 1):
        model = adapter.clone_model(model)
        assert paddle_model_device(model, ((inputs,), {})) == device
        backend = TorchLensSplitBackend(device="cpu").trace(
            model, (inputs,), boundary="50%", trainable=False,
            batch_axes={"/args/0": 0}, dynamic_batch=(1, 4),
        )
        assert str(backend.device) == device
        assert backend.runtime.batch_validation["status"] == "passed"
        with native_device_scope("paddle", device):
            for batch_size in (3, 2):
                batch = inputs[:batch_size]
                output = backend.run_suffix(backend.run_prefix(batch))
                assert paddle_tensor_device(output) == device
                np.testing.assert_array_equal(output.numpy(), model(batch).numpy())
        if device != "cpu":
            with pytest.raises(ValueError, match="stateless Paddle runtime"):
                paddle_model_device(model, (inputs, paddle.ones([1])))
    assert paddle.get_device() == "cpu"
