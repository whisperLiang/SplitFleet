"""Fixed captures must avoid dynamic graph walks while dynamic replay still resizes."""

import numpy as np
import pytest

from tests.integration.test_torchlens_optional_frameworks import _run_isolated


def test_fixed_capture_skips_shape_graph_walks_and_checks_input_shapes(request, monkeypatch):
    if _run_isolated(request, extra_env={"DEV": "CPU", "DEBUG": "0"}):
        return
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor
    from torchlens.split.adapters.tinygrad import _TinygradGeneratedSegmentBase
    from torchlens.split.errors import SplitBoundaryError

    from splitfleet.autosplit import prepare_torchlens_runtime

    class Model:
        def __call__(self, inputs):
            return inputs.reshape(2, 4).relu() * 2

    model = Model()
    inputs = Tensor.arange(8).reshape(2, 4).float()
    handle = prepare_torchlens_runtime(
        model, inputs, boundary="50%", trainable=False,
        batch_axes={}, dynamic_batch=(1, 4),
    )

    def unexpected_shape_walk(*args, **kwargs):
        raise AssertionError("fixed capture traversed a UOp graph for dynamic shapes")

    monkeypatch.setattr(
        _TinygradGeneratedSegmentBase, "_rewrite_shape_uop_tree", unexpected_shape_walk,
    )
    monkeypatch.setattr(
        _TinygradGeneratedSegmentBase, "_rewrite_parameter_expand_tree", unexpected_shape_walk,
    )

    actual = handle.backend.run_suffix(handle.backend.run_prefix(inputs))
    np.testing.assert_array_equal(actual.numpy(), model(inputs).numpy())
    with pytest.raises(SplitBoundaryError, match="non-batch input.*shape changed"):
        handle.backend.run_prefix(Tensor.zeros(2, 5))


def test_dynamic_capture_rewrites_reshape_for_new_batch_size(request):
    if _run_isolated(request, extra_env={"DEV": "CPU", "DEBUG": "0"}):
        return
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor

    from splitfleet.autosplit import prepare_torchlens_runtime

    class Model:
        def __call__(self, inputs):
            return inputs.reshape(inputs.shape[0], 2, 2).relu() * 2

    model = Model()
    handle = prepare_torchlens_runtime(
        model, Tensor.ones(2, 4), boundary="50%", trainable=False,
        batch_axes={"/args/0": 0}, dynamic_batch=(1, 4),
    )
    inputs = Tensor.arange(12).reshape(3, 4).float()
    actual = handle.backend.run_suffix(handle.backend.run_prefix(inputs))

    assert actual.shape == (3, 2, 2)
    np.testing.assert_array_equal(actual.numpy(), model(inputs).numpy())
