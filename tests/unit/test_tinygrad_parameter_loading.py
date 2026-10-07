"""Round weight loads must preserve graph identity and independent replicas."""

import numpy as np
import pytest

from tests.integration.test_torchlens_optional_frameworks import _run_isolated


def _model():
    from tinygrad import Tensor

    class Model:
        def __init__(self):
            self.weight = Tensor(np.array([2., 3.], np.float32), device="CPU").realize()
            self.weight.requires_grad = True
            self.tied_weight = self.weight

        def __call__(self, inputs):
            return (inputs * self.weight).relu() + .25

    return Model()


def test_weight_reload_preserves_named_cut_and_live_replay(request):
    if _run_isolated(request, extra_env={"DEV": "CPU", "DEBUG": "0"}):
        return
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor
    from splitfleet.autosplit import prepare_torchlens_runtime
    from splitfleet.backends import BACKEND_ADAPTERS

    model = _model()
    inputs = Tensor(np.array([[1., -2.], [3., 4.]], np.float32), device="CPU").realize()
    adapter = BACKEND_ADAPTERS.create("tinygrad")
    options = {"trainable": True, "batch_axes": {}, "dynamic_batch": (2, 2)}
    before = prepare_torchlens_runtime(model, inputs, boundary="50%", **options)
    replacement = np.array([5., 6.], np.float32)
    adapter.load_ndarrays(model, [replacement, replacement])
    # Reuse the actual named cut, as the server does for a suffix replica.
    after = prepare_torchlens_runtime(model, inputs, boundary=before.plan.boundary, **options)
    assert [(n.canonical_id, n.op_type) for n in before.runtime.trace_graph.nodes] == [
        (n.canonical_id, n.op_type) for n in after.runtime.trace_graph.nodes
    ]
    for handle in (before, after):
        replay = handle.backend.run_suffix(handle.backend.run_prefix(inputs))
        np.testing.assert_allclose(replay.numpy(), model(inputs).numpy())
    assert model.weight.requires_grad is True
    assert model.weight is model.tied_weight


def test_replica_sgd_updates_independent_storage_and_retains_tied_parameters(request):
    if _run_isolated(request, extra_env={"DEV": "CPU", "DEBUG": "0"}):
        return
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor
    from splitfleet.backends import BACKEND_ADAPTERS

    Tensor.training = True
    model = _model()
    adapter = BACKEND_ADAPTERS.create("tinygrad")
    initial = adapter.export_ndarrays(model)
    replica = adapter.clone_model(model)
    assert replica.weight is replica.tied_weight
    assert replica.weight is not model.weight
    assert adapter.state_manifest(replica) == adapter.state_manifest(model)
    opt = adapter.build_optimizer(replica, {"name": "sgd", "lr": .05})
    opt.zero_grad()
    inputs = Tensor(np.array([[1., 2.], [3., 4.]], np.float32), device="CPU").realize()
    replica(inputs).sum().backward()
    opt.step()
    for original, expected in zip(adapter.export_ndarrays(model), initial, strict=True):
        np.testing.assert_array_equal(original, expected)
    for updated, expected in zip(adapter.export_ndarrays(replica), initial, strict=True):
        assert not np.array_equal(updated, expected)
