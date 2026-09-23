"""Split residual training must update every parameter like native tinygrad SGD."""

import numpy as np
import pytest

from tests.integration.test_resnet18_all_backends_all_nodes import _build_tinygrad_resnet18
from tests.integration.test_torchlens_optional_frameworks import _run_isolated


@pytest.mark.parametrize("view", ["reshape", "transpose", "slice"])
def test_tinygrad_parameter_view_replay_preserves_values_after_state_load(request, view):
    if _run_isolated(request, extra_env={"DEV": "CPU", "DEBUG": "0"}):
        return
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor

    from splitfleet.autosplit import prepare_torchlens_runtime
    from splitfleet.backends.utils import adapter_for

    class Model:
        def __init__(self):
            if view == "reshape":
                self.weight = Tensor([1., 2., 3., 4.]).reshape(2, 2)
            elif view == "transpose":
                self.weight = Tensor([[1., 2.], [3., 4.]]).transpose(0, 1)
            else:
                self.weight = Tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])[:2, :2]
            self.weight.requires_grad = True

        def __call__(self, inputs):
            return (inputs @ self.weight).relu()

    model = Model()
    inputs = Tensor([[2., 3.]])
    adapter = adapter_for(model, inputs)
    runtime = prepare_torchlens_runtime(
        model, inputs, boundary="50%", trainable=True,
        dynamic_batch=(1, 1), batch_axes={},
    )
    for replacement in (None, np.array([[.75, 1.], [.75, 1.]], dtype=np.float32)):
        if replacement is not None:
            adapter.load_ndarrays(model, [replacement])
        expected = model(inputs).numpy().copy()
        for training in (False, True):
            boundary = runtime.backend.run_prefix(inputs, training=training)
            actual = runtime.backend.run_suffix(boundary).numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_tinygrad_residual_split_update_matches_native_sgd(request):
    if _run_isolated(request, extra_env={"DEV": "CPU", "DEBUG": "0"}):
        return
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor
    from tinygrad.nn.optim import SGD
    from tinygrad.nn.state import get_state_dict

    from splitfleet.autosplit import prepare_torchlens_runtime
    from splitfleet.backends.utils import adapter_for
    from splitfleet.split_engine import graph_contract_for_runtime_handle
    from splitfleet.transport.split_wire import boundary_to_envelope, envelope_to_boundary

    learning_rate = 1e-4
    native_model, native_inputs, native_targets, loss_fn = _build_tinygrad_resnet18()
    native_parameters = get_state_dict(native_model)
    assert len(native_parameters) == 18
    native_optimizer = SGD(list(native_parameters.values()), lr=learning_rate)
    with Tensor.train():
        native_optimizer.zero_grad()
        native_loss = loss_fn(native_model(native_inputs), native_targets[0])
        native_loss_value = native_loss.item()
        native_loss.backward()
        native_gradients = {
            name: parameter.grad.numpy().copy()
            for name, parameter in native_parameters.items()
        }
        native_optimizer.step()
    expected_parameters = {
        name: parameter.numpy().copy()
        for name, parameter in native_parameters.items()
    }

    model, inputs, targets, loss_fn = _build_tinygrad_resnet18()
    adapter = adapter_for(model, inputs)
    parameters = get_state_dict(model)
    initial_parameters = {
        name: parameter.numpy().copy() for name, parameter in parameters.items()
    }
    runtime = prepare_torchlens_runtime(
        model, inputs, boundary="50%", trainable=True,
        dynamic_batch=(2, 2), batch_axes={},
    )
    split_gradients = {}
    wrapped_optimizer = adapter.build_optimizer(model, {"name": "sgd", "lr": learning_rate})

    class RecordingOptimizer:
        def zero_grad(self):
            return wrapped_optimizer.zero_grad()

        def step(self):
            # tinygrad gradients are lazy: retain their values before parameter
            # assignment can change buffers referenced by their expression.
            for name, parameter in parameters.items():
                if parameter.grad is not None:
                    gradient = parameter.grad.numpy().copy()
                    split_gradients[name] = split_gradients.get(name, 0) + gradient
            return wrapped_optimizer.step()

    optimizer = RecordingOptimizer()
    boundary = runtime.backend.run_prefix(inputs, training=True)
    contract = graph_contract_for_runtime_handle(runtime)
    envelope = boundary_to_envelope(
        boundary, round_id=1, client_id="residual-sgd", step_id="1",
        plan_id=runtime.plan.plan_id, split_id=contract.split_id,
        canonical_graph_hash=contract.canonical_graph_hash,
        boundary_schema_hash=contract.boundary_schema_hash, model_version=1,
    )
    remote_boundary = envelope_to_boundary(envelope, runtime.runtime, None)
    split_loss_values = []

    def split_loss(output, target):
        value = loss_fn(output, target)
        split_loss_values.append(value.item())
        return value

    loss, boundary_gradients = runtime.backend.train_suffix(
        remote_boundary, targets[0], loss_fn=split_loss, optimizer=optimizer,
    )
    assert boundary_gradients
    runtime.backend.backward_prefix(boundary, boundary_gradients, optimizer=optimizer)

    assert np.isfinite(adapter.scalar_value(loss))
    np.testing.assert_allclose(split_loss_values, [native_loss_value], rtol=1e-6)
    assert split_gradients.keys() == native_gradients.keys()
    for name, parameter in parameters.items():
        np.testing.assert_allclose(
            split_gradients[name], native_gradients[name], rtol=2e-5, atol=2e-6,
            err_msg=f"split gradient differs for {name}",
        )
        actual = parameter.numpy()
        assert not np.array_equal(actual, initial_parameters[name]), name
        np.testing.assert_allclose(
            actual, expected_parameters[name], rtol=1e-6, atol=1e-7,
            err_msg=f"split SGD update differs for {name}",
        )
