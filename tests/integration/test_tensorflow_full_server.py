"""TensorFlow full-server training must update both split segments."""

import numpy as np
import pytest

from tests.integration.test_torchlens_optional_frameworks import _run_isolated


@pytest.mark.parametrize("custom_optimizer", [False, True])
def test_tensorflow_full_server_optimizer_covers_both_segments(request, custom_optimizer):
    if _run_isolated(request, extra_env={"CUDA_VISIBLE_DEVICES": ""}):
        return
    tf = pytest.importorskip("tensorflow")
    from splitfleet.autosplit import AutoSplitSession
    from splitfleet.server.server_model.autosplit_server_model import AutoSplitServerModel
    from splitfleet.server.stage_runtime.manager import StageRuntimeManager
    from splitfleet.tasks import ModelInputs, TaskBatch
    from splitfleet.tasks.transport import encode_task_batch

    model = tf.keras.Sequential([
        tf.keras.layers.Input((4,)),
        tf.keras.layers.Dense(3, activation="relu", kernel_initializer="ones"),
        tf.keras.layers.Dense(2, kernel_initializer="ones"),
    ])
    inputs, targets = tf.ones((2, 4)), tf.zeros((2, 2))
    loss_fn = tf.keras.losses.MeanSquaredError()
    initial = model.get_weights()
    session = AutoSplitSession()
    handle = session.prepare_runtime(model, (inputs,), boundary="50%", dynamic_batch=(2, 2))
    manager = StageRuntimeManager(autosplit_session=session)
    manager.bind_runtime_handle(handle)
    learning_rate = .03 if custom_optimizer else .01
    factory = (lambda _: tf.keras.optimizers.SGD(learning_rate)) if custom_optimizer else None
    server = AutoSplitServerModel(runtime_manager=manager, model=model,
                                  loss_fn=loss_fn, optimizer_fn=factory)
    server.configure_fit(initial, {})
    payload = encode_task_batch(TaskBatch(ModelInputs((inputs,)), targets), backend="tf")
    response, = server.train_task_batch([payload])

    with tf.GradientTape() as tape:
        expected_loss = loss_fn(model(inputs), targets)
    gradients = tape.gradient(expected_loss, model.trainable_variables)
    tf.keras.optimizers.SGD(learning_rate).apply_gradients(zip(gradients, model.trainable_variables))
    np.testing.assert_allclose(response["loss"], [expected_loss.numpy()], rtol=1e-5)
    actual = server.get_parameters()
    for before, expected, updated in zip(initial, model.get_weights(), actual, strict=True):
        assert not np.array_equal(before, updated)
        np.testing.assert_allclose(updated, expected, rtol=1e-5, atol=1e-6)
