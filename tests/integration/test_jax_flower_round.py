"""Exercise functional JAX updates through the Flower prefix/suffix round."""

from __future__ import annotations

import numpy as np
import pytest

from tests.integration.test_torchlens_optional_frameworks import _run_isolated


def test_jax_flower_round_matches_native_sgd(request) -> None:
    if _run_isolated(request, extra_env={"CUDA_VISIBLE_DEVICES": ""}):
        return
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
    from splitfleet.common import ServerModelFitIns
    from splitfleet.server.stage_runtime.manager import StageRuntimeManager
    from splitfleet.server.strategy import AutoSplitStrategy
    from splitfleet.tasks import ImageClassificationTask, ModelInputs, TaskBatch
    from tests.test_flower_autosplit_round_loop import InProcessServerModelProxy

    params = {
        "a": jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 20,
        "b": jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 15,
    }

    def model(parameters, inputs):
        return jax.nn.relu(inputs @ parameters["a"]) @ parameters["b"]

    inputs_and_targets = [
        (jnp.ones((2, 4)), jnp.array([0, 2])),
        (jnp.arange(8, dtype=jnp.float32).reshape(2, 4) / 8, jnp.array([1, 0])),
    ]
    expected = dict(params)
    expected_losses = []
    for inputs, targets in inputs_and_targets:
        def loss_fn(parameters):
            log_probs = jax.nn.log_softmax(model(parameters, inputs))
            return -jnp.mean(log_probs[jnp.arange(targets.shape[0]), targets])

        loss, gradients = jax.value_and_grad(loss_fn)(expected)
        expected_losses.append(float(loss))
        expected = jax.tree_util.tree_map(
            lambda value, gradient: value - 0.01 * gradient, expected, gradients
        )

    task = ImageClassificationTask()
    batches = [
        TaskBatch(ModelInputs((params, inputs)), targets, num_examples=2)
        for inputs, targets in inputs_and_targets
    ]
    strategy = AutoSplitStrategy(
        model=model,
        sample_inputs=batches[0],
        task=task,
        batch_axes={"/args/1": 0},
    )
    manager = StageRuntimeManager(autosplit_session=strategy.autosplit_session)
    strategy.bind_stage_runtime_manager(manager)
    config = strategy._autosplit_config(1)
    initial_parameters = strategy.initialize_server_parameters()
    server = strategy._make_server_model()
    server.configure_fit(
        ServerModelFitIns(parameters=initial_parameters, config=config, sid="")
    )
    client = AutoSplitSplitLearningClient(
        model=model,
        sample_inputs=batches[0],
        train_data=batches,
        task=task,
        batch_axes={"/args/1": 0},
        functional_update_fn=lambda parameters, result: jax.tree_util.tree_map(
            lambda value, gradient: value - 0.01 * gradient,
            parameters,
            result["inputs"][0],
        ),
    )
    client.server_model_proxy = InProcessServerModelProxy(server_model=server)

    updated_parameters, num_examples, metrics = client.fit(initial_parameters, config)
    server_result = server.get_fit_result()

    assert num_examples == server_result.config["num_examples"] == 4
    assert metrics["loss"] == pytest.approx(np.mean(expected_losses), rel=1e-5)
    assert server_result.config["avg_loss"] == pytest.approx(metrics["loss"])
    for initial, actual, reference in zip(
        initial_parameters, updated_parameters, jax.tree_util.tree_leaves(expected)
    ):
        assert not np.allclose(initial, reference)
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
