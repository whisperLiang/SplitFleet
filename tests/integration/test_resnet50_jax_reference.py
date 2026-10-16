"""Canonical JAX ResNet50 structure, live state and native training checks."""

import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from tests.integration.resnet50_reference.jax_model import build_model, export_state
from experiments.resnet50_quality.model_contract import (
    NAMES,
    SCHEMA,
    TRAINABLE_NAMES,
    TRAINABLE_PARAMETERS,
    canonical_state,
    validate_state,
)


@pytest.fixture(scope="module")
def initial_state():
    return canonical_state(233)


def test_schema_roundtrip_and_live_parameter_inputs(initial_state):
    model = build_model(initial_state, device="cpu")
    assert tuple(model.state_tensors()) == NAMES
    assert set(model.params) == set(TRAINABLE_NAMES)
    assert set(model.params).isdisjoint(model.buffers)
    assert model.trainable_parameter_count == TRAINABLE_PARAMETERS == 23_528_522
    exported = export_state(model)
    for row in SCHEMA:
        np.testing.assert_array_equal(exported[row.name], initial_state[row.name])
        assert exported[row.name].shape == row.shape
        assert str(exported[row.name].dtype) == row.dtype
    images = jax.device_put(np.zeros((1, 3, 64, 64), dtype=np.float32), model.jax_device)
    bound_inputs = (model.params, images)
    before = model(*bound_inputs)["logits"]
    replacement = {**model.params, "fc.bias": model.params["fc.bias"] + np.float32(0.25)}
    model.set_params(replacement)
    assert bound_inputs[0] is model.params
    after = model(*bound_inputs)["logits"]
    np.testing.assert_allclose(np.asarray(after - before), np.full((1, 10), 0.25), rtol=0, atol=1e-7)
    assert after.dtype == jnp.float32
    assert model(*bound_inputs)["batch_statistics"] == {}


def test_batch_norm_matches_torch_and_commits_only_owned_statistics(initial_state):
    torch = pytest.importorskip("torch")
    model = build_model(initial_state, device="cpu")
    images = np.arange(2 * 64 * 2 * 2, dtype=np.float32).reshape(2, 64, 2, 2) / np.float32(100)
    native = jax.device_put(images, model.jax_device)
    stats = {}
    actual = model._batch_norm(native, model.params, "bn1", stats, training=True)
    running_mean = torch.zeros(64, dtype=torch.float32)
    running_var = torch.ones(64, dtype=torch.float32)
    expected = torch.nn.functional.batch_norm(
        torch.from_numpy(images), running_mean, running_var,
        torch.ones(64), torch.zeros(64), training=True, momentum=0.1, eps=1e-5,
    )
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=1e-6, atol=1e-6)
    assert int(model.buffers["bn1.num_batches_tracked"]) == 0
    np.testing.assert_array_equal(np.asarray(model.buffers["bn1.running_mean"]), np.zeros(64))
    np.testing.assert_array_equal(np.asarray(model.buffers["bn1.running_var"]), np.ones(64))
    assert int(stats["bn1"]["count"]) == 8
    assert stats["bn1"]["count"].dtype == jnp.int64
    np.testing.assert_allclose(np.asarray(stats["bn1"]["variance"]), images.var(axis=(0, 2, 3)), rtol=1e-6)
    short_stats = {}
    model._batch_norm(native[:1], model.params, "bn1", short_stats, training=True)
    assert int(short_stats["bn1"]["count"]) == 4
    np.testing.assert_allclose(np.asarray(short_stats["bn1"]["mean"]), images[:1].mean(axis=(0, 2, 3)), rtol=1e-6)

    stats["layer1.0.bn1"] = stats["bn1"]
    model.apply_statistics(stats, names=["layer1.0.bn1"])
    assert int(model.buffers["layer1.0.bn1.num_batches_tracked"]) == 1
    assert int(model.buffers["bn1.num_batches_tracked"]) == 0
    model.apply_statistics(stats, names=["bn1"])
    assert int(model.buffers["layer1.0.bn1.num_batches_tracked"]) == 1
    assert int(model.buffers["bn1.num_batches_tracked"]) == 1
    np.testing.assert_allclose(np.asarray(model.buffers["bn1.running_mean"]), running_mean.numpy(), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(model.buffers["bn1.running_var"]), running_var.numpy(), rtol=1e-6, atol=1e-7)
    evaluation = model._batch_norm(native, model.params, "bn1", {}, training=False)
    expected_eval = torch.nn.functional.batch_norm(
        torch.from_numpy(images), running_mean, running_var,
        torch.ones(64), torch.zeros(64), training=False, eps=1e-5,
    )
    np.testing.assert_allclose(np.asarray(evaluation), expected_eval.numpy(), rtol=1e-6, atol=1e-6)


def test_int64_counters_and_failed_statistics_are_atomic(initial_state):
    state = dict(initial_state)
    state["bn1.num_batches_tracked"] = np.asarray(2**40, dtype=np.int64)
    model = build_model(state, device="cpu")
    good = {"mean": np.zeros(64, dtype=np.float32), "variance": np.ones(64, dtype=np.float32), "count": 4}
    bad = {**good, "count": 1}
    with pytest.raises(ValueError, match="greater than one"):
        model.apply_statistics({"bn1": good, "layer1.0.bn1": bad})
    assert int(model.buffers["bn1.num_batches_tracked"]) == 2**40
    assert int(model.buffers["layer1.0.bn1.num_batches_tracked"]) == 0
    model.apply_statistics({"bn1": good})
    assert int(model.buffers["bn1.num_batches_tracked"]) == 2**40 + 1
    assert model.buffers["bn1.num_batches_tracked"].dtype == jnp.int64
    assert all(value.dtype == jnp.float32 for value in model.params.values())
    bound_params, bound_buffers = model.params, model.buffers
    model.load_state(initial_state)
    assert model.params is bound_params and model.buffers is bound_buffers
    assert int(model.buffers["bn1.num_batches_tracked"]) == 0
    with pytest.raises(ValueError, match="Unknown BatchNorm names"):
        model.apply_statistics({"bn1": good}, names=["missing_bn"])
    with pytest.raises(ValueError, match="Missing BatchNorm statistics"):
        model.apply_statistics({"bn1": good}, names=["layer1.0.bn1"])


def test_primitive_batch_norm_native_gradients_match_population_formula(initial_state):
    model = build_model(initial_state, device="cpu")
    random = np.random.default_rng(183)
    images = random.normal(size=(2, 64, 2, 2)).astype(np.float32)
    gamma = random.uniform(.5, 1.5, size=64).astype(np.float32)
    beta = random.normal(size=64).astype(np.float32)
    weights = random.normal(size=images.shape).astype(np.float32)

    def objective(value, scale, offset):
        output = model._batch_norm(value, {"bn1.weight": scale, "bn1.bias": offset},
                                   "bn1", {}, training=True)
        return jax.lax.reduce_sum(jax.lax.mul(output, jnp.asarray(weights)), (0, 1, 2, 3))

    actual = jax.grad(objective, argnums=(0, 1, 2))(
        jnp.asarray(images), jnp.asarray(gamma), jnp.asarray(beta))
    values, derivatives = images.astype(np.float64), weights.astype(np.float64)
    axes = (0, 2, 3)
    centered = values - values.mean(axis=axes)[None, :, None, None]
    deviation = np.sqrt((centered * centered).mean(axis=axes) + 1e-5)
    normalized = centered / deviation[None, :, None, None]
    samples = images.shape[0] * images.shape[2] * images.shape[3]
    offset_gradient = derivatives.sum(axis=axes)
    scale_gradient = (derivatives * normalized).sum(axis=axes)
    input_gradient = (gamma / deviation)[None, :, None, None] * (
        derivatives - offset_gradient[None, :, None, None] / samples
        - normalized * scale_gradient[None, :, None, None] / samples)
    for received, expected in zip(actual, (input_gradient, scale_gradient, offset_gradient), strict=True):
        assert received.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(received), expected, atol=1e-5, rtol=1e-4)
    assert int(model.buffers["bn1.num_batches_tracked"]) == 0


@pytest.mark.integration
def test_full_native_ce_sgd_and_bn_on_two_real_cifar_images(initial_state, device="cpu", compile_step=False):
    archive = Path(__file__).resolve().parents[2] / "data" / "cifar-10-python.tar.gz"
    if not archive.is_file():
        pytest.skip("The full native minibatch check requires the local CIFAR-10 archive")
    Image = pytest.importorskip("PIL.Image")
    with tarfile.open(archive, "r:gz") as handle:
        source = handle.extractfile("cifar-10-batches-py/data_batch_1")
        assert source is not None
        batch = pickle.load(source, encoding="bytes")
    original = batch[b"data"][:2].reshape(2, 3, 32, 32).transpose(0, 2, 3, 1)
    images = np.stack([
        np.asarray(Image.fromarray(image).resize((64, 64), Image.Resampling.BILINEAR))
        for image in original
    ]).transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255)
    labels = np.asarray(batch[b"labels"][:2], dtype=np.int32)
    model = build_model(initial_state, device=device).train()
    images = jax.device_put(images, model.jax_device)
    labels = jax.device_put(labels, model.jax_device)

    def value_and_statistics(params):
        output = model(params, images)
        log_probs = jax.nn.log_softmax(output["logits"], axis=1)
        ce = -jnp.take_along_axis(log_probs, labels[:, None], axis=1).mean(dtype=jnp.float32)
        return ce, output

    gradient_fn = jax.value_and_grad(value_and_statistics, has_aux=True)
    if compile_step:
        gradient_fn = jax.jit(gradient_fn)
    (loss, output), gradients = gradient_fn(model.params)
    loss.block_until_ready()
    assert np.isfinite(float(loss)) and float(loss) > 0
    assert output["logits"].shape == (2, 10)
    assert loss.dtype == output["logits"].dtype == jnp.float32
    assert set(output["batch_statistics"]) == set(model.batch_norm_names)
    assert len(model.batch_norm_names) == 53
    for name, width, spatial in (
        ("layer1.2.bn3", 256, 16), ("layer2.3.bn3", 512, 8),
        ("layer3.5.bn3", 1024, 4), ("layer4.2.bn3", 2048, 2),
    ):
        statistics = output["batch_statistics"][name]
        assert statistics["mean"].shape == (width,)
        assert int(statistics["count"]) == 2 * spatial**2
    assert set(gradients) == set(TRAINABLE_NAMES)
    for name, gradient in gradients.items():
        assert gradient.dtype == jnp.float32
        assert gradient.shape == model.params[name].shape
        assert np.isfinite(np.asarray(gradient)).all(), name
    assert all(int(model.buffers[name + ".num_batches_tracked"]) == 0 for name in model.batch_norm_names)
    before = {name: np.asarray(model.params[name]).copy() for name in ("conv1.weight", "fc.weight")}
    updated = jax.tree_util.tree_map(lambda parameter, gradient: parameter - np.float32(0.05) * gradient, model.params, gradients)
    model.set_params(updated)
    model.apply_statistics(output["batch_statistics"])
    for name, previous in before.items():
        assert np.any(np.asarray(model.params[name]) != previous), name
    assert all(int(model.buffers[name + ".num_batches_tracked"]) == 1 for name in model.batch_norm_names)
    final_state = export_state(model)
    validate_state(final_state)
    assert all(str(final_state[row.name].dtype) == row.dtype for row in SCHEMA)
    print(f"native JAX CE/SGD OK: device={device}, batch=2, loss={float(loss):.8f}, parameters={model.trainable_parameter_count}, BN layers={len(model.batch_norm_names)}")
