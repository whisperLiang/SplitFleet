"""Native state layouts, proven split ownership and exact integer aggregation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from splitfleet.autosplit import AutoSplitSession
from splitfleet.autosplit.state_ownership import (
    DIGEST_KEY,
    aggregate_owned,
    export_owned,
    ownership_digest,
    read_ownership,
    state_ownership,
)
from splitfleet.backends import BACKEND_ADAPTERS
from splitfleet.backends.utils import adapter_for
from tests.integration.test_torchlens_optional_frameworks import _run_isolated


def test_integer_maximum_is_exact_and_streaming_while_float_weights_are_averaged():
    initial = [np.asarray([0, 0], dtype=np.float32), np.asarray(2**60 + 100, dtype=np.int64)]
    manifest = {"names": ["weight", "counter"], "owners": ["prefix", "suffix"]}

    def updates():
        yield manifest, [np.asarray([1, 3], dtype=np.float32)], [np.asarray(2**60 + 1, dtype=np.int64)], 1
        yield manifest, [np.asarray([5, 7], dtype=np.float32)], [np.asarray(2**60 + 3, dtype=np.int64)], 3

    result = aggregate_owned(initial, updates(), integer_policy="maximum")
    np.testing.assert_array_equal(result[0], np.asarray([4, 6], dtype=np.float32))
    assert result[1].shape == () and result[1].dtype == np.int64
    assert result[1].item() == 2**60 + 3  # No float64 conversion; no initial maximum.
    with pytest.raises(ValueError, match="integer aggregation policy"):
        aggregate_owned(initial, updates(), integer_policy="unsupported")
    small = [np.zeros(2, np.float32), np.asarray(0, np.int64)]
    weighted = aggregate_owned(small, [
        (manifest, [np.ones(2, np.float32)], [np.asarray(1, np.int64)], 1),
        (manifest, [np.ones(2, np.float32)], [np.asarray(5, np.int64)], 3),
    ])
    assert weighted[1].item() == 4


def test_jax_ownership_uses_input_purpose_and_survives_parameter_replacement():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    device = jax.devices("cpu")[0]

    class Model:
        training = True

        def __init__(self):
            self.params = {
                "first": jax.device_put(np.ones((4, 4), np.float32), device),
                "second": jax.device_put(np.full((4, 2), 0.5, np.float32), device),
            }
            self.buffers = {"counter": jax.device_put(np.asarray(2**60 + 1, np.int64), device)}

        def __call__(self, params, images):
            return (images @ params["first"]) @ params["second"]

        def state_tensors(self):
            return {**self.params, **self.buffers}

        def state_associations(self):
            return {"counter": ("first",)}

        def load_state(self, state):
            self.params.update({name: jax.device_put(state[name], device) for name in self.params})
            self.buffers.update({name: jax.device_put(state[name], device) for name in self.buffers})

    model = Model()
    images = jax.device_put(np.ones((3, 4), np.float32), device)
    handle = AutoSplitSession().prepare_runtime(
        model, (model.params, images), boundary="50%", trainable=True,
        batch_axes={"/args/1": 0},
    )
    adapter = adapter_for(model, (model.params, images))
    schema = adapter.state_manifest(model).schema_hash
    before = state_ownership(handle, schema)
    assert dict(zip(before["names"], before["owners"])) == {
        "first": "prefix", "second": "suffix", "counter": "prefix",
    }
    # Every parameter input is physically before the first matmul in the native
    # plan; that source position must not assign the suffix matrix to prefix.
    assert set(handle.runtime.trace_graph.input_node_ids).issubset(handle.runtime.plan.prefix_node_ids)
    model.params.update(jax.tree.map(lambda value: value + np.float32(0.1), model.params))
    assert state_ownership(handle, schema) == before
    config = {"autosplit_state_ownership": json.dumps(before)}
    assert read_ownership(config, handle, schema) == before
    exported = export_owned(model, before, "prefix")
    assert exported[-1].shape == () and exported[-1].item() == 2**60 + 1
    arrays = adapter.export_ndarrays(model)
    arrays[-1] = np.asarray(2**60 + 7, dtype=np.int64)
    bound_params = model.params
    adapter.load_ndarrays(model, arrays)
    assert model.params is bound_params
    assert int(model.buffers["counter"]) == 2**60 + 7
    assert all(value.dtype == jnp.float32 for value in model.params.values())
    model.state_associations = None
    model.apply_statistics = lambda stats: None
    with pytest.raises(ValueError, match="no declared ownership association"):
        state_ownership(handle, schema)


def test_declared_buffer_ownership_rejects_ambiguous_trainable_anchors():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    class Model:
        training = True

        def __init__(self):
            self.params = {"first": jnp.ones((4, 4)), "second": jnp.ones((4, 2))}
            self.counter = jnp.asarray(0, dtype=jnp.int64)

        def __call__(self, params, images):
            return (images @ params["first"]) @ params["second"]

        def state_tensors(self):
            return {**self.params, "counter": self.counter}

        def state_associations(self):
            return {"counter": ("first", "second")}

    model = Model()
    images = jnp.ones((2, 4))
    handle = AutoSplitSession().prepare_runtime(model, (model.params, images), boundary="50%", trainable=True)
    adapter = adapter_for(model, (model.params, images))
    with pytest.raises(ValueError, match="ambiguous ownership"):
        state_ownership(handle, adapter.state_manifest(model).schema_hash)


def test_tensorflow_native_hook_preserves_hwio_and_scalar_int64_with_atomic_loads():
    tf = pytest.importorskip("tensorflow")

    class Model(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.kernel = self.add_weight(shape=(3, 3, 2, 4), dtype="float32")
            self.counter = self.add_weight(shape=(), dtype="int64", initializer="zeros", trainable=False)

        def canonical_variables(self):
            return {"conv.weight": self.kernel, "bn.num_batches_tracked": self.counter}

    model = Model()
    adapter = BACKEND_ADAPTERS.create("tf")
    manifest = adapter.state_manifest(model)
    assert [(row.name, row.shape, row.dtype) for row in manifest.entries] == [
        ("conv.weight", (3, 3, 2, 4), "float32"), ("bn.num_batches_tracked", (), "int64"),
    ]
    arrays = [np.ones((3, 3, 2, 4), np.float32), np.asarray(2**60 + 3, np.int64)]
    adapter.load_ndarrays(model, arrays)
    actual = adapter.export_ndarrays(model)
    assert actual[0].shape == (3, 3, 2, 4)
    assert actual[1].shape == () and actual[1].item() == 2**60 + 3
    broken = [np.zeros_like(arrays[0]), arrays[1].astype(np.float64)]
    with pytest.raises(ValueError, match="shape or dtype"):
        adapter.load_ndarrays(model, broken)
    np.testing.assert_array_equal(model.kernel.numpy(), arrays[0])
    broken = [np.full_like(arrays[0], np.nan), arrays[1]]
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        adapter.load_ndarrays(model, broken)
    np.testing.assert_array_equal(model.kernel.numpy(), arrays[0])
    legacy = tf.keras.Sequential([tf.keras.layers.Dense(2)])
    legacy.build((None, 4))
    assert all(row.name.startswith(f"{index}:") for index, row in enumerate(adapter.state_manifest(legacy).entries))


def verify_resnet50_native_stage_ownership(backend, *, device="cpu"):
    """Bounded external-process helper: capture one full model, then native cuts."""
    from experiments.resnet50_quality.model_contract import SCHEMA, canonical_state
    from tests.integration.resnet50_reference.native import (
        build_model, call_inputs, configure_backend, native_batch, training_mode,
    )

    configure_backend(backend)
    model = build_model(backend, canonical_state(233), device)
    training_mode(backend, model)
    batch = 2
    images, _ = native_batch(backend, np.zeros((batch, 3, 64, 64), np.float32), np.zeros(batch, np.int32), device)
    inputs = call_inputs(backend, model, images)
    handle = AutoSplitSession(device=device).prepare_runtime(
        model, inputs, boundary="50%", trainable=True,
        batch_axes={"/args/1" if backend == "jax" else "/args/0": 0}, dynamic_batch=(1, 16),
    )
    adapter = adapter_for(model, inputs)
    schema = adapter.state_manifest(model).schema_hash
    graph = handle.runtime.trace_graph
    results = []
    for stage, channels, spatial in ((1, 256, 16), (2, 512, 8), (3, 1024, 4)):
        shape = (spatial, spatial, channels) if backend == "tensorflow" else (channels, spatial, spatial)
        matches = [node for node in graph.nodes if not (node.is_input or node.is_output or node.is_buffer_only_source)
                   and tuple((node.output_shape or ())[1:]) == shape
                   and ("float32" in str(node.dtype) or str(node.dtype) == "float")]
        assert matches, (backend, stage)
        # This identifies an actual terminal native activation.  State owners
        # are still obtained exclusively from the plan and native references.
        chosen = matches[-1]
        cut = handle.backend.repartition("after:" + chosen.canonical_id)
        manifest = state_ownership(cut, schema)
        owners = dict(zip(manifest["names"], manifest["owners"], strict=True))
        expected = {row.name: "prefix" if row.stage <= stage else "suffix" for row in SCHEMA}
        assert owners == expected, {name: (owners.get(name), expected[name])
                                   for name in expected if owners.get(name) != expected[name]}
        assert manifest[DIGEST_KEY] == ownership_digest(manifest)
        prefix, suffix = export_owned(model, manifest, "prefix"), export_owned(model, manifest, "suffix")
        assert len(prefix) + len(suffix) == len(SCHEMA)
        results.append({"stage": stage, "boundary": chosen.canonical_id,
                        "prefix_tensors": len(prefix), "suffix_tensors": len(suffix),
                        "prefix_parameters": sum(int(np.prod(row.shape)) for row in SCHEMA if row.trainable and row.stage <= stage),
                        "suffix_parameters": sum(int(np.prod(row.shape)) for row in SCHEMA if row.trainable and row.stage > stage)})
    print({"backend": backend, "graph_nodes": len(graph.nodes), "ownership": results}, flush=True)
    return results


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["pytorch", "tensorflow", "paddle", "jax"])
def test_full_resnet50_native_stage_ownership(request, backend):
    # Full optional captures are explicit opt-in; small state/ownership tests
    # above run normally without launching four full ResNet50 captures in CI.
    import os
    if os.environ.get("SPLITFLEET_FULL_OWNERSHIP_TESTS") != "1":
        pytest.skip("Set SPLITFLEET_FULL_OWNERSHIP_TESTS=1 for full native ResNet50 captures")
    if _run_isolated(request, extra_env={"CUDA_VISIBLE_DEVICES": "", "DEV": "CPU",
                                        "CPU_COUNT": "1", "THREADS": "1", "JAX_PLATFORMS": "cpu"}):
        return
    module = {"pytorch": "torch", "tensorflow": "tensorflow"}.get(backend, backend)
    pytest.importorskip(module)
    verify_resnet50_native_stage_ownership(backend)
