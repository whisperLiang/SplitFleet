"""Native state layouts, proven split ownership and exact integer aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from splitfleet.autosplit import AutoSplitSession
from splitfleet.autosplit.state_ownership import (
    DIGEST_KEY,
    aggregate_owned,
    export_owned,
    ownership_digest,
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
        batch_axes={"/args/0": 0}, dynamic_batch=(1, 16),
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
@pytest.mark.parametrize("backend", ["pytorch", "tensorflow", "paddle"])
def test_full_resnet50_native_stage_ownership(request, backend):
    # Full optional captures are explicit opt-in; small state/ownership tests
    # above run normally without launching four full ResNet50 captures in CI.
    import os
    if os.environ.get("SPLITFLEET_FULL_OWNERSHIP_TESTS") != "1":
        pytest.skip("Set SPLITFLEET_FULL_OWNERSHIP_TESTS=1 for full native ResNet50 captures")
    if _run_isolated(request, extra_env={"CUDA_VISIBLE_DEVICES": "", "DEV": "CPU",
                                        "CPU_COUNT": "1", "THREADS": "1"}):
        return
    module = {"pytorch": "torch", "tensorflow": "tensorflow"}.get(backend, backend)
    pytest.importorskip(module)
    verify_resnet50_native_stage_ownership(backend)
