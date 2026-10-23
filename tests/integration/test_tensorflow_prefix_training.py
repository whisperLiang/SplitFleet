"""Compiled native prefixes preserve independent, transported full ResNet50 SGD."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.integration
def test_tensorflow_compiled_prefix_preserves_two_transported_native_steps():
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("Native TensorFlow is unavailable")
    from splitfleet.autosplit import AutoSplitSession
    from splitfleet.autosplit.tensorflow_clone import clone_tensorflow_runtime_handle
    from splitfleet.autosplit.state_ownership import state_ownership
    from splitfleet.backends.utils import adapter_for
    from splitfleet.runtime.owned_auxiliary_state import statistics_for_stage
    from splitfleet.runtime.native_frontier import frontier_proof
    from splitfleet.split_engine import graph_contract_for_runtime_handle
    from splitfleet.tasks.losses import sparse_cross_entropy
    from splitfleet.transport import encode_boundary, decode_boundary, encode_gradients, decode_gradients
    from splitfleet.transport.split_wire import (
        boundary_to_envelope, envelope_to_boundary, gradients_to_envelope, envelope_to_gradients,
    )
    from experiments.resnet50_quality.model_contract import canonical_state, NAMES, TRAINABLE_NAMES
    from tests.integration.resnet50_reference.array_helpers import (
        RecordingOptimizer, canonical_array, stage_boundaries,
    )
    from tests.integration.resnet50_reference.native import (
        build_model, configure_backend, export_state, native_batch, numpy_value, train_step, training_mode,
    )

    configure_backend("tensorflow")
    device = os.environ.get("SPLITFLEET_TF_PREFIX_DEVICE", "cpu")
    initial = canonical_state(233)
    random = np.random.default_rng(41)
    batches = [native_batch("tensorflow", random.normal(size=(size, 3, 64, 64)).astype(np.float32),
                            random.integers(0, 10, size, dtype=np.int64), device) for size in (16, 12)]
    reference = build_model("tensorflow", initial, device)
    expected = [train_step("tensorflow", reference, images, labels, capture=True) for images, labels in batches]
    expected_state = export_state("tensorflow", reference)
    del reference
    prefix, suffix = [build_model("tensorflow", initial, device) for _ in range(2)]
    for model in (prefix, suffix):
        training_mode("tensorflow", model, True)
    sample = batches[0][0][:2]
    client = AutoSplitSession(device=device).prepare_runtime(prefix, (sample,), trainable=True,
        batch_axes={"/args/0": 0}, dynamic_batch=(1, 16))
    client = client.backend.repartition(stage_boundaries(client, "tensorflow")["stage2"]["boundary"])
    server = clone_tensorflow_runtime_handle(client, suffix)
    assert server is not None and server.model is not client.model
    ownership = state_ownership(client, adapter_for(prefix, (sample,)).state_manifest(prefix).schema_hash)
    contract = graph_contract_for_runtime_handle(client)

    class Recorder(RecordingOptimizer):
        def __init__(self, model):
            super().__init__("tensorflow", model)
            self.records = []

        def apply_gradients(self, pairs):
            self.gradients = {}
            result = super().apply_gradients(pairs)
            self.records.append(self.gradients)
            return result

    prefix_opt, suffix_opt = Recorder(prefix), Recorder(suffix)
    received, losses = [], []
    objective = lambda output, labels: sparse_cross_entropy(output["logits"], labels)
    for step, (images, labels) in enumerate(batches):
        boundary = client.backend.run_prefix(images, training=True)
        envelope = boundary_to_envelope(boundary, round_id=1, client_id="tf-prefix-client", step_id=str(step),
            plan_id=client.plan.plan_id, split_id=contract.split_id,
            canonical_graph_hash=contract.canonical_graph_hash,
            boundary_schema_hash=contract.boundary_schema_hash, model_version=1)
        wire = encode_boundary(envelope)
        decoded = decode_boundary(wire)
        remote = envelope_to_boundary(decoded, server.runtime, device)
        assert "tf_tape" not in remote.metadata
        received.append({"batch_size": envelope.batch_size, "fields": len(envelope.tensors), "encoded_bytes": len(wire)})
        loss, gradients = server.backend.train_suffix(remote, labels, loss_fn=objective, optimizer=suffix_opt)
        losses.append(float(numpy_value(loss)))
        returned = envelope_to_gradients(decode_gradients(encode_gradients(
            gradients_to_envelope(decoded, gradients))), device)
        client.backend.backward_prefix(boundary, returned, optimizer=prefix_opt)
        statistics = server.backend.train_suffix_auxiliary
        for model, stage in ((prefix, "prefix"), (suffix, "suffix")):
            model.apply_statistics(statistics_for_stage(model, statistics, ownership, stage))
    assert [r["batch_size"] for r in received] == [16, 12]
    required_fields = len(frontier_proof(client.runtime.trace_graph, client.runtime.plan)["required_field_ids"])
    assert all(row["fields"] == required_fields and row["encoded_bytes"] < 3_000_000 for row in received), received
    assert len(client.runtime.segments.training_prefix.functions) == 2
    owners = dict(zip(ownership["names"], ownership["owners"], strict=True))
    states = {"prefix": export_state("tensorflow", prefix), "suffix": export_state("tensorflow", suffix)}
    for name in NAMES:
        actual = states[owners[name]][name]
        if name.endswith("num_batches_tracked"):
            assert actual.item() == 2
            np.testing.assert_array_equal(actual, expected_state[name], err_msg=name)
        else:
            np.testing.assert_allclose(actual, expected_state[name], atol=1e-5, rtol=1e-4, err_msg=name)
    for index in range(2):
        combined = {}
        for recorder in (prefix_opt, suffix_opt):
            assert len(recorder.records) == 2
            assert not combined.keys() & recorder.records[index].keys()
            combined.update(recorder.records[index])
        assert set(combined) == set(TRAINABLE_NAMES)
        for name in TRAINABLE_NAMES:
            np.testing.assert_allclose(combined[name], canonical_array("tensorflow", name,
                expected[index]["gradients"][name]), atol=1e-5, rtol=1e-4, err_msg=f"step {index + 1}: {name}")
        np.testing.assert_allclose(losses[index], expected[index]["loss"], atol=1e-5, rtol=1e-4)
    receipt = {"status": "passed", "backend": "tensorflow", "device": device,
        "steps": 2, "batch_sizes": [16, 12], "checked_gradients_per_step": 161,
        "checked_model_states": 320, "checked_bn_counters": 53, "boundaries": received,
        "atol": 1e-5, "rtol": 1e-4,
        "scope": "independent native client/suffix models with actual boundary and gradient codecs; no LAN"}
    if os.environ.get("SPLITFLEET_TF_PREFIX_RECEIPT"):
        Path(os.environ["SPLITFLEET_TF_PREFIX_RECEIPT"]).write_text(json.dumps(receipt, indent=2) + "\n")
