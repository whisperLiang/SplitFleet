"""Real ResNet-50 SGD remains correct after removing redundant BN operands."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_paddle_full_resnet50_frontier_preserves_two_native_sgd_steps():
    if os.environ.get("SPLITFLEET_PADDLE_FRONTIER_CHILD") != "1":
        python = ROOT / ".venv-paddle-gpu/bin/python"
        if not python.exists() or importlib.util.find_spec("paddle") is None:
            pytest.skip("The isolated native Paddle runtime is unavailable")
        environment = dict(os.environ, SPLITFLEET_PADDLE_FRONTIER_CHILD="1",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if environment.get("SPLITFLEET_PADDLE_FRONTIER_DEVICE", "cpu") == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run([str(python), "-m", "pytest", __file__, "-q", "-o", "addopts="],
                                cwd=ROOT, env=environment, capture_output=True, text=True, timeout=240)
        assert result.returncode == 0, result.stdout + result.stderr
        return
    sys.modules["tensorflow"] = None
    import paddle
    from flwr.common import FitIns, Message, Metadata, ndarrays_to_parameters
    from flwr.app.message_type import MessageType
    try:
        from flwr.compat.common.recorddict_compat import fitins_to_recorddict
    except ImportError:
        from flwr.common.recorddict_compat import fitins_to_recorddict
    from splitfleet.autosplit import AutoSplitSession
    from splitfleet.backends.utils import adapter_for
    from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
    from splitfleet.client.grpc.message_handler import handle_message
    from splitfleet.common.constants import CLIENT_ID_CONFIG_KEY
    from splitfleet.common import BatchData, ServerModelFitIns
    from splitfleet.common.serde import control_code_from_proto, control_code_to_proto, from_grpc_format, to_grpc_format
    from splitfleet.proto.server_model_pb2 import BatchData as WireBatchData
    from splitfleet.runtime.native_frontier import frontier_proof
    from splitfleet.server.stage_runtime.manager import StageRuntimeManager
    from splitfleet.server.strategy import AutoSplitStrategy
    from splitfleet.tasks import ImageClassificationTask, ModelInputs, TaskBatch
    from splitfleet.transport import decode_boundary
    from experiments.resnet50_quality.model_contract import canonical_state, NAMES, TRAINABLE_NAMES
    from tests.integration.resnet50_reference.array_helpers import stage_boundaries
    from tests.integration.resnet50_reference.native import (
        build_model, configure_backend, export_state, native_batch, numpy_value, optimizer, train_step,
    )

    configure_backend("paddle")
    device = os.environ.get("SPLITFLEET_PADDLE_FRONTIER_DEVICE", "cpu")
    initial = canonical_state(233)
    random = np.random.default_rng(41)
    batches = [native_batch("paddle", random.normal(size=(size, 3, 64, 64)).astype(np.float32),
                            random.integers(0, 10, size, dtype=np.int64), device) for size in (16, 12)]
    reference = build_model("paddle", initial, device)
    expected = [train_step("paddle", reference, images, labels, capture=True) for images, labels in batches]
    expected_state = export_state("paddle", reference)
    del reference
    template, prefix = [build_model("paddle", initial, device) for _ in range(2)]
    sample = batches[0][0][:2]
    handle = AutoSplitSession(device=device).prepare_runtime(template, (sample,), trainable=True,
        batch_axes={"/args/0": 0}, dynamic_batch=(1, 16))
    boundary = stage_boundaries(handle, "paddle")["stage2"]["boundary"]

    class Recorder:
        def __init__(self, model):
            self.model, self.native = model, optimizer("paddle", model, .05)
            self.records = []

        def __getattr__(self, name):
            return getattr(self.native, name)

        def step(self):
            self.records.append({name: numpy_value(value.grad) for name, value in self.model.named_parameters()
                                 if value.grad is not None})
            return self.native.step()

    recorders = []

    def make_optimizer(model):
        value = Recorder(model)
        recorders.append(value)
        return value

    strategy = AutoSplitStrategy(model=template, sample_inputs=(sample,),
        task=ImageClassificationTask(), boundary=boundary, batch_axes={"/args/0": 0},
        dynamic_batch=(1, 16), optimizer_fn=make_optimizer, runtime_device=device,
        aggregation_policy="splitfed", owned_state_exchange=True)
    manager = StageRuntimeManager(autosplit_session=strategy.autosplit_session)
    strategy.bind_stage_runtime_manager(manager)
    config = strategy._autosplit_config(1)
    adapter = adapter_for(template, (sample,))
    arrays = adapter.export_ndarrays(template)
    cid = "private-paddle-frontier-client"
    server = strategy._make_server_model()
    server.configure_fit(ServerModelFitIns(parameters=arrays, config=dict(config), sid=cid))
    client = AutoSplitSplitLearningClient(model=prefix, sample_inputs=(sample,),
        train_data=[TaskBatch(ModelInputs((images,)), labels, len(labels)) for images, labels in batches],
        task=ImageClassificationTask(), batch_axes={"/args/0": 0}, device=device, optimizer_fn=make_optimizer)
    received = []

    class NativeStub:
        def UnaryRequest(self, request, **kwargs):
            data = from_grpc_format(request.data)
            envelope = decode_boundary(data["boundary"])
            assert envelope.client_id == request.cid == cid
            received.append({"batch_size": envelope.batch_size, "tensor_fields": len(envelope.tensors),
                             "tensor_bytes": sum(len(t.payload) for t in envelope.tensors),
                             "encoded_bytes": len(data["boundary"])})
            reply = server.train_tail([BatchData(data=data,
                control_code=control_code_from_proto(request.control_code), metadata=dict(request.metadata))])[0]
            return WireBatchData(data=to_grpc_format(reply.data),
                control_code=control_code_to_proto(reply.control_code), metadata=reply.metadata)

    instruction = Message(content=fitins_to_recorddict(FitIns(ndarrays_to_parameters(arrays),
        {**config, CLIENT_ID_CONFIG_KEY: cid}), keep_input=True), metadata=Metadata(
            run_id=0, message_id="paddle-frontier-fit", src_node_id=0, dst_node_id=1,
            reply_to_message_id="", group_id="", created_at=time.time(), ttl=3600., message_type=MessageType.TRAIN))
    handle_message(lambda _: client, instruction, NativeStub())
    assert server.num_examples == 28 and len(server._processed_steps) == 2
    assert [r["batch_size"] for r in received] == [16, 12]
    assert all(r["tensor_fields"] == 73 and r["encoded_bytes"] < 3_000_000 for r in received)
    runtime = server._require_runtime_handle().runtime
    proof = frontier_proof(runtime.trace_graph, runtime.plan)
    assert len(proof["original_native_field_ids"]) == 145
    assert len(proof["required_field_ids"]) == 73 and len(proof["removed_field_ids"]) == 72
    owners = dict(zip(server.state_ownership["names"], server.state_ownership["owners"], strict=True))
    states = {"prefix": export_state("paddle", client.model), "suffix": export_state("paddle", server.model)}
    for name in NAMES:
        actual = states[owners[name]][name]
        if name.endswith("num_batches_tracked"):
            assert actual.item() == 2
            np.testing.assert_array_equal(actual, expected_state[name], err_msg=name)
        else:
            np.testing.assert_allclose(actual, expected_state[name], atol=1e-5, rtol=1e-4, err_msg=name)
    assert len(recorders) == 2
    for index in range(2):
        combined = {}
        for recorder in recorders:
            assert len(recorder.records) == 2
            assert not combined.keys() & recorder.records[index].keys()
            combined.update(recorder.records[index])
        assert set(combined) == set(TRAINABLE_NAMES)
        for name in TRAINABLE_NAMES:
            np.testing.assert_allclose(combined[name], expected[index]["gradients"][name],
                                       atol=1e-5, rtol=1e-4, err_msg=f"step {index + 1}: {name}")
    expected_loss = sum(size * value["loss"] for size, value in zip((16, 12), expected, strict=True)) / 28
    np.testing.assert_allclose(server.loss_total / 28, expected_loss, atol=1e-5, rtol=1e-4)
    receipt = {"status": "passed", "device": device, "backend": "paddle",
               "batch_sizes": [16, 12], "steps": 2, "trainable_gradients_per_step": 161,
               "model_state_tensors_checked": 320, "bn_counters": 53,
               "received_boundaries": received, "original_boundary_fields": 145,
               "fixed_boundary_fields": 73, "atol": 1e-5, "rtol": 1e-4,
               "transport": "physical Flower dispatch and protobuf codec with in-process native server stub",
               "scope": "Bounded two-step numerical regression, not a 50-round distributed experiment"}
    if os.environ.get("SPLITFLEET_PADDLE_FRONTIER_RECEIPT"):
        Path(os.environ["SPLITFLEET_PADDLE_FRONTIER_RECEIPT"]).write_text(json.dumps(receipt, indent=2) + "\n")
