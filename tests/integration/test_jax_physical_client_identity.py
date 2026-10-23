"""The physical Flower handler must preserve private JAX boundary identity."""

from dataclasses import replace
import time

import numpy as np
import pytest


@pytest.mark.parametrize("tamper", [None, "client_id", "round_id"])
def test_physical_jax_resident_updates_use_the_bound_rpc_identity(tamper):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
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
    from splitfleet.common import BatchData, ServerModelFitIns
    from splitfleet.common.constants import CLIENT_ID_CONFIG_KEY
    from splitfleet.common.serde import control_code_from_proto, control_code_to_proto, from_grpc_format, to_grpc_format
    from splitfleet.proto.server_model_pb2 import BatchData as WireBatchData
    from splitfleet.runtime.functional_sgd import OwnedFunctionalSGD
    from splitfleet.server.stage_runtime.manager import StageRuntimeManager
    from splitfleet.server.strategy import AutoSplitStrategy
    from splitfleet.tasks import ImageClassificationTask, ModelInputs, TaskBatch
    from splitfleet.transport import decode_boundary, encode_boundary

    class Classifier:
        def __init__(self):
            self.params = {"a": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 50,
                           "b": jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 20}
            self.training = True

        def set_params(self, values):
            self.params.update(values)

        def state_tensors(self):
            return dict(self.params)

        def load_state(self, values):
            self.params = {name: jnp.asarray(value) for name, value in values.items()}

        def __call__(self, params, inputs):
            return jax.nn.relu(inputs @ params["a"]) @ params["b"]

    template, prefix = Classifier(), Classifier()
    images = jnp.asarray([[1., 2., 3.], [3., 2., 1.]], dtype=jnp.float32)
    labels = jnp.asarray([0, 1], dtype=jnp.int32)
    session = AutoSplitSession()
    handle = session.prepare_runtime(template, (template.params, images),
                                     batch_axes={"/args/1": 0}, dynamic_batch=(1, 16))
    cut = next(node for node in handle.runtime.trace_graph.nodes if node.op_type == "dot_general")
    strategy = AutoSplitStrategy(model=template, sample_inputs=(template.params, images),
        task=ImageClassificationTask(), boundary="after:" + cut.canonical_id,
        batch_axes={"/args/1": 0}, dynamic_batch=(1, 16),
        optimizer_fn=lambda model: OwnedFunctionalSGD(model, .05),
        aggregation_policy="splitfed", owned_state_exchange=True)
    manager = StageRuntimeManager(autosplit_session=strategy.autosplit_session)
    strategy.bind_stage_runtime_manager(manager)
    cid, round_id = "physical-private-client", 3
    config = strategy._autosplit_config(round_id)
    adapter = adapter_for(template, (template.params, images))
    initial = adapter.export_ndarrays(template)
    server = strategy._make_server_model()
    server.configure_fit(ServerModelFitIns(parameters=initial, config=dict(config), sid=cid))
    client = AutoSplitSplitLearningClient(model=prefix, sample_inputs=(prefix.params, images),
        train_data=[TaskBatch(ModelInputs((prefix.params, images)), labels, 2)],
        task=ImageClassificationTask(), batch_axes={"/args/1": 0},
        optimizer_fn=lambda model: OwnedFunctionalSGD(model, .05))
    seen = []

    class NativeStub:
        def UnaryRequest(self, request, **kwargs):
            data = from_grpc_format(request.data)
            envelope = decode_boundary(data["boundary"])
            seen.append((request.cid, envelope.client_id, envelope.round_id))
            assert request.cid == cid
            if tamper:
                data["boundary"] = encode_boundary(replace(envelope, **{
                    tamper: "another-client" if tamper == "client_id" else round_id + 1}))
            response = server.train_tail([BatchData(data=data,
                control_code=control_code_from_proto(request.control_code), metadata=dict(request.metadata))])[0]
            return WireBatchData(data=to_grpc_format(response.data),
                control_code=control_code_to_proto(response.control_code), metadata=response.metadata)

    instruction = Message(content=fitins_to_recorddict(FitIns(ndarrays_to_parameters(initial),
        {**config, CLIENT_ID_CONFIG_KEY: cid}), keep_input=True), metadata=Metadata(
            run_id=0, message_id="physical-fit", src_node_id=0, dst_node_id=1,
            reply_to_message_id="", group_id="", created_at=time.time(), ttl=3600., message_type=MessageType.TRAIN))
    if tamper:
        with pytest.raises(RuntimeError, match="incorrect client/round identity"):
            handle_message(lambda _: client, instruction, NativeStub())
        assert not server._processed_steps and server.num_examples == 0
        return

    expected_loss, gradients = jax.value_and_grad(lambda parameters: -jnp.mean(
        jax.nn.log_softmax(template(parameters, images))[jnp.arange(2), labels]))(template.params)
    expected = jax.tree_util.tree_map(lambda value, gradient: value - np.float32(.05) * gradient,
                                     template.params, gradients)
    handle_message(lambda _: client, instruction, NativeStub())
    assert seen == [(cid, cid, round_id)]
    assert CLIENT_ID_CONFIG_KEY not in next(iter(instruction.content.configs_records.values()))
    assert server.num_examples == 2
    np.testing.assert_allclose(server.loss_total / 2, float(expected_loss), atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(client.model.params["a"]), np.asarray(expected["a"]), atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(server.model.params["b"]), np.asarray(expected["b"]), atol=1e-5, rtol=1e-4)
