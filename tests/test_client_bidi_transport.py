from concurrent.futures import ThreadPoolExecutor

import grpc
import numpy as np
import pytest
from flwr.common import EvaluateIns, FitIns, ndarrays_to_parameters, serde
from flwr.proto import transport_pb2, transport_pb2_grpc

from splitfleet.client.app import start_client
from splitfleet.client.grpc.connection import init_connection
from splitfleet.client.numpy_client import NumPyClient
from splitfleet.common import ControlCode
from splitfleet.common.constants import CLIENT_ID_CONFIG_KEY
from splitfleet.common.serde import control_code_from_proto, control_code_to_proto
from splitfleet.proto import server_model_pb2, server_model_pb2_grpc


class _FlowerService(transport_pb2_grpc.FlowerServiceServicer):
    def __init__(self):
        self.responses = []

    def Join(self, requests, context):
        parameters = ndarrays_to_parameters([])
        config = {CLIENT_ID_CONFIG_KEY: "client-7"}
        yield transport_pb2.ServerMessage(
            fit_ins=serde.fit_ins_to_proto(FitIns(parameters, dict(config)))
        )
        self.responses.append(next(requests))
        yield transport_pb2.ServerMessage(
            evaluate_ins=serde.evaluate_ins_to_proto(EvaluateIns(parameters, dict(config)))
        )
        self.responses.append(next(requests))
        yield transport_pb2.ServerMessage(
            reconnect_ins=transport_pb2.ServerMessage.ReconnectIns(seconds=0)
        )


class _ModelService(server_model_pb2_grpc.ServerModelServicer):
    def __init__(self):
        self.client_ids = []
        self.closed_streams = 0

    def UnaryRequest(self, request, context):
        self.client_ids.append(request.cid)
        return request

    def StreamRequest(self, requests, context):
        for request in requests:
            control = control_code_from_proto(request.control_code)
            if control == ControlCode.INIT_STREAM:
                self.client_ids.append(request.cid)
            elif control == ControlCode.DO_CLOSE_STREAM:
                self.closed_streams += 1
                yield server_model_pb2.BatchData(
                    control_code=control_code_to_proto(ControlCode.STREAM_CLOSED_OK)
                )
                return
            else:
                yield request


class _Client(NumPyClient):
    def fit(self, parameters, config):
        assert CLIENT_ID_CONFIG_KEY not in config
        self.server_model_proxy.numpy()
        result = self.server_model_proxy.train_batch(inputs=np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result["inputs"], [1.0, 2.0])
        return parameters, 2, {"loss": 0.25}

    def evaluate(self, parameters, config):
        self.server_model_proxy.numpy()
        result = self.server_model_proxy.evaluate_batch(
            inputs=np.array([3.0]), _streams_=False
        )
        np.testing.assert_array_equal(result["inputs"], [3.0])
        return 0.5, 1, {}


@pytest.fixture
def rpc_server():
    flower, model = _FlowerService(), _ModelService()
    with ThreadPoolExecutor(max_workers=4) as executor:
        server = grpc.server(executor)
        transport_pb2_grpc.add_FlowerServiceServicer_to_server(flower, server)
        server_model_pb2_grpc.add_ServerModelServicer_to_server(model, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            yield f"127.0.0.1:{port}", flower, model
        finally:
            server.stop(0).wait()


def test_bidi_runner_handles_flower_and_model_rpcs(rpc_server):
    address, flower, model = rpc_server
    start_client(server_address=address, client=_Client().to_client(), max_retries=1)
    fit, evaluation = flower.responses
    assert fit.fit_res.num_examples == 2
    assert evaluation.evaluate_res.loss == pytest.approx(0.5)
    assert model.client_ids == ["client-7", "client-7"]
    assert model.closed_streams == 1


def test_bidi_runner_propagates_client_failures(rpc_server):
    class BrokenClient(_Client):
        def fit(self, parameters, config):
            raise ValueError("client training failed")

    address, _, _ = rpc_server
    with pytest.raises(ValueError, match="client training failed"):
        start_client(server_address=address, client=BrokenClient().to_client(), max_retries=1)


@pytest.mark.parametrize("transport", ["grpc-rere", "rest", None])
def test_bidi_runner_rejects_unsupported_transports(transport):
    with pytest.raises(ValueError, match="Unsupported transport"):
        init_connection(transport, "127.0.0.1:8080")


def test_bidi_runner_rejects_invalid_address():
    with pytest.raises(ValueError, match="cannot be parsed"):
        init_connection("grpc-bidi", "invalid-address")
