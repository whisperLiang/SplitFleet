from queue import Queue
from typing import Optional
from logging import DEBUG

from flwr.common import log

from splitfleet.common import BatchData
from splitfleet.common.serde import (
    control_code_to_proto,
    control_code_from_proto,
    from_grpc_format,
    to_grpc_format,
    ControlCode,
)
from splitfleet.server.server_model.proxy.server_model_proxy import ServerModelProxy
from splitfleet.proto import server_model_pb2_grpc
from splitfleet.proto import server_model_pb2


class FutureResponse:
    def __init__(self, response_future):
        self.response_future = response_future

    def get_response(self):
        return self.response_future()



# pylint: disable=no-member
class GrpcServerModelProxy(ServerModelProxy):

    """Existing split RPC protocol, with optional deadlines and call credentials."""
    def __init__(
        self,
        *,
        stub: server_model_pb2_grpc.ServerModelStub,
        cid: str,
        default_timeout: Optional[float] = None,
        metadata=(),
    ) -> None:
        super().__init__(cid=cid)
        log(DEBUG, "Initializing proxy for CID %s", cid)
        self.stub = stub
        self.request_queue = None
        self.default_timeout = default_timeout
        self.call_metadata = tuple(metadata)

    def _initialize_stream(self):
        self.request_queue = Queue(maxsize=20)
        kwargs = {}
        if self.default_timeout is not None:
            kwargs["timeout"] = self.default_timeout
        if self.call_metadata:
            kwargs["metadata"] = self.call_metadata
        self.response_stream = self.stub.StreamRequest(iter(self.request_queue.get, None), **kwargs)
        self.request_queue.put(server_model_pb2.BatchData(
            control_code=control_code_to_proto(ControlCode.INIT_STREAM),
            cid=self.cid,
        ))

    def _blocking_request(
        self,
        method: str,
        batch_data: BatchData,
        _streams_: bool,
        _timeout_: Optional[float],
    ) -> BatchData:
        """Issue a blocking request to the server over gRPC

        Parameters
        ----------
        method : str
            name of the method to be invoked
        batch_data : BatchData
            data to be sent to the server
        _timeout_ : Optional[float]
            optionally pass the maximum time allowed for the server to return a response

        Returns
        -------
        BatchData
            data returned by the server
        """
        if _streams_:
            if self.request_queue is None:
                self._initialize_stream()

            ins = server_model_pb2.BatchData(
                method=method,
                data=to_grpc_format(batch_data.data),
                control_code=control_code_to_proto(batch_data.control_code),
                metadata=batch_data.metadata,
            )
            self.request_queue.put(ins)
            res = next(self.response_stream)
        else:
            kwargs = {}
            timeout = _timeout_ if _timeout_ is not None else self.default_timeout
            if timeout is not None:
                kwargs["timeout"] = timeout
            if self.call_metadata:
                kwargs["metadata"] = self.call_metadata
            res = self.stub.UnaryRequest(
                server_model_pb2.BatchData(
                    method=method,
                    data=to_grpc_format(batch_data.data),
                    control_code=control_code_to_proto(batch_data.control_code),
                    cid=self.cid,
                    metadata=batch_data.metadata,
                ),
                **kwargs,
            )
        return BatchData(
            data=from_grpc_format(res.data),
            control_code=control_code_from_proto(res.control_code),
            metadata=dict(res.metadata),
        )

    def _streaming_request(self, method, batch_data, _streams_, _timeout_):
        assert _streams_, "Not implemented yet for unary requests"
        if self.request_queue is None:
            self._initialize_stream()
        ins = server_model_pb2.BatchData(
            method=method,
            data=to_grpc_format(batch_data.data),
            control_code=control_code_to_proto(batch_data.control_code),
            metadata=batch_data.metadata,
        )
        self.request_queue.put(ins)

    def _nonblocking_request(self, method, batch_data, _timeout_, _streams_):
        assert _streams_, "Not implemented yet for unary requests"
        if self.request_queue is None:
            self._initialize_stream()

        ins = server_model_pb2.BatchData(
            method=method,
            data=to_grpc_format(batch_data.data),
            control_code=control_code_to_proto(batch_data.control_code),
            metadata=batch_data.metadata,
        )
        self.request_queue.put(ins)
        def get_future_response():
            res = next(self.response_stream)
            batch_data = BatchData(
                data=from_grpc_format(res.data),
                control_code=control_code_from_proto(res.control_code),
                metadata=dict(res.metadata),
            )
            return self._parse_response_args(batch_data)

        return FutureResponse(get_future_response)

    def close_stream(self):
        if self.request_queue is None:
            return
        self.request_queue.put(server_model_pb2.BatchData(
            control_code=control_code_to_proto(ControlCode.DO_CLOSE_STREAM)
        ))
        res = next(self.response_stream)
        assert control_code_from_proto(res.control_code) == ControlCode.STREAM_CLOSED_OK

    def cancel(self):
        """Release a failed/cancelled streaming call without waiting for a reply."""
        stream = self.__dict__.get("response_stream")
        if stream is not None:
            stream.cancel()
        if self.request_queue is not None:
            try:
                self.request_queue.put_nowait(None)
            except Exception:
                pass
        self.request_queue = None
