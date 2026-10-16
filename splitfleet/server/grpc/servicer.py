import asyncio
import time
import traceback
from collections.abc import AsyncIterable
from concurrent.futures import ThreadPoolExecutor

from splitfleet.common import BatchData, ControlCode
from splitfleet.common.constants import (
    TRANSPORT_SERVER_ELAPSED_NS_METADATA_KEY,
    TRANSPORT_SERVER_RECEIVE_NS_METADATA_KEY,
    TRANSPORT_SERVER_SEND_NS_METADATA_KEY,
)
from splitfleet.common.serde import (
    control_code_from_proto,
    control_code_to_proto,
    from_grpc_format,
    to_grpc_format,
)
from splitfleet.proto import server_model_pb2, server_model_pb2_grpc
from splitfleet.server.server_model.manager import ServerModelManager
from splitfleet.server.server_model.utils import ClientRequestGroup


class EventWithReturnValue(asyncio.Event):
    def __init__(self):
        super().__init__()
        self.result = None
        self.error = None

    def set_result(self, result: BatchData):
        self.result = result
        self.set()

    def get_result(self) -> BatchData:
        if self.error is not None:
            raise self.error
        return self.result

    def set_error(self, error):
        self.error = error
        self.set()


# pylint: disable=no-member
class ServerModelServicer(server_model_pb2_grpc.ServerModelServicer):
    def __init__(self, server_model_manager: ServerModelManager, *, execution_concurrency=None,
                 control_concurrency=None) -> None:
        super().__init__()
        self.server_model_manager = server_model_manager
        self._lock = asyncio.Lock()
        self._executor = None
        self._execution_slots = None
        self._control_executor = None
        self._control_slots = None
        self._computations = set()
        self.peak_executions = 0
        self.active_executions = 0
        self.queued_executions = 0
        self.queue_wait_sec = 0.
        if execution_concurrency is not None:
            if execution_concurrency < 1:
                raise ValueError("execution_concurrency must be positive")
            self._executor = ThreadPoolExecutor(max_workers=execution_concurrency,
                                                thread_name_prefix="splitfleet-suffix")
            self._execution_slots = asyncio.Semaphore(execution_concurrency)
        if control_concurrency is not None:
            if control_concurrency < 1:
                raise ValueError("control_concurrency must be positive")
            self._control_executor = ThreadPoolExecutor(max_workers=control_concurrency,
                                                       thread_name_prefix="splitfleet-lifecycle")
            self._control_slots = asyncio.Semaphore(control_concurrency)

    async def _compute_response(
        self,
        cid: str,
        request: server_model_pb2.BatchData
    ) -> BatchData:
        server_receive_ns = time.time_ns()
        server_started_ns = time.perf_counter_ns()
        data = BatchData(
            data = from_grpc_format(request.data),
            control_code=control_code_from_proto(request.control_code),
            metadata=dict(request.metadata),
        )
        method_name = request.method
        event = EventWithReturnValue()
        async with self._lock:
            request_group, callback_fn = self.server_model_manager.strategy.route_client_request(
                cid=cid,
                method_name=method_name,
            )
            request_group.add(data, event, cid)
            is_new = request_group.is_new()
            self.server_model_manager.strategy.mark_ready_requests()

        if is_new:
            try:
                await request_group._is_ready.wait()
                if self._executor is None:
                    self._trigger_computation(request_group, method_name=method_name)
                else:
                    await self.execute_group(request_group, method_name)
            except BaseException as exc:
                for pending in request_group.events:
                    pending.set_error(exc)
                raise
            finally:
                if callback_fn:
                    callback_fn()

        await event.wait()
        response = event.get_result()
        response.metadata[TRANSPORT_SERVER_RECEIVE_NS_METADATA_KEY] = str(
            server_receive_ns
        )
        response.metadata[TRANSPORT_SERVER_SEND_NS_METADATA_KEY] = str(
            time.time_ns()
        )
        response.metadata[TRANSPORT_SERVER_ELAPSED_NS_METADATA_KEY] = str(time.perf_counter_ns() - server_started_ns)
        return response

    def _trigger_computation(self, request_group: ClientRequestGroup, method_name: str):
        events, res = self._compute_group(request_group, method_name)
        for e, r in zip(events, res):
            e.set_result(r)

    def _compute_group(self, request_group, method_name):
        batches, events = request_group.get_data()
        sid = request_group.sid
        server_model = self.server_model_manager.get_server_model(sid)
        method = getattr(server_model, method_name)
        try:
            res = method(batches)
        except Exception:
            traceback.print_exc()
            raise

        return events, res

    async def execute(self, function, *args, control=False):
        """Optional bounded execution; keep the RPC loop responsive while busy."""
        queued = time.perf_counter()
        slots = self._control_slots if control else self._execution_slots
        executor = self._control_executor if control else self._executor
        self.queued_executions += 1
        try:
            await slots.acquire()
        finally:
            self.queued_executions -= 1
        try:
            queue_sec = time.perf_counter() - queued
            if not control:
                self.queue_wait_sec += queue_sec
                self.active_executions += 1
                self.peak_executions = max(self.peak_executions, self.active_executions)
            future = asyncio.get_running_loop().run_in_executor(executor, function, *args)
            self._computations.add(future)
            try:
                cancelled = None
                while True:
                    try:
                        result = await asyncio.shield(future)
                        break
                    except asyncio.CancelledError as exc:
                        # Deadlines and round reconciliation can both cancel a
                        # request. Neither may free a running Torch execution.
                        if future.cancelled():
                            raise
                        cancelled = exc
                    except Exception:
                        if cancelled is not None:
                            raise cancelled
                        raise
                if cancelled is not None:
                    raise cancelled
                return result, queue_sec
            finally:
                self._computations.discard(future)
                if not control:
                    self.active_executions -= 1
        finally:
            slots.release()

    async def execute_control(self, function, *args):
        return await self.execute(function, *args, control=True)

    async def execute_group(self, group, method_name):
        (events, responses), queue_sec = await self.execute(self._compute_group, group, method_name)
        for event, response in zip(events, responses):
            response.metadata["suffix_server_queue_sec"] = str(queue_sec)
            event.set_result(response)

    async def drain(self):
        if self._computations:
            await asyncio.gather(
                *(asyncio.shield(future) for future in tuple(self._computations)),
                return_exceptions=True,
            )

    def shutdown_executor(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        if self._control_executor is not None:
            self._control_executor.shutdown(wait=True, cancel_futures=True)

    def _to_grpc(self, data: BatchData) -> server_model_pb2.BatchData:
        return server_model_pb2.BatchData(
            data=to_grpc_format(data.data),
            control_code=control_code_to_proto(data.control_code),
            metadata=data.metadata,
        )

    async def UnaryRequest(
        self,
        request,
        context
    ) -> server_model_pb2.BatchData:
        _ = (context, )
        return self._to_grpc(await self._compute_response(request.cid, request))

    async def StreamRequest(
        self,
        request_iterator: AsyncIterable[server_model_pb2.BatchData],
        context,
    ) -> AsyncIterable[server_model_pb2.BatchData]:
        _ = (context, )
        req = await request_iterator.__anext__()
        assert control_code_from_proto(req.control_code) == ControlCode.INIT_STREAM
        cid = req.cid
        async for req in request_iterator:
            if control_code_from_proto(req.control_code) == ControlCode.DO_CLOSE_STREAM:
                break
            res = await self._compute_response(cid, req)

            if res is not None:
                yield self._to_grpc(res)

        async with self._lock:
            self.server_model_manager.strategy.mark_client_as_done(cid)
        yield server_model_pb2.BatchData(
            control_code=control_code_to_proto(ControlCode.STREAM_CLOSED_OK)
        )
