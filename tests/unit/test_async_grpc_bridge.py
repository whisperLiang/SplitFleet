"""Cross-thread notification and lifecycle contracts of the async bridge."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.server.superlink.fleet.grpc_bidi.grpc_bridge import (
    GrpcBridgeClosed,
    InsWrapper,
    ResWrapper,
    Status,
)

from splitfleet.server.grpc.async_grpc_bridge import AsyncGrpcBridge


def test_instruction_queued_before_async_consumer_is_delivered():
    bridge = AsyncGrpcBridge()
    instruction = InsWrapper(server_message=ServerMessage(), timeout=1)
    response = ResWrapper(client_message=ClientMessage())

    async def consume():
        iterator = bridge.ins_wrapper_async_iterator()
        try:
            assert await asyncio.wait_for(anext(iterator), 3) is instruction
            bridge.set_res_wrapper(response)
        finally:
            await iterator.aclose()

    with ThreadPoolExecutor(max_workers=1) as executor:
        request = executor.submit(bridge.request, instruction)
        try:
            # Ensure the request predates the async consumer and its event loop.
            with bridge._cv:
                assert bridge._cv.wait_for(
                    lambda: bridge._status == Status.INS_WRAPPER_AVAILABLE,
                    timeout=3,
                )
            asyncio.run(consume())
            assert request.result(timeout=3) is response
        finally:
            # A proxy can also be closed after the consumer's loop has ended.
            bridge.close()
    with pytest.raises(GrpcBridgeClosed):
        bridge.request(instruction)


def test_close_from_worker_wakes_idle_async_consumer():
    async def exercise():
        bridge = AsyncGrpcBridge()
        iterator = bridge.ins_wrapper_async_iterator()
        waiting = asyncio.create_task(anext(iterator))
        try:
            await asyncio.sleep(0)
            await asyncio.to_thread(bridge.close)
            with pytest.raises(GrpcBridgeClosed):
                await asyncio.wait_for(waiting, 3)
        finally:
            bridge.close()
            if not waiting.done():
                waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            await iterator.aclose()

    asyncio.run(exercise())
