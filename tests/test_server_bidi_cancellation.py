"""Real asyncio gRPC stream tests for cleanup, without model training."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest
from flwr.common import FitIns, ndarrays_to_parameters, serde
from flwr.proto import transport_pb2, transport_pb2_grpc
from flwr.server.client_manager import SimpleClientManager
from flwr.server.superlink.fleet.grpc_bidi.grpc_bridge import GrpcBridgeClosed

from splitfleet.server.grpc.flower_servicer import FlowerServiceServicer


async def _wait_for(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), timeout=3)


async def _exercise(mode):
    manager = SimpleClientManager()
    server = grpc.aio.server()
    transport_pb2_grpc.add_FlowerServiceServicer_to_server(
        FlowerServiceServicer(manager), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    queue = asyncio.Queue()

    async def requests():
        while True:
            yield await queue.get()

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        call = transport_pb2_grpc.FlowerServiceStub(channel).Join(requests())
        try:
            await _wait_for(lambda: manager.num_available() == 1)
            proxy = next(iter(manager.all().values()))
            if mode in ("idle_cancel", "shutdown"):
                if mode == "idle_cancel":
                    call.cancel()
                else:
                    await server.stop(0)
                await _wait_for(lambda: manager.num_available() == 0)
                with pytest.raises(GrpcBridgeClosed):
                    proxy.bridge._raise_if_closed()
                return

            timeout = 0.05 if mode == "timeout" else None
            fit = asyncio.create_task(asyncio.to_thread(
                proxy.fit, FitIns(ndarrays_to_parameters([]), {"round": 1}),
                timeout, 1,
            ))
            instruction = await asyncio.wait_for(call.read(), 3)
            assert instruction.fit_ins.config["round"].sint64 == 1
            if mode == "inflight_cancel":
                call.cancel()
                with pytest.raises(GrpcBridgeClosed):
                    await asyncio.wait_for(fit, 3)
            elif mode == "timeout":
                with pytest.raises(grpc.aio.AioRpcError) as err:
                    await asyncio.wait_for(call.read(), 3)
                assert err.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
                with pytest.raises(GrpcBridgeClosed):
                    await asyncio.wait_for(fit, 3)
            else:
                response = transport_pb2.ClientMessage.FitRes(
                    status=transport_pb2.Status(code=transport_pb2.OK),
                    parameters=serde.parameters_to_proto(ndarrays_to_parameters([])),
                    num_examples=7,
                )
                await queue.put(transport_pb2.ClientMessage(fit_res=response))
                result = await asyncio.wait_for(fit, 3)
                assert result.num_examples == 7
                call.cancel()
            await _wait_for(lambda: manager.num_available() == 0)
            # A fresh connection must not coexist with an abandoned old proxy.
            second = transport_pb2_grpc.FlowerServiceStub(channel).Join(requests())
            await _wait_for(lambda: manager.num_available() == 1)
            assert next(iter(manager.all())) != proxy.cid
            second.cancel()
            await _wait_for(lambda: manager.num_available() == 0)
        finally:
            call.cancel()
            await server.stop(0)


@pytest.mark.parametrize("mode", [
    "idle_cancel", "inflight_cancel", "timeout", "normal_response", "shutdown",
])
def test_native_async_bidi_cleans_up_connections(mode):
    asyncio.run(_exercise(mode))


async def _exercise_many_idle_connections(mode):
    # A single shared worker makes starvation deterministic without relying on
    # the host's CPU count or the default executor's platform-specific size.
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
    manager = SimpleClientManager()
    server = grpc.aio.server()
    transport_pb2_grpc.add_FlowerServiceServicer_to_server(
        FlowerServiceServicer(manager), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    calls = []

    async def requests(queue):
        while True:
            yield await queue.get()

    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = transport_pb2_grpc.FlowerServiceStub(channel)
            for count in range(40):
                queue = asyncio.Queue()
                calls.append(stub.Join(requests(queue)))
                await _wait_for(lambda: manager.num_available() == count + 1)

            # Idle streams must leave the executor available for run_fl.
            assert await asyncio.wait_for(asyncio.to_thread(lambda: True), 1)
            proxy = list(manager.all().values())[-1]
            call = calls[-1]
            for round_id in (1, 2):
                timeout = 0.05 if mode == "timeout" else 1
                fit = asyncio.create_task(asyncio.to_thread(
                    proxy.fit,
                    FitIns(ndarrays_to_parameters([]), {"round": round_id}),
                    timeout, round_id,
                ))
                instruction = await asyncio.wait_for(call.read(), 3)
                assert instruction.fit_ins.config["round"].sint64 == round_id
                if mode == "timeout":
                    with pytest.raises(grpc.aio.AioRpcError) as err:
                        await asyncio.wait_for(call.read(), 3)
                    assert err.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
                    with pytest.raises(GrpcBridgeClosed):
                        await asyncio.wait_for(fit, 3)
                    await _wait_for(lambda: manager.num_available() == 39)
                    break

                response = transport_pb2.ClientMessage.FitRes(
                    status=transport_pb2.Status(code=transport_pb2.OK),
                    parameters=serde.parameters_to_proto(ndarrays_to_parameters([])),
                    num_examples=round_id,
                )
                await queue.put(transport_pb2.ClientMessage(fit_res=response))
                result = await asyncio.wait_for(fit, 3)
                assert result.num_examples == round_id
                assert manager.num_available() == 40
    finally:
        for call in calls:
            call.cancel()
        await server.stop(0)
        await _wait_for(lambda: manager.num_available() == 0)


@pytest.mark.parametrize("mode", ["normal_response", "timeout"])
def test_idle_bidi_connections_do_not_starve_training_executor(mode):
    asyncio.run(_exercise_many_idle_connections(mode))
