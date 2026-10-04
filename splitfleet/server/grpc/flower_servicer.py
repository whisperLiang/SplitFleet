import asyncio
import uuid
from typing import AsyncIterator, Callable
from logging import WARNING

import grpc
from flwr.common.logger import log

from flwr.proto import transport_pb2_grpc  # pylint: disable=E0611
from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    ServerMessage,
)
from flwr.server.client_manager import ClientManager
from flwr.server.superlink.fleet.grpc_bidi.grpc_bridge import (
    GrpcBridge,
    GrpcBridgeClosed,
    ResWrapper,
)
from flwr.server.superlink.fleet.grpc_bidi.grpc_client_proxy import GrpcClientProxy

from splitfleet.server.grpc.async_grpc_bridge import AsyncGrpcBridge


def default_bridge_factory() -> AsyncGrpcBridge:
    """Return a bridge supporting nonblocking async instruction delivery."""
    return AsyncGrpcBridge()


def default_grpc_client_proxy_factory(cid: str, bridge: GrpcBridge) -> GrpcClientProxy:
    """Return GrpcClientProxy instance."""
    return GrpcClientProxy(cid=cid, bridge=bridge)


class FlowerServiceServicer(transport_pb2_grpc.FlowerServiceServicer):
    """Bridge Flower's blocking proxies to the native asyncio gRPC service.

    Cancellation closes the bridge and unregisters the proxy, including while
    Join waits for its next instruction. A disconnected in-flight instruction
    fails; it is never replayed against a potentially updated training state.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        grpc_bridge_factory: Callable[[], AsyncGrpcBridge] = default_bridge_factory,
        grpc_client_proxy_factory: Callable[
            [str, GrpcBridge], GrpcClientProxy
        ] = default_grpc_client_proxy_factory,
    ) -> None:
        self.client_manager: ClientManager = client_manager
        self.grpc_bridge_factory = grpc_bridge_factory
        self.client_proxy_factory = grpc_client_proxy_factory

    async def Join(  # pylint: disable=invalid-name
        self,
        request_iterator: AsyncIterator[ClientMessage],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[ServerMessage]:
        cid: str = uuid.uuid4().hex
        bridge = self.grpc_bridge_factory()
        client_proxy = self.client_proxy_factory(cid, bridge)
        registered = self.client_manager.register(client_proxy)

        def close_bridge(_context=None) -> None:
            bridge.close()
            if registered:
                self.client_manager.unregister(client_proxy)

        # Use the asyncio callback; the sync migration context's add_callback
        # did not reliably run on cancellation (grpc/grpc#38346).
        context.add_done_callback(close_bridge)
        try:
            if not registered:
                return
            async for ins_wrapper in bridge.ins_wrapper_async_iterator():
                yield ins_wrapper.server_message
                try:
                    client_message = await asyncio.wait_for(
                        request_iterator.__anext__(), timeout=ins_wrapper.timeout
                    )
                except asyncio.TimeoutError:
                    await context.abort(
                        code=grpc.StatusCode.DEADLINE_EXCEEDED,
                        details=f"Timeout of {ins_wrapper.timeout}sec was exceeded.",
                    )
                    return
                bridge.set_res_wrapper(ResWrapper(client_message=client_message))
        except (StopAsyncIteration, GrpcBridgeClosed):
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log(WARNING, "Bidi Join ended for client %s: %s: %s",
                cid, type(exc).__name__, exc)
            raise
        finally:
            close_bridge()
