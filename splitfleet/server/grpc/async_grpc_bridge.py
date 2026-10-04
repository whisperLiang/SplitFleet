"""Async instruction delivery for Flower's synchronous client proxies."""

import asyncio
from collections.abc import AsyncIterator

from flwr.server.superlink.fleet.grpc_bidi.grpc_bridge import (
    GrpcBridge,
    InsWrapper,
    Status,
)


class AsyncGrpcBridge(GrpcBridge):
    """Wake an async consumer without dedicating a thread to an idle bridge.

    Flower retains ownership of the request/response state machine and the
    blocking proxy API. Each bridge has one async instruction consumer.
    """

    def __init__(self) -> None:
        super().__init__()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._instruction_available = asyncio.Event()

    def _transition(self, next_status: Status) -> None:
        super()._transition(next_status)
        if next_status not in (Status.INS_WRAPPER_AVAILABLE, Status.CLOSED):
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._instruction_available.set)
        except RuntimeError:
            # Closing a proxy after its event loop shuts down must still wake
            # Flower's blocking request callers via the parent transition.
            if not loop.is_closed():
                raise

    async def ins_wrapper_async_iterator(self) -> AsyncIterator[InsWrapper]:
        self._loop = asyncio.get_running_loop()
        iterator = self.ins_wrapper_iterator()
        while True:
            with self._cv:
                self._raise_if_closed()
                self._instruction_available.clear()
                # The condition uses a reentrant lock. Advancing Flower's
                # iterator only when ready preserves its transitions without
                # ever executing a blocking condition wait on the event loop.
                ins_wrapper = (
                    next(iterator)
                    if self._status == Status.INS_WRAPPER_AVAILABLE
                    else None
                )
            if ins_wrapper is None:
                await self._instruction_available.wait()
            else:
                yield ins_wrapper
