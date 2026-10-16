import asyncio
import socket
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from threading import Event
from types import SimpleNamespace

import grpc
import pytest

from splitfleet.common import BatchData, ControlCode, ServerModelFitIns
from splitfleet.server.grpc.servicer import ServerModelServicer
from splitfleet.server.server_model.utils import ClientRequestGroup
from splitfleet.server.stage_runtime.leased_manager import LeasedServerModelManager
from splitfleet.transport import BoundaryEnvelope, encode_boundary
from splitfleet.virtual.suffix_service import (
    LeasedSuffixServicer,
    SuffixService,
    suffix_connection,
)


class IndependentStrategy:
    def route_client_request(self, cid, method_name):
        return ClientRequestGroup(cid), None

    def mark_ready_requests(self):
        pass


class EchoModel:
    def __init__(self, entered, release):
        self.entered, self.release = entered, release

    def echo(self, batches):
        self.entered.set()
        assert self.release.wait(timeout=5)
        return batches


class EchoManager:
    def __init__(self, entered, release):
        self.strategy = IndependentStrategy()
        self.model = EchoModel(entered, release)

    def get_server_model(self, sid):
        return self.model


class BlockingSuffix(EchoModel):
    def to_server_model(self):
        return self

    def configure_fit(self, config):
        pass

    def train_tail(self, batches):
        return self.echo(batches)


def test_unary_deadline_cancels_call_and_drains_execution_safely():
    entered, release = Event(), Event()
    manager = EchoManager(entered, release)
    manager.strategy.mark_ready_requests = lambda: None

    # Independent groups are immediately ready, as in the virtual runtime.
    def route(cid, method_name):
        group = ClientRequestGroup(cid)
        group.mark_as_ready()
        return group, None

    manager.strategy.route_client_request = route
    servicer = ServerModelServicer(manager, execution_concurrency=1)
    service = SuffixService(manager, servicer=servicer).start()
    try:
        with suffix_connection(service.address, "client", timeout=2) as proxy:
            with pytest.raises(grpc.RpcError) as error:
                proxy.echo(
                    BatchData(
                        data={"payload": b"boundary"}, control_code=ControlCode.OK
                    ),
                    _streams_=False,
                    _timeout_=0.05,
                )
            assert error.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
            assert entered.is_set()
        release.set()
        asyncio.run_coroutine_threadsafe(servicer.drain(), service.loop).result(
            timeout=5
        )
        assert not servicer._computations
        assert servicer.active_executions == 0
    finally:
        release.set()
        service.stop()


def test_dynamic_port_and_occupied_port_failure_cleanup():
    service = SuffixService(EchoManager(Event(), Event())).start()
    assert service.port > 0
    service.stop()
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        blocked = SuffixService(None, port=occupied.getsockname()[1])
        with pytest.raises((OSError, RuntimeError)):
            blocked.start()
        assert blocked.loop is None


def leased_echo_manager(entered, release, released_models):
    delegate = LeasedServerModelManager(
        lambda: BlockingSuffix(entered, release),
        capacity=1,
        state_store=None,
        release_model=released_models.append,
    )
    delegate.initialize_server_models(
        [ServerModelFitIns(parameters=[], config={}, sid=cid) for cid in ("1", "2")]
    )
    strategy = IndependentStrategy()
    strategy._cid_to_sid_mapping = {"1": "1", "2": "2"}

    def route(cid, method_name):
        group = ClientRequestGroup(cid)
        group.mark_as_ready()
        return group, None

    strategy.route_client_request = route
    return SimpleNamespace(
        strategy=strategy,
        _delegate=delegate,
        get_server_model=delegate.get_server_model,
        acquire_replica=delegate.acquire,
        finish_replica=delegate.finish,
    )


@pytest.mark.parametrize("reconcile", [False, True])
def test_abort_waits_for_timed_out_computation_before_admitting_next_replica(reconcile):
    entered, release = Event(), Event()
    released_models = []
    manager = leased_echo_manager(entered, release, released_models)
    delegate = manager._delegate
    servicer = LeasedSuffixServicer(
        manager,
        capacity=1,
        concurrency=2,
        state_store=None,
        seed=3,
    )
    servicer.prepare_round(["1", "2"], 1)
    service = SuffixService(manager, servicer=servicer).start()
    request = BatchData(data={}, control_code=ControlCode.OK)
    try:
        with (
            suffix_connection(
                service.address,
                "1",
                timeout=3,
                metadata=(("authorization", f"Bearer {servicer.tokens['1']}"),),
            ) as first,
            suffix_connection(
                service.address,
                "2",
                timeout=3,
                metadata=(("authorization", f"Bearer {servicer.tokens['2']}"),),
            ) as second,
            ThreadPoolExecutor(max_workers=2) as callers,
        ):
            try:
                first.virtual_acquire(request, _streams_=False)
                with pytest.raises(grpc.RpcError) as error:
                    first.train_tail(
                        BatchData(
                            data={
                                "boundary": encode_boundary(
                                    BoundaryEnvelope(
                                        tensors=(), client_id="1", round_id=1
                                    )
                                )
                            },
                            control_code=ControlCode.OK,
                        ),
                        _streams_=False,
                        _timeout_=0.05,
                    )
                assert error.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
                assert entered.is_set()
                aborting = callers.submit(first.virtual_abort, request, _streams_=False)
                acquiring = callers.submit(
                    second.virtual_acquire, request, _streams_=False
                )
                with pytest.raises(FutureTimeout):
                    aborting.result(timeout=0.1)
                assert not acquiring.done()
                assert set(delegate.live) == {"1"}
                assert not released_models
                assert servicer.active_executions == 1
                if reconcile:
                    cleanup = asyncio.run_coroutine_threadsafe(
                        servicer.reconcile(), service.loop
                    )
                    with pytest.raises(FutureTimeout):
                        cleanup.result(timeout=0.1)
                    assert set(delegate.live) == {"1"}
                    assert not released_models
                    assert servicer.active_executions == 1
                    release.set()
                    cleanup.result(timeout=3)
                    for call in (aborting, acquiring):
                        with pytest.raises(grpc.RpcError):
                            call.result(timeout=3)
                    assert not delegate.live
                    assert not servicer._requests
                    assert not servicer._leases
                    assert servicer._replica_slots._value == 1
                    assert len(released_models) == 1
                    return
                release.set()
                aborting.result(timeout=3)
                acquiring.result(timeout=3)
                assert set(delegate.live) == {"2"}
                assert delegate.peak_live == 1
                second.virtual_abort(request, _streams_=False)
                assert not delegate.live
                assert len(released_models) == 2
            finally:
                release.set()
    finally:
        release.set()
        service.stop()


@pytest.mark.parametrize("fail", [False, True])
def test_repeated_cancellation_keeps_execution_slot_until_thread_finishes(fail):
    async def exercise():
        entered, release = Event(), Event()
        servicer = ServerModelServicer(None, execution_concurrency=2)

        def compute():
            entered.set()
            assert release.wait(timeout=5)
            if fail:
                raise RuntimeError("Failed computation after cancellation")

        running = asyncio.create_task(servicer.execute(compute))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            running.cancel()
            await asyncio.sleep(0)
            running.cancel()
            await asyncio.sleep(0)
            assert not running.done()
            assert servicer.active_executions == 1
            assert len(servicer._computations) == 1
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await running
            assert servicer.active_executions == 0
            assert not servicer._computations
        finally:
            release.set()
            await asyncio.gather(running, return_exceptions=True)
            servicer.shutdown_executor()

    asyncio.run(exercise())


def test_abort_timeout_while_queued_for_lifecycle_retains_replica_capacity():
    release = Event()
    manager = leased_echo_manager(Event(), release, [])
    servicer = LeasedSuffixServicer(
        manager,
        capacity=1,
        concurrency=2,
        state_store=None,
        seed=3,
    )
    servicer.prepare_round(["1", "2"], 1)
    service = SuffixService(manager, servicer=servicer).start()
    request = BatchData(data={}, control_code=ControlCode.OK)
    busy = []
    try:
        with suffix_connection(
            service.address,
            "1",
            timeout=3,
            metadata=(("authorization", f"Bearer {servicer.tokens['1']}"),),
        ) as proxy:
            proxy.virtual_acquire(request, _streams_=False)
            for _ in range(1):
                entered = Event()

                def hold_lane(entered=entered):
                    entered.set()
                    assert release.wait(timeout=5)

                busy.append(
                    asyncio.run_coroutine_threadsafe(
                        servicer.execute_control(hold_lane),
                        service.loop,
                    )
                )
                assert entered.wait(timeout=3)
            with pytest.raises(grpc.RpcError) as error:
                proxy.virtual_abort(request, _streams_=False, _timeout_=0.05)
            assert error.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
            # Synchronize with the RPC loop after the deadline cancellation.
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0), service.loop).result(
                timeout=3
            )
            assert set(manager._delegate.live) == {"1"}
            assert servicer._leases == {"1"}
            assert servicer._replica_slots._value == 0
            release.set()
            for call in busy:
                call.result(timeout=3)
            proxy.virtual_abort(request, _streams_=False)
            assert not manager._delegate.live
            assert servicer._replica_slots._value == 1
    finally:
        release.set()
        service.stop()


def test_slow_lifecycle_work_cannot_take_a_training_execution_lane():
    async def exercise():
        entered, release = Event(), Event()
        servicer = ServerModelServicer(None, execution_concurrency=1, control_concurrency=1)

        def prepare():
            entered.set()
            assert release.wait(5)

        preparing = asyncio.create_task(servicer.execute_control(prepare))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            value, _ = await asyncio.wait_for(servicer.execute(lambda: 'trained'), timeout=1)
            assert value == 'trained' and servicer.peak_executions == 1
            assert not preparing.done()
        finally:
            release.set()
            await preparing
            servicer.shutdown_executor()

    asyncio.run(exercise())
