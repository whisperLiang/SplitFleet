import asyncio
import time
from threading import Event

import grpc
import pytest
import torch

from experiments.tcp_pacing import PacedProxy
from experiments.virtual_sfl.workloads import TinyClassifier
from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.common import BatchData, ControlCode, ServerModelFitIns
from splitfleet.common.constants import CLIENT_ID_CONFIG_KEY
from splitfleet.server.grpc.servicer import ServerModelServicer
from splitfleet.server.server_model.utils import ClientRequestGroup
from splitfleet.server.stage_runtime.manager import StageRuntimeManager
from splitfleet.server.stage_runtime.storage import RoundStateStore
from splitfleet.server.strategy import AutoSplitStrategy
from splitfleet.tasks import ImageClassificationTask
from splitfleet.virtual.config import Network
from splitfleet.virtual.network import TaskNetwork
from splitfleet.virtual.suffix_service import (
    LeasedSuffixServicer,
    SuffixService,
    suffix_connection,
)
from tests.virtual.test_suffix_rpc import EchoManager


def test_authenticated_lease_rejects_replay_and_releases_real_suffix_graph(tmp_path):
    torch.set_num_threads(1)
    model = TinyClassifier()
    data = (torch.randn(2, 4), torch.tensor([0, 1]))
    sample = ImageClassificationTask().prepare_batch(data)
    strategy = AutoSplitStrategy(
        model=model,
        sample_inputs=sample,
        task=ImageClassificationTask(),
        boundary="after:head",
        aggregation_policy="splitfed",
        owned_state_exchange=True,
    )
    store = RoundStateStore(tmp_path)
    manager = StageRuntimeManager(
        init_server_model_fn=strategy.init_server_model_fn,
        autosplit_session=strategy.autosplit_session,
        replica_capacity=1,
        state_store=store,
    )
    strategy.bind_stage_runtime_manager(manager)
    manager.set_strategy(strategy)
    strategy._cid_to_sid_mapping = {"1": "1"}
    strategy._round_active_clients = ["1"]
    config = strategy._autosplit_config(1)
    config.update({CLIENT_ID_CONFIG_KEY: "1", "virtual_partition_id": 0})
    parameters = strategy.backend_adapter.export_ndarrays(model)
    manager.initialize_server_models(
        [ServerModelFitIns(parameters=parameters, config=config, sid="1")]
    )
    servicer = LeasedSuffixServicer(
        manager, capacity=1, concurrency=1, state_store=store, seed=3
    )
    servicer.prepare_round(["1"], 1)
    service = SuffixService(manager, servicer=servicer).start()
    request = BatchData(data={}, control_code=ControlCode.OK)
    try:
        with suffix_connection(service.address, "1", timeout=3) as unauthenticated:
            with pytest.raises(grpc.RpcError) as denied:
                unauthenticated.virtual_acquire(request, _streams_=False)
            assert denied.value.code() == grpc.StatusCode.UNAUTHENTICATED
        with suffix_connection(
            service.address,
            "1",
            timeout=3,
            metadata=(("authorization", f"Bearer {servicer.tokens['1']}"),),
        ) as proxy:
            proxy.virtual_acquire(request, _streams_=False)
            assert len(manager._delegate.live) == 1
            with pytest.raises(grpc.RpcError) as replay:
                proxy.virtual_acquire(request, _streams_=False)
            assert replay.value.code() == grpc.StatusCode.ALREADY_EXISTS
            client = AutoSplitSplitLearningClient(
                model=model,
                sample_inputs=sample,
                train_data=[data],
                task=ImageClassificationTask(),
                max_cached_runtimes=1,
            )
            client.server_model_proxy = proxy
            _, count, metrics = client.fit(parameters, config)
            assert count == 2 and metrics["gradient_download_bytes"] > 0
            proxy.virtual_commit(request, _streams_=False)
            assert not manager._delegate.live
            assert not manager._clone_runtime_cache
            result = manager.collect_server_fit_results()[0]
            assert result.config["suffix_optimization_steps"] == 1
            assert result.config["suffix_changed_tensors"] > 0
            with pytest.raises(grpc.RpcError) as replay:
                proxy.virtual_acquire(request, _streams_=False)
            assert replay.value.code() == grpc.StatusCode.ALREADY_EXISTS
            assert manager._delegate.total_created == 1
    finally:
        service.stop()
        manager.end_round()
        store.clear()


def test_tcp_constraint_delays_actual_protocol_bytes_by_the_declared_rate():
    entered, released = Event(), Event()
    released.set()
    manager = EchoManager(entered, released)

    def route(cid, method_name):
        group = ClientRequestGroup(cid)
        group.mark_as_ready()
        return group, None

    manager.strategy.route_client_request = route
    service = SuffixService(
        manager, servicer=ServerModelServicer(manager, execution_concurrency=1)
    ).start()
    payload = bytes(65536)
    try:
        with (
            TaskNetwork(
                service.address, Network(upload_mbps=0.5, download_mbps=2.0), round_id=1
            ) as path,
            suffix_connection(path.address, "client", timeout=5) as proxy,
        ):
            start = time.perf_counter()
            reply = proxy.echo(
                BatchData(data={"payload": payload}, control_code=ControlCode.OK),
                _streams_=False,
            )
            elapsed = time.perf_counter() - start
            assert reply.data["payload"] == payload
            assert elapsed >= len(payload) * 8 / 500000 * 0.85
            metrics = path.metrics()
            assert metrics["tcp_upload_bytes"] >= len(payload)
            assert metrics["tcp_upload_pacing_wait_sec"] > 0.5
    finally:
        service.stop()


@pytest.mark.parametrize("rate", [1000, None])
def test_tcp_pipeline_preserves_throughput_across_propagation_delay(tmp_path, rate):
    async def exercise():
        payload = bytes(range(256)) * (8 * 1024 * 1024 // 256)
        received = asyncio.get_running_loop().create_future()

        async def target(reader, writer):
            try:
                value = await reader.readexactly(len(payload))
                received.set_result((value, time.perf_counter()))
            finally:
                writer.close()
                await writer.wait_closed()

        target_server = await asyncio.start_server(target, "127.0.0.1", 0)
        proxy = PacedProxy(
            target=target_server.sockets[0].getsockname(),
            control=lambda: {"upload_mbps": rate, "rtt_ms": 100, "round_id": 1},
            output=tmp_path / "receipt.json",
        )
        proxy_server = await asyncio.start_server(proxy.handle, "127.0.0.1", 0)
        writer = None
        try:
            _, writer = await asyncio.open_connection(
                *proxy_server.sockets[0].getsockname()
            )
            started = time.perf_counter()
            writer.write(payload)
            await writer.drain()
            value, finished = await asyncio.wait_for(received, timeout=10)
            elapsed = finished - started
            expected = 0.05 + (len(payload) * 8 / (rate * 1_000_000) if rate else 0)
            assert value == payload
            assert elapsed >= expected * 0.85
            # Leave scheduling slack while rejecting the old queue's repeated
            # latency penalty (over four seconds for this transfer).
            assert elapsed < expected + 0.4
            assert proxy.counts["1"]["upload_bytes"] == len(payload)
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            proxy_server.close()
            target_server.close()
            await proxy_server.wait_closed()
            await target_server.wait_closed()
            for task in tuple(proxy.tasks):
                task.cancel()
            await asyncio.gather(*tuple(proxy.tasks), return_exceptions=True)

    asyncio.run(exercise())


def test_uncapped_pipeline_stops_reading_when_destination_is_blocked(tmp_path):
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()

        class Source:
            bytes_read = 0

            async def read(self, size):
                if self.bytes_read >= 8 * 1024 * 1024:
                    return b""
                self.bytes_read += size
                return bytes(size)

        class BlockedWriter:
            bytes_written = 0

            def write(self, block):
                self.bytes_written += len(block)

            async def drain(self):
                entered.set()
                await release.wait()

            def write_eof(self):
                pass

        proxy = PacedProxy(
            target=("127.0.0.1", 1),
            control=dict,
            output=tmp_path / "receipt.json",
        )
        source, writer = Source(), BlockedWriter()
        pumping = asyncio.create_task(proxy.pump(source, writer, "download"))
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            before = source.bytes_read
            for _ in range(20):
                await asyncio.sleep(0)
            assert source.bytes_read == before
            release.set()
            await asyncio.wait_for(pumping, timeout=3)
            assert writer.bytes_written == source.bytes_read == 8 * 1024 * 1024
        finally:
            release.set()
            pumping.cancel()
            await asyncio.gather(pumping, return_exceptions=True)

    asyncio.run(exercise())
