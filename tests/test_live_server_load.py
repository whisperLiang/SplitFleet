"""Real executor observations reach placement; synthetic queue events do not."""

import asyncio
import json
import socket
import time
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from splitfleet.server.grpc.servicer import ServerModelServicer
from splitfleet.server.telemetry import LiveServerTelemetry, NvidiaDeviceSampler
from splitfleet.server.placement.greedy import GreedyConfig, GreedyPlacementPolicy, PlacementFeedback, StaticCandidateProvider
from tests.unit.placement_fixtures import _candidate


async def _until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(poll(), 3)


@pytest.mark.parametrize("cancel_waiter", [False, True])
def test_actual_executor_queue_is_measured_and_wait_survives_round_boundary(cancel_waiter):
    async def exercise():
        load = LiveServerTelemetry(sampler=lambda: {})
        load.begin_round(1)
        servicer = ServerModelServicer(None, execution_concurrency=1, telemetry=load)
        entered, release = Event(), Event()

        def work():
            entered.set()
            assert release.wait(3)

        first = asyncio.create_task(servicer.execute(work))
        await _until(entered.is_set)
        second = asyncio.create_task(servicer.execute(lambda: 2))
        try:
            await _until(lambda: load.server_context()["queue_length"] == 1)
            assert load.server_context()["active_jobs"] == 1
            if cancel_waiter:
                second.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await second
                assert load.server_context()["queue_length"] == 0
            else:
                await asyncio.sleep(.02)
        finally:
            release.set()
            await first
            if not cancel_waiter:
                assert (await second)[0] == 2
            await servicer.drain()
            servicer.shutdown_executor()
        previous = load.server_context()
        assert previous["active_jobs"] == previous["queue_length"] == 0
        if not cancel_waiter:
            assert previous["queue_wait_p95_ms"] >= 10
        load.begin_round(2)
        current = load.server_context()
        assert current["queue_wait_mean_ms"] == previous["queue_wait_mean_ms"]
        assert current["queue_observation_round"] == 1
        assert current["execution_concurrency"] == 1

    asyncio.run(exercise())


def test_running_cancellation_does_not_report_idle_before_worker_finishes():
    async def exercise():
        load = LiveServerTelemetry(sampler=lambda: {})
        servicer = ServerModelServicer(None, execution_concurrency=1, telemetry=load)
        entered, release = Event(), Event()

        def work():
            entered.set()
            assert release.wait(3)

        task = asyncio.create_task(servicer.execute(work))
        await _until(entered.is_set)
        task.cancel()
        await asyncio.sleep(.01)
        assert load.server_context()["active_jobs"] == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert load.server_context()["active_jobs"] == 0
        servicer.shutdown_executor()

    asyncio.run(exercise())


def test_executor_submission_failure_releases_capacity_and_observed_activity():
    async def exercise():
        load = LiveServerTelemetry(sampler=lambda: {})
        servicer = ServerModelServicer(None, execution_concurrency=1, telemetry=load)
        servicer.shutdown_executor()
        with pytest.raises(RuntimeError, match="cannot schedule new futures"):
            await servicer.execute(lambda: None)
        values = load.server_context()
        assert values["active_jobs"] == values["queue_length"] == 0
        assert servicer.active_executions == 0
        assert servicer._execution_slots._value == 1

    asyncio.run(exercise())


def test_round_without_execution_does_not_reuse_older_queue_measurements():
    load = LiveServerTelemetry(sampler=lambda: {})
    load.begin_round(1)
    load.enqueued()
    load.started(.1)
    load.finished()
    load.begin_round(2)
    assert load.server_context()["queue_observation_round"] == 1
    load.begin_round(3)
    assert "queue_wait_mean_ms" not in load.server_context()


def test_gpu_sampling_maps_cuda_device_by_uuid_and_never_uses_visible_index(monkeypatch):
    import torch
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index:
        SimpleNamespace(uuid="selected-uuid") if index == 1 else pytest.fail("Wrong CUDA device"))

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="73, 250, 1000\n")

    monkeypatch.setattr("splitfleet.server.telemetry.subprocess.run", run)
    values = NvidiaDeviceSampler("cuda:1")()
    assert "--id=GPU-selected-uuid" in calls[0]
    assert values["gpu_utilization"] == .73
    assert values["memory_free_ratio"] == .25




def test_missing_failed_or_expired_hardware_measurements_are_not_zero_load():
    now = [10.]
    load = LiveServerTelemetry(sampler=lambda: {"gpu_utilization": 80.}, clock=lambda: now[0])
    load.sample_hardware()
    assert load.server_context()["gpu_utilization"] == 80
    now[0] += 4
    assert "gpu_utilization" not in load.server_context()

    def missing():
        raise FileNotFoundError("No GPU monitoring API")

    load._sampler = missing
    load.sample_hardware()
    values = load.server_context()
    assert "gpu_utilization" not in values
    assert values["hardware_error"] == "FileNotFoundError"
    assert "queue_wait_mean_ms" not in values


















def test_native_grpc_transmits_execution_measurements_and_cleans_up_sampler():
    import grpc
    from flwr.server.client_manager import SimpleClientManager
    from splitfleet.server.app import start_grpc_server
    from splitfleet.server.server_model.utils import ClientRequestGroup
    from splitfleet.proto import server_model_pb2, server_model_pb2_grpc

    class Manager:
        def __init__(self):
            self.strategy = self

        def route_client_request(self, **kwargs):
            group = ClientRequestGroup(kwargs["cid"])
            group.mark_as_ready()
            return group, None

        def mark_ready_requests(self):
            pass

        def get_server_model(self, sid):
            return self

        def echo(self, batches):
            time.sleep(.02)
            return batches

    async def exercise():
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            address = f"127.0.0.1:{reserved.getsockname()[1]}"
        load = LiveServerTelemetry(sampler=lambda: {"gpu_utilization": 35., "memory_free_ratio": .7})
        server = await start_grpc_server(SimpleClientManager(), Manager(), address,
                                        telemetry=load, execution_concurrency=1)
        try:
            async with grpc.aio.insecure_channel(address) as channel:
                stub = server_model_pb2_grpc.ServerModelStub(channel)
                responses = await asyncio.gather(*[
                    stub.UnaryRequest(server_model_pb2.BatchData(cid=cid, method="echo"), timeout=3)
                    for cid in ("a", "b")])
                for response in responses:
                    values = json.loads(response.metadata["splitfleet_server_load"])
                    assert values["active_jobs"] == 1
                    assert values["execution_concurrency"] == 1
                    assert values["gpu_utilization"] == 35
                assert max(float(r.metadata["suffix_server_queue_sec"]) for r in responses) >= .01
                assert load.server_context()["queue_wait_samples"] == 2
        finally:
            await server.stop(0)
            await server.splitfleet_servicer.drain()
            server.splitfleet_servicer.shutdown_executor()
        assert load._thread is None

    asyncio.run(exercise())


def test_native_training_entrypoint_automatically_binds_the_live_monitor(monkeypatch):
    from flwr.server.client_manager import SimpleClientManager
    from flwr.server import ServerConfig
    from flwr.server.history import History
    from splitfleet.server import app

    policy = GreedyPlacementPolicy(candidate_provider=StaticCandidateProvider([_candidate("cut", .25)]))
    strategy = SimpleNamespace(placement_policy=policy, runtime_device="cpu")
    manager = SimpleClientManager()
    server = SimpleNamespace(strategy=strategy, server_model_manager=None, client_manager=lambda: manager)
    observed = []

    def run_fl(*, server, config):
        load = server.strategy.placement_policy.server_load_provider
        assert isinstance(load, LiveServerTelemetry)
        observed.append(load.server_context()["source"])
        return History()

    monkeypatch.setattr(app, "run_fl", run_fl)
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{reserved.getsockname()[1]}"
    assert isinstance(app.start_server(server_address=address, server=server, config=ServerConfig(num_rounds=1)), History)
    assert observed == ["measured_server_runtime"]
    assert policy.server_load_provider._thread is None




