"""The physical comparison starts a round only when all six workers are ready."""

from __future__ import annotations

import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import copy

import pytest

from experiments.orchestrate_physical_multitask import RoundBarrier, validate_result


def test_round_barrier_releases_all_six_workers_together() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    address = f"127.0.0.1:{port}"
    identities = {f"worker-{index}" for index in range(6)}
    barrier = RoundBarrier(address, identities, rounds=1)
    barrier.start()

    def arrive(identity: str) -> bytes:
        with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
            connection.settimeout(2)
            connection.sendall(json.dumps({"round_id": 1, "id": identity}).encode() + b"\n")
            return connection.recv(1)

    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            first_five = [pool.submit(arrive, f"worker-{index}") for index in range(5)]
            assert not any(future.done() for future in first_five)
            sixth = pool.submit(arrive, "worker-5")
            assert [future.result(timeout=3) for future in (*first_five, sixth)] == [b"1"] * 6
        assert barrier.error is None
        assert len(barrier.records) == 1
        receipt = barrier.records[0]
        assert receipt["ready_ids"] == sorted(identities)
        assert max(receipt["ready_monotonic_ns"].values()) <= receipt["release_started_monotonic_ns"]
        assert receipt["release_started_monotonic_ns"] <= receipt["release_finished_monotonic_ns"]
    finally:
        barrier.stop()


def test_round_barrier_tracks_three_rounds_without_reusing_readiness() -> None:
    identities = {f"worker-{index}" for index in range(6)}
    barrier = RoundBarrier("127.0.0.1:0", identities, rounds=3)
    port = barrier.listener.getsockname()[1]
    barrier.start()
    def arrive(identity):
        releases = []
        for round_id in range(1, 4):
            with socket.create_connection(("127.0.0.1", port), timeout=3) as connection:
                connection.sendall(json.dumps({"round_id": round_id, "id": identity}).encode() + b"\n")
                releases.append(connection.recv(1))
        return releases
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            assert list(pool.map(arrive, sorted(identities))) == [[b"1"] * 3] * 6
        barrier.thread.join(timeout=3)
        assert barrier.error is None
        assert [r["round_id"] for r in barrier.records] == [1, 2, 3]
        for record in barrier.records:
            assert record["ready_ids"] == sorted(identities)
            assert max(record["ready_monotonic_ns"].values()) <= record["release_started_monotonic_ns"]
        assert barrier.records[0]["release_finished_monotonic_ns"] <= min(barrier.records[1]["ready_monotonic_ns"].values())
        assert barrier.diagnostic()["pending_round"]["missing_ids"] == []
    finally:
        barrier.stop()


def test_round_barrier_preserves_missing_worker_and_protocol_error() -> None:
    barrier = RoundBarrier("127.0.0.1:0", {"ready", "missing"}, rounds=2)
    port = barrier.listener.getsockname()[1]
    barrier.start()
    accepted = threading.Event()
    def arrive():
        with socket.create_connection(("127.0.0.1", port), timeout=3) as connection:
            connection.sendall(b'{"round_id":1,"id":"ready"}\n')
            accepted.set()
            return connection.recv(1)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(arrive)
            assert accepted.wait(timeout=3)
            # A wrong round must abort the barrier and unblock the earlier
            # arrival without issuing a successful readiness receipt.
            with socket.create_connection(("127.0.0.1", port), timeout=3) as connection:
                connection.sendall(b'{"round_id":2,"id":"missing"}\n')
                assert connection.recv(1) == b""
            assert waiting.result(timeout=3) == b""
        barrier.thread.join(timeout=3)
        report = barrier.diagnostic()
        assert report["pending_round"]["ready_ids"] == ["ready"]
        assert report["pending_round"]["missing_ids"] == ["missing"]
        assert report["pending_round"]["last_message"] == {"round_id": 2, "id": "missing"}
        assert report["error"]["type"] == "RuntimeError"
        assert barrier.records == []
    finally:
        barrier.stop()


def test_round_barrier_records_truncated_message_without_releasing_workers() -> None:
    barrier = RoundBarrier("127.0.0.1:0", {"worker"}, rounds=1)
    port = barrier.listener.getsockname()[1]
    barrier.start()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3) as connection:
            connection.sendall(b'{"round_id":1')
            connection.shutdown(socket.SHUT_WR)
            assert connection.recv(1) == b""
        barrier.thread.join(timeout=3)
        assert "incomplete" in barrier.diagnostic()["error"]["message"]
        assert barrier.records == []
    finally:
        barrier.stop()


def _complete_short_round():
    hosts = [{"id": f"host-{index}"} for index in range(3)]
    identities = [f"{host['id']}-{kind}" for host in hosts for kind in ("cpu", "gpu")]
    bundle = {"data_content_hash": "data", "partition_hash": "partition",
              "initial_model_hash": "initial", "train_size": 6,
              "image_model": "resnet50_pretrained",
              "partition_sizes": {str(i): 1 for i in range(6)},
              "assignments": {str(i): [i] for i in range(6)}}
    result = {**bundle, "task": "image_classification", "method": "fedavg", "rounds": 1,
              "image_model": "resnet50_pretrained", "pretrain_checkpoint_sha256": None,
              "final_model_hash": "updated", "expected_clients": 6, "fit_failures": [],
              "evaluation_records": [{"round_id": 1, "metrics": {"accuracy": 0.5}}],
              "fit_records": [{"round_id": 1, "cid": identity, "num_examples": 1,
                               "metrics": {"logical_client_id": identity, "pid": index + 1,
                                           "device": "cpu" if index % 2 == 0 else "cuda:0",
                                           "num_batches": 1, "partition_hash": "partition", "task_loss": 0.5,
                                           "fit_started_unix_ns": (index + 1) * 1_000_000_000,
                                           "fit_finished_unix_ns": (index + 1) * 1_000_000_000 + 10_000_000}}
                              for index, identity in enumerate(identities)]}
    receipts = [{"round_id": 1, "ready_ids": identities,
                 "ready_monotonic_ns": {identity: index + 1 for index, identity in enumerate(identities)},
                 "release_started_monotonic_ns": 7, "release_finished_monotonic_ns": 8}]
    return hosts, bundle, result, receipts


def test_clock_offsets_do_not_invalidate_a_complete_barrier_controlled_round():
    hosts, bundle, result, receipts = _complete_short_round()
    report = validate_result(result, task="image_classification", method="fedavg", rounds=1,
                             hosts=hosts, batch_size=1, bundle=bundle, barrier_records=receipts)
    assert report["valid"] and report["barrier_verified"]
    assert report["fit_interval_overlap_sec"]["1"] < 0
    assert len(report["warnings"]) == 1


@pytest.mark.parametrize("mutation", ["missing_receipt", "early_release", "missing_worker"])
def test_readiness_validation_rejects_incomplete_or_premature_rounds(mutation):
    hosts, bundle, result, receipts = _complete_short_round()
    receipts = copy.deepcopy(receipts)
    if mutation == "missing_receipt":
        receipts.clear()
    elif mutation == "early_release":
        receipts[0]["release_started_monotonic_ns"] = 1
    else:
        result["fit_records"].pop()
    report = validate_result(result, task="image_classification", method="fedavg", rounds=1,
                             hosts=hosts, batch_size=1, bundle=bundle, barrier_records=receipts)
    assert not report["valid"]
