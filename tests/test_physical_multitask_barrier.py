"""The physical comparison starts a round only when all six workers are ready."""

from __future__ import annotations

import json
import socket
from concurrent.futures import ThreadPoolExecutor

from experiments.orchestrate_physical_multitask import RoundBarrier


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
    finally:
        barrier.stop()
