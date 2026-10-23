"""Active directional goodput probes on the existing split RPC channel."""

from __future__ import annotations

import math
from statistics import median
import time

from .typing import BatchData, ControlCode


PROBE_CONFIG_KEY = "splitfleet_active_network_probe_round"
PROBE_RESULT_KEY = "splitfleet_active_network_probe_receipt"
PROBE_METHOD = "splitfleet_transport_probe"
PROBE_SCHEMA = "splitfleet.active-network-probe.v1"
PROBE_SOURCE = "measured_split_rpc"
PROBE_MODE = "active_split_rpc_probe_each_training_round"
# A 1 KiB baseline can trigger delayed small-message delivery on the physical
# SSH/HTTP2 path. Warm each payload shape and use a multi-packet baseline so
# size-dependent transmission remains resolvable on the fast link as well.
SMALL_BYTES = 65_536
LARGE_BYTES = 2_097_152
PROBE_REPEATS = 3
PROBE_ATTEMPTS = 3


class UnresolvedNetworkProbe(ValueError):
    """Valid RPC samples whose size-dependent duration is hidden by jitter."""


def probe_response(data, metadata):
    """Serve bounded, stateless probe traffic without touching a model."""
    payload = data.get("payload")
    if not isinstance(payload, bytes) or not 1 <= len(payload) <= LARGE_BYTES:
        raise ValueError("Invalid active network probe payload")
    try:
        reply_bytes = int(metadata["reply_bytes"])
        round_id = int(metadata["round_id"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Invalid active network probe metadata") from exc
    if not 1 <= reply_bytes <= LARGE_BYTES or round_id < 1:
        raise ValueError("Invalid active network probe size or round")
    return BatchData(data={"payload": bytes(reply_bytes)}, control_code=ControlCode.OK,
                     metadata={"request_bytes": str(len(payload)),
                               "reply_bytes": str(reply_bytes), "round_id": str(round_id)})


def measure_network(proxy, *, round_id, timeout=60):
    """Measure RTT and each direction using only the client's monotonic clock.

    All probes run in one property handler on its training RPC channel. Model
    construction, Flower task dispatch and connection setup are outside the
    sample windows. The result contains observations, never configured rates.
    """
    if type(round_id) is not int or round_id < 1:
        raise ValueError("Active network probe requires a positive integer round")
    payloads = {size: bytes(size) for size in (SMALL_BYTES, LARGE_BYTES)}

    def sample(request_bytes, reply_bytes):
        request = BatchData(data={"payload": payloads[request_bytes]},
                            control_code=ControlCode.OK,
                            metadata={"reply_bytes": str(reply_bytes), "round_id": str(round_id)})
        began = time.perf_counter_ns()
        response = proxy._blocking_request(PROBE_METHOD, request, _streams_=False, _timeout_=timeout)
        elapsed_ms = (time.perf_counter_ns() - began) / 1e6
        if (response.control_code != ControlCode.OK
                or response.data.get("payload") != payloads[reply_bytes]
                or response.metadata.get("request_bytes") != str(request_bytes)
                or response.metadata.get("reply_bytes") != str(reply_bytes)
                or response.metadata.get("round_id") != str(round_id)
                or not math.isfinite(elapsed_ms) or elapsed_ms <= 0):
            raise ValueError("Active network probe reply failed validation")
        return {"request_bytes": request_bytes, "reply_bytes": reply_bytes,
                "roundtrip_ms": elapsed_ms}

    began = time.perf_counter()
    shapes = ((SMALL_BYTES, SMALL_BYTES), (LARGE_BYTES, SMALL_BYTES),
              (SMALL_BYTES, LARGE_BYTES))
    warmup = [sample(request, reply) for request, reply in shapes]
    rejected_attempts = []
    for attempt in range(PROBE_ATTEMPTS):
        samples = []
        for _ in range(PROBE_REPEATS):
            samples.extend((sample(SMALL_BYTES, SMALL_BYTES),
                            sample(LARGE_BYTES, SMALL_BYTES),
                            sample(SMALL_BYTES, LARGE_BYTES)))
        receipt = {"schema": PROBE_SCHEMA, "source": PROBE_SOURCE, "round_id": round_id,
                   "samples": samples, "warmup_samples": warmup,
                   "rejected_attempts": rejected_attempts,
                   "elapsed_sec": time.perf_counter() - began}
        try:
            estimate_network(receipt, round_id=round_id)
        except UnresolvedNetworkProbe:
            if attempt + 1 == PROBE_ATTEMPTS:
                raise
            rejected_attempts.append({"samples": samples})
        else:
            return receipt


def estimate_network(receipt, *, round_id):
    """Estimate application goodput by varying one direction at a time.

    Subtracting small-RPC RTT cancels the other direction and fixed overhead.
    No subtraction uses clocks from different hosts. RTT still includes the
    RPC's fixed processing overhead; these are application-level estimates.
    """
    if (receipt.get("schema") != PROBE_SCHEMA or receipt.get("source") != PROBE_SOURCE
            or type(receipt.get("round_id")) is not int or receipt["round_id"] != round_id):
        raise ValueError("Active network probe must match the current round and RPC source")
    groups = {(SMALL_BYTES, SMALL_BYTES): [], (LARGE_BYTES, SMALL_BYTES): [],
              (SMALL_BYTES, LARGE_BYTES): []}
    for sample in receipt.get("samples", []):
        request, reply = sample.get("request_bytes"), sample.get("reply_bytes")
        if type(request) is not int or type(reply) is not int or (request, reply) not in groups:
            raise ValueError("Active network probe has invalid directional payload sizes")
        elapsed = sample.get("roundtrip_ms")
        if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed) or elapsed <= 0):
            raise ValueError("Active network probe requires finite positive measured durations")
        groups[(request, reply)].append(float(elapsed))
    if any(len(rows) != PROBE_REPEATS for rows in groups.values()):
        raise ValueError("Active network probe requires complete repeated directional samples")
    rtt_ms = median(groups[(SMALL_BYTES, SMALL_BYTES)])
    up_ms = median(groups[(LARGE_BYTES, SMALL_BYTES)]) - rtt_ms
    down_ms = median(groups[(SMALL_BYTES, LARGE_BYTES)]) - rtt_ms
    if up_ms <= 0 or down_ms <= 0:
        raise UnresolvedNetworkProbe(
            "Active network probe could not resolve directional goodput "
            f"(small RTT={rtt_ms:.6f} ms, upload delta={up_ms:.6f} ms, "
            f"download delta={down_ms:.6f} ms)")
    bits = (LARGE_BYTES - SMALL_BYTES) * 8
    context = {"upload_mbps": bits / (up_ms * 1000),
               "download_mbps": bits / (down_ms * 1000), "rtt_ms": rtt_ms}
    if any(not math.isfinite(value) or value <= 0 for value in context.values()):
        raise ValueError("Active network probe produced invalid goodput or RTT")
    return context
