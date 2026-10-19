from splitfleet.server.placement.greedy import SplitCandidateDescriptor

def _candidate(boundary: str, position: float) -> SplitCandidateDescriptor:
    prefix = int(position * 10)
    return SplitCandidateDescriptor(
        boundary=boundary,
        split_id=boundary,
        graph_position_ratio=position,
        prefix_node_count=prefix,
        suffix_node_count=10 - prefix,
        total_node_count=10,
        boundary_forward_bytes=1000,
        boundary_gradient_bytes=1000,
        boundary_tensor_count=1,
        prefix_parameter_bytes=1000,
        suffix_parameter_bytes=1000,
        client_memory_bytes=None,
        server_memory_bytes=None,
        trainable=True,
        feature_abi_id=f"abi-{boundary}",
        graph_signature="graph",
        framework_backend="pytorch",
    )


def _network_probe_receipt(round_id, *, upload_mbps, download_mbps, rtt_ms):
    """Synthetic directional RPC observations for deterministic controller tests."""
    from splitfleet.common.network_probe import (
        LARGE_BYTES, SMALL_BYTES, PROBE_REPEATS, PROBE_SCHEMA, PROBE_SOURCE,
    )
    samples = [
        {"request_bytes": request, "reply_bytes": reply,
         "roundtrip_ms": rtt_ms + (request - SMALL_BYTES) * 8 / (upload_mbps * 1000)
                         + (reply - SMALL_BYTES) * 8 / (download_mbps * 1000)}
        for _ in range(PROBE_REPEATS)
        for request, reply in ((SMALL_BYTES, SMALL_BYTES), (LARGE_BYTES, SMALL_BYTES),
                               (SMALL_BYTES, LARGE_BYTES))
    ]
    return {"schema": PROBE_SCHEMA, "source": PROBE_SOURCE, "round_id": round_id,
            "samples": samples, "warmup_samples": [samples[0]],
            "elapsed_sec": sum(row["roundtrip_ms"] for row in samples) / 1000}
