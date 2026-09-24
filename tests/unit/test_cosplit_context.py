from __future__ import annotations

import numpy as np

from splitfleet.server.placement.cosplit_ucb import ContextEncoder, SplitCandidateDescriptor


def _candidate() -> SplitCandidateDescriptor:
    return SplitCandidateDescriptor(
        boundary="after:any_operation_17",
        split_id="split-17",
        graph_position_ratio=0.4,
        prefix_node_count=4,
        suffix_node_count=6,
        total_node_count=10,
        boundary_forward_bytes=4096,
        boundary_gradient_bytes=4096,
        boundary_tensor_count=2,
        prefix_parameter_bytes=8192,
        suffix_parameter_bytes=16384,
        client_memory_bytes=None,
        server_memory_bytes=None,
        trainable=True,
        feature_abi_id="abi",
        graph_signature="graph",
    )


def test_context_dimensions_are_stable_and_model_name_agnostic() -> None:
    encoder = ContextEncoder()
    candidate = _candidate()
    assert encoder.edge_context(candidate).shape == (encoder.edge_dimension,)
    assert encoder.network_context(candidate, direction="upload").shape == (encoder.network_dimension,)
    assert encoder.server_context(candidate, max_concurrency=2).shape == (encoder.server_dimension,)
    assert encoder.switch_context(candidate, previous=None).shape == (encoder.switch_dimension,)
    assert all(np.isfinite(encoder.edge_context(candidate)))


def test_missing_telemetry_has_explicit_indicators() -> None:
    encoder = ContextEncoder()
    candidate = _candidate()
    edge = encoder.edge_context(candidate, {})
    network = encoder.network_context(candidate, {}, direction="download")
    server = encoder.server_context(candidate, {}, max_concurrency=1)
    switch = encoder.switch_context(candidate, previous=None, telemetry={})
    assert tuple(edge[[7, 9, 11]]) == (1.0, 1.0, 1.0)
    assert tuple(network[[7, 9, 11]]) == (1.0, 1.0, 1.0)
    assert tuple(server[[4, 6, 8, 11]]) == (1.0, 1.0, 1.0, 1.0)
    assert tuple(switch[[4, 6, 8]]) == (1.0, 1.0, 1.0)
