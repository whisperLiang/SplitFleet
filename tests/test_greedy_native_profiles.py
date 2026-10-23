"""Native framework metadata must produce comparable compute/wire costs."""

from types import SimpleNamespace

import numpy as np
import pytest

from splitfleet.server.placement.greedy.profiles import graph_workload_features


@pytest.mark.parametrize("dtype", ["torch.float32", "paddle.float32", "<dtype: 'float32'>", "float32"])
def test_native_convolution_weights_and_frontier_dtype_determine_cost(dtype):
    weights = np.zeros((8, 3, 3, 3), dtype=np.float32)
    convolution = SimpleNamespace(canonical_id="conv", output_shape=(2, 8, 4, 4),
        op_type="conv2d", param_refs=[SimpleNamespace(handle=weights)], requires_grad=True, dtype=dtype)
    tail = SimpleNamespace(canonical_id="tail", output_shape=(2, 8, 2, 2),
        op_type="conv2d", param_refs=[SimpleNamespace(handle=weights)], requires_grad=True, dtype=dtype)
    candidate = SimpleNamespace(boundary="after:conv", metadata={"boundary_tensor_labels": ["conv"]},
        boundary_forward_bytes=512)
    runtime = SimpleNamespace(trace_graph=SimpleNamespace(nodes=[convolution, tail]), traced_batch_size=2)
    provider = SimpleNamespace(get_candidates=lambda **_: [candidate],
                               _backends={True: SimpleNamespace(runtime=runtime)})

    row = graph_workload_features(provider)[candidate.boundary]
    assert row["gradient_volume_proxy_bytes_per_sample"] == 8 * 4 * 4 * 4
    assert row["total_mac_units_per_sample"] == (8 * 4 * 4 + 8 * 2 * 2) * 3 * 3 * 3
    assert row["prefix_mac_fraction"] == .8


def test_unknown_frontier_dtype_cannot_silently_change_network_cost():
    from splitfleet.server.placement.greedy.profiles import _dtype_itemsize

    with pytest.raises(ValueError, match="unsupported"):
        _dtype_itemsize("native.unknown")


def test_jax_vjp_gradient_volume_excludes_resident_weights_and_integer_metadata():
    conv = SimpleNamespace(canonical_id="conv", output_shape=(2, 8, 4, 4),
        op_type="conv2d", param_refs=[], requires_grad=None, dtype="float32")
    weights = SimpleNamespace(canonical_id="weights", output_shape=(8, 3, 3, 3),
        op_type="input", param_refs=[], requires_grad=None, dtype="float32", is_input=True)
    counter = SimpleNamespace(canonical_id="counter", output_shape=(), op_type="sum",
        param_refs=[], requires_grad=None, dtype="int64")
    candidate = SimpleNamespace(boundary="after:conv", framework_backend="jax",
        metadata={"boundary_tensor_labels": ["conv", "weights", "counter"],
                  "resident_parameter_labels": ["weights"]}, boundary_forward_bytes=512)
    runtime = SimpleNamespace(trace_graph=SimpleNamespace(nodes=[weights, conv, counter]), traced_batch_size=2)
    provider = SimpleNamespace(get_candidates=lambda **_: [candidate],
                               _backends={True: SimpleNamespace(runtime=runtime)})
    row = graph_workload_features(provider)[candidate.boundary]
    assert row["gradient_volume_proxy_bytes_per_sample"] == 8 * 4 * 4 * 4


def test_transport_cost_uses_shape_resolved_gradient_bytes_for_the_real_batch():
    from splitfleet.server.placement.greedy import GreedyPlacementPolicy

    policy = GreedyPlacementPolicy(candidate_provider=None)
    policy.profiles = {"after:conv": {"gradient_volume_proxy_bytes_per_sample": 999999}}
    candidate = SimpleNamespace(boundary="after:conv", prefix_parameter_bytes=32,
        boundary_forward_bytes=100, metadata={"boundary_forward_bytes_by_batch_size": {16: 4096},
        "boundary_gradient_bytes_by_batch_size": {"16": 4096}})
    activation, gradient, _, _ = policy._payload_volumes(candidate, 16)
    assert activation == gradient == 4096
