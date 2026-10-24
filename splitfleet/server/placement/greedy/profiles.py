"""Captured computation and differentiable-frontier volume proxies."""

import math
import re
import numpy as np


def _dtype_name(dtype):
    name = str(dtype).rsplit(".", 1)[-1]
    match = re.fullmatch(r"<dtype: '([^']+)'>", name)
    if match:
        name = match.group(1)
    return name


def _dtype_itemsize(dtype):
    name = _dtype_name(dtype)
    if name == "bfloat16":
        return 2
    try:
        return np.dtype(name).itemsize
    except TypeError as error:
        raise ValueError("Captured differentiable frontier dtype is unsupported") from error

def _structural_fallback(catalog):
    """Interpolate by graph position when native MAC metadata is unavailable."""
    result = {}
    for candidate in catalog:
        forward = candidate.boundary_forward_bytes
        gradient = candidate.boundary_gradient_bytes
        result[candidate.boundary] = {
            "prefix_mac_fraction": candidate.graph_position_ratio,
            "gradient_volume_proxy_bytes_per_sample": float(gradient if gradient is not None else forward or 0),
            "l2s_structural": [candidate.suffix_node_count, 0, 0, forward or 0,
                               gradient or 0, candidate.suffix_parameter_bytes or 0, 0, 0],
            "structural_feature_scope": "graph-position fallback; catalog byte proxies",
        }
    return result


def graph_workload_features(provider):
    """Eight L2S-like structural features plus three SFL exchange features.

    MACs are calculated from captured output/parameter shapes. BatchNorm,
    pooling and residual elementwise work enter the miscellaneous activation
    group, an explicit extension beyond the original sequential CNN model.
    Gradient volume is a captured differentiable-frontier proxy, not wire bytes.
    """
    catalog = provider.get_candidates(training=True)
    if not hasattr(provider, "_backends"):
        return _structural_fallback(catalog)
    graph = provider._backends[True].runtime.trace_graph
    nodes = list(graph.nodes)
    by_id = {node.canonical_id: node for node in nodes}
    index = {node.canonical_id: position for position, node in enumerate(nodes)}
    batch = provider._backends[True].runtime.traced_batch_size
    rows = []
    parameter_seen = set()
    for node in nodes:
        shape = node.output_shape
        elements = math.prod(shape) / batch if isinstance(shape, tuple) and all(isinstance(v, int) and v > 0 for v in shape) else 0
        parameters = [ref.handle for ref in node.param_refs
                      if getattr(getattr(ref, "handle", None), "shape", None) is not None]
        kind = str(node.op_type).lower()
        group = 0 if "conv" in kind else 1 if kind in {"linear", "addmm"} else 2
        cost = elements
        weights = [parameter for parameter in parameters if len(parameter.shape) >= 2]
        if group == 0 and weights:
            cost = elements * math.prod(weights[0].shape[1:])
        elif group == 1 and weights:
            cost = elements * weights[0].shape[-1]
        if any(getattr(node, field, False) for field in ("is_input", "is_output", "is_buffer", "is_buffer_only_source", "is_param_source")):
            cost = 0
        values = np.zeros(6, dtype=np.float64)
        values[group] = cost
        for parameter in parameters:
            if id(parameter) not in parameter_seen:
                values[group + 3] += math.prod(parameter.shape)
                parameter_seen.add(id(parameter))
        rows.append(values)
    cumulative = np.vstack([np.zeros(6), np.cumsum(rows, axis=0)])
    total = cumulative[-1]
    if total[:2].sum() <= 0:
        return _structural_fallback(catalog)
    result = {}
    for candidate in catalog:
        kind, target = candidate.boundary.split(":", 1)
        prefix_end = index[target] + int(kind == "after")
        prefix = cumulative[prefix_end]
        suffix = total - prefix
        gradient_proxy = 0
        for label in candidate.metadata["boundary_tensor_labels"]:
            node = by_id[label]
            differentiable = bool(node.requires_grad)
            if differentiable and isinstance(node.output_shape, tuple) and all(isinstance(v, int) for v in node.output_shape):
                gradient_proxy += math.prod(node.output_shape) * _dtype_itemsize(node.dtype) / batch
        result[candidate.boundary] = {
            "l2s_structural": [*suffix[:3], candidate.boundary_forward_bytes,
                               gradient_proxy, *suffix[3:]],
            "prefix_mac_fraction": float(prefix[:2].sum() / total[:2].sum()),
            "gradient_volume_proxy_bytes_per_sample": float(gradient_proxy),
            "total_mac_units_per_sample": float(total[:2].sum()),
            "structural_feature_scope": "captured_shapes; misc includes BN/pool/residual; gradient volume is a proxy",
        }
    return result
