"""L2S controller adaptation and a profile greedy reference.

Controllers share physical split execution, admission, state exchange and timing.
L2S's decision equation and optional escaping rule are retained, with three
state/framing features added for roundwise SFL. The physical catalog contains
two-sided cuts; its fully-local escaping action is consequently unavailable.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from experiments.baselines.independent_ucb import IndependentUCB
from experiments.physical_placement import PhysicalIndependentUCB, save_json


def graph_workload_features(provider):
    """Eight L2S-like structural features plus three SFL exchange features.

    MACs are calculated from captured output/parameter shapes. BatchNorm,
    pooling and residual elementwise work enter the miscellaneous activation
    group, an explicit extension beyond the original sequential CNN model.
    Gradient volume is a captured differentiable-frontier proxy, not wire bytes.
    """
    catalog = provider.get_candidates(training=True)
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
        parameters = [ref.handle for ref in node.param_refs if isinstance(getattr(ref, "handle", None), torch.Tensor)]
        kind = str(node.op_type).lower()
        group = 0 if "conv" in kind else 1 if kind in {"linear", "addmm"} else 2
        cost = elements
        weights = [parameter for parameter in parameters if parameter.ndim >= 2]
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
                values[group + 3] += parameter.numel()
                parameter_seen.add(id(parameter))
        rows.append(values)
    cumulative = np.vstack([np.zeros(6), np.cumsum(rows, axis=0)])
    total = cumulative[-1]
    if total[:2].sum() <= 0:
        raise ValueError("Captured graph has no supported convolution/linear workload")
    result = {}
    for candidate in catalog:
        kind, target = candidate.boundary.split(":", 1)
        prefix_end = index[target] + int(kind == "after")
        prefix = cumulative[prefix_end]
        suffix = total - prefix
        gradient_proxy = 0
        for label in candidate.metadata["boundary_tensor_labels"]:
            node = by_id[label]
            if node.requires_grad and isinstance(node.output_shape, tuple) and all(isinstance(v, int) for v in node.output_shape):
                dtype = getattr(torch, str(node.dtype).removeprefix("torch."), None)
                if not isinstance(dtype, torch.dtype):
                    raise ValueError("Captured differentiable frontier dtype is unsupported")
                gradient_proxy += math.prod(node.output_shape) * torch.empty((), dtype=dtype).element_size() / batch
        result[candidate.boundary] = {
            "l2s_structural": [*suffix[:3], candidate.boundary_forward_bytes,
                               gradient_proxy, *suffix[3:]],
            "prefix_mac_fraction": float(prefix[:2].sum() / total[:2].sum()),
            "gradient_volume_proxy_bytes_per_sample": float(gradient_proxy),
            "total_mac_units_per_sample": float(total[:2].sum()),
            "structural_feature_scope": "captured_shapes; misc includes BN/pool/residual; gradient volume is a proxy",
        }
    return result


class LinUCBE:
    """L2S Algorithm 1: latency LCB, ridge updates and optional forced escaping."""

    def __init__(self, dimension, *, alpha=1.0, ridge=1.0, horizon=10, mu=.25):
        if dimension < 1 or alpha < 0 or ridge <= 0 or horizon < 1 or not 0 < mu < .5:
            raise ValueError("Invalid LinUCB-E configuration")
        self.a = np.eye(dimension) * ridge
        self.b = np.zeros(dimension)
        self.alpha = alpha
        self.interval = max(1, math.ceil(horizon ** mu))
        self.updates = 0

    def scores(self, features, frontend_seconds):
        features = np.asarray(features, dtype=np.float64)
        frontend_seconds = np.asarray(frontend_seconds, dtype=np.float64)
        inverse_x = np.linalg.solve(self.a, features.T).T
        theta = np.linalg.solve(self.a, self.b)
        return frontend_seconds + features @ theta - self.alpha * np.sqrt(np.maximum(np.sum(features * inverse_x, axis=1), 0))

    def choose(self, features, frontend_seconds, *, round_id, fully_local_index=None):
        scores = self.scores(features, frontend_seconds)
        if fully_local_index is not None and round_id % self.interval == 0:
            scores[fully_local_index] = math.inf
        if not np.isfinite(scores).any():
            raise ValueError("No available LinUCB-E action")
        return int(np.argmin(scores))

    def observe(self, features, offloaded_seconds):
        x = np.asarray(features, dtype=np.float64)
        if not np.isfinite(x).all() or not math.isfinite(offloaded_seconds) or offloaded_seconds < 0:
            raise ValueError("Invalid observed offloaded latency")
        self.a += np.outer(x, x)
        self.b += x * offloaded_seconds
        self.updates += 1


class AdaptiveReference(PhysicalIndependentUCB):
    """Private measured bootstrap plus a separately declared decision rule."""

    def __init__(self, *, candidate_provider, config, algorithm, horizon, output):
        if algorithm not in ("L2S", "Greedy"):
            raise ValueError("Unknown adaptive reference")
        super().__init__(candidate_provider=candidate_provider, config=config)
        self.algorithm = algorithm
        self.horizon = horizon
        self.output = Path(output)
        self.profiles = None
        self.linear = {}
        self.cost_tables = {}
        self.pending_features = {}
        self.decision_receipts = []
        self.state_file = None

    def _initialize_reference(self):
        if self.profiles is not None:
            return
        self.profiles = graph_workload_features(self.candidate_provider)
        catalog = tuple(self.candidate_provider.get_candidates(training=True))
        server = {row["boundary"]: row["local_tail_service_ms"] / 1000 for row in self.telemetry_provider.server_receipt["records"]}
        for cid, child in self.clients.items():
            receipt = child.telemetry_provider.receipt
            samples = {row["boundary"]: row for row in receipt["client_samples"]}
            anchors = []
            for boundary, row in samples.items():
                anchors.append((self.profiles[boundary]["prefix_mac_fraction"],
                    (row["client_forward_ms"] + row["client_backward_ms"]) / 1000,
                    server[boundary], boundary))
            self.cost_tables[cid] = {"anchors": sorted(anchors), "observed_front": {},
                "observed_tail": {}, "network_seconds_per_byte": 0., "network_intercept_seconds": 0.,
                "state_seconds_per_byte": 0., "state_intercept_seconds": 0.}
            table = self.cost_tables[cid]
            echo = receipt["transport_bootstrap"]["samples"]
            small = np.mean([row["roundtrip_ms"] / 1000 for row in echo if row["payload_bytes"] == 65536])
            large = np.mean([row["roundtrip_ms"] / 1000 for row in echo if row["payload_bytes"] == 2097152])
            table["network_seconds_per_byte"] = max((large-small)/(2*(2097152-65536)), 0)
            table["network_intercept_seconds"] = max(small - 2*65536*table["network_seconds_per_byte"], 0)
            exchange = receipt["state_exchange_bootstrap"]["samples"]
            low = np.mean([row["roundtrip_ms"] / 1000 for row in exchange if row["request_bytes"] + row["reply_bytes"] == 2*65536])
            high = np.mean([row["roundtrip_ms"] / 1000 for row in exchange if row["request_bytes"] + row["reply_bytes"] == 2097152+65536])
            table["state_seconds_per_byte"] = max((high-low)/(2097152-65536), 0)
            table["state_intercept_seconds"] = max(low - 2*65536*table["state_seconds_per_byte"], 0)
            self.linear[cid] = LinUCBE(11, horizon=self.horizon)

    def _front(self, cid, candidate):
        table = self.cost_tables[cid]
        if candidate.boundary in table["observed_front"]:
            return table["observed_front"][candidate.boundary]
        anchors = table["anchors"]
        return float(np.interp(self.profiles[candidate.boundary]["prefix_mac_fraction"],
                               [row[0] for row in anchors], [row[1] for row in anchors]))

    def _features(self, cid, candidates):
        props = self.telemetry_provider.client_context(cid)
        batch, count = int(props["batch_size"]), int(props["num_batches"])
        domain = tuple(self.candidate_provider.get_candidates(training=True))
        raw = []
        for candidate in domain:
            profile = self.profiles[candidate.boundary]
            values = np.asarray(profile["l2s_structural"]) * batch * count
            values[3] = candidate.metadata["boundary_forward_bytes_by_batch_size"][batch] * count
            upload = candidate.metadata["optimizer_prefix_parameter_bytes"]
            raw.append([*values, float(upload), float(self.state_download_bytes), 1.])
        raw = np.asarray(raw)
        normalized = raw / np.maximum(np.max(raw, axis=0), 1) / math.sqrt(raw.shape[1])
        lookup = {candidate.boundary: row for candidate, row in zip(domain, normalized)}
        return np.asarray([lookup[candidate.boundary] for candidate in candidates])

    def plan_round(self, *, round_id, client_ids, training):
        if not training:
            # Flower requests an evaluation plan even when fraction_evaluate=0.
            # Reuse the evaluation catalog without changing training state.
            return super().plan_round(round_id=round_id, client_ids=client_ids,
                                      training=False)
        self._initialize_reference()
        assignment = {}
        for cid in sorted(client_ids):
            child = self.clients[cid]
            child.state_download_bytes = self.state_download_bytes
            candidates = tuple(child._catalog())
            props = {**child._client_telemetry(cid), **child.telemetry_provider.clients[cid]}
            server = child._server_telemetry()
            feasible = tuple(candidate for candidate in candidates if child.feasibility_filter.check(candidate,
                client_id=cid, round_id=round_id, capabilities={**props, **server}).feasible)
            if not feasible:
                raise ValueError("No admitted feasible reference action")
            count = int(props["num_batches"])
            front = np.asarray([self._front(cid, candidate) * count for candidate in feasible])
            features = self._features(cid, feasible)
            if self.algorithm == "L2S":
                position = self.linear[cid].choose(features, front, round_id=round_id)
                chosen = feasible[position]
                self.pending_features[(round_id, cid)] = features[position]
                scores = self.linear[cid].scores(features, front)
            elif self.algorithm == "Greedy":
                table = self.cost_tables[cid]
                anchors = table["anchors"]
                scores = []
                batch = int(props["batch_size"])
                for candidate, frontend in zip(feasible, front):
                    fraction = self.profiles[candidate.boundary]["prefix_mac_fraction"]
                    tail = table["observed_tail"].get(candidate.boundary, float(np.interp(fraction, [row[0] for row in anchors], [row[2] for row in anchors])))
                    activation = candidate.metadata["boundary_forward_bytes_by_batch_size"][batch]
                    gradient = self.profiles[candidate.boundary]["gradient_volume_proxy_bytes_per_sample"] * batch
                    split_network = table["network_intercept_seconds"] + (activation + gradient) * table["network_seconds_per_byte"]
                    state_network = table["state_intercept_seconds"] + (self.state_download_bytes + candidate.metadata["optimizer_prefix_parameter_bytes"]) * table["state_seconds_per_byte"]
                    scores.append(frontend + count * (tail + split_network) + state_network)
                position = int(np.argmin(scores));chosen = feasible[position]
            else:
                raise ValueError("Unknown adaptive reference")
            assignment[cid] = chosen.boundary
            child._round_assignments[(round_id, True)] = {cid: chosen.boundary}
            child._round_batch_counts[(round_id, True)] = {cid: count}
            child._round_contexts[(round_id, True, cid, chosen.boundary)] = (
                child._execution_profile(cid, props, chosen), child._contexts(client_id=cid,
                    candidate=chosen, client_telemetry=props, server_telemetry=server))
            child.round_diagnostics[round_id] = {"reference_algorithm": self.algorithm,
                "assignment": {cid: chosen.boundary}, "estimates": {}, "learner_update_counts": {}}
            self.decision_receipts.append({"round_id": round_id, "logical_client_id": props["logical_client_id"],
                "algorithm": self.algorithm, "boundary": chosen.boundary,
                "predicted_frontend_seconds": float(self._front(cid, chosen) * count),
                "candidate_count": len(feasible), "score_seconds": None if scores is None else float(scores[position]),
                "prefix_mac_fraction": self.profiles[chosen.boundary]["prefix_mac_fraction"],
                "fully_local_escape_available": False, "decision_granularity": "federated_round"})
        return assignment

    def observe_round(self, *, round_id, feedback):
        feedback = tuple(feedback)
        IndependentUCB.observe_round(self, round_id=round_id, feedback=feedback)
        candidates = {value.boundary: value for value in self.candidate_provider.get_candidates(training=True)}
        for observation in feedback:
            if not observation.success:
                continue
            cid = observation.client_id
            props = self.telemetry_provider.clients[cid]
            count = int(observation.num_batches)
            if count < 1 or observation.completion_ms is None or observation.completion_ms <= 0:
                raise ValueError("Reference controllers require actual fit-RPC time and batches")
            front = (observation.client_forward_ms + observation.client_backward_ms) / 1000
            table = self.cost_tables[cid]
            table["observed_front"][observation.boundary] = front
            table["observed_tail"][observation.boundary] = observation.server_service_ms / 1000
            if self.algorithm == "L2S":
                edge = max(observation.completion_ms / 1000 - front * count, 0)
                self.linear[cid].observe(self.pending_features.pop((round_id, cid)), edge)
            if self.algorithm == "Greedy":
                candidate = candidates[observation.boundary]
                batch = int(props["batch_size"])
                volume = (candidate.metadata["boundary_forward_bytes_by_batch_size"][batch]
                          + self.profiles[candidate.boundary]["gradient_volume_proxy_bytes_per_sample"] * batch)
                if volume > 0 and observation.network_roundtrip_ms is not None:
                    table["network_seconds_per_byte"] = max((observation.network_roundtrip_ms / 1000 - table["network_intercept_seconds"]) / volume, 0)
                exchange = self.state_download_bytes + candidate.metadata["optimizer_prefix_parameter_bytes"]
                if exchange > 0 and observation.state_exchange_ms is not None:
                    table["state_seconds_per_byte"] = max((observation.state_exchange_ms / 1000 - table["state_intercept_seconds"]) / exchange, 0)
        save_json(self.output.with_suffix(".reference.json"), {"algorithm": self.algorithm,
            "decisions": self.decision_receipts,
            "l2s_updates": {cid: learner.updates for cid, learner in self.linear.items()},
            "fully_local_escape_available": False, "shared_sfl_pipeline": True})


__all__ = ["AdaptiveReference", "LinUCBE", "graph_workload_features"]
