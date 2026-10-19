"""L2S comparator on the common measured Greedy calibration and SFL runtime."""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from splitfleet.server.placement.greedy import GreedyPlacementPolicy
from splitfleet.server.placement.greedy.profiles import graph_workload_features


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


class AdaptiveReference(GreedyPlacementPolicy):
    """LinUCB-E for L2S, with Greedy retained as an explicit naming alias."""

    def __init__(self, *, candidate_provider, config, algorithm, horizon, output):
        if algorithm not in ("L2S", "Greedy"):
            raise ValueError("Choose L2S or Greedy")
        self.algorithm = algorithm
        self.name = "l2s" if algorithm == "L2S" else "greedy"
        self.uses_active_network_probes = algorithm == "Greedy"
        self.horizon = horizon
        self.output = Path(output)
        self.linear = {}
        super().__init__(candidate_provider=candidate_provider, config=config)

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

    def _score(self, cid, candidate, *, batch, count, network_context=None):
        if self.algorithm == "Greedy":
            return super()._score(cid, candidate, batch=batch, count=count,
                                  network_context=network_context)
        learner = self.linear.setdefault(cid, LinUCBE(11, horizon=self.horizon))
        features = self._features(cid, [candidate])
        return float(learner.scores(features, [self._front(cid, candidate) * count])[0])

    def observe_round(self, *, round_id, feedback):
        feedback = tuple(feedback)
        fresh = [o for o in feedback if (int(round_id), o.client_id, o.boundary) not in self._observed_actions]
        super().observe_round(round_id=round_id, feedback=feedback)
        if self.algorithm == "L2S":
            for observation in fresh:
                if not observation.success:
                    continue
                if observation.completion_ms is None or observation.completion_ms <= 0:
                    raise ValueError("L2S requires an actual whole-fit RPC duration")
                candidate = self._catalog_by_boundary[observation.boundary]
                front = (observation.client_forward_ms + observation.client_backward_ms) / 1000
                edge = max(observation.completion_ms / 1000 - front * observation.num_batches, 0)
                self.linear[observation.client_id].observe(self._features(observation.client_id, [candidate])[0], edge)
            self._save_state()

    def state_dict(self):
        state = super().state_dict()
        state["linear"] = {cid: {"a": model.a.tolist(), "b": model.b.tolist(), "updates": model.updates}
                           for cid, model in self.linear.items()}
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for cid, value in state.get("linear", {}).items():
            model = LinUCBE(11, horizon=self.horizon)
            model.a = np.asarray(value["a"], dtype=np.float64)
            model.b = np.asarray(value["b"], dtype=np.float64)
            model.updates = int(value["updates"])
            self.linear[cid] = model


__all__ = ["AdaptiveReference", "LinUCBE", "graph_workload_features"]
