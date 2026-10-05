"""Independent contextual-bandit reference using the CoSplit-UCB contracts.

Every client owns its edge, network, server and switch learners. No learned
state is shared and no other client's server queue enters its decision. This
is an experimental reference, not a reproduction of an adaptive-SFL paper.
"""

from __future__ import annotations

from dataclasses import replace

from splitfleet.server.placement.cosplit_ucb import CoSplitUCBConfig, CoSplitUCBPlacementPolicy
from splitfleet.server.placement.cosplit_ucb.solver import GlobalPlacementSolver


class _IndependentSolver(GlobalPlacementSolver):
    def solve(self, estimates, *, use_upper=False, batch_counts=None):
        if len(estimates) != 1:
            raise ValueError("Independent decisions must contain exactly one client")
        cid, options = next(iter(estimates.items()))
        feasible = [option for option in options if option.feasible]
        if not feasible:
            raise ValueError(f"Client {cid!r} has no feasible candidates")
        # Production exploitation uses the mean simulation, with confidence
        # bounds governing safe exploration. Reuse that same score so learner
        # sharing and joint queues are the only disabled mechanisms.
        return {cid: min(feasible, key=lambda option: (
            self.simulate({cid: option}, use_upper=use_upper, batch_counts=batch_counts).objective,
            option.boundary))}


class IndependentUCB:
    """Use isolated CoSplit component encoders and feedback handling per client.

    The independent score is the production single-client simulation cost.
    Residence, feasibility and safe confidence-bound exploration reuse the production policy,
    with the exploration budget applying per independent client.
    """

    def __init__(self, *, candidate_provider, config=None, telemetry_provider=None):
        self.candidate_provider = candidate_provider
        self.config = config or CoSplitUCBConfig()
        if self.config.state_path:
            raise ValueError("Independent-UCB state is exported explicitly per client")
        self.telemetry_provider = telemetry_provider
        self.clients = {}

    def _client(self, cid):
        cid = str(cid)
        if cid not in self.clients:
            policy = CoSplitUCBPlacementPolicy(candidate_provider=self.candidate_provider,
                config=replace(self.config, state_path=None), telemetry_provider=self.telemetry_provider)
            policy.solver = _IndependentSolver(server_concurrency=self.config.server_concurrency,
                                               max_coordinate_passes=0)
            self.clients[cid] = policy
        return self.clients[cid]

    def bind_clients(self, clients, round_id):
        bind = getattr(self.telemetry_provider, "bind_clients", None)
        if bind:
            bind(clients, round_id)

    def plan_round(self, *, round_id, client_ids, training):
        normalized = sorted({str(cid) for cid in client_ids})
        return {cid: self._client(cid).plan_round(round_id=round_id, client_ids=[cid], training=training)[cid]
                for cid in normalized}

    def observe_round(self, *, round_id, feedback):
        grouped = {}
        for observation in feedback:
            if observation.client_id not in self.clients:
                raise ValueError("Feedback received for an unplanned client")
            grouped.setdefault(observation.client_id, []).append(observation)
        for cid in sorted(grouped):
            self.clients[cid].observe_round(round_id=round_id, feedback=grouped[cid])

    def observe_failure(self, *, round_id, client_id, boundary=None, kind="runtime", reason=None):
        self._client(client_id).observe_failure(round_id=round_id, client_id=client_id,
                                               boundary=boundary, kind=kind, reason=reason)

    def state_dict(self):
        return {cid: policy.state_dict() for cid, policy in sorted(self.clients.items())}
