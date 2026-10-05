"""Ablation: client-isolated component learners with the shared fleet solver."""

from contextlib import contextmanager

from splitfleet.server.placement.cosplit_ucb import CoSplitUCBConfig, CoSplitUCBPlacementPolicy
from splitfleet.server.placement.cosplit_ucb.feedback import aggregate_feedback
from splitfleet.server.placement.cosplit_ucb.learners import CooperativeLearners


class NoCooperation(CoSplitUCBPlacementPolicy):
    """Reuse contexts, constraints, feedback, exploration and the joint solver.

    Only the edge/network/server/switch learner bundle is isolated per client.
    Experimental snapshots are exported explicitly; automatic warm starts are
    refused to prevent mixing the production and ablation state envelopes.
    """

    def __init__(self, *, config=None, **kwargs):
        config = config or CoSplitUCBConfig()
        if config.state_path:
            raise ValueError("No-cooperation ablation needs fresh learners without automatic warm starts")
        self.client_learners = {}
        super().__init__(config=config, **kwargs)

    def _bundle(self, cid):
        cid = str(cid)
        if cid not in self.client_learners:
            self.client_learners[cid] = CooperativeLearners(self.config, self.context_encoder)
        return self.client_learners[cid]

    @contextmanager
    def _use_bundle(self, cid):
        original = self.learners
        self.learners = self._bundle(cid)
        try:
            yield
        finally:
            self.learners = original

    def _estimate_round(self, *, round_id, client_ids, training, allowed_boundaries=None):
        estimates, locked = {}, set()
        for cid in sorted({str(value) for value in client_ids}):
            with self._use_bundle(cid):
                if training:
                    self.learners.advance_round(round_id)
                values, resident = super()._estimate_round(round_id=round_id, client_ids=[cid], training=training,
                                                           allowed_boundaries=allowed_boundaries)
                estimates.update(values)
                locked.update(resident)
        return estimates, locked

    def observe_round(self, *, round_id, feedback):
        values = aggregate_feedback(feedback)
        planned = self._round_assignments.get((int(round_id), True), {})
        if any(value.round_id != round_id or planned.get(value.client_id) != value.boundary for value in values):
            raise ValueError("Feedback does not match the planned round/client/cut")
        for value in values:
            with self._use_bundle(value.client_id):
                super().observe_round(round_id=round_id, feedback=[value])

    def state_dict(self):
        state = super().state_dict()
        state["experimental_ablation"] = "no_cooperation"
        state["client_learners"] = {cid: bundle.state_dict() for cid, bundle in sorted(self.client_learners.items())}
        return state

    def load_state_dict(self, state):
        if state.get("experimental_ablation") != "no_cooperation":
            raise ValueError("Expected an explicit no-cooperation experiment snapshot")
        super().load_state_dict(state)
        self.client_learners.clear()
        for cid, values in state["client_learners"].items():
            self._bundle(cid).load_state_dict(values)
