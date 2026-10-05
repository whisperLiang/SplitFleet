"""Ablation: independent mean-cost decisions with no shared-server queues."""

from splitfleet.server.placement.cosplit_ucb import CoSplitUCBPlacementPolicy
from splitfleet.server.placement.cosplit_ucb.solver import GlobalPlacementSolver, PlacementSimulation


class NoQueueSolver(GlobalPlacementSolver):
    def simulate(self, assignment, *, use_upper=False, batch_counts=None):
        timelines = {}
        for cid, value in assignment.items():
            timelines.update(super().simulate({cid: value}, use_upper=use_upper,
                                               batch_counts=batch_counts).timelines)
        return PlacementSimulation(timelines=timelines,
            max_client_completion_ms=max((value.completion_ms for value in timelines.values()), default=0),
            sum_client_completion_ms=sum(value.completion_ms for value in timelines.values()))

    def solve(self, estimates, *, use_upper=False, batch_counts=None):
        assignment = {}
        for cid, options in sorted(estimates.items()):
            valid = [value for value in options if value.feasible]
            if not valid:
                raise ValueError(f"Client {cid!r} has no feasible candidates")
            assignment[cid] = min(valid, key=lambda value: (
                self.simulate({cid: value}, use_upper=use_upper, batch_counts=batch_counts).objective,
                value.boundary))
        return assignment


class NoJointSolver(CoSplitUCBPlacementPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = NoQueueSolver(server_concurrency=self.config.server_concurrency,
                                   max_coordinate_passes=0)
