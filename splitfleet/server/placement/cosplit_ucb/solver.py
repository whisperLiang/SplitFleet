"""Joint makespan placement under bounded server concurrency."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Mapping, Sequence

from .types import CandidateEstimate


@dataclass(frozen=True)
class ClientTimeline:
    client_id: str
    boundary: str
    arrival_ms: float
    server_start_ms: float
    server_finish_ms: float
    queue_ms: float
    completion_ms: float


@dataclass(frozen=True)
class PlacementSimulation:
    timelines: Mapping[str, ClientTimeline]
    max_client_completion_ms: float
    sum_client_completion_ms: float

    @property
    def objective(self) -> tuple[float, float]:
        return (self.max_client_completion_ms, self.sum_client_completion_ms)


def _component(value: float, uncertainty: float, use_upper: bool) -> float:
    return max(float(value) + (float(uncertainty) if use_upper else 0.0), 0.0)


class GlobalPlacementSolver:
    """Greedy plus coordinate search using full server-lane simulation."""

    def __init__(self, *, server_concurrency: int = 1, max_coordinate_passes: int = 3) -> None:
        if server_concurrency < 1 or max_coordinate_passes < 0:
            raise ValueError("invalid solver parameters")
        self.server_concurrency = int(server_concurrency)
        self.max_coordinate_passes = int(max_coordinate_passes)

    def simulate(
        self,
        assignment: Mapping[str, CandidateEstimate],
        *,
        use_upper: bool = False,
        batch_counts: Mapping[str, int] | None = None,
    ) -> PlacementSimulation:
        """Simulate sequential client batches on shared suffix-server lanes.

        Component predictions describe a representative batch. Repartitioning
        happens once before that client's first batch; each later batch starts
        after its preceding boundary gradient and client backward step.
        """

        lanes = [0.0] * self.server_concurrency
        counts = {str(cid): int((batch_counts or {}).get(cid, 1)) for cid in assignment}
        if any(count < 1 for count in counts.values()):
            raise ValueError("batch counts must be positive")
        jobs: list[tuple[float, str, int, CandidateEstimate, float, float, float, float, float]] = []
        for client_id, estimate in assignment.items():
            switch = _component(estimate.switch_mean_ms, estimate.switch_uncertainty_ms, use_upper)
            forward = _component(
                estimate.client_forward_mean_ms,
                estimate.client_forward_uncertainty_ms,
                use_upper,
            )
            upload = _component(
                estimate.network_upload_mean_ms,
                estimate.network_upload_uncertainty_ms,
                use_upper,
            )
            service = _component(
                estimate.server_service_mean_ms,
                estimate.server_service_uncertainty_ms,
                use_upper,
            )
            download = _component(
                estimate.network_download_mean_ms,
                estimate.network_download_uncertainty_ms,
                use_upper,
            )
            backward = _component(
                estimate.client_backward_mean_ms,
                estimate.client_backward_uncertainty_ms,
                use_upper,
            )
            jobs.append((switch + forward + upload, str(client_id), 0, estimate, forward, upload, service, download, backward))
        heapq.heapify(jobs)

        timelines: dict[str, ClientTimeline] = {}
        while jobs:
            arrival, client_id, batch_index, estimate, forward, upload, service, download, backward = heapq.heappop(jobs)
            lane_index = min(range(len(lanes)), key=lambda index: (lanes[index], index))
            start = max(arrival, lanes[lane_index])
            finish = start + service
            lanes[lane_index] = finish
            completion = finish + download + backward
            previous = timelines.get(client_id)
            timelines[client_id] = ClientTimeline(
                client_id=client_id,
                boundary=estimate.boundary,
                arrival_ms=arrival if previous is None else previous.arrival_ms,
                server_start_ms=start if previous is None else previous.server_start_ms,
                server_finish_ms=finish,
                queue_ms=max(start - arrival, 0.0) + (previous.queue_ms if previous else 0.0),
                completion_ms=completion,
            )
            if batch_index + 1 < counts[client_id]:
                heapq.heappush(
                    jobs,
                    (completion + forward + upload, client_id, batch_index + 1, estimate, forward, upload, service, download, backward),
                )
        completions = [item.completion_ms for item in timelines.values()]
        return PlacementSimulation(
            timelines=timelines,
            max_client_completion_ms=max(completions, default=0.0),
            sum_client_completion_ms=sum(completions),
        )

    def solve(
        self,
        estimates: Mapping[str, Sequence[CandidateEstimate]],
        *,
        use_upper: bool = False,
        batch_counts: Mapping[str, int] | None = None,
    ) -> dict[str, CandidateEstimate]:
        """Return a deterministic joint assignment without exponential search."""

        feasible: dict[str, list[CandidateEstimate]] = {}
        for client_id, options in estimates.items():
            valid = sorted(
                (value for value in options if value.feasible),
                key=lambda value: value.boundary,
            )
            if not valid:
                reasons = sorted({value.infeasible_reason or "unknown" for value in options})
                raise ValueError(f"client {client_id!r} has no feasible split candidates: {reasons}")
            feasible[str(client_id)] = valid

        def standalone(option: CandidateEstimate) -> float:
            multiplier = int((batch_counts or {}).get(option.client_id, 1))
            total = option.ucb_total_without_queue_ms if use_upper else option.mean_total_without_queue_ms
            switch = option.switch_mean_ms + (option.switch_uncertainty_ms if use_upper else 0.0)
            return switch + multiplier * (total - switch)

        order = sorted(
            feasible,
            key=lambda cid: (-min(standalone(option) for option in feasible[cid]), cid),
        )
        assignment: dict[str, CandidateEstimate] = {}
        for client_id in order:
            choice = min(
                feasible[client_id],
                key=lambda option: (
                    self.simulate({**assignment, client_id: option}, use_upper=use_upper, batch_counts=batch_counts).objective,
                    option.boundary,
                ),
            )
            assignment[client_id] = choice

        for _ in range(self.max_coordinate_passes):
            changed = False
            for client_id in sorted(feasible):
                current = assignment[client_id]
                choice = min(
                    feasible[client_id],
                    key=lambda option: (
                        self.simulate({**assignment, client_id: option}, use_upper=use_upper, batch_counts=batch_counts).objective,
                        option.boundary,
                    ),
                )
                if choice.boundary != current.boundary:
                    assignment[client_id] = choice
                    changed = True
            if not changed:
                break
        return assignment


__all__ = [
    "ClientTimeline",
    "GlobalPlacementSolver",
    "PlacementSimulation",
]
