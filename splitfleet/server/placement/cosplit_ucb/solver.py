"""Joint makespan placement under bounded server concurrency."""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

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
    worker_start_ms: float = 0.


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

    def __init__(
        self, *, server_concurrency: int = 1, max_coordinate_passes: int = 3,
        simulation_cache_size: int = 8192,
        client_concurrency: int | None = None,
    ) -> None:
        if server_concurrency < 1 or max_coordinate_passes < 0 or simulation_cache_size < 0:
            raise ValueError("invalid solver parameters")
        self.server_concurrency = int(server_concurrency)
        self.max_coordinate_passes = int(max_coordinate_passes)
        self.simulation_cache_size = int(simulation_cache_size)
        self._simulation_cache: OrderedDict[tuple, PlacementSimulation] = OrderedDict()
        self.simulation_cache_hits = 0
        self.simulation_cache_misses = 0
        if client_concurrency is not None and client_concurrency < 1:
            raise ValueError("client_concurrency must be positive")
        self.client_concurrency = client_concurrency
        self.dispatch_order: tuple[str, ...] = ()
        self.startup_costs_ms: Mapping[str, float] = {}
        self.cold_prepare_costs_ms: Mapping[str, float] = {}

    def clear_cache(self) -> None:
        """Release simulations and reset per-plan diagnostic counters."""

        self._simulation_cache.clear()
        self.simulation_cache_hits = 0
        self.simulation_cache_misses = 0

    def _simulate_without_queue(self, assignment, counts, components, ordered):
        """Use independent client cycles when every active worker has a lane."""
        pending = deque(ordered)
        releases = []
        timelines = {}

        def start_client(cid, worker_start):
            switch, forward, upload, service, download, backward, roundtrip, exchange = components[cid]
            startup = max(float(self.startup_costs_ms.get(cid, 0.)), 0.)
            prepare = max(switch, float(self.cold_prepare_costs_ms.get(cid, 0.)))
            first_arrival = worker_start + startup + prepare + forward + upload
            arrival = first_arrival
            # Keep the original per-batch arithmetic order. Multiplying a
            # cycle duration changes floating-point ties in placement search.
            for _ in range(counts[cid] - 1):
                finish = arrival + service
                completion = finish + download + backward + roundtrip
                arrival = completion + forward + upload
            finish = arrival + service
            completion = finish + download + backward + roundtrip
            completion += exchange
            # A zero-duration final batch can release a worker at the same
            # instant as its arrival. Preserve the event-level tie handling
            # of the general simulator for that case and nonfinite sums.
            if not math.isfinite(completion) or completion <= arrival:
                return False
            timelines[cid] = ClientTimeline(cid, assignment[cid].boundary,
                first_arrival, first_arrival, finish, 0., completion, worker_start)
            heapq.heappush(releases, (completion, cid))
            return True

        for _ in range(min(len(ordered), self.client_concurrency or len(ordered))):
            if not start_client(pending.popleft(), 0.):
                return None
        while pending:
            worker_finish, _ = heapq.heappop(releases)
            if not start_client(pending.popleft(), worker_finish):
                return None
        # The original simulator inserts records at their first batch arrival.
        # Retain that order, including the summation order of the objective.
        timelines = dict(sorted(timelines.items(), key=lambda row: (row[1].arrival_ms, row[0])))
        completions = [row.completion_ms for row in timelines.values()]
        return PlacementSimulation(MappingProxyType(timelines),
            max(completions, default=0.), sum(completions))

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

        counts = {str(cid): int((batch_counts or {}).get(cid, 1)) for cid in assignment}
        if any(count < 1 for count in counts.values()):
            raise ValueError("batch counts must be positive")
        components = {
            str(cid): tuple(_component(mean, radius, use_upper) for mean, radius in (
                (estimate.switch_mean_ms, estimate.switch_uncertainty_ms),
                (estimate.client_forward_mean_ms, estimate.client_forward_uncertainty_ms),
                (estimate.network_upload_mean_ms, estimate.network_upload_uncertainty_ms),
                (estimate.server_service_mean_ms, estimate.server_service_uncertainty_ms),
                (estimate.network_download_mean_ms, estimate.network_download_uncertainty_ms),
                (estimate.client_backward_mean_ms, estimate.client_backward_uncertainty_ms),
                (estimate.network_roundtrip_mean_ms, estimate.network_roundtrip_uncertainty_ms),
                (estimate.state_exchange_mean_ms, estimate.state_exchange_uncertainty_ms),
            )) for cid, estimate in assignment.items()
        }
        if self.simulation_cache_size:
            # Timings depend on component costs, not boundary names. Preserve
            # all six components and their arithmetic order; do not combine
            # forward/upload or backward/download and change rounding.
            key = (self.server_concurrency, self.client_concurrency, self.dispatch_order,
                   tuple(sorted(self.startup_costs_ms.items())), tuple(sorted(self.cold_prepare_costs_ms.items())),
                   bool(use_upper), tuple(sorted(
                (cid, counts[cid], values) for cid, values in components.items()
            )))
            cached = self._simulation_cache.get(key)
            if cached is not None:
                self._simulation_cache.move_to_end(key)
                self.simulation_cache_hits += 1
                if all(cached.timelines[cid].boundary == estimate.boundary for cid, estimate in assignment.items()):
                    return cached
                return PlacementSimulation(
                    timelines=MappingProxyType({
                        cid: replace(timeline, boundary=assignment[cid].boundary)
                        for cid, timeline in cached.timelines.items()
                    }),
                    max_client_completion_ms=cached.max_client_completion_ms,
                    sum_client_completion_ms=cached.sum_client_completion_ms,
                )
        self.simulation_cache_misses += 1
        # The heap orders equal availability by lane index, matching the
        # original scan without visiting every lane for every simulated batch.
        lanes = [(0.0, index) for index in range(self.server_concurrency)]
        jobs: list[tuple[float, str, int, CandidateEstimate, float, float, float, float, float, float]] = []
        worker_starts = {}
        priority = {cid: i for i, cid in enumerate(self.dispatch_order)}
        ordered = sorted(assignment, key=lambda cid: (priority.get(cid, len(priority)), str(cid)))
        active_clients = min(len(ordered), self.client_concurrency or len(ordered))
        if self.server_concurrency >= active_clients:
            simulation = self._simulate_without_queue(assignment, counts, components, ordered)
            if simulation is not None:
                if self.simulation_cache_size:
                    self._simulation_cache[key] = simulation
                    if len(self._simulation_cache) > self.simulation_cache_size:
                        self._simulation_cache.popitem(last=False)
                return simulation
        pending = deque(
            ordered[self.client_concurrency:]
            if self.client_concurrency is not None else []
        )
        initial_clients = ordered[:self.client_concurrency] if self.client_concurrency is not None else assignment
        worker_completions: list[tuple[float, str]] = []

        def enqueue(client_id, worker_start):
            estimate = assignment[client_id]
            switch, forward, upload, service, download, backward, roundtrip, _exchange = components[str(client_id)]
            worker_starts[str(client_id)] = worker_start
            startup = max(float(self.startup_costs_ms.get(str(client_id), 0.)), 0.)
            prepare = max(switch, float(self.cold_prepare_costs_ms.get(str(client_id), 0.)))
            heapq.heappush(jobs, (worker_start + startup + prepare + forward + upload,
                str(client_id), 0, estimate, forward, upload, service, download, backward, roundtrip))

        for client_id in initial_clients:
            enqueue(client_id, 0.)
        heapq.heapify(jobs)

        # Keep scalar accumulators during scheduling; constructing an immutable
        # timeline for every batch adds no information to the final result.
        records: dict[str, list[float]] = {}
        while jobs or worker_completions:
            # A client can finish downloading/backward/state exchange before
            # an earlier suffix request's client releases its Worker. Process
            # those releases alongside batch arrivals in chronological order.
            if worker_completions and (
                not jobs or worker_completions[0][0] <= jobs[0][0]
            ):
                worker_finish, _ = heapq.heappop(worker_completions)
                if pending:
                    enqueue(pending.popleft(), worker_finish)
                continue
            arrival, client_id, batch_index, estimate, forward, upload, service, download, backward, roundtrip = heapq.heappop(jobs)
            available, lane_index = lanes[0]
            start = max(arrival, available)
            finish = start + service
            heapq.heapreplace(lanes, (finish, lane_index))
            # Combined transport overhead is client think time; it is not
            # divided into unmeasured one-way network phases.
            completion = finish + download + backward + roundtrip
            previous = records.get(client_id)
            if previous is None:
                records[client_id] = [arrival, start, finish, max(start - arrival, 0.0) + 0.0, completion]
            else:
                previous[2] = finish
                previous[3] = max(start - arrival, 0.0) + previous[3]
                previous[4] = completion
            if batch_index + 1 < counts[client_id]:
                heapq.heappush(
                    jobs,
                    (completion + forward + upload, client_id, batch_index + 1, estimate, forward, upload, service, download, backward, roundtrip),
                )
            else:
                records[client_id][4] += components[client_id][7]
                if pending:
                    heapq.heappush(worker_completions, (records[client_id][4], client_id))
        timelines = {
            cid: ClientTimeline(cid, assignment[cid].boundary, *values, worker_starts.get(cid, 0.))
            for cid, values in records.items()
        }
        completions = [item.completion_ms for item in timelines.values()]
        simulation = PlacementSimulation(
            timelines=MappingProxyType(timelines),
            max_client_completion_ms=max(completions, default=0.0),
            sum_client_completion_ms=sum(completions),
        )
        if self.simulation_cache_size:
            self._simulation_cache[key] = simulation
            if len(self._simulation_cache) > self.simulation_cache_size:
                self._simulation_cache.popitem(last=False)
        return simulation

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
            exchange = option.state_exchange_mean_ms + (option.state_exchange_uncertainty_ms if use_upper else 0.0)
            return switch + exchange + multiplier * (total - switch - exchange)

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
