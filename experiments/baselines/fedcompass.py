"""FedCompass scheduling and delta aggregation on a coordinator monotonic clock.

Port of Li et al., ICLR 2024, Algorithms 1--3, cross-checked against
APPFL/FedCompass af96db2204795b0b0addfdec219b2a541f4b1ab3.
The event loop replaces upstream timer threads. Polynomial weights decay as in
the paper; sample-size weights match the other physical comparison methods.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping

import torch

from experiments.common.training import aggregate_named_states


@dataclass(frozen=True)
class FedCompassConfig:
    min_local_steps: int = 20
    max_local_steps: int = 100
    speed_momentum: float = 0.9
    latest_time_factor: float = 1.2
    alpha: float = 0.9
    staleness_exponent: float = 0.5

    def __post_init__(self):
        if (type(self.min_local_steps) is not int or type(self.max_local_steps) is not int
                or not 1 <= self.min_local_steps <= self.max_local_steps):
            raise ValueError("FedCompass requires 1 <= min_local_steps <= max_local_steps")
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"FedCompass {name} must be finite")
        if not 0 <= self.speed_momentum <= 1 or not 0 < self.alpha <= 1:
            raise ValueError("Invalid FedCompass speed momentum or mixing coefficient")
        if self.latest_time_factor < 1 or self.staleness_exponent < 0:
            raise ValueError("FedCompass requires a valid deadline factor and decaying staleness")


@dataclass
class Assignment:
    client_id: str
    local_steps: int
    model_version: int
    group_id: int | None
    started: float


@dataclass
class Update:
    assignment: Assignment
    base: Mapping[str, torch.Tensor]
    trained: Mapping[str, torch.Tensor]
    receipt: dict


@dataclass
class ArrivalGroup:
    expected: float
    deadline: float
    pending: set[str] = field(default_factory=set)
    arrived: list[Update] = field(default_factory=list)
    closed: bool = False


class FedCompass:
    """Single-owner, arrival-driven scheduler; it never waits for the whole fleet."""

    def __init__(self, model: torch.nn.Module, partition_sizes: Mapping[str, int],
                 config: FedCompassConfig | None = None, *,
                 clock: Callable[[], float] | None = None, deadline: float | None = None):
        self.model = model
        self.config = config or FedCompassConfig()
        if deadline is not None and (clock is None or not math.isfinite(deadline) or deadline <= 0):
            raise ValueError("FedCompass deadlines require a clock and a finite positive budget")
        self.clock, self.deadline = clock, deadline
        if not partition_sizes or any(type(n) is not int or n <= 0 for n in partition_sizes.values()):
            raise ValueError("FedCompass requires nonempty positive client partitions")
        self.partition_sizes = dict(partition_sizes)
        total = sum(partition_sizes.values())
        self.weights = {cid: n / total for cid, n in partition_sizes.items()}
        self.parameter_names = set(dict(model.named_parameters(remove_duplicate=False)))
        self.version = 0
        self.speeds: dict[str, float] = {}
        self.active: dict[str, Assignment] = {}
        self.groups: dict[int, ArrivalGroup] = {}
        self.buffer: dict[str, Update] = {}
        self.aggregations: list[dict] = []
        self.group_events: list[dict] = []
        self._group_counter = 0

    def state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}

    def initialize(self, now: float) -> list[Assignment]:
        if self.active or self.version:
            raise RuntimeError("FedCompass can only initialize once")
        jobs = [Assignment(cid, self.config.max_local_steps, 0, None, now)
                for cid in self.partition_sizes]
        self.active = {job.client_id: job for job in jobs}
        return jobs

    def _check_state(self, state):
        current = self.model.state_dict()
        if set(state) != set(current):
            raise ValueError("FedCompass update does not match the model schema")
        for name, value in state.items():
            if value.shape != current[name].shape or value.dtype != current[name].dtype:
                raise ValueError(f"FedCompass tensor schema mismatch: {name}")
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite FedCompass tensor: {name}")

    def receive(self, update: Update, now: float) -> list[Assignment]:
        job = update.assignment
        if self.active.get(job.client_id) is not job or now < job.started:
            raise ValueError("Unexpected, duplicated or invalid FedCompass assignment")
        self._check_state(update.base)
        self._check_state(update.trained)
        del self.active[job.client_id]
        measured = max(now - job.started, 1e-9) / job.local_steps
        old = self.speeds.get(job.client_id, measured)
        m = self.config.speed_momentum
        self.speeds[job.client_id] = (1 - m) * old + m * measured
        group = self.groups.get(job.group_id)
        if job.group_id is None:
            if not self._aggregate([update], now, None, "initial_arrival"):
                return []
            return [self._assign(job.client_id, now)]
        if group is None or group.closed or now > group.deadline:
            if group is not None:
                group.pending.discard(job.client_id)
            previous = self.buffer.get(job.client_id)
            if previous is not None:
                previous.receipt["update_status"] = "superseded_late_buffer"
            update.receipt["update_status"] = "late_buffer"
            self.buffer[job.client_id] = update
            return [self._assign(job.client_id, now)]
        group.pending.remove(job.client_id)
        group.arrived.append(update)
        update.receipt["update_status"] = "waiting_for_group"
        return self._close(job.group_id, now, "group_complete") if not group.pending else []

    def expire(self, now: float, max_version: int | None = None) -> list[Assignment]:
        released = []
        for gid, group in list(self.groups.items()):
            if self.deadline is not None and self.clock() >= self.deadline:
                break
            if max_version is not None and self.version >= max_version:
                break
            if not group.closed and now >= group.deadline:
                released.extend(self._close(gid, now, "group_deadline"))
            if group.closed and not group.pending:
                self.groups.pop(gid, None)
        return released

    def next_deadline(self) -> float | None:
        times = [group.deadline for group in self.groups.values() if not group.closed]
        return min(times) if times else None

    def _close(self, gid: int, now: float, reason: str) -> list[Assignment]:
        group = self.groups[gid]
        group.closed = True
        arrived = group.arrived
        group.arrived = []
        self.group_events.append({"group_id": gid, "reason": reason,
                                  "elapsed_sec": now, "missing_clients": sorted(group.pending),
                                  "expected_arrival_elapsed_sec": group.expected,
                                  "group_deadline_elapsed_sec": group.deadline,
                                  "arrived_clients": [u.assignment.client_id for u in arrived]})
        if not arrived:
            return []
        merged = dict(self.buffer)
        for update in arrived:
            cid = update.assignment.client_id
            if cid in merged:
                merged[cid].receipt["update_status"] = "superseded_late_buffer"
            merged[cid] = update
        updates = list(merged.values())
        self.buffer.clear()
        if not self._aggregate(updates, now, gid, reason):
            return []
        return [self._assign(u.assignment.client_id, now)
                for u in sorted(arrived, key=lambda u: self.speeds[u.assignment.client_id])]

    def _aggregate(self, updates: list[Update], now: float, gid: int | None, reason: str):
        previous = self.state()
        state = dict(previous)
        buffer_names = set(state) - self.parameter_names
        # Buffers are local states, not parameter deltas. This also preserves
        # integer dtypes, scalar shape and original names.
        averaged_buffers = aggregate_named_states(
            [{name: u.trained[name] for name in buffer_names} for u in updates],
            [self.partition_sizes[u.assignment.client_id] for u in updates],
        ) if buffer_names else {}
        weights, stale = [], []
        for update in updates:
            job = update.assignment
            delay = self.version - job.model_version
            if delay < 0:
                raise ValueError("FedCompass update has a future model version")
            scale = self.config.alpha * self.weights[job.client_id] * (1 + delay) ** -self.config.staleness_exponent
            for name in self.parameter_names:
                state[name] = state[name] - scale * (update.base[name] - update.trained[name]).to(state[name])
            weights.append(scale)
            stale.append(delay)
        state.update(averaged_buffers)
        self._check_state(state)
        self.model.load_state_dict(state, strict=True)
        committed_at = self.clock() if self.clock is not None else now
        if self.deadline is not None and committed_at >= self.deadline:
            self.model.load_state_dict(previous, strict=True)
            for update in updates:
                update.receipt["update_status"] = "discarded_at_stop"
            return False
        self.version += 1
        for update, scale, delay in zip(updates, weights, stale, strict=True):
            update.receipt.update(update_status="aggregated", aggregated_model_version=self.version,
                                  aggregation_elapsed_sec=committed_at, staleness=delay, aggregation_weight=scale)
        self.aggregations.append({"model_version": self.version, "elapsed_sec": committed_at,
                                  "group_id": gid, "reason": reason,
                                  "dispatch_ids": [u.receipt.get("dispatch_id") for u in updates],
                                  "clients": [u.assignment.client_id for u in updates],
                                  "staleness": stale, "weights": weights})
        return True

    def _assign(self, cid: str, now: float) -> Assignment:
        speed = self.speeds[cid]
        low, high = self.config.min_local_steps, self.config.max_local_steps
        candidates = []
        for gid, group in self.groups.items():
            if group.closed:
                continue
            steps = math.floor((group.expected - now) / speed)
            if low <= steps <= high:
                candidates.append((steps, gid))
        if candidates:
            steps, gid = max(candidates, key=lambda pair: pair[0])
            self.groups[gid].pending.add(cid)
        else:
            # Algorithm 2: seek a later group that can align with an existing
            # group's next fast-client arrival before falling back to Q_max.
            options = []
            for group in self.groups.values():
                if group.closed or now >= group.deadline:
                    continue
                members = group.pending | {u.assignment.client_id for u in group.arrived}
                fastest = min(self.speeds[member] for member in members)
                steps = math.floor((group.deadline + fastest * high - now) / speed)
                if steps <= high:
                    options.append(steps)
            steps = max(low, max(options)) if options else high
            gid = self._group_counter
            self._group_counter += 1
            duration = steps * speed
            self.groups[gid] = ArrivalGroup(now + duration,
                                            now + self.config.latest_time_factor * duration, {cid})
        job = Assignment(cid, steps, self.version, gid, now)
        self.active[cid] = job
        return job

    def stop(self):
        for update in self.buffer.values():
            update.receipt["update_status"] = "unaggregated_at_stop"
        for group in self.groups.values():
            for update in group.arrived:
                update.receipt["update_status"] = "unaggregated_at_stop"
        self.buffer.clear()
        self.groups.clear()
        self.active.clear()
