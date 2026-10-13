"""FedAsync interpolation and FedBuff mean-delta aggregation.

FedAsync follows Xie et al., arXiv:1903.03934v5, Algorithm 1 (Poly).
FedBuff follows Nguyen et al., AISTATS 2022, Algorithms 1--2, without DP.
The physical harness supplies actual arrival times and fixed local steps.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from experiments.baselines.fedcompass import Assignment, Update


@dataclass(frozen=True)
class AsyncFLConfig:
    method: str
    local_steps: int
    alpha: float = .9
    staleness_exponent: float = .5
    proximal_mu: float = .005
    buffer_size: int = 2
    server_learning_rate: float = 1.0

    def __post_init__(self):
        if self.method not in ("fedasync", "fedbuff"):
            raise ValueError("Expected FedAsync or FedBuff")
        if type(self.local_steps) is not int or self.local_steps < 1:
            raise ValueError("Fixed local steps must be a positive integer")
        if type(self.buffer_size) is not int or self.buffer_size < 2:
            raise ValueError("FedBuff needs at least two updates per buffer")
        numbers = (self.alpha, self.staleness_exponent, self.proximal_mu, self.server_learning_rate)
        if (not all(math.isfinite(x) for x in numbers) or not 0 < self.alpha <= 1
                or self.staleness_exponent < 0 or self.proximal_mu < 0
                or self.server_learning_rate <= 0):
            raise ValueError("Invalid asynchronous aggregation parameters")


class AsyncFL:
    """One outstanding assignment per client; arrivals never wait for all clients."""

    def __init__(self, model, partition_sizes, config: AsyncFLConfig):
        if not partition_sizes or any(type(n) is not int or n < 1 for n in partition_sizes.values()):
            raise ValueError("Client partitions must be positive")
        self.model, self.config = model, config
        self.parameter_names = set(dict(model.named_parameters(remove_duplicate=False)))
        self.partition_sizes = dict(partition_sizes)
        self.version = 0
        self.active = {}
        self.buffer = []
        self.aggregations = []
        self.groups, self.group_events, self.speeds = {}, [], {}
        self.clock = self.deadline = None
        self.initialized = False
        self.received_dispatches = set()

    def state(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _check_state(self, state):
        reference = self.model.state_dict()
        if set(state) != set(reference):
            raise ValueError("Async update state keys differ")
        for name, value in state.items():
            expected = reference[name]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"Async update shape or dtype differs: {name}")
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite asynchronous state: {name}")

    def initialize(self, now):
        if self.initialized:
            raise RuntimeError("Async controller cannot initialize twice")
        self.initialized = True
        jobs = [Assignment(cid, self.config.local_steps, 0, None, now) for cid in self.partition_sizes]
        self.active = {job.client_id: job for job in jobs}
        return jobs

    def _commit(self, updates, now):
        current = self.state()
        candidate = {name: value.clone() for name, value in current.items()}
        delays = [self.version - update.assignment.model_version for update in updates]
        if any(delay < 0 for delay in delays):
            raise ValueError("A future model version cannot contribute")
        if self.config.method == "fedasync":
            weight = self.config.alpha * (1 + delays[0]) ** -self.config.staleness_exponent
            weights = [weight]
        else:
            weights = [self.config.server_learning_rate / self.config.buffer_size] * len(updates)
        for name, value in current.items():
            if value.is_floating_point() or value.is_complex():
                if self.config.method == "fedasync":
                    candidate[name] = (1 - weights[0]) * value + weights[0] * updates[0].trained[name]
                elif name in self.parameter_names:
                    for update, weight in zip(updates, weights, strict=True):
                        candidate[name] += weight * (update.trained[name] - update.base[name])
                else:
                    # Running statistics are states, not optimizer deltas.
                    candidate[name] = torch.stack([u.trained[name] for u in updates]).mean(dim=0)
            else:
                # Counters/flags are state, not optimizer deltas. Match the
                # harness's latest-contribution rule without changing dtype.
                chosen = max(updates, key=lambda u: u.assignment.model_version)
                candidate[name] = chosen.trained[name].clone()
        self._check_state(candidate)
        if self.deadline is not None and self.clock() >= self.deadline:
            return False
        self.model.load_state_dict(candidate)
        committed = self.clock() if self.clock is not None else now
        if self.deadline is not None and committed >= self.deadline:
            self.model.load_state_dict(current)
            return False
        self.version += 1
        for update, delay, weight in zip(updates, delays, weights, strict=True):
            update.receipt.update(update_status="aggregated", staleness=delay,
                aggregation_weight=weight, aggregation_elapsed_sec=committed,
                aggregated_model_version=self.version)
        self.aggregations.append({"model_version": self.version, "elapsed_sec": committed,
            "dispatch_ids": [u.receipt["dispatch_id"] for u in updates],
            "aggregation_rule": "stale_model_interpolation" if self.config.method == "fedasync"
                                else "mean_model_delta", "buffer_size": len(updates),
            "floating_buffer_rule": "stale_model_interpolation" if self.config.method == "fedasync"
                                    else "uniform_completed_state_average"})
        return True

    def receive(self, update: Update, now):
        job = update.assignment
        if self.active.get(job.client_id) is not job:
            raise ValueError("Async update does not match the active assignment")
        dispatch_id = update.receipt["dispatch_id"]
        if dispatch_id in self.received_dispatches:
            raise ValueError("Async dispatch was received twice")
        self._check_state(update.base)
        self._check_state(update.trained)
        self.received_dispatches.add(dispatch_id)
        del self.active[job.client_id]
        if self.deadline is not None and self.clock() >= self.deadline:
            return []
        update.receipt["update_status"] = "unaggregated_at_stop"
        if self.config.method == "fedasync":
            if not self._commit([update], now):
                update.receipt["update_status"] = "discarded_at_stop"
                return []
        else:
            self.buffer.append(update)
            if len(self.buffer) == self.config.buffer_size:
                if not self._commit(self.buffer, now):
                    return []
                self.buffer.clear()
        next_job = Assignment(job.client_id, self.config.local_steps, self.version, None, now)
        self.active[job.client_id] = next_job
        return [next_job]

    def expire(self, now, max_updates=None):
        return []

    def next_deadline(self):
        return None

    def stop(self):
        # An incomplete FedBuff buffer remains excluded at the deadline.
        self.buffer.clear()
        self.active.clear()
