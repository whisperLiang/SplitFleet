"""Exact offline oracle for small measured candidate-cost tables.

The oracle is not deployable and cannot initialize an online learner. Its
certificate is conditional on the recorded costs and shared-server simulation,
not a guarantee of the physical runtime's global minimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

from splitfleet.server.placement.cosplit_ucb.solver import GlobalPlacementSolver, PlacementSimulation
from splitfleet.server.placement.cosplit_ucb.types import CandidateEstimate


@dataclass(frozen=True)
class OracleResult:
    assignment: dict[str, CandidateEstimate]
    simulation: PlacementSimulation
    evaluated_assignments: int
    exact: bool = True
    deployable: bool = False


def offline_oracle(estimates, *, server_concurrency=1, batch_counts=None, max_assignments=1_000_000):
    ids = sorted(estimates)
    options = [sorted((option for option in estimates[cid] if option.feasible), key=lambda option: option.boundary)
               for cid in ids]
    if not ids or any(not values for values in options):
        raise ValueError("Oracle requires at least one feasible candidate per client")
    if any(value.client_id != cid for cid, values in zip(ids, options) for value in values):
        raise ValueError("Oracle cost rows must match their client identity")
    if any(value.uncertainty_total_ms != 0 for values in options for value in values):
        raise ValueError("Offline oracle requires recorded cost rows without bandit uncertainty")
    count = math.prod(len(values) for values in options)
    if count > max_assignments:
        raise ValueError(f"Exact oracle needs {count} assignments; limit is {max_assignments}. Explicitly restrict and label the candidate domain.")
    simulator = GlobalPlacementSolver(server_concurrency=server_concurrency)
    best = None
    for values in product(*options):
        assignment = dict(zip(ids, values))
        simulation = simulator.simulate(assignment, batch_counts=batch_counts)
        objective = (simulation.objective, tuple(value.boundary for value in values))
        if best is None or objective < best[0]:
            best = objective, assignment, simulation
    return OracleResult(assignment=best[1], simulation=best[2], evaluated_assignments=count)
