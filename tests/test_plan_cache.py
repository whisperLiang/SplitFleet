from __future__ import annotations

from slbd.autosplit.cache import PlanCacheEntry, PlanCacheStore
from slbd.autosplit.types import PlacementConstraint, PlacementObjective, WorkerSpec


def test_worker_signature_is_stable() -> None:
    signature_a = PlanCacheStore.worker_signature(
        [
            WorkerSpec(worker_id="b", device="cpu"),
            WorkerSpec(worker_id="a", device="cuda:0"),
        ]
    )
    signature_b = PlanCacheStore.worker_signature(
        [
            WorkerSpec(worker_id="a", device="cuda:0"),
            WorkerSpec(worker_id="b", device="cpu"),
        ]
    )
    assert signature_a == signature_b


def test_cache_entry_matches_full_signature() -> None:
    constraints = PlacementConstraint()
    objective = PlacementObjective()
    entry = PlanCacheEntry(
        model_name="Demo",
        graph_signature="graph",
        cutoffs=[1, 3],
        stage_to_worker={"stage-a": "worker-0"},
        score=1.5,
        worker_signature="worker-0",
        constraint_signature=constraints.__dict__,
        objective_signature=objective.__dict__,
    )

    assert entry.matches(
        model_name="Demo",
        graph_signature="graph",
        worker_signature="worker-0",
        constraints=constraints,
        objective=objective,
    )
