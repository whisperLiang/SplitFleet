"""Persistent cache for autosplit partition and placement choices."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional

from splitfleet.autosplit.types import PlacementConstraint, PlacementObjective, WorkerSpec


@dataclass
class PlanCacheEntry:
    """Serializable summary of one chosen partition/placement."""

    model_name: str
    graph_signature: str
    cutoffs: list[int]
    stage_to_worker: Dict[str, str]
    score: float
    worker_signature: str
    constraint_signature: Dict[str, Any]
    objective_signature: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(
        self,
        *,
        model_name: str,
        graph_signature: str,
        worker_signature: str,
        constraints: PlacementConstraint,
        objective: PlacementObjective,
    ) -> bool:
        return (
            self.model_name == model_name
            and self.graph_signature == graph_signature
            and self.worker_signature == worker_signature
            and self.constraint_signature == asdict(constraints)
            and self.objective_signature == asdict(objective)
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlanCacheEntry":
        return cls(
            model_name=str(payload["model_name"]),
            graph_signature=str(payload["graph_signature"]),
            cutoffs=[int(value) for value in payload.get("cutoffs", [])],
            stage_to_worker={
                str(stage_id): str(worker_id)
                for stage_id, worker_id in dict(payload.get("stage_to_worker", {})).items()
            },
            score=float(payload.get("score", 0.0)),
            worker_signature=str(payload.get("worker_signature", "")),
            constraint_signature=dict(payload.get("constraint_signature", {})),
            objective_signature=dict(payload.get("objective_signature", {})),
            metadata=dict(payload.get("metadata", {})),
        )


class PlanCacheStore:
    """Filesystem-backed cache keyed by model name."""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir

    def _path_for(self, model_name: str) -> str:
        safe_name = model_name.replace("\\", "_").replace("/", "_")
        return os.path.join(self.root_dir, f"{safe_name}.json")

    def load(self, model_name: str) -> Optional[PlanCacheEntry]:
        path = self._path_for(model_name)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return None
        return PlanCacheEntry.from_dict(payload)

    def save(self, entry: PlanCacheEntry) -> None:
        os.makedirs(self.root_dir, exist_ok=True)
        path = self._path_for(entry.model_name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(entry.to_dict(), handle, indent=2, sort_keys=True)

    @staticmethod
    def worker_signature(worker_specs: Iterable[WorkerSpec]) -> str:
        parts = []
        for spec in sorted(worker_specs, key=lambda item: item.worker_id):
            parts.append(
                ":".join(
                    [
                        spec.worker_id,
                        spec.device,
                        str(spec.bandwidth_mbps),
                        str(spec.memory_bytes),
                        "1" if spec.online else "0",
                        ",".join(spec.tags),
                    ]
                )
            )
        return "|".join(parts)
