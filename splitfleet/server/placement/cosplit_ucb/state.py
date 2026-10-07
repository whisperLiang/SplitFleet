"""Validated persistence for CoSplit-UCB sufficient statistics."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .types import ALGORITHM_NAME, FEATURE_SCHEMA


class BanditStateStore:
    """Persist JSON-compatible bandit state without requiring warm start."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None

    def envelope(
        self,
        state: Mapping[str, Any],
        *,
        graph_signature: str,
        backend: str,
        feature_schema: str = FEATURE_SCHEMA,
    ) -> dict[str, Any]:
        return {
            "algorithm": ALGORITHM_NAME,
            "feature_schema": str(feature_schema),
            "graph_signature": str(graph_signature),
            "backend": str(backend),
            "state": dict(state),
        }

    def validate(
        self,
        value: Mapping[str, Any],
        *,
        graph_signature: str,
        backend: str,
        feature_schema: str = FEATURE_SCHEMA,
    ) -> Mapping[str, Any]:
        if value.get("algorithm") != ALGORITHM_NAME:
            raise ValueError("CoSplit-UCB algorithm mismatch")
        if value.get("feature_schema") != str(feature_schema):
            raise ValueError("CoSplit-UCB feature schema mismatch")
        if str(value.get("graph_signature")) != str(graph_signature):
            raise ValueError("CoSplit-UCB graph signature mismatch")
        if str(value.get("backend")) != str(backend):
            raise ValueError("CoSplit-UCB backend mismatch")
        state = value.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("CoSplit-UCB state payload is invalid")
        return state

    def save(self, value: Mapping[str, Any]) -> None:
        """Atomically save state when a path was configured."""

        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self) -> dict[str, Any] | None:
        """Load persisted state, or return ``None`` for a cold start."""

        if self.path is None or not self.path.exists():
            return None
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("CoSplit-UCB state file must contain an object")
        return value


__all__ = ["BanditStateStore"]
