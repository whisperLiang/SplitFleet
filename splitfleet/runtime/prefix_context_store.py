"""Process-local, consume-once storage for graph-connected prefix state."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class PrefixContextKey:
    round_id: int
    client_id: str
    step_id: str


class PrefixContextStore:
    def __init__(self) -> None:
        self._contexts: dict[PrefixContextKey, Any] = {}
        self._lock = RLock()

    def put(self, round_id: int, client_id: str, step_id: str, context: Any) -> PrefixContextKey:
        key = PrefixContextKey(int(round_id), str(client_id), str(step_id))
        with self._lock:
            if key in self._contexts:
                raise KeyError(f"Prefix context already exists for {key}")
            self._contexts[key] = context
        return key

    def pop(self, round_id: int, client_id: str, step_id: str) -> Any:
        key = PrefixContextKey(int(round_id), str(client_id), str(step_id))
        with self._lock:
            try:
                return self._contexts.pop(key)
            except KeyError as exc:
                raise KeyError(f"No pending prefix context for {key}") from exc

    def discard_round(self, round_id: int, client_id: str | None = None) -> int:
        with self._lock:
            keys = [
                key for key in self._contexts
                if key.round_id == int(round_id) and (client_id is None or key.client_id == str(client_id))
            ]
            for key in keys:
                del self._contexts[key]
            return len(keys)

    def __len__(self) -> int:
        with self._lock:
            return len(self._contexts)
