"""Explicit split-engine registry; no backend is selected implicitly."""

from __future__ import annotations

from typing import Callable

from splitfleet.split_engine.base import SplitEngine


class SplitEngineRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], SplitEngine]] = {}

    def register(self, name: str, factory: Callable[[], SplitEngine], *, replace: bool = False) -> None:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("Engine name must not be empty")
        if key in self._factories and not replace:
            raise KeyError(f"Split engine {key!r} is already registered")
        self._factories[key] = factory

    def create(self, name: str) -> SplitEngine:
        key = str(name).strip().lower()
        try:
            return self._factories[key]()
        except KeyError as exc:
            raise KeyError(f"Unknown split engine {key!r}; available: {sorted(self._factories)}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
