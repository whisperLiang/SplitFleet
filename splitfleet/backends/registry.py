"""Explicit backend adapter registry for state and tensor codecs."""

from __future__ import annotations

from typing import Callable

from splitfleet.backends.base import BackendAdapter


class BackendAdapterRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], BackendAdapter]] = {}

    def register(self, name: str, factory: Callable[[], BackendAdapter], *, replace: bool = False) -> None:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("Backend name must not be empty")
        if key in self._factories and not replace:
            raise KeyError(f"Backend adapter {key!r} is already registered")
        self._factories[key] = factory

    def create(self, name: str) -> BackendAdapter:
        key = str(name).strip().lower()
        try:
            return self._factories[key]()
        except KeyError as exc:
            raise KeyError(f"Unknown backend adapter {key!r}; available: {sorted(self._factories)}") from exc


BACKEND_ADAPTERS = BackendAdapterRegistry()
