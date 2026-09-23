"""Explicit backend adapter registry for state and tensor codecs."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Callable

from splitfleet.backends.base import BackendAdapter


@dataclass(frozen=True)
class BackendAvailability:
    """Dependency discovery only; model/split support is checked by TorchLens."""

    name: str
    available: bool
    missing_modules: tuple[str, ...]
    install_extra: str | None = None


class BackendAdapterRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], BackendAdapter]] = {}
        self._dependencies: dict[str, tuple[str, ...]] = {}
        self._extras: dict[str, str | None] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], BackendAdapter],
        *,
        replace: bool = False,
        required_modules: tuple[str, ...] = (),
        install_extra: str | None = None,
    ) -> None:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("Backend name must not be empty")
        if key in self._factories and not replace:
            raise KeyError(f"Backend adapter {key!r} is already registered")
        self._factories[key] = factory
        self._dependencies[key] = tuple(required_modules)
        self._extras[key] = install_extra

    def names(self) -> tuple[str, ...]:
        """List registered names, including aliases, without loading frameworks."""
        return tuple(sorted(self._factories))

    def availability(self) -> tuple[BackendAvailability, ...]:
        """Discover missing dependencies without importing optional frameworks.

        An installed framework can still fail to load (for example a missing
        native library). Runtime preparation and validation remain mandatory.
        """
        reports = []
        for name in self.names():
            missing = []
            for module in self._dependencies[name]:
                try:
                    present = find_spec(module) is not None
                except (ImportError, ValueError):
                    present = False
                if not present:
                    missing.append(module)
            reports.append(BackendAvailability(name, not missing, tuple(missing), self._extras[name]))
        return tuple(reports)

    def create(self, name: str) -> BackendAdapter:
        key = str(name).strip().lower()
        try:
            factory = self._factories[key]
        except KeyError as exc:
            raise KeyError(f"Unknown backend adapter {key!r}; available: {sorted(self._factories)}") from exc
        return factory()


BACKEND_ADAPTERS = BackendAdapterRegistry()
