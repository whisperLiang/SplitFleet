"""Persistent cache for autosplit partition and placement choices.

Optimized with:
- In-memory LRU cache layer for fast repeated access
- Async file I/O for non-blocking saves
- Thread-safe operations
- TTL-based cache invalidation
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
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


@dataclass
class _CacheEntry:
    """Internal cache entry with timestamp for TTL support."""
    entry: PlanCacheEntry
    timestamp: float


class PlanCacheStore:
    """Filesystem-backed cache with in-memory LRU layer.

    Features:
    - Two-tier caching: memory (LRU) + disk (JSON)
    - Thread-safe operations with fine-grained locking
    - Async disk writes for non-blocking saves
    - TTL-based memory cache invalidation
    """

    def __init__(
        self,
        root_dir: str,
        *,
        max_memory_cache: int = 32,
        ttl_seconds: float = 3600.0,
    ) -> None:
        """Initialize the cache store.

        Args:
            root_dir: Directory for persistent cache files.
            max_memory_cache: Maximum number of entries in memory cache.
            ttl_seconds: Time-to-live for memory cache entries (0 = no TTL).
        """
        self.root_dir = root_dir
        self._max_memory_cache = max_memory_cache
        self._ttl_seconds = ttl_seconds

        # Thread-safe LRU cache using OrderedDict
        self._memory_cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._cache_lock = threading.RLock()

        # Thread pool for async disk writes
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache_writer")

    def _path_for(self, model_name: str) -> str:
        safe_name = model_name.replace("\\", "_").replace("/", "_")
        return os.path.join(self.root_dir, f"{safe_name}.json")

    def load(self, model_name: str) -> Optional[PlanCacheEntry]:
        """Load a cache entry, checking memory cache first.

        Args:
            model_name: Name of the model to load cache for.

        Returns:
            Cached entry or None if not found/expired.
        """
        # Check memory cache first (fast path)
        with self._cache_lock:
            cached = self._memory_cache.get(model_name)
            if cached is not None:
                # Check TTL
                if self._ttl_seconds > 0:
                    if time.time() - cached.timestamp > self._ttl_seconds:
                        del self._memory_cache[model_name]
                    else:
                        # Move to end for LRU
                        self._memory_cache.move_to_end(model_name)
                        return cached.entry
                else:
                    self._memory_cache.move_to_end(model_name)
                    return cached.entry

        # Fall back to disk
        entry = self._load_from_disk(model_name)
        if entry is not None:
            with self._cache_lock:
                self._add_to_memory_cache(model_name, entry)
        return entry

    def _load_from_disk(self, model_name: str) -> Optional[PlanCacheEntry]:
        """Load entry from disk (internal method)."""
        path = self._path_for(model_name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                return None
            return PlanCacheEntry.from_dict(payload)
        except (json.JSONDecodeError, OSError, KeyError):
            return None

    def save(self, entry: PlanCacheEntry) -> None:
        """Save entry to memory cache and asynchronously to disk.

        Args:
            entry: Cache entry to save.
        """
        # Update memory cache immediately
        with self._cache_lock:
            self._add_to_memory_cache(entry.model_name, entry)

        # Async disk write
        self._executor.submit(self._save_to_disk, entry)

    def _save_to_disk(self, entry: PlanCacheEntry) -> None:
        """Save entry to disk (internal method, runs in thread)."""
        try:
            os.makedirs(self.root_dir, exist_ok=True)
            path = self._path_for(entry.model_name)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(entry.to_dict(), handle, indent=2, sort_keys=True)
        except OSError:
            pass  # Silently fail disk writes

    def _add_to_memory_cache(self, model_name: str, entry: PlanCacheEntry) -> None:
        """Add entry to memory cache with LRU eviction (must hold lock)."""
        # Remove if exists (for move-to-end behavior)
        if model_name in self._memory_cache:
            del self._memory_cache[model_name]

        # Evict oldest if at capacity
        while len(self._memory_cache) >= self._max_memory_cache:
            self._memory_cache.popitem(last=False)

        self._memory_cache[model_name] = _CacheEntry(
            entry=entry,
            timestamp=time.time(),
        )

    def invalidate(self, model_name: str) -> bool:
        """Remove entry from both memory and disk cache.

        Args:
            model_name: Name of model to invalidate.

        Returns:
            True if entry was removed, False if not found.
        """
        found = False
        with self._cache_lock:
            if model_name in self._memory_cache:
                del self._memory_cache[model_name]
                found = True

        path = self._path_for(model_name)
        if os.path.exists(path):
            try:
                os.remove(path)
                found = True
            except OSError:
                pass

        return found

    def clear_memory_cache(self) -> int:
        """Clear all entries from memory cache.

        Returns:
            Number of entries cleared.
        """
        with self._cache_lock:
            count = len(self._memory_cache)
            self._memory_cache.clear()
            return count

    def shutdown(self) -> None:
        """Shutdown the async executor (call on program exit)."""
        self._executor.shutdown(wait=True)

    @staticmethod
    def worker_signature(worker_specs: Iterable[WorkerSpec]) -> str:
        """Generate a signature string from worker specs.

        Optimized with list comprehension and join.
        """
        parts = [
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
            for spec in sorted(worker_specs, key=lambda item: item.worker_id)
        ]
        return "|".join(parts)
