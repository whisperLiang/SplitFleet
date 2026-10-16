"""Bounded, private execution slots in one sequential Flower worker process.

Logical identity, task data, optimizer and RNG live in each task. A completed
slot can retain its native model and graph; the caller must reset every model
state before another task executes. Failed slots are always discarded.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from threading import RLock


class WorkerSlots:
    def __init__(self, *, scope, capacity):
        if not scope or type(capacity) is not int or capacity < 1:
            raise ValueError("Worker slots require an explicit run scope and positive capacity")
        self.scope, self.capacity = str(scope), capacity
        self.pid = os.getpid()
        self.idle = OrderedDict()
        self.active = {}
        self.lock = RLock()
        self.created = self.reused = self.peak_resident = 0

    def acquire(self, key, factory, cleanup):
        with self.lock:
            self._require_process()
            if key in self.active:
                raise RuntimeError("A worker execution slot is already active")
            item = self.idle.pop(key, None)
            reused = item is not None
            if item is None:
                if len(self.active) >= self.capacity:
                    raise RuntimeError("Worker slot capacity exceeded; active slots cannot be evicted")
                if len(self.active) + len(self.idle) >= self.capacity:
                    _, (previous, release) = self.idle.popitem(last=False)
                    release(previous)
                value = factory()
                item = value, cleanup
                self.created += 1
            else:
                self.reused += 1
            self.active[key] = item
            self.peak_resident = max(self.peak_resident, len(self.active) + len(self.idle))
            return item[0], reused

    def finish(self, key, value, *, success):
        with self.lock:
            self._require_process()
            actual, cleanup = self.active.pop(key)
            if actual is not value:
                self.active[key] = actual, cleanup
                raise RuntimeError("Worker execution slot identity mismatch")
            if success:
                self.idle[key] = actual, cleanup
            else:
                cleanup(actual)

    def clear(self):
        with self.lock:
            self._require_process()
            if self.active:
                raise RuntimeError("Cannot clear a worker pool with active executions")
            for value, cleanup in self.idle.values():
                cleanup(value)
            self.idle.clear()

    def _require_process(self):
        if os.getpid() != self.pid:
            raise RuntimeError("Worker execution pools cannot migrate to a different process")


_PROCESS_POOL = None
_PROCESS_LOCK = RLock()


def worker_slots(*, scope, capacity):
    """Return one bounded pool for this run in the actual Ray worker process."""
    global _PROCESS_POOL
    with _PROCESS_LOCK:
        if _PROCESS_POOL is not None and _PROCESS_POOL.pid != os.getpid():
            # A fork cannot reuse the parent's model or execution context.
            _PROCESS_POOL = None
        if _PROCESS_POOL is not None and (
            _PROCESS_POOL.scope != str(scope) or _PROCESS_POOL.capacity != capacity
        ):
            _PROCESS_POOL.clear()
            _PROCESS_POOL = None
        if _PROCESS_POOL is None:
            _PROCESS_POOL = WorkerSlots(scope=scope, capacity=capacity)
        return _PROCESS_POOL


def discard_worker_slots(scope):
    """Invalidate idle slots after outer Flower validation/serialization fails."""
    with _PROCESS_LOCK:
        if _PROCESS_POOL is not None and _PROCESS_POOL.scope == str(scope):
            _PROCESS_POOL.clear()
