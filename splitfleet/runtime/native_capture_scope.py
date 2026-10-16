"""Exclude temporary process-wide capture hooks from native execution."""

from contextlib import contextmanager
from threading import Condition, RLock, get_ident


class NativeCaptureGate:
    """Reentrant execution readers and a fair, exclusive capture writer.

    Native execution readers overlap. Capture writers exclusively own temporary
    process-wide hooks, and nested native calls may complete while a writer waits.
    """

    def __init__(self):
        self.condition = Condition(RLock())
        self.readers = {}
        self.writer = None
        self.write_depth = 0
        self.waiting_writers = 0

    @contextmanager
    def execution(self):
        owner = get_ident()
        with self.condition:
            # A nested native method must finish even if capture is waiting.
            if owner not in self.readers and self.writer != owner:
                self.condition.wait_for(
                    lambda: self.writer is None and self.waiting_writers == 0
                )
            self.readers[owner] = self.readers.get(owner, 0) + 1
        try:
            yield
        finally:
            with self.condition:
                self.readers[owner] -= 1
                if not self.readers[owner]:
                    del self.readers[owner]
                self.condition.notify_all()

    @contextmanager
    def capture(self):
        owner = get_ident()
        with self.condition:
            if self.writer == owner:
                self.write_depth += 1
            else:
                if owner in self.readers:
                    raise RuntimeError(
                        "Cannot capture native operations inside an active execution scope"
                    )
                self.waiting_writers += 1
                try:
                    self.condition.wait_for(
                        lambda: self.writer is None and not self.readers
                    )
                    self.writer, self.write_depth = owner, 1
                finally:
                    self.waiting_writers -= 1
                    self.condition.notify_all()
        try:
            yield
        finally:
            with self.condition:
                self.write_depth -= 1
                if not self.write_depth:
                    self.writer = None
                self.condition.notify_all()
