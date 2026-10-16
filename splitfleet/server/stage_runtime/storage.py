"""Coordinator-local, spillable state sequences for bounded round aggregation."""

import pickle
import uuid
from collections.abc import Sequence
from pathlib import Path


class StoredSequence(Sequence):
    """Load one update at a time; never retain executable model/graph state."""

    def __init__(self, path, count):
        self.path, self.count = Path(path), count

    def __len__(self):
        return self.count

    def __iter__(self):
        with self.path.open("rb") as stream:
            for _ in range(self.count):
                yield pickle.load(stream)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return list(self)[key]
        if key < 0:
            key += self.count
        for i, value in enumerate(self):
            if i == key:
                return value
        raise IndexError(key)


class RoundStateStore:
    """Trusted local files, confined to one run and deleted at the round boundary."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def put(self, values):
        path = self.directory / f"{uuid.uuid4().hex}.pkl"
        count = 0
        with path.open("xb") as stream:
            for value in values:
                pickle.dump(value, stream, protocol=5)
                count += 1
        return StoredSequence(path, count)

    def clear(self):
        for path in self.directory.glob("*.pkl"):
            path.unlink()
