"""Explicit model calls and task batches, independent of a tensor framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelInputs:
    """A model call. A list is one argument; a tuple represents several arguments."""

    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.args, tuple):
            raise TypeError("ModelInputs.args must be a tuple (use (images,) for an image list).")
        if not isinstance(self.kwargs, Mapping) or not all(isinstance(k, str) for k in self.kwargs):
            raise TypeError("ModelInputs.kwargs must map string argument names to values.")
        object.__setattr__(self, "kwargs", dict(self.kwargs))

    @classmethod
    def from_value(cls, value: Any) -> ModelInputs:
        if isinstance(value, cls):
            return value
        return cls(args=value if isinstance(value, tuple) else (value,))

    def map_values(self, transform: Callable[[Any], Any]) -> ModelInputs:
        """Apply a recursive tensor-tree transform separately to args and kwargs."""
        return ModelInputs(args=transform(self.args), kwargs=transform(dict(self.kwargs)))


def infer_num_examples(value: Any) -> int:
    """Infer a leading tensor dimension; image-list tasks supply their own count."""
    if isinstance(value, ModelInputs):
        value = (value.args, value.kwargs)
    if hasattr(value, "shape"):
        if len(value.shape):
            return int(value.shape[0])
        raise ValueError("A scalar has no batch dimension.")
    children = value.values() if isinstance(value, Mapping) else value
    if isinstance(value, (Mapping, list, tuple)):
        for child in children:
            try:
                return infer_num_examples(child)
            except ValueError:
                pass
    raise ValueError("Cannot infer example count; provide TaskBatch.num_examples explicitly.")


@dataclass(frozen=True)
class TaskBatch:
    inputs: ModelInputs
    targets: Any = None
    num_examples: int | None = None

    def __post_init__(self) -> None:
        call = ModelInputs.from_value(self.inputs)
        object.__setattr__(self, "inputs", call)
        count = infer_num_examples(call) if self.num_examples is None else self.num_examples
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("TaskBatch.num_examples must be a positive integer.")
        object.__setattr__(self, "num_examples", count)


@runtime_checkable
class TaskAdapter(Protocol):
    """Task implementations translate data-loader batches and define the objective."""

    def prepare_batch(self, batch: Any, *, training: bool = True) -> TaskBatch: ...

    def loss(self, outputs: Any, targets: Any = None) -> Any: ...


def prepare_task_batch(batch: Any, *, task: TaskAdapter | None = None,
                       batch_adapter: Callable | None = None, training: bool = True) -> TaskBatch:
    if isinstance(batch, TaskBatch):
        return batch
    if batch_adapter is not None:
        batch = batch_adapter(batch)
        if isinstance(batch, TaskBatch):
            return batch
    elif task is not None:
        return task.prepare_batch(batch, training=training)
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return TaskBatch(ModelInputs.from_value(batch[0]), batch[1])
    raise ValueError("Expected TaskBatch or (inputs, targets); provide a task or batch_adapter for mappings.")
