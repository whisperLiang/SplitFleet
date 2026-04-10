"""Execution-plan data structures for graph replay and graph partitioning.

These classes are intentionally lightweight: they store only the metadata needed
to rebuild one concrete traced execution graph and execute it node-by-node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import weakref

from torch import nn


@dataclass(frozen=True, slots=True)
class ParentRef:
    """Placeholder pointing at a previously computed node output."""

    node_idx: int
    path: Optional[Tuple[Any, ...]] = None
    frozen_value: Optional[Any] = None


@dataclass(frozen=True, slots=True)
class ParamRef:
    """Placeholder pointing at a live model parameter."""

    address: str


@dataclass(frozen=True, slots=True)
class BufferRef:
    """Placeholder pointing at a live model buffer."""

    address: str


@dataclass(frozen=True, slots=True)
class TensorConst:
    """Frozen tensor constant captured in a call template."""

    tensor: Any


class BoundaryPayload(TypedDict):
    """Serialized boundary-tensor payload used by partitioned replay/training."""

    cut_id: str
    labels: List[str]
    tensors: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass(slots=True)
class ExecNode:
    """One replayable node in a compiled execution plan."""

    idx: int
    label: str
    op: Any
    parents: List[int]
    parent_arg_locs: List[Any]
    const_args_template: Any
    const_kwargs_template: Any
    is_input: bool
    is_output: bool
    is_buffer: bool
    is_internal_init: bool
    is_inplace: bool
    rng_state: Optional[Dict[str, Any]]
    output_selector: Optional[Any]
    num_users: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        func_name = self.meta.get("func_name") or getattr(self.op, "__name__", None) or "none"
        flags = []
        if self.is_input:
            flags.append("input")
        if self.is_output:
            flags.append("output")
        if self.is_buffer:
            flags.append("buffer")
        if self.is_internal_init:
            flags.append("internal")
        flag_str = f", flags={'+'.join(flags)}" if flags else ""
        return (
            f"ExecNode(idx={self.idx}, label={self.label!r}, op={func_name!r}, "
            f"parents={self.parents}, users={self.num_users}{flag_str})"
        )


@dataclass(slots=True)
class ExecutionPlan:
    """Lightweight replay plan for one concrete traced execution graph."""

    nodes: List[ExecNode]
    node_to_index: Dict[str, int]
    input_specs: Any
    output_specs: Any
    graph_signature: str
    model_name: str
    default_device: Optional[str] = None
    shape_constraints: Optional[Any] = None
    dtype_constraints: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    input_node_indices: List[int] = field(default_factory=list)
    output_node_indices: List[int] = field(default_factory=list)
    buffer_node_indices: List[int] = field(default_factory=list)
    _model_ref: Optional[weakref.ReferenceType[nn.Module]] = field(default=None, repr=False)

    @property
    def model(self) -> Optional[nn.Module]:
        """Return the live model associated with the plan, if still alive."""

        if self._model_ref is None:
            return None
        return self._model_ref()

    def set_model(self, model: nn.Module) -> None:
        """Attach the plan to a live model."""

        self._model_ref = weakref.ref(model)

    def __repr__(self) -> str:
        return (
            f"ExecutionPlan(model_name={self.model_name!r}, nodes={len(self.nodes)}, "
            f"inputs={len(self.input_node_indices)}, outputs={len(self.output_node_indices)}, "
            f"graph_signature={self.graph_signature[:12]!r})"
        )


@dataclass(slots=True)
class FrontierSplit:
    """Valid frontier split of a concrete execution DAG."""

    split_id: str
    boundary_labels: List[str]
    boundary_indices: List[int]
    prefix_node_indices: List[int]
    suffix_node_indices: List[int]
    boundary_schema: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"FrontierSplit(split_id={self.split_id!r}, boundary={self.boundary_labels}, "
            f"prefix={len(self.prefix_node_indices)}, suffix={len(self.suffix_node_indices)})"
        )
