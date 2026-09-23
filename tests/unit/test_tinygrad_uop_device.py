"""Device rewriting must preserve shared UOp graphs without expanding paths."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from torchlens.split.adapters import tinygrad as adapter


@dataclass
class ReadBudget:
    limit: int
    reads: int = 0

    def read(self):
        self.reads += 1
        assert self.reads <= self.limit, "device rewrite repeatedly traversed shared paths"


class UOp:
    """A replaceable UOp with observable edges and immutable replacement semantics."""

    def __init__(self, op, sources=(), *, arg=None, dtype="float32", metadata=(), budget=None):
        self.op = op
        self._sources = sources
        self.arg = arg
        self.dtype = dtype
        self.metadata = metadata
        self.budget = budget

    @property
    def src(self):
        if self.budget is not None:
            self.budget.read()
        return self._sources

    def replace(self, **changes):
        assert set(changes) <= {"src", "arg"}
        return UOp(
            self.op,
            changes.get("src", self._sources),
            arg=changes.get("arg", self.arg),
            dtype=self.dtype,
            metadata=self.metadata,
            budget=self.budget,
        )


@pytest.fixture
def device_op(monkeypatch):
    device = object()
    monkeypatch.setattr(adapter, "_tinygrad_ops", lambda: SimpleNamespace(DEVICE=device))
    return device


def test_none_device_keeps_graph_without_inspecting_it(monkeypatch):
    def unexpected_ops():
        raise AssertionError("a missing target must not inspect tinygrad operations")

    monkeypatch.setattr(adapter, "_tinygrad_ops", unexpected_ops)
    root = UOp("add", budget=ReadBudget(0))

    assert adapter._rewrite_tinygrad_uop_device(root, None) is root


@pytest.mark.parametrize("target", ["CPU", "CUDA"])
def test_shared_device_graph_is_visited_once_per_node(device_op, target):
    # Thirty-two shared doublings have only 33 nodes but over 8 billion paths.
    depth = 32
    budget = ReadBudget(2 * depth)
    root = UOp(device_op, arg="CPU", budget=budget)
    for index in range(depth):
        root = UOp("add", (root, root), arg=index, budget=budget)

    result = adapter._rewrite_tinygrad_uop_device(root, target)

    assert budget.reads <= 2 * depth
    assert (result is root) == (target == "CPU")
    # Inspect private sources so the assertion itself consumes no read budget.
    for index in reversed(range(depth)):
        assert result.arg == index
        assert result._sources[0] is result._sources[1]
        result = result._sources[0]
    assert result.arg == target


def test_device_rewrite_preserves_operand_order_metadata_and_original(device_op):
    device = UOp(device_op, arg="CPU")
    left = UOp("constant", (device,), arg=2)
    right = UOp("constant", (device,), arg=3)
    root = UOp("subtract", (left, right, left), arg="keep", dtype="float64", metadata=("tag",))

    result = adapter._rewrite_tinygrad_uop_device(root, "CUDA")

    assert result.op == root.op
    assert (result.arg, result.dtype, result.metadata) == ("keep", "float64", ("tag",))
    assert [source.arg for source in result.src] == [2, 3, 2]
    assert result.src[0] is result.src[2]
    assert result.src[0].src[0] is result.src[1].src[0]
    assert result.src[0].src[0].arg == "CUDA"
    assert root.src == (left, right, left)
    assert device.arg == "CPU"


def test_deep_device_graph_does_not_use_python_recursion(device_op):
    depth = 4000
    budget = ReadBudget(2 * depth)
    root = UOp(device_op, arg="CPU", budget=budget)
    for _ in range(depth):
        root = UOp("identity", (root,), budget=budget)

    result = adapter._rewrite_tinygrad_uop_device(root, "CUDA")

    for _ in range(depth):
        assert result.op == "identity"
        result = result._sources[0]
    assert result.arg == "CUDA"


def test_device_rewrite_leaves_opaque_sources_unchanged(device_op):
    opaque = object()
    root = UOp("pair", (opaque, UOp(device_op, arg="CPU")))

    result = adapter._rewrite_tinygrad_uop_device(root, "CUDA")

    assert result.src[0] is opaque
    assert result.src[1].arg == "CUDA"


def test_device_rewrite_does_not_cache_across_calls(device_op):
    device = UOp(device_op, arg="CPU")
    root = UOp("identity", (device,))
    assert adapter._rewrite_tinygrad_uop_device(root, "CPU") is root

    device.arg = "CUDA"
    result = adapter._rewrite_tinygrad_uop_device(root, "CPU")

    assert result is not root
    assert result.src[0].arg == "CPU"
    assert device.arg == "CUDA"
