"""Captured model-tensor ownership must determine live tinygrad buffer binding."""

from types import SimpleNamespace
from enum import Enum

import pytest

from torchlens.split.adapters.tinygrad import _TinygradGeneratedSegmentBase


class Ops(Enum):
    BUFFER = "buffer"
    RESHAPE = "reshape"
    PERMUTE = "permute"


class UOp:
    def __init__(self, shape, *sources, op=None):
        self.shape = shape
        self.src = sources
        self.op = op or (Ops.RESHAPE if sources else Ops.BUFFER)

    def toposort(self):
        seen = set()
        stack = [self]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            yield node
            stack.extend(node.src)


class Tensor:
    def __init__(self, uop, *, shape=None, owner=None):
        self.uop = uop
        self.shape = uop.shape if shape is None else shape
        self.owner = self if owner is None else owner
        self.reshape_calls = 0

    def reshape(self, shape):
        self.reshape_calls += 1
        return Tensor(UOp(shape, self.uop), owner=self.owner)


def capture(uop, live_tensor=None, *, param_refs=(), op_type="buffer"):
    return SimpleNamespace(
        op_type=op_type,
        target=SimpleNamespace(uop=uop, live_tensor=live_tensor),
        param_refs=param_refs,
    )


def segment(nodes):
    result = object.__new__(_TinygradGeneratedSegmentBase)
    result.graph = SimpleNamespace(nodes=nodes)
    result._backend = SimpleNamespace(is_tensor=lambda value: isinstance(value, Tensor))
    return result


@pytest.fixture(autouse=True)
def tinygrad_operations(monkeypatch):
    monkeypatch.setattr("torchlens.split.adapters.tinygrad._tinygrad_ops", lambda: Ops)


def test_exact_tensor_capture_wins_over_unrelated_ancestor_parameter():
    buffer = UOp((1,))
    owner = Tensor(buffer)
    stem = Tensor(UOp((1,)))
    source = capture(buffer, owner)
    descendant = capture(
        UOp((1,), stem.uop, buffer),
        param_refs=(SimpleNamespace(_param_ref=stem),),
    )

    assert segment([source, descendant])._live_child_param_source_value(source) is owner


def test_captured_ancestry_finds_owner_without_parameter_annotations():
    buffer = UOp((4,))
    view_uop = UOp((4,), buffer)
    owner = Tensor(view_uop)
    source = capture(buffer)
    owner_capture = capture(view_uop, owner, op_type="reshape")

    assert segment([source, owner_capture])._live_child_param_source_value(source) is owner


def test_captured_ownership_survives_live_state_replacement():
    buffer = UOp((4,))
    view_uop = UOp((4,), buffer)
    owner = Tensor(view_uop)
    source = capture(buffer)
    owner_capture = capture(view_uop, owner, op_type="reshape")
    # State assignment changes the tensor's UOp but must retain its ownership.
    owner.uop = UOp((4,))

    assert segment([source, owner_capture])._live_child_param_source_value(source) is owner


def test_matrix_owner_is_flattened_once_despite_repeated_capture():
    buffer = UOp((6,))
    view_uop = UOp((2, 3), buffer)
    owner = Tensor(view_uop)
    source = capture(buffer)
    first = capture(view_uop, owner, op_type="reshape")
    second = capture(UOp((2, 3), view_uop), owner, op_type="reshape")

    result = segment([source, first, second])._live_child_param_source_value(source)

    assert result.shape == (6,)
    assert result.owner is owner
    assert owner.reshape_calls == 1


def test_ambiguous_tensor_owners_are_not_arbitrarily_selected():
    buffer = UOp((4,))
    first_uop, second_uop = UOp((4,), buffer), UOp((4,), buffer)
    source = capture(buffer)
    candidates = [
        source,
        capture(first_uop, Tensor(first_uop)),
        capture(second_uop, Tensor(second_uop)),
    ]

    assert segment(candidates)._live_child_param_source_value(source) is None


def test_parameter_references_alone_do_not_establish_buffer_ownership():
    buffer = UOp((1,))
    unrelated = Tensor(UOp((1,)))
    source = capture(buffer)
    descendant = capture(
        UOp((1,), buffer, unrelated.uop),
        param_refs=(SimpleNamespace(_param_ref=unrelated),),
    )

    assert segment([source, descendant])._live_child_param_source_value(source) is None


def test_permuted_owner_cannot_be_flattened_into_original_buffer():
    buffer = UOp((4,))
    matrix = UOp((2, 2), buffer)
    permuted = UOp((2, 2), matrix, op=Ops.PERMUTE)
    owner = Tensor(permuted)
    source = capture(buffer)
    owner_capture = capture(permuted, owner, op_type="permute")

    assert segment([source, owner_capture])._live_child_param_source_value(source) is None
    assert owner.reshape_calls == 0


@pytest.mark.parametrize("exact", [False, True])
def test_incompatible_owner_shape_is_rejected(exact):
    buffer = UOp((4,))
    owner_uop = UOp((6,), buffer)
    owner = Tensor(owner_uop)
    source = capture(buffer, owner if exact else None)
    owner_capture = capture(owner_uop, owner)

    assert segment([source, owner_capture])._live_child_param_source_value(source) is None


def test_non_buffer_node_has_no_buffer_owner():
    uop = UOp((1,))
    node = capture(uop, Tensor(uop), op_type="multiply")

    assert segment([node])._live_child_param_source_value(node) is None
