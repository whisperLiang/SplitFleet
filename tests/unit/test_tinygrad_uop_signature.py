"""Structural identity and bounded work for TorchLens's tinygrad signatures."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import sys

import pytest

from torchlens.backends.tinygrad._uop_graph import _uop_signature


@dataclass
class SourceReads:
    limit: int
    count: int = 0


class UOp:
    """Minimal UOp protocol, with a guard against exponential traversal."""

    def __init__(self, op, *, dtype="float32", arg=None, src=(), reads=None):
        self.op = SimpleNamespace(name=op)
        self.dtype = dtype
        self.arg = arg
        self._src = tuple(src)
        self.reads = reads

    @property
    def src(self):
        if self.reads is not None:
            self.reads.count += 1
            assert self.reads.count <= self.reads.limit, (
                "Signature traversal repeatedly expanded shared subgraphs"
            )
        return self._src


def _sum():
    return UOp("ADD", src=(UOp("CONST", arg=1), UOp("CONST", arg=2)))


def test_equivalent_graphs_have_equal_signatures_regardless_of_object_sharing():
    shared = _sum()
    shared_graph = UOp("MUL", src=(shared, shared))
    copied_graph = UOp("MUL", src=(_sum(), _sum()))
    independently_copied_graph = UOp("MUL", src=(_sum(), _sum()))

    assert _uop_signature(shared_graph) == _uop_signature(copied_graph)
    assert _uop_signature(copied_graph) == _uop_signature(independently_copied_graph)


@pytest.mark.parametrize("changed", ["op", "dtype", "arg", "order", "arity", "descendant"])
def test_signatures_distinguish_graph_structure_and_metadata(changed):
    first, second = UOp("CONST", arg=1), UOp("CONST", arg=2)
    reference = UOp("SUB", src=(first, second))
    fields = {"op": "SUB", "src": (first, second)}
    if changed == "op":
        fields["op"] = "ADD"
    elif changed == "dtype":
        fields["dtype"] = "float64"
    elif changed == "arg":
        fields["arg"] = (1, 2)
    elif changed == "order":
        fields["src"] = (second, first)
    elif changed == "arity":
        fields["src"] = (first, second, second)
    else:
        fields["src"] = (first, UOp("CONST", arg=3))

    assert _uop_signature(reference) != _uop_signature(UOp(**fields))


def test_metadata_fields_are_unambiguously_delimited():
    first = UOp("CUSTOM", dtype="float32", arg="extra:value")
    second = UOp("CUSTOM", dtype="float32:extra", arg="value")

    assert _uop_signature(first) != _uop_signature(second)


def test_deep_graph_signature_does_not_depend_on_python_recursion_limit():
    root = UOp("CONST", arg=1)
    for _ in range(sys.getrecursionlimit() + 100):
        root = UOp("NEG", src=(root,))

    signature = _uop_signature(root)

    assert isinstance(signature, str)
    assert 0 < len(signature) <= 128


def test_shared_graph_signature_visits_each_node_a_bounded_number_of_times():
    levels = 80
    node_count, edge_count = levels + 1, 2 * levels
    reads = SourceReads(limit=4 * (node_count + edge_count))
    root = UOp("CONST", arg=1, reads=reads)
    for _ in range(levels):
        root = UOp("ADD", src=(root, root), reads=reads)

    signature = _uop_signature(root)

    assert 0 < len(signature) <= 128
    assert node_count <= reads.count <= reads.limit
