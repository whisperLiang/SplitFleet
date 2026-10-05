from types import SimpleNamespace

import pytest

from experiments.ablations.layer_level_candidates import LayerCandidateProvider, module_end_boundaries
from splitfleet.server.placement.cosplit_ucb import StaticCandidateProvider
from tests.unit.test_cosplit_policy import _candidate


def node(label, path=None, **flags):
    return SimpleNamespace(canonical_id=label, module_path=path, **flags)


def test_module_endpoints_follow_semantic_paths_and_skip_input_output_nodes():
    graph = SimpleNamespace(nodes=[node("input", "block", is_input=True), node("one", "block.linear"),
        node("two", "block.relu"), node("three", "head"), node("output", "head", is_output=True)])
    assert module_end_boundaries(graph) == {
        "after:one": ["block.linear"], "after:two": ["block", "block.relu"], "after:three": ["head"]}


def test_layer_provider_reuses_original_valid_descriptors_and_canonical_identities():
    values = [_candidate("after:one", .1), _candidate("before:middle", .5), _candidate("after:two", .7)]
    full = StaticCandidateProvider(values)
    layer = LayerCandidateProvider(full, {"after:one": ["layer1"], "after:two": ["layer2"],
                                         "after:rejected": ["output"]})
    selected = layer.get_candidates(training=True)
    assert selected == (values[0], values[2])
    assert selected[0] is values[0] and selected[1] is values[2]
    assert {candidate.split_id for candidate in selected} <= {candidate.split_id for candidate in values}
    with pytest.raises(ValueError, match="outside"):
        layer.get_placement_plan("before:middle")
    received = []
    full.get_placement_plan = lambda boundary, **kwargs: received.append((boundary, kwargs)) or "original plan"
    assert layer.get_placement_plan("after:one", worker_specs=["edge"]) == "original plan"
    assert received == [("after:one", {"training": True, "worker_specs": ["edge"]})]


def test_missing_module_metadata_is_explicitly_refused():
    with pytest.raises(ValueError, match="metadata"):
        LayerCandidateProvider(StaticCandidateProvider([]), {})
