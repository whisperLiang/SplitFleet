"""Container-held tinygrad state must retain its parameter and module identity."""

import pytest

from tests.integration.test_torchlens_optional_frameworks import _run_isolated


@pytest.mark.parametrize("block_count", [1, 2])
def test_native_capture_discovers_every_residual_parameter(request, block_count):
    if _run_isolated(request, extra_env={"DEV": "CPU", "DEBUG": "0"}):
        return
    pytest.importorskip("tinygrad")
    from tinygrad.nn.state import get_state_dict

    from splitfleet.autosplit import prepare_torchlens_runtime
    from tests.integration.test_resnet18_all_backends_all_nodes import _build_tinygrad_resnet18

    # Diagnostic fixtures isolate discovery; the exhaustive eighteen-layer
    # integration test continues to exercise all eight blocks unchanged.
    model, inputs, _, _ = _build_tinygrad_resnet18()
    model.blocks = model.blocks[:block_count]
    handle = prepare_torchlens_runtime(
        model, inputs, boundary="50%", trainable=True,
        batch_axes={}, dynamic_batch=(2, 2),
    )
    parameters = {
        parameter.address: parameter
        for node in handle.runtime.trace_graph.nodes
        for parameter in node.param_refs
    }
    expected = get_state_dict(model)

    assert len(parameters) == 2 + 2 * block_count
    assert set(parameters) == set(expected)
    for address, parameter in parameters.items():
        assert parameter._param_ref is expected[address]


def test_nested_container_paths_use_actual_callable_module_parents(request):
    if _run_isolated(request, extra_env={"DEV": "CPU", "DEBUG": "0"}):
        return
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor
    from torchlens.backends.tinygrad.backend import (
        _synthetic_stack_for_address,
        _tinygrad_metadata_top_level,
        discover_tinygrad_module_tree,
    )

    class Scale:
        def __init__(self):
            self.weight = Tensor([0.5]).realize()

        def __call__(self, value):
            return value * self.weight

    class Group:
        def __init__(self):
            self.operations = {"tail": (Scale(),)}

        def __call__(self, value):
            return self.operations["tail"][0](value)

    class Model:
        def __init__(self):
            self.stages = [Group()]
            self.state = {"gain": (Tensor([1.25]).realize(),)}

        def __call__(self, value):
            return self.stages[0](value) * self.state["gain"][0]

    tree = discover_tinygrad_module_tree(Model())
    assert tree is not None
    assert tree.param_owner_by_address == {
        "state.gain.0": "self",
        "stages.0.operations.tail.0.weight": "stages.0.operations.tail.0",
    }
    assert set(tree.metadata) == {"self", "stages.0", "stages.0.operations.tail.0"}
    frames = _synthetic_stack_for_address("stages.0.operations.tail.0", tree)
    assert [frame.address for frame in frames] == [
        "self", "stages.0", "stages.0.operations.tail.0",
    ]
    assert [frame.module_type for frame in frames] == ["Model", "Group", "Scale"]
    assert _tinygrad_metadata_top_level("stages.0", tree.metadata["stages.0"], tree.metadata)
    assert not _tinygrad_metadata_top_level(
        "stages.0.operations.tail.0", tree.metadata["stages.0.operations.tail.0"], tree.metadata,
    )


def test_shared_module_and_tensor_aliases_survive_recursive_containers(request):
    if _run_isolated(request, extra_env={"DEV": "CPU", "DEBUG": "0"}):
        return
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor
    from torchlens.backends.tinygrad.backend import discover_tinygrad_module_tree

    class Scale:
        def __init__(self):
            self.weight = Tensor([0.5]).realize()

        def __call__(self, value):
            return value * self.weight

    class Model:
        def __init__(self):
            shared = Scale()
            tensor = Tensor([1.25]).realize()
            self.layers = [shared, shared]
            self.weights = {"first": tensor, "second": tensor}
            self.cycle = []
            self.cycle.append(self.cycle)
            self.reference = self

        def __call__(self, value):
            return self.layers[1](self.layers[0](value)) * self.weights["first"]

    tree = discover_tinygrad_module_tree(Model())
    assert tree is not None
    assert len(tree.param_address_by_uop_id) == 2
    assert tree.param_owner_by_address == {
        "weights.first": "self", "weights.second": "self", "layers.0.weight": "layers.0",
    }
    assert tree.metadata["layers.0"]["all_addresses"] == ["layers.0", "layers.1"]
    assert tree.metadata["self"]["address_children"] == ["layers.0"]
    assert set(tree.metadata) == {"self", "layers.0"}


def test_namedtuple_state_paths_match_tinygrad_state_dict(request):
    if _run_isolated(request, extra_env={"DEV": "CPU", "DEBUG": "0"}):
        return
    pytest.importorskip("tinygrad")
    from collections import namedtuple

    from tinygrad import Tensor
    from tinygrad.nn.state import get_state_dict
    from torchlens.backends.tinygrad.backend import discover_tinygrad_module_tree

    from splitfleet.autosplit import prepare_torchlens_runtime

    State = namedtuple("State", ["weight", "bias"])
    Blocks = namedtuple("Blocks", ["scale"])

    class Scale:
        def __init__(self):
            self.gain = Tensor([0.5]).realize()

        def __call__(self, value):
            return value * self.gain

    class Model:
        def __init__(self):
            self.state = State(Tensor([2.0]).realize(), Tensor([0.25]).realize())
            self.blocks = Blocks(Scale())

        def __call__(self, value):
            return self.blocks.scale(value * self.state.weight + self.state.bias)

    model = Model()
    tree = discover_tinygrad_module_tree(model)
    assert tree is not None
    expected = get_state_dict(model)
    assert set(tree.param_owner_by_address) == set(expected) == {
        "state.weight", "state.bias", "blocks.scale.gain",
    }
    assert tree.metadata["self"]["address_children"] == ["blocks.scale"]
    handle = prepare_torchlens_runtime(
        model, Tensor.ones(2, 1), boundary="50%", trainable=False,
        batch_axes={}, dynamic_batch=(2, 2),
    )
    parameters = {
        parameter.address: parameter
        for node in handle.runtime.trace_graph.nodes
        for parameter in node.param_refs
    }
    assert set(parameters) == set(expected)
    for address, parameter in parameters.items():
        assert parameter._param_ref is expected[address]
