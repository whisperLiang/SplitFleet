from __future__ import annotations

import pytest
import torch

from splitfleet.backends import TorchBackendAdapter
from splitfleet.split_engine import (
    ContractValidationError,
    GraphContract,
    SplitEngineRegistry,
    compare_contracts,
    validate_contract,
)


def test_contract_ignores_device_metadata_but_reports_first_identity_difference() -> None:
    expected = GraphContract(backend="torch", canonical_graph_hash="g", metadata={"device": "cpu"})
    actual = GraphContract(backend="torch", canonical_graph_hash="g", metadata={"device": "cuda:0"})
    validate_contract(expected, actual)

    mismatch = compare_contracts(expected, GraphContract(backend="tensorflow", canonical_graph_hash="g"))
    assert mismatch is not None
    assert mismatch.field == "backend"
    with pytest.raises(ContractValidationError, match="backend"):
        validate_contract(expected, GraphContract(backend="tensorflow", canonical_graph_hash="g"))


def test_torch_state_manifest_is_value_independent_and_round_trips() -> None:
    adapter = TorchBackendAdapter()
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    before = adapter.state_manifest(source)
    with torch.no_grad():
        source.weight.add_(5)
    assert adapter.state_manifest(source).schema_hash == before.schema_hash

    state = adapter.export_state(source)
    adapter.load_state(target, state)
    for left, right in zip(source.parameters(), target.parameters()):
        assert torch.equal(left, right)


def test_torch_ndarray_export_is_an_independent_snapshot() -> None:
    adapter = TorchBackendAdapter()
    model = torch.nn.Linear(2, 1)
    snapshot = adapter.export_ndarrays(model)

    with torch.no_grad():
        model.weight.add_(1)

    assert not torch.equal(torch.from_numpy(snapshot[0]), model.weight)


def test_engine_registry_requires_explicit_unique_names() -> None:
    registry = SplitEngineRegistry()
    marker = object()
    registry.register("demo", lambda: marker)  # type: ignore[arg-type]
    assert registry.create("DEMO") is marker
    with pytest.raises(KeyError, match="already registered"):
        registry.register("demo", lambda: marker)  # type: ignore[arg-type]
