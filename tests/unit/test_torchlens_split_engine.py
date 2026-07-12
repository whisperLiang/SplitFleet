from __future__ import annotations

import copy
import pytest
import torch
from dataclasses import replace
from torchlens.split import SplitFeatures, SplitRequest, percent

from splitfleet.split_engine import TorchLensSplitEngine


def _request(training: bool = True) -> SplitRequest:
    return SplitRequest(
        percent(50),
        backend="torch",
        features=SplitFeatures(dynamic_batch=(2, 8), training=training),
    )


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )


def test_independent_models_export_same_device_neutral_contract() -> None:
    torch.manual_seed(4)
    first, second = _model(), _model()
    second.load_state_dict(first.state_dict())
    sample = torch.randn(2, 4)
    engine = TorchLensSplitEngine()
    left = engine.export_contract(engine.prepare(first, sample, _request(False)))
    right = engine.export_contract(engine.prepare(second, sample.clone(), _request(False)))

    assert left.canonical_graph_hash == right.canonical_graph_hash
    assert left.split_id == right.split_id
    assert left.boundary_schema_hash == right.boundary_schema_hash
    assert left.digest == right.digest


def test_engine_training_keeps_context_local_and_consumes_once() -> None:
    model = _model()
    engine = TorchLensSplitEngine()
    handle = engine.prepare(model, torch.randn(2, 4), _request(True))
    handle.metadata.update(round_id=2, client_id="client-a", plan_id="plan", model_version=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    boundary, token = engine.run_prefix(handle, torch.randn(3, 4), training=True)
    assert token is not None
    assert all(isinstance(tensor.payload, bytes) for tensor in boundary.tensors)
    result = engine.run_suffix(handle, boundary, torch.randn(3, 2), optimizer)
    assert result.gradients is not None
    assert result.loss is not None
    engine.backward_prefix(handle, token, result.gradients, optimizer)

    with pytest.raises(KeyError, match="No pending"):
        engine.backward_prefix(handle, token, result.gradients, optimizer)


def test_engine_rejects_cross_backend_boundary() -> None:
    engine = TorchLensSplitEngine()
    handle = engine.prepare(_model(), torch.randn(2, 4), _request(False))
    boundary, _ = engine.run_prefix(handle, torch.randn(2, 4), training=False)
    boundary = replace(boundary, backend="tensorflow")
    with pytest.raises(ValueError, match="backend mismatch"):
        engine.run_suffix(handle, boundary)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_prefix_to_gpu_suffix_has_same_contract_and_trains() -> None:
    cpu_model = _model()
    gpu_model = copy.deepcopy(cpu_model).cuda()
    client_engine, server_engine = TorchLensSplitEngine(), TorchLensSplitEngine()
    client = client_engine.prepare(cpu_model, torch.randn(2, 4), _request(True))
    server = server_engine.prepare(gpu_model, torch.randn(2, 4, device="cuda"), _request(True))
    client.metadata.update(round_id=1, client_id="cpu-client", plan_id="plan", model_version=1)

    client_contract = client_engine.export_contract(client)
    server_contract = server_engine.export_contract(server)
    assert client_contract.canonical_graph_hash == server_contract.canonical_graph_hash
    assert client_contract.boundary_schema_hash == server_contract.boundary_schema_hash

    boundary, token = client_engine.run_prefix(client, torch.randn(3, 4), training=True)
    result = server_engine.run_suffix(
        server,
        boundary,
        torch.randn(3, 2, device="cuda"),
        torch.optim.SGD(gpu_model.parameters(), lr=0.01),
    )
    assert token is not None and result.gradients is not None
    client_engine.backward_prefix(
        client,
        token,
        result.gradients,
        torch.optim.SGD(cpu_model.parameters(), lr=0.01),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_prefix_to_gpu_suffix_trains_with_independent_runtimes() -> None:
    client_model = _model().cuda()
    server_model = copy.deepcopy(client_model).cuda()
    client_engine, server_engine = TorchLensSplitEngine(), TorchLensSplitEngine()
    client = client_engine.prepare(client_model, torch.randn(2, 4, device="cuda"), _request(True))
    server = server_engine.prepare(server_model, torch.randn(2, 4, device="cuda"), _request(True))
    client.metadata.update(round_id=2, client_id="gpu-client", plan_id="plan", model_version=2)

    boundary, token = client_engine.run_prefix(
        client, torch.randn(3, 4, device="cuda"), training=True
    )
    result = server_engine.run_suffix(
        server,
        boundary,
        torch.randn(3, 2, device="cuda"),
        torch.optim.SGD(server_model.parameters(), lr=0.01),
    )
    assert token is not None and result.gradients is not None
    client_engine.backward_prefix(
        client,
        token,
        result.gradients,
        torch.optim.SGD(client_model.parameters(), lr=0.01),
    )
