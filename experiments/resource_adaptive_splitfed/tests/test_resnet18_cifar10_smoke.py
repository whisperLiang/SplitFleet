from __future__ import annotations

import copy
import os

import pytest
import torch
from torch import nn

from experiments.resource_adaptive_splitfed.logical_state import (
    LogicalClientModelState,
    aggregate_named_states,
)
from experiments.resource_adaptive_splitfed.model_data import build_model, cifar10_datasets
from experiments.resource_adaptive_splitfed.resource_emulator import ResourceEmulator
from experiments.resource_adaptive_splitfed.resource_monitor import ResourceMonitor, ServerJobPool
from experiments.resource_adaptive_splitfed.split_candidates import discover_split_candidates
from experiments.resource_adaptive_splitfed.training_runtime import LogicalClientRuntime


def _real_cifar_batch():
    download = os.environ.get("SPLITFLEET_TEST_DOWNLOAD_CIFAR10") == "1"
    try:
        train, _ = cifar10_datasets("data", download=download, max_train_samples=8)
    except RuntimeError as exc:
        pytest.skip(
            "Real CIFAR-10 is not present. Set SPLITFLEET_TEST_DOWNLOAD_CIFAR10=1 "
            f"for this required hardware/integration smoke test: {exc}"
        )
    images = torch.stack([train[index][0] for index in range(2)])
    labels = torch.tensor([train[index][1] for index in range(2)])
    return images, labels


@pytest.mark.integration
def test_real_resnet18_cifar10_three_cuts_switch_wire_train_and_fedavg() -> None:
    torch.manual_seed(2025)
    images, labels = _real_cifar_batch()
    factory = lambda: build_model("resnet18")
    base = factory().train()
    initial = {name: tensor.detach().clone() for name, tensor in base.state_dict().items()}
    candidates = {item.split_key: item for item in discover_split_candidates(base, images)}
    keys = ("stem", "layer2", "layer4")
    client_states = []
    for index, split_key in enumerate(keys):
        runtime = LogicalClientRuntime(
            str(index), factory, images, candidates, device="cpu", learning_rate=1e-4
        )
        logical = LogicalClientModelState(
            str(index), {name: tensor.clone() for name, tensor in initial.items()}
        )
        handle, switch = runtime.activate(logical, split_key)
        reference = factory().train()
        reference.load_state_dict(initial)
        with torch.no_grad():
            expected = reference(images)
            actual = handle.backend.run_suffix(handle.backend.run_prefix(images))
        assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)
        measurement = runtime.train_batch(
            handle,
            images,
            labels,
            round_id=1,
            loss_fn=nn.CrossEntropyLoss(),
            server_pool=ServerJobPool(1),
            network=ResourceEmulator({}).link,
            energy=ResourceMonitor("cpu").energy,
        )
        assert torch.isfinite(torch.tensor(measurement.loss))
        assert measurement.boundary_forward_bytes > 0
        assert measurement.boundary_gradient_bytes > 0
        assert measurement.client_backward_ms >= 0
        logical.capture_model(runtime.model)
        client_states.append(logical.full_state_dict)

    aggregated = aggregate_named_states(client_states, [2, 2, 2])
    assert list(aggregated) == list(initial)
    assert all(torch.isfinite(value).all() for value in aggregated.values() if value.is_floating_point())

    switching = LogicalClientRuntime("switch", factory, images, candidates, learning_rate=1e-4)
    logical = LogicalClientModelState("switch", {name: tensor.clone() for name, tensor in initial.items()})
    early, _ = switching.activate(logical, "stem")
    reference = factory().train()
    reference.load_state_dict(initial)
    with torch.no_grad():
        expected_early = reference(images)
        actual_early = early.backend.run_suffix(early.backend.run_prefix(images))
    assert torch.allclose(expected_early, actual_early, atol=1e-5, rtol=1e-4)
    middle, _ = switching.activate(logical, "layer2")
    reference.load_state_dict(initial)
    with torch.no_grad():
        expected_middle = reference(images)
        actual_middle = middle.backend.run_suffix(middle.backend.run_prefix(images))
    assert torch.allclose(expected_middle, actual_middle, atol=1e-5, rtol=1e-4)
    full, _ = switching.activate(logical, "full_local")
    reference.load_state_dict(initial)
    with torch.no_grad():
        expected_full = reference(images)
        actual_full = switching.model(images)
    assert torch.allclose(expected_full, actual_full, atol=1e-5, rtol=1e-4)
    early_again, _ = switching.activate(logical, "stem")
    reference.load_state_dict(initial)
    with torch.no_grad():
        expected_again = reference(images)
        actual_again = early_again.backend.run_suffix(early_again.backend.run_prefix(images))
    assert torch.allclose(expected_again, actual_again, atol=1e-5, rtol=1e-4)
    assert early is early_again
    assert early.plan.feature_abi_id != middle.plan.feature_abi_id
    assert full is None
    assert set(switching.model.state_dict()) == set(initial)
