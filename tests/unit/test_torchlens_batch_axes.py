from __future__ import annotations

import copy

import pytest
import torch

from splitfleet.autosplit.torchlens_backend import prepare_torchlens_runtime


class ExternalWeights(torch.nn.Module):
    def forward(self, weights, inputs):
        return torch.relu(inputs @ weights["projection"])


def test_declared_batch_axis_ignores_leading_parameter_tree() -> None:
    model = ExternalWeights().eval()
    weights = {"projection": torch.randn(8, 5)}
    handle = prepare_torchlens_runtime(
        model, (weights, torch.randn(2, 8)), trainable=False,
        batch_axes={"/args/1": 0}, dynamic_batch=(2, 4),
    )
    inputs = torch.randn(3, 8)
    boundary = handle.backend.run_prefix(weights, inputs)
    assert handle.plan.trace_batch_size == 2
    assert boundary.batch_size == 3
    assert boundary.metadata["runtime_batch_size"] == 3
    torch.testing.assert_close(handle.backend.run_suffix(boundary), model(weights, inputs))
    with pytest.raises(ValueError, match="outside configured dynamic_batch"):
        handle.backend.run_prefix(weights, torch.randn(5, 8))


class SequenceFirst(torch.nn.Module):
    def forward(self, sequence):
        return torch.relu(sequence * 2)


def test_batch_axis_need_not_be_leading_dimension() -> None:
    model = SequenceFirst().eval()
    handle = prepare_torchlens_runtime(
        model, torch.randn(7, 2, 4), trainable=False,
        batch_axes={"/args/0": 1}, dynamic_batch=(2, 4),
    )
    inputs = torch.randn(7, 3, 4)
    boundary = handle.backend.run_prefix(inputs)
    assert boundary.batch_size == 3
    torch.testing.assert_close(handle.backend.run_suffix(boundary), model(inputs))


class ImageList(torch.nn.Module):
    def forward(self, images):
        return [torch.relu(image * 2) for image in images]


def test_static_image_list_does_not_use_channels_as_batch_size() -> None:
    model = ImageList().eval()
    images = [torch.randn(3, 5, 6), torch.randn(3, 7, 8)]
    handle = prepare_torchlens_runtime(
        model, images, trainable=False, batch_axes={}, dynamic_batch=(1, 2),
    )
    boundary = handle.backend.run_prefix(images)
    assert handle.plan.trace_batch_size is None
    assert boundary.batch_size == boundary.metadata["runtime_batch_size"]
    for expected, actual in zip(model(images), handle.backend.run_suffix(boundary)):
        torch.testing.assert_close(actual, expected)


def test_capture_releases_wrappers_before_replica_deepcopy() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)).eval()
    model.tl_user_marker = "keep me"
    model[0].tl_user_marker = "also keep me"
    sample = torch.randn(2, 4)
    first = prepare_torchlens_runtime(model, sample, trainable=False)
    replica = copy.deepcopy(model)
    second = prepare_torchlens_runtime(replica, sample, trainable=False)
    assert model.tl_user_marker == replica.tl_user_marker == "keep me"
    assert replica[0].tl_user_marker == "also keep me"
    assert all("forward" not in vars(module) for module in model.modules())
    assert first.plan.feature_abi_id == second.plan.feature_abi_id
    boundary = second.backend.run_prefix(sample)
    torch.testing.assert_close(second.backend.run_suffix(boundary), replica(sample))


def test_batchnorm_probe_refuses_unverified_extrapolation() -> None:
    from torchlens.split.errors import SplitBoundaryError
    from tests.test_autosplit_runtime import BatchNormNet

    model = BatchNormNet().train()
    handle = prepare_torchlens_runtime(model, torch.randn(2, 4), boundary="after:fc1", dynamic_batch=(2, 4))
    assert handle.runtime.batch_validation["mode"] == "captured_only"
    inputs = torch.randn(2, 4)
    expected = copy.deepcopy(model)(inputs)
    torch.testing.assert_close(handle.backend.run_suffix(handle.backend.run_prefix(inputs)), expected)
    with pytest.raises(SplitBoundaryError, match="batch.*probe"):
        handle.backend.run_prefix(torch.randn(3, 4), training=True)


def test_keyword_only_tensor_order_matches_capture_and_replay() -> None:
    from tests.test_flower_task_round_loop import KeywordClassifier

    model = KeywordClassifier().eval()
    kwargs = {"input_ids": torch.tensor([[1, 2, 0], [4, 5, 6]]),
              "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]])}
    handle = prepare_torchlens_runtime(model, (), sample_kwargs=kwargs, boundary="after:embedding")
    for supplied in (kwargs, dict(reversed(list(kwargs.items())))):
        boundary = handle.backend.run_prefix(input_kwargs=supplied)
        torch.testing.assert_close(handle.backend.run_suffix(boundary), model(**kwargs))


def test_input_contract_symbolizes_only_the_declared_axis() -> None:
    from splitfleet.split_engine.torchlens_engine import _tensor_schema

    schema = _tensor_schema({"args": ({"weights": torch.zeros(8, 5)}, torch.zeros(7, 2, 4))},
                            batch_axes={"/args/1": 1})
    assert schema["args"][0]["weights"]["shape"] == [8, 5]
    assert schema["args"][1]["shape"] == [7, "B", 4]
