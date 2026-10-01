"""The physical benchmark must train every disjoint private partition."""

import math
from types import SimpleNamespace

import pytest
import torch

from experiments.physical_multitask import (
    FullClient, RoundBatches, _make_candidate_provider, load_bundle, prepare_bundle,
)
from experiments.orchestrate_physical_multitask import _run_one
from experiments.rfdetr_nano_physical import RFDETRCandidateProvider
from experiments.unified_multitask.data import load_workload
from experiments.unified_multitask.models import ImageClassifier, TextClassifier
from splitfleet.server.placement.cosplit_ucb import TorchLensCandidateProvider
from splitfleet.autosplit.types import PlacementConstraint, PlacementObjective
from splitfleet.tasks import ModelInputs


def test_private_bundles_and_full_local_epoch(tmp_path, monkeypatch):
    import experiments.physical_multitask as physical

    fixture = load_workload(
        "image_classification", source="fixture", seed=19,
        max_train_samples=61, max_test_samples=17,
    )
    monkeypatch.setattr(physical, "load_workload", lambda *args, **kwargs: fixture)
    monkeypatch.setattr(physical, "_wait_for_round_release", lambda *args: None)

    server_path = tmp_path / "cifar.pt"
    manifest = prepare_bundle(
        task="image_classification", data_root="unused", output=server_path,
        seed=19, train_samples=61, test_samples=17, batch_size=8,
    )
    server_workload, server_bundle = load_bundle(server_path)
    assert server_bundle["role"] == "server"
    assert len(server_workload.test_dataset) == 17
    assert torch.all(server_workload.train_dataset[0][0] == 0.5)
    assert sum(manifest["partition_sizes"].values()) == 61

    all_indices = []
    for index in range(6):
        worker_path = tmp_path / f"cifar.client_{index}.pt"
        workload, bundle = load_bundle(worker_path)
        assert bundle["role"] == "client"
        assert len(workload.test_dataset) == 0
        all_indices.extend(bundle["source_indices"])

        batches = RoundBatches(workload, seed=19, client_index=index, batch_size=8)
        assert sum(len(raw[0]) for raw in batches) == len(bundle["source_indices"])

        client = FullClient(
            workload=workload, bundle=bundle, identity=f"test-{index}",
            client_index=index, method="fedavg", device="cpu",
            learning_rate=0.01, barrier="unused",
            optimizer_name="adam" if index == 0 else "sgd",
        )
        parameters = [tensor.numpy() for tensor in bundle["initial_model_state"].values()]
        _, examples, metrics = client.fit(parameters, {"round_id": 1})
        assert examples == len(bundle["source_indices"])
        assert metrics["num_batches"] == math.ceil(examples / 8)
        assert math.isfinite(metrics["task_loss"])

    assert sorted(all_indices) == list(range(61))


def test_failed_split_prewarm_stops_before_connecting_to_the_server(tmp_path, monkeypatch):
    import experiments.physical_multitask as physical

    fixture = load_workload(
        "image_classification", source="fixture", seed=19,
        max_train_samples=24, max_test_samples=12,
    )
    monkeypatch.setattr(physical, "load_workload", lambda *args, **kwargs: fixture)
    bundle_path = tmp_path / "image_classification.pt"
    prepare_bundle(task="image_classification", data_root="unused", output=bundle_path,
                   seed=19, train_samples=24, test_samples=12, batch_size=2)
    attempts = []

    def failed_prewarm(self, **kwargs):
        attempts.append(kwargs)
        raise RuntimeError("unsupported training graph")

    def connect(**kwargs):
        pytest.fail("a failed runtime must not start a training worker")

    monkeypatch.setattr(physical.SplitClient, "prewarm_runtime", failed_prewarm)
    monkeypatch.setattr(physical, "split_start_client", connect)
    args = SimpleNamespace(
        bundle=tmp_path / "image_classification.client_0.pt", client_index=0,
        client_id="test-cpu", device="cpu", method="splitfed_fixed",
        optimizer="sgd", learning_rate=0.01, fixed_boundary="50%", barrier="unused",
    )
    with pytest.raises(RuntimeError, match="unsupported training graph"):
        physical.run_client(args)
    assert len(attempts) == 1


def test_text_candidate_window_covers_actual_training_batch():
    sample = ModelInputs(
        (), {"input_ids": torch.full((128, 64), 2, dtype=torch.long),
             "attention_mask": torch.ones((128, 64), dtype=torch.long)},
    )
    provider = TorchLensCandidateProvider(
        model=TextClassifier(100), sample_inputs=sample,
        batch_axes={"/kwargs/input_ids": 0, "/kwargs/attention_mask": 0},
        dynamic_batch=(1, 128),
    )
    assert provider.get_candidates(training=True)


def test_physical_candidates_require_a_trainable_client_prefix():
    provider = _make_candidate_provider(
        model=ImageClassifier(10), sample_inputs=torch.full((4, 3, 32, 32), 0.5),
        bundle={"task": "image_classification", "batch_size": 4, "image_model": "small"},
    )
    candidates = provider.get_candidates(training=True)
    assert candidates
    assert all(candidate.prefix_node_count >= 2 for candidate in candidates)
    assert all(candidate.metadata["prefix_trainable_parameter_count"] >= 1
               for candidate in candidates)
    assert all(candidate.boundary != "before:conv2d_1_1:1" for candidate in candidates)


@pytest.mark.parametrize("image_model", ["small", "rfdetr_nano"])
def test_physical_runner_exposes_all_cuts_for_every_model(image_model):
    provider = _make_candidate_provider(
        model=torch.nn.Sequential(*[
            layer for _ in range(6) for layer in (torch.nn.Linear(4, 4), torch.nn.ReLU())
        ]),
        sample_inputs=torch.randn(1, 4),
        bundle={"task": "object_detection", "batch_size": 1, "image_model": image_model},
    )
    candidates = provider.get_candidates(training=True)
    assert provider.catalog_diagnostics[True]["scope"] == "all_valid_operation_boundaries"
    assert len(candidates) > 3
    assert any(candidate.boundary.startswith("before:") for candidate in candidates)
    assert any(candidate.boundary.startswith("after:") for candidate in candidates)


def test_rfdetr_provider_prepares_evaluation_cut_from_evaluation_catalog():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)).train()
    provider = RFDETRCandidateProvider(model=model, sample_inputs=torch.randn(1, 4))
    candidates = provider.get_candidates(training=False)

    placement = provider.get_placement_plan(
        candidates[0].boundary, worker_specs=[], constraints=PlacementConstraint(),
        objective=PlacementObjective(), training=False,
    )

    handle = placement.metadata["_runtime_handle"]
    assert handle.model.training is False
    assert handle.plan.trainable is False
    assert model.training is True


def test_rfdetr_splitfleet_refuses_missing_device_profiles(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="requires device profiles"):
        _run_one(
            {"server": {}, "hosts": []}, task="object_detection", method="splitfleet",
            bundle_path=tmp_path / "bundle.pt", bundle={"image_model": "rfdetr_nano"},
            remote_root="/tmp/unused", run_dir=output, rounds=3, timeout=1,
            learning_rate=1e-5, optimizer="adam", fixed_boundary="50%",
        )
    assert not output.exists()
