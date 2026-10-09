"""Storage defaults must preserve training inputs and numerical evidence."""
import json

import pytest
import torch

from experiments.common.artifact_retention import (
    cleanup_input_bundles, file_digest, verify_final_state_receipt, write_training_result,
)
from experiments.common.bundle_storage import (
    LEGACY_SCHEMA, bundle_dependencies, derive_bundle, read_bundle, write_bundle,
)
from experiments.common.identity import tensor_state_hash
from experiments.unified_multitask.data import _dataset_pair_hash


def payload(state, index=0):
    items = [(torch.full((3, 4, 4), float(index)), index)]
    return {"schema": LEGACY_SCHEMA, "task": "image_classification", "role": "client",
            "initial_model_state": state, "initial_model_hash": tensor_state_hash(state),
            "train_items": items, "test_items": [], "local_data_hash": _dataset_pair_hash(items, []),
            "online_calibration_boundaries": ["50%"]}


def test_role_and_layer_bundles_share_assets_without_sharing_runtime_storage(tmp_path):
    state = torch.nn.Linear(512, 512).state_dict()
    paths = []
    for index in range(4):
        path = tmp_path / f"client_{index}.pt"
        write_bundle(path, payload(state, index))
        layer = tmp_path / "layer" / path.name
        derive_bundle(path, layer, {"online_calibration_boundaries": ["25%"]})
        paths.append(path)
        assert bundle_dependencies(path) == bundle_dependencies(layer)
        assert read_bundle(path)["online_calibration_boundaries"] == ["50%"]
        assert read_bundle(layer)["online_calibration_boundaries"] == ["25%"]
    first, second = (read_bundle(path) for path in paths[:2])
    assert tensor_state_hash(first["initial_model_state"]) == tensor_state_hash(state)
    assert torch.equal(first["train_items"][0][0], torch.zeros(3, 4, 4))
    assert torch.equal(second["train_items"][0][0], torch.ones(3, 4, 4))
    first["initial_model_state"]["weight"].add_(1)
    assert tensor_state_hash(second["initial_model_state"]) == tensor_state_hash(state)
    stored = sum(path.stat().st_size for path in tmp_path.rglob("*.pt"))
    model_bytes = sum(value.numel() * value.element_size() for value in state.values())
    assert stored < 2 * model_bytes  # Eight role/variant descriptors, one weight copy.


@pytest.mark.parametrize("asset_index", [0, 1])
def test_corrupt_shared_weights_or_data_are_rejected(tmp_path, asset_index):
    path = tmp_path / "client.pt"
    write_bundle(path, payload(torch.nn.Linear(2, 2).state_dict()))
    asset = bundle_dependencies(path)[asset_index]
    with asset.open("ab") as stream:
        stream.write(b"unexpected bytes")
    with pytest.raises(ValueError, match="asset changed"):
        read_bundle(path)


def test_legacy_bundle_can_be_read_and_derived_without_modifying_it(tmp_path):
    path = tmp_path / "legacy.pt"
    original = payload(torch.nn.Linear(2, 2).state_dict())
    torch.save(original, path)
    digest = file_digest(path)
    assert tensor_state_hash(read_bundle(path)["initial_model_state"]) == original["initial_model_hash"]
    derived = tmp_path / "layer" / path.name
    derive_bundle(path, derived, {"online_calibration_boundaries": ["25%"]})
    assert file_digest(path) == digest
    assert tensor_state_hash(read_bundle(derived)["initial_model_state"]) == original["initial_model_hash"]


@pytest.mark.parametrize("save_model", [False, True])
def test_final_state_is_verified_with_optional_weight_retention(tmp_path, save_model):
    state = torch.nn.Linear(2, 2).state_dict()
    result = {"task": "classification", "method": "fedavg", "final_model_hash": tensor_state_hash(state),
              "evaluation_records": [{"accuracy": .75}], "fit_failures": []}
    output = tmp_path / "result.json"
    write_training_result(output, result, state, save_model=save_model)
    restored = json.loads(output.read_text())
    assert restored["evaluation_records"] == result["evaluation_records"]
    assert restored["fit_failures"] == []
    verify_final_state_receipt(restored)
    assert output.with_suffix(".model.pt").exists() == save_model
    if save_model:
        saved = torch.load(output.with_suffix(".model.pt"), weights_only=True)["state_dict"]
        assert tensor_state_hash(saved) == restored["final_model_hash"]
    restored["final_state_verification"]["state_hash"] = "incorrect"
    with pytest.raises(ValueError, match="verification"):
        verify_final_state_receipt(restored)


def test_nonfinite_or_mismatched_state_cannot_publish_success(tmp_path):
    state = {"weight": torch.ones(1)}
    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="result hash"):
        write_training_result(output, {"final_model_hash": "incorrect"}, state)
    state["weight"].fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        write_training_result(output, {"final_model_hash": tensor_state_hash(state)}, state)
    assert not output.exists() and not list(tmp_path.glob("*.pt"))


def test_finished_input_cleanup_keeps_metrics_failure_receipts_and_manifests(tmp_path):
    inputs = tmp_path / "bundles"
    path = inputs / "client.pt"
    write_bundle(path, payload(torch.nn.Linear(2, 2).state_dict()))
    manifest = inputs / "client.manifest.json"
    manifest.write_text('{"partition_hash":"frozen"}')
    result = tmp_path / "result.json"
    result.write_text('{"status":"failed","reason":"nonfinite loss"}')
    before = {p: file_digest(p) for p in (manifest, result)}
    cleanup_input_bundles(inputs, tmp_path / "input_storage.json")
    assert not list(inputs.rglob("*.pt"))
    assert all(file_digest(p) == digest for p, digest in before.items())
    receipt = json.loads((tmp_path / "input_storage.json").read_text())
    assert receipt["status"] == "pruned" and len(receipt["files"]) == 3
