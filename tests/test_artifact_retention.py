import json

import pytest

from experiments.common import artifact_retention as retention


def test_live_file_changes_cannot_use_archived_verification(tmp_path, monkeypatch):
    path = tmp_path / "result.model.pt"
    path.write_bytes(b"original checkpoint")
    digest = retention.file_digest(path)
    assert retention.verify_retained_or_pruned(path, digest) == "retained_hash_checked"
    monkeypatch.setattr(retention, "RECEIPT", tmp_path / "nonexistent.json")
    path.write_bytes(b"changed checkpoint")
    with pytest.raises(AssertionError, match="File changed"):
        retention.verify_retained_or_pruned(path, digest)


def test_only_explicitly_verified_pruned_weights_can_use_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "ROOT", tmp_path)
    monkeypatch.setattr(retention, "RECEIPT", tmp_path / "receipt.json")
    path = tmp_path / "result.model.pt"
    digest = "a" * 64
    row = {"sha256": digest, "size_bytes": 100, "kind": "training_checkpoint", "deleted": True,
           "checkpoint_verification": {"status": "passed", "finite": True}}
    receipt = {"schema": "splitfleet.storage-pruning.v1", "status": "passed",
               "removed": {path.name: row}}
    retention.RECEIPT.write_text(json.dumps(receipt))
    assert retention.verify_retained_or_pruned(path, digest) == "checkpoint_verified_before_pruning"
    with pytest.raises(AssertionError, match="receipt hash differs"):
        retention.verify_retained_or_pruned(path, "b" * 64)
    with pytest.raises(AssertionError, match="Unrecorded missing"):
        retention.verify_retained_or_pruned(tmp_path / "result.json", digest)
    row["kind"] = "input_bundle"
    retention.RECEIPT.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):
        retention.verify_retained_or_pruned(path, digest)
    row["kind"] = "training_checkpoint"
    row["checkpoint_verification"]["finite"] = False
    retention.RECEIPT.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):
        retention.verify_retained_or_pruned(path, digest)
