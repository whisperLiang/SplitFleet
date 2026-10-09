"""Input-admission failures must not masquerade as numerical successes."""

import hashlib
import json

import pytest

from experiments.analysis import canonical_cifar_correctness as correctness


def test_missing_native_framework_is_an_explicit_unsupported_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(correctness.importlib.util, "find_spec", lambda name: None)
    result = correctness.validate_cifar_backend("tf", tmp_path / "unread_bundle")
    assert result["status"] == "unsupported"
    assert result["nodes"] == []
    assert "tensorflow" in result["reason"]


def test_changed_real_input_file_is_refused_before_native_execution(tmp_path):
    sample = tmp_path / "sample.npz"
    original = b"frozen representative images"
    sample.write_bytes(original)
    (tmp_path / "config.json").write_text(json.dumps({
        "files": {sample.name: hashlib.sha256(original).hexdigest()}}))
    sample.write_bytes(b"changed representative images")
    with pytest.raises(ValueError, match="Frozen bundle changed: sample.npz"):
        correctness.CanonicalCifarTraining("torch", tmp_path)


@pytest.mark.parametrize("backends", [[], ["torch", "torch"], ["unknown"]])
def test_empty_or_invalid_backend_population_cannot_pass_a_contract_run(tmp_path, backends):
    output = tmp_path / "never_created"
    with pytest.raises(ValueError, match="nonempty set"):
        correctness.run(output, tmp_path / "unread_bundle", backends=backends)
    assert not output.exists()
