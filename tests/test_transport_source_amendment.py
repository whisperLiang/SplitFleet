"""Transport-only exceptions cannot authorize changes to training source."""
from pathlib import Path
import hashlib
import json
import pytest
from experiments.analysis.transport_source_amendment import comparison_source, FILES, OLD, NEW


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(folder):
    hashes = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob("*.py")}
    return {"source_hashes": hashes, "source_identity": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), "packages": {"torch": "same"}}


@pytest.fixture
def amendment(tmp_path):
    old, new, phase = [tmp_path / n for n in ("old", "new", "phase")]
    for folder, fragment in ((old, OLD), (new, NEW), (phase / "runtime_snapshot", NEW)):
        for name in (*FILES, "experiments/training.py"):
            p = folder / name; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(fragment + b"\n" if name in FILES else b"training = 'unchanged'\n")
    before, after = manifest(old), manifest(new)
    paths = [tmp_path / n for n in ("before.json", "after.json")]
    bound = {str(p):write(p, v) for p,v in zip(paths, (before, after))}
    receipt = {"schema": "splitfleet.transport-grace-source-amendment.v1", "created_unix": 1,
        "allowed_files": list(FILES), "original_grace_ms": 10000, "amended_grace_ms": 60000,
        "original_runtime_manifest_path": str(paths[0]), "amended_runtime_manifest_path": str(paths[1]),
        "original_source_root": str(old), "amended_source_root": str(new),
        "original_source_identity": before["source_identity"], "amended_source_identity": after["source_identity"],
        "bound_artifacts_sha256": bound}
    write(phase / "source_amendment.json", receipt)
    plan = {"source_identity": after["source_identity"], "source_root": str(new)}
    return phase, after, plan, {"started_unix": 2}, receipt, before


def test_transport_normalization_keeps_actual_identity_separate(amendment):
    root, runtime, plan, ledger, _, before = amendment
    identity, receipt, artifacts = comparison_source(root, runtime, plan, ledger)
    assert identity == before["source_identity"] != runtime["source_identity"]
    assert len(receipt["allowed_files"]) == 2 and artifacts


def test_transport_amendment_must_precede_execution(amendment):
    root, runtime, plan, _, _, _ = amendment
    with pytest.raises(ValueError, match="retrospectively"):
        comparison_source(root, runtime, plan, {"started_unix": 1})


def test_transport_exception_cannot_hide_training_file_change(amendment):
    root, runtime, plan, ledger, _, _ = amendment
    (root / "runtime_snapshot/experiments/training.py").write_text("training = 'changed'")
    with pytest.raises(ValueError, match="source file changed"):
        comparison_source(root, runtime, plan, ledger)


def test_transport_exception_cannot_add_unbound_runtime_code(amendment):
    root, runtime, plan, ledger, _, _ = amendment
    (root / "runtime_snapshot/extra.py").write_text("extra = True")
    with pytest.raises(ValueError, match="unbound source files"):
        comparison_source(root, runtime, plan, ledger)


def test_transport_exception_rejects_dependency_change_even_rehashed(amendment):
    root, runtime, plan, ledger, receipt, _ = amendment
    path = Path(receipt["amended_runtime_manifest_path"])
    value = json.loads(path.read_text());value["packages"]["torch"] = "changed"
    receipt["bound_artifacts_sha256"][str(path)] = write(path, value)
    write(root / "source_amendment.json", receipt)
    with pytest.raises(ValueError, match="dependency environment"):
        comparison_source(root, runtime, plan, ledger)
