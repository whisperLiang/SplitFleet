"""A baseline repair cannot authorize unrelated training or dependency edits."""
from pathlib import Path
import hashlib
import json
import shutil
import pytest
from experiments.analysis.baseline_source_amendment import (
    baseline_source, expected_after, shared_execution_identity, FILES, EDITS,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(folder):
    hashes = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob("*.py")}
    return {"source_hashes": hashes, "source_identity": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "packages": {"torch": "same"}}


@pytest.fixture
def correction(tmp_path):
    old, new, phase = [tmp_path / n for n in ("old", "new", "phase")]
    for name in FILES:
        previous = b"# source identity fixture only\n" + b"\n".join(before for before, _ in EDITS[name])
        for folder, content in ((old, previous), (new, expected_after(previous, name))):
            p = folder / name; p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(content)
    for name in ("splitfleet/client/grpc/connection.py", "splitfleet/server/app.py"):
        for folder in (old, new):
            p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'("grpc.keepalive_timeout_ms", 60_000)\n')
    for folder in (old,new):
        p=folder/"experiments/training.py";p.write_text("training = 'unchanged'\n")
    shutil.copytree(new, phase / "runtime_snapshot")
    before, after = manifest(old), manifest(new)
    paths = [tmp_path / n for n in ("before.json", "after.json")]
    receipt={"schema":"splitfleet.fedbuff-buffer-source-amendment.v1","created_unix":1,"allowed_files":list(FILES),
        "affected_aggregation_method":"fedbuff","quality_or_timing_selection_gate":False,
        "original_runtime_manifest_path":str(paths[0]),"amended_runtime_manifest_path":str(paths[1]),
        "original_source_root":str(old),"amended_source_root":str(new),
        "original_source_identity":before["source_identity"],"amended_source_identity":after["source_identity"],
        "bound_artifacts_sha256":{str(p):write(p,v) for p,v in zip(paths,(before,after))}}
    write(phase/"baseline_source_amendment.json",receipt)
    return phase, after, {"source_identity":after["source_identity"],"source_root":str(new)}, {"started_unix":2}, receipt, before, old


def test_shared_execution_keeps_the_changed_adapter_identity_separate(correction):
    root,runtime,plan,ledger,_,before,old=correction
    receipt,artifacts,guards=baseline_source(root,runtime,plan,ledger)
    assert receipt and artifacts and runtime["source_identity"] != before["source_identity"]
    old_phase=old.parent/"old_phase";shutil.copytree(old,old_phase/"runtime_snapshot")
    assert shared_execution_identity(root,runtime,guards)==shared_execution_identity(old_phase,before)


def test_baseline_repair_must_precede_dispatch(correction):
    root,runtime,plan,_,_,_,_=correction
    with pytest.raises(ValueError,match="retrospective"):
        baseline_source(root,runtime,plan,{"started_unix":1})


def test_evaluation_guard_cannot_hide_an_optimizer_change_even_rehashed(correction):
    root,runtime,plan,ledger,receipt,_,_=correction
    name="experiments/common/workload_training.py"
    for folder in (root/"runtime_snapshot",Path(receipt["amended_source_root"])):
        p=folder/name;p.write_bytes(p.read_bytes()+b"\nlearning_rate_override = 0.1\n")
    runtime.update(manifest(Path(receipt["amended_source_root"])))
    p=Path(receipt["amended_runtime_manifest_path"]);receipt["bound_artifacts_sha256"][str(p)]=write(p,runtime)
    plan["source_identity"]=receipt["amended_source_identity"]=runtime["source_identity"]
    write(root/"baseline_source_amendment.json",receipt)
    with pytest.raises(ValueError,match="beyond the declared"):
        baseline_source(root,runtime,plan,ledger)


def test_guard_normalization_requires_the_bound_amendment(correction):
    root,runtime,*_=correction
    with pytest.raises(ValueError,match="not bound"):
        shared_execution_identity(root,runtime)


def test_new_unamended_run_keeps_actual_evaluation_guard_hashes(correction, tmp_path):
    root, runtime, *_ = correction
    fresh = tmp_path / "fresh"
    shutil.copytree(root / "runtime_snapshot", fresh / "runtime_snapshot")
    receipt, artifacts, guards = baseline_source(fresh, runtime, {}, {})
    assert receipt is None and artifacts == guards == {}
    expected = {name: digest for name, digest in runtime["source_hashes"].items() if name != FILES[0]}
    for name in ("splitfleet/client/grpc/connection.py", "splitfleet/server/app.py"):
        expected[name] = hashlib.sha256(b'("grpc.keepalive_timeout_ms", 10_000)\n').hexdigest()
    actual = shared_execution_identity(fresh, runtime, guards)
    assert actual == hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    _, _, historical_guards = baseline_source(root, runtime, correction[2], correction[3])
    assert actual != shared_execution_identity(root, runtime, historical_guards)


def test_baseline_repair_rejects_dependency_change_even_rebound(correction):
    root,runtime,plan,ledger,receipt,_,_=correction
    p=Path(receipt["amended_runtime_manifest_path"]);value=json.loads(p.read_text());value["packages"]["torch"]="changed"
    receipt["bound_artifacts_sha256"][str(p)]=write(p,value);write(root/"baseline_source_amendment.json",receipt)
    with pytest.raises(ValueError,match="dependency environment"):
        baseline_source(root,runtime,plan,ledger)
