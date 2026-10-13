"""Validate a whole-population replacement after a reproduced method defect."""
from pathlib import Path
import hashlib
import json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_replacement(root):
    root = Path(root).resolve()
    path = root / "replacement_population.json"
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != "splitfleet.method-correction-population-replacement.v1"
            or Path(receipt["original_root"]).resolve() != root
            or receipt.get("expected_primary_jobs") != 160
            or receipt.get("quality_or_timing_selection_gate") is not False
            or receipt.get("hyperparameters_or_seeds_changed") is not False
            or receipt.get("original_statuses_modified") is not False):
        raise ValueError("Invalid whole-population method correction")
    artifacts = {str(path): sha(path)}
    for name, expected in receipt["bound_artifacts_sha256"].items():
        if sha(name) != expected:
            raise ValueError("Retired population artifact changed: " + name)
        artifacts[name] = expected
    corrected = Path(receipt["corrected_root"]).resolve()
    old_plan = json.loads((root / "frozen_plan.json").read_text())
    new_plan = json.loads((corrected / "frozen_plan.json").read_text())
    old_runtime = json.loads((root / "runtime_manifest.json").read_text())
    identity = hashlib.sha256(json.dumps(old_runtime["source_hashes"], sort_keys=True).encode()).hexdigest()
    if identity != old_runtime["source_identity"]:
        raise ValueError("Retired source identity does not match its file hashes")
    ignored = {"source_root", "source_identity"}
    if ({k:v for k,v in old_plan.items() if k not in ignored} != {k:v for k,v in new_plan.items() if k not in ignored}
            or old_plan["source_identity"] != receipt["original_source_identity"]
            or old_runtime["source_identity"] != receipt["original_source_identity"]
            or new_plan["source_identity"] != receipt["corrected_source_identity"]):
        raise ValueError("Replacement changes methods, seeds or training/budget settings")
    from experiments.www2027_study import schemes_for
    methods = schemes_for(old_plan)
    if sum(len(s["seeds"]) * len(s["tasks"]) * len(methods) for s in old_plan["stages"]) != 160:
        raise ValueError("Replacement substitutes the original population")
    for name, expected in old_runtime["source_hashes"].items():
        source = root / "runtime_snapshot" / name
        if sha(source) != expected:
            raise ValueError("Retired runtime changed: " + name)
        artifacts[str(source)] = expected
    ledger = json.loads((root / "ledger.json").read_text())
    if not ledger.get("finished_unix", float("inf")) < receipt["created_unix"]:
        raise ValueError("Retirement was not bound after the controlled stop")
    attempts = []
    for job in ledger["jobs"]:
        p = root / "physical" / job["run_id"] / "attempts.json"
        if p.exists():
            if str(p) not in artifacts:
                raise ValueError("Every retired attempt record must be bound")
            attempts.extend(json.loads(p.read_text()))
    successes = sum(a["status"] == "completed" for a in attempts)
    paired = sum(j["status"] == "completed" for j in ledger["jobs"]) * len(methods)
    interrupted = sum(a["status"] == "running" for a in attempts)
    if (successes != receipt["retired_successful_method_attempts"]
            or paired != receipt["retired_complete_paired_endpoints"]
            or successes-paired != receipt["retired_unpaired_successes"]
            or interrupted != receipt["retained_controlled_interrupted_method_preparations"]):
        raise ValueError("Retirement conceals or changes attempted outcomes")
    reproductions = [Path(p) for p in artifacts if p.endswith("/reproduction.json")]
    if len(reproductions) != 1:
        raise ValueError("Retirement requires a bound independent defect reproduction")
    proof = json.loads(reproductions[0].read_text())
    if (proof.get("schema") != "splitfleet.fedbuff-buffer-regression.v1"
            or proof.get("test_fixture_only") is not True or proof.get("included_in_primary") is not False
            or proof.get("final_state_all_finite") is not True
            or proof.get("evaluation_predictions_all_finite") is not False
            or proof["committed_running_variances"][-1] >= 0
            or proof["source_sha256"] != old_runtime["source_hashes"]["experiments/baselines/async_fl.py"]):
        raise ValueError("Retirement lacks a reproduced implementation defect")
    new_ledger = corrected / "ledger.json"
    if new_ledger.exists():
        started = json.loads(new_ledger.read_text())["started_unix"]
        admission = corrected / "source_admission.json"
        if not admission.exists():
            raise ValueError("Corrected population started without source admission")
        gate = json.loads(admission.read_text())
        if (gate.get("status") != "passed" or gate.get("actual_runtime_source_identity") != new_plan["source_identity"]
                or not receipt["created_unix"] < started or not gate["created_unix"] < started):
            raise ValueError("Corrected population was not prospectively admitted")
        for name, expected in gate["bound_artifacts_sha256"].items():
            if sha(name) != expected:
                raise ValueError("Corrected population admission artifact changed: " + name)
            artifacts[name] = expected
        artifacts[str(admission)] = sha(admission)
    return corrected, receipt, artifacts
