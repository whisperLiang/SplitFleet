"""Corrections cannot relax hyperparameter or per-run identity checks."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.www2027_study import cosplit_metadata_correction, validate_cosplit_config


def correction_record(tmp_path):
    plan = {"cosplit_hyperparameters": {"seed": 10400, "state_path": "/pilot/result.bandit.json", "alpha_edge": 1.0}}
    frozen = tmp_path / "frozen_plan.json"
    frozen.write_text(json.dumps(plan))
    runtime = {"source_identity": "frozen-source", "source_hashes": {"experiments/physical_multitask.py": "physical-hash"}}
    (tmp_path / "runtime_manifest.json").write_text(json.dumps(runtime))
    record = {"schema": "splitfleet.cosplit-run-metadata-correction.v1", "metadata_only": True,
        "training_parameters_changed": False, "runs_restarted": False,
        "original_frozen_plan_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "runtime_source_identity": runtime["source_identity"],
        "frozen_physical_module_sha256": runtime["source_hashes"]["experiments/physical_multitask.py"],
        "original_fields": {key: plan["cosplit_hyperparameters"][key] for key in ("seed", "state_path")},
        "corrected_field_rules": {"seed": "paired_seed_from_frozen_stage", "state_path": "result_path_with_bandit_json_suffix"}}
    return plan, record


def test_explicit_correction_still_binds_identity_and_numeric_settings(tmp_path):
    plan, record = correction_record(tmp_path)
    (tmp_path / "cosplit_run_metadata_correction.json").write_text(json.dumps(record))
    correction = cosplit_metadata_correction(tmp_path, plan)
    result_path = tmp_path / "fresh_run" / "result.json"
    actual = {"seed": 10501, "state_path": str(result_path.with_suffix(".bandit.json")), "alpha_edge": 1.0}
    validate_cosplit_config(actual, plan["cosplit_hyperparameters"], seed=10501,
                           result_path=result_path, correction=correction)
    with pytest.raises(ValueError, match="seed"):
        validate_cosplit_config(actual, plan["cosplit_hyperparameters"], seed=10501, result_path=result_path)
    for field, value in (("seed", 10502), ("state_path", "/other/result.bandit.json"), ("alpha_edge", .5)):
        changed = {**actual, field: value}
        with pytest.raises(ValueError, match=field):
            validate_cosplit_config(changed, plan["cosplit_hyperparameters"], seed=10501,
                                   result_path=result_path, correction=correction)


def test_derived_run_identity_is_checked_without_copied_pilot_fields(tmp_path):
    path = tmp_path / "new_block" / "result.json"
    expected = {"alpha_edge": 1.0}
    actual = {"alpha_edge": 1.0, "seed": 10501, "state_path": str(path.with_suffix(".bandit.json"))}
    validate_cosplit_config(actual, expected, seed=10501, result_path=path, require_run_identity=True)
    for key, value in (("seed", 10502), ("state_path", "/pilot/result.bandit.json")):
        with pytest.raises(ValueError, match=key):
            validate_cosplit_config({**actual, key: value}, expected, seed=10501,
                                    result_path=path, require_run_identity=True)


@pytest.mark.parametrize("field,value", [
    ("training_parameters_changed", True), ("runs_restarted", True),
    ("original_frozen_plan_sha256", "different-plan"),
    ("runtime_source_identity", "different-source"),
    ("frozen_physical_module_sha256", "different-physical-code"),
    ("corrected_field_rules", {"seed": "paired_seed_from_frozen_stage", "alpha_edge": "relaxed"}),
    ("original_fields", {"seed": 7401, "state_path": "/pilot/result.bandit.json"}),
])
def test_correction_cannot_expand_scope_or_change_provenance(tmp_path, field, value):
    plan, record = correction_record(tmp_path)
    changed = copy.deepcopy(record)
    changed[field] = value
    (tmp_path / "cosplit_run_metadata_correction.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Invalid CoSplit run-metadata correction"):
        cosplit_metadata_correction(tmp_path, plan)
