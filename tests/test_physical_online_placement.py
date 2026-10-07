"""The canonical physical runner uses native CoSplit-UCB defaults."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.physical_multitask import _make_placement_policy
from splitfleet.server.placement.cosplit_ucb import SafeExplorationController
from tests.test_cosplit_calibration import _bootstrap_policy


def test_physical_policy_uses_default_controller_without_experiment_switches(tmp_path):
    policy = _make_placement_policy(
        provider=object(), bundle={"seed": 7401},
        args=SimpleNamespace(min_residence_rounds=2, output=tmp_path / "result.json"),
    )
    assert type(policy.exploration_controller) is SafeExplorationController
    assert policy.config.safe_exploration_epsilon == .05
    assert policy.context_encoder.edge_dimension == 13
    assert policy.context_encoder.feature_schema == "cosplit_context"


def test_missing_worker_batch_context_is_refused_before_cost_updates(tmp_path):
    policy, worker, props = _bootstrap_policy(tmp_path)
    props.pop("num_batches")
    with pytest.raises(ValueError, match="positive num_batches"):
        policy.bind_clients([worker], 1)
    assert policy.learners.server.model.num_updates == 0


def test_only_current_telemetry_provider_is_bound():
    from splitfleet.server.placement.cosplit_ucb import CoSplitUCBPlacementPolicy
    calls = []
    telemetry = SimpleNamespace(bind_clients=lambda *args: calls.append(args))
    policy = CoSplitUCBPlacementPolicy(candidate_provider=object(), telemetry_provider=telemetry)
    policy.bind_clients([], 2)
    assert calls == [([], 2)]


def test_study_command_uses_default_implementation():
    from experiments.www2027_study import command_for
    plan = dict(deployment="deployment.json", data_root="data", workers=6, dirichlet_alpha=10)
    stage = dict(name="nano", seeds=[7401], tasks=["object_detection"], image_model="rfdetr_nano",
        model_id="rfdetr_nano", checkpoint="weights.pth", train_samples=240, test_samples=200,
        rounds=10, batch_size=1, optimizer="adam", learning_rate=1e-5, timeout_per_scheme_sec=14400)
    command, _ = command_for(plan, stage, 0, Path("new-run"))
    assert "--model" in command
    assert all(flag not in command for flag in ("--device-profiles", "--placement-mode", "--online-initialization"))


def test_analysis_excludes_missing_online_learning_or_changed_parameters(tmp_path, monkeypatch):
    from experiments import www2027_study as study
    from tests.test_www2027_study import fake_audit, nano_primary_block
    monkeypatch.setattr(study, "summarize", fake_audit)
    plan = nano_primary_block(tmp_path)
    plan["cosplit_hyperparameters"] = {"discount_gamma": .98}
    path = tmp_path / "physical/timing_s1/splitfleet/result.json"
    result = json.loads(path.read_text())
    result.update(online_cost_learning=True, cosplit_config={"discount_gamma": .98})
    study.save_json(path, result)
    assert study.analyze(plan, tmp_path)["reports"][0]["paired_n"] == 1
    for field, value, reason in (("online_cost_learning", False, "online learning"),
                                 ("cosplit_config", {"discount_gamma": .9}, "parameters differ")):
        study.save_json(path, {**result, field: value})
        analysis = study.analyze(plan, tmp_path)
        assert analysis["reports"][0]["paired_n"] == 0
        assert reason in analysis["exclusions"][0]["reason"]
