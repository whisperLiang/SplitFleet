"""The CLI FedAvg path uses the same classification objective and SGD."""

import copy
import json

import pytest
import torch

from experiments.virtual_sfl import __main__ as cli
from experiments.virtual_sfl.workloads import TinyClassifier, build_spec, fixture_dataset, train_classification_batch
from splitfleet.virtual import VirtualConfig


def test_full_model_trainer_matches_direct_sgd():
    model = TinyClassifier()
    reference = copy.deepcopy(model)
    inputs, targets = fixture_dataset(8).tensors
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
    expected_loss = torch.nn.functional.cross_entropy(reference(inputs), targets)
    expected_loss.backward()
    expected_optimizer.step()
    count, loss = train_classification_batch(model, (inputs, targets), optimizer, "cpu", .01)
    assert count == 8 and loss == pytest.approx(float(expected_loss.detach()))
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["splitfleet", "fedavg"])
def test_frozen_cli_dispatches_the_configured_training_method(tmp_path, monkeypatch, method):
    from splitfleet.virtual import fedavg

    output = tmp_path / "run"
    output.mkdir()
    config = VirtualConfig(workload={"source": "fixture", "samples": 8, "training_method": method})
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config.as_dict()))
    (output / "result.json").write_text('{"status": "running"}')
    seen = []

    def runner(config, spec, *, output):
        seen.append(method)
        assert (spec.trainer is not None) == (method == "fedavg")
        (output / "validation.json").write_text('{"status": "passed"}')
        (output / "round_metrics.jsonl").write_text('{"round_id": 1}\n')
        return {"status": "passed"}

    def wrong_runner(*args, **kwargs):
        pytest.fail("CLI selected the other training method")

    monkeypatch.setattr(cli, "run_virtual", runner if method == "splitfleet" else wrong_runner)
    monkeypatch.setattr(fedavg, "run_fedavg", runner if method == "fedavg" else wrong_runner)
    monkeypatch.setattr(cli, "__file__", str(output / "runtime_snapshot/experiments/virtual_sfl/__main__.py"))
    monkeypatch.setattr("sys.argv", ["virtual_sfl", "run", "--frozen", "--config", str(config_path), "--output", str(output)])
    cli.main()
    assert seen == [method]


def test_invalid_training_method_fails_before_loading_data():
    with pytest.raises(ValueError, match="training_method"):
        build_spec(VirtualConfig(workload={"training_method": "typo"}))


@pytest.mark.parametrize("first_run_fails", [False, True])
def test_campaign_runs_methods_in_order_and_preserves_second_job_on_failure(tmp_path, monkeypatch, first_run_fails):
    from experiments.virtual_sfl import comparison

    source = tmp_path / "campaign.json"
    source.write_text(json.dumps({"runs": {
        method: {"workload": {"source": "fixture", "training_method": method}}
        for method in ("splitfleet", "fedavg")
    }}))
    output = tmp_path / "output"
    seen = []

    def run(config, path):
        method = config.workload["training_method"]
        state = json.loads((output / "campaign_status.json").read_text())
        assert state["jobs"][method]["status"] == "running"
        if method == "splitfleet":
            assert state["jobs"]["fedavg"]["status"] == "pending"
        seen.append(method)
        if method == "splitfleet" and first_run_fails:
            raise RuntimeError("simulated hard exit")
        return {"status": "passed"}

    monkeypatch.setattr(cli, "frozen_run", run)
    monkeypatch.setattr(comparison, "summarize_comparison", lambda directory: {})
    monkeypatch.setattr("sys.argv", ["virtual_sfl", "run", "--config", str(source), "--output", str(output)])
    if first_run_fails:
        with pytest.raises(RuntimeError, match="Campaign contains failed runs"):
            cli.main()
    else:
        cli.main()
    assert seen == ["splitfleet", "fedavg"]
    report = json.loads((output / "campaign_status.json").read_text())
    assert report["jobs"]["fedavg"]["status"] == "passed"
    assert report["status"] == ("failed" if first_run_fails else "passed")
