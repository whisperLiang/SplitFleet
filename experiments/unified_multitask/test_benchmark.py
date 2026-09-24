from __future__ import annotations

import json

import pytest
import torch

from experiments.unified_multitask.data import _subset, load_workload
from experiments.unified_multitask.run import _train_client, run_benchmark
from experiments.unified_multitask.run_matrix import run_matrix
from experiments.unified_multitask.summarize import summarize
from experiments.unified_multitask.validate import validate_run


@pytest.mark.parametrize(
    "task",
    (
        "image_classification",
        "text_classification",
        "object_detection",
        "semantic_segmentation",
    ),
)
def test_four_task_splitfed_matches_full_local_model_updates(task: str) -> None:
    workload = load_workload(task, source="fixture", seed=17)
    torch.manual_seed(17)
    model = workload.model_factory()
    initial = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    common = dict(
        workload=workload,
        global_state=initial,
        indices=[0, 1, 2, 3],
        seed=17,
        round_id=1,
        client_id="0",
        batch_size=2,
        max_batches=2,
        learning_rate=0.01,
        proximal_mu=0.01,
        device=torch.device("cpu"),
    )
    full, _ = _train_client(method="fedavg", boundary=None, **common)
    split, telemetry = _train_client(method="splitfed_fixed", boundary="50%", **common)

    assert telemetry["boundary_upload_bytes"] > 0
    assert telemetry["boundary_download_bytes"] > 0
    assert telemetry["switch_ms"] is None
    for name in full:
        torch.testing.assert_close(split[name], full[name], rtol=2e-4, atol=2e-6, msg=name)


def test_fedprox_changes_a_multi_batch_local_update() -> None:
    workload = load_workload("image_classification", source="fixture", seed=19)
    torch.manual_seed(19)
    initial = {name: tensor.detach().clone() for name, tensor in workload.model_factory().state_dict().items()}
    common = dict(
        workload=workload,
        global_state=initial,
        indices=[0, 1, 2, 3],
        boundary=None,
        seed=19,
        round_id=1,
        client_id="0",
        batch_size=2,
        max_batches=2,
        learning_rate=0.01,
        proximal_mu=5.0,
        device=torch.device("cpu"),
    )
    fedavg, _ = _train_client(method="fedavg", **common)
    fedprox, _ = _train_client(method="fedprox", **common)

    assert any(not torch.equal(fedavg[name], fedprox[name]) for name in fedavg)


def test_real_pilot_subset_spans_dataset_instead_of_taking_first_rows() -> None:
    source = torch.utils.data.TensorDataset(torch.arange(120))
    selected = _subset(source, 8)
    assert selected.indices[0] == 0
    assert selected.indices[-1] == 119
    assert len(set(selected.indices)) == 8


def test_text_macro_f1_counts_all_four_classes_when_subset_misses_classes() -> None:
    workload = load_workload("text_classification", source="fixture")
    assert workload.task.evaluate([0, 0], [0, 0])["macro_f1"] == pytest.approx(0.25)


def test_validator_rejects_missing_round_records(tmp_path) -> None:
    workload = load_workload("image_classification", source="fixture", seed=23)
    run_dir = run_benchmark(workload, method="fedavg", output=tmp_path / "run", seed=23)
    assert validate_run(run_dir)["valid"]
    (run_dir / "round_metrics.jsonl").write_text("", encoding="utf-8")

    report = validate_run(run_dir)
    assert not report["valid"]
    assert any("Round records" in error for error in report["errors"])


def test_matrix_launcher_runs_and_validates_each_method_sequentially(tmp_path, monkeypatch) -> None:
    calls = []

    def record(command, *, check, env):
        assert check is True
        assert env["OMP_NUM_THREADS"] == "1"
        calls.append(command)

    monkeypatch.setattr("experiments.unified_multitask.run_matrix.subprocess.run", record)
    destinations = run_matrix(
        tasks=("image_classification",),
        methods=("fedavg", "splitfed_fixed"),
        output_root=tmp_path,
        run_prefix="serial",
        source="fixture",
        data_root="data",
        seed=1,
        rounds=1,
        max_train_samples=8,
        max_test_samples=4,
        batch_size=2,
        max_batches_per_client=1,
    )

    assert len(destinations) == 2
    assert [command[2] for command in calls] == [
        "experiments.unified_multitask.run",
        "experiments.unified_multitask.validate",
        "experiments.unified_multitask.run",
        "experiments.unified_multitask.validate",
    ]


def test_summary_does_not_pair_different_local_batch_budgets(tmp_path) -> None:
    workload = load_workload("image_classification", source="fixture", seed=29)
    run_benchmark(
        workload, method="fedavg", output=tmp_path / "baseline", seed=29,
        max_batches_per_client=1,
    )
    run_benchmark(
        workload, method="splitfleet", output=tmp_path / "proposed", seed=29,
        max_batches_per_client=2,
    )

    report = summarize(tmp_path, tmp_path / "summary.json")
    assert report["valid_runs"] == 2
    assert report["comparisons"] == []


def test_summary_does_not_pair_different_execution_devices(tmp_path) -> None:
    workload = load_workload("image_classification", source="fixture", seed=31)
    baseline = run_benchmark(workload, method="fedavg", output=tmp_path / "baseline", seed=31)
    run_benchmark(workload, method="splitfleet", output=tmp_path / "proposed", seed=31)
    metadata_path = baseline / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["device"] = "cuda:0"
    metadata["device_name"] = "different accelerator"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = summarize(tmp_path, tmp_path / "summary.json")
    assert report["valid_runs"] == 2
    assert report["comparisons"] == []


def test_summary_distinguishes_fixed_boundary_variants(tmp_path) -> None:
    workload = load_workload("image_classification", source="fixture", seed=37)
    run_benchmark(workload, method="splitfleet", output=tmp_path / "proposed", seed=37)
    for boundary in ("25%", "50%"):
        run_benchmark(
            workload, method="splitfed_fixed", fixed_boundary=boundary,
            output=tmp_path / f"fixed_{boundary}", seed=37,
        )

    report = summarize(tmp_path, tmp_path / "summary.json")
    assert report["valid_runs"] == 3
    assert {row["baseline_config"]["fixed_boundary"] for row in report["comparisons"]} == {"25%", "50%"}
