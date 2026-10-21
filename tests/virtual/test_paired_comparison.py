"""A speed comparison cannot silently mix different training inputs."""

import csv
import json

import pytest

from experiments.virtual_sfl.comparison import summarize_comparison


def prepare_pair(root):
    for method, seconds in (("splitfleet", 10), ("fedavg", 20)):
        run = root / method
        run.mkdir()
        values = {
            "config.json": {"rounds": 1, "seed": 233, "virtual_clients": {"total": 2},
                            "workload": {"training_method": method}},
            "validation.json": {"status": "passed", "initial_model_hash": "same-model"},
            "partition_manifest.json": {"clients": {"0": {"hash": "same-partition"}},
                                        "data": {"data_content_hash": "same-data"}},
            "runtime_manifest.json": {"source_identity": "same-source"},
            "model_metrics.json": {"evaluations": [{"round_id": 1, "elapsed_wall_sec": 9,
                "metrics": {"loss": .8, "accuracy": .5}}]},
            "invocation_timing.json": {"wall_sec": seconds, "child_exit_code": 0},
        }
        for name, value in values.items():
            (run / name).write_text(json.dumps(value))
        (run / "round_metrics.jsonl").write_text(json.dumps({
            "round_id": 1, "successful_clients": 2, "failed_clients": 0,
            "train_loss": 1.0, "round_wall_sec": 8,
        }) + "\n")


def test_verified_pair_exports_round_curves_and_measured_time(tmp_path):
    prepare_pair(tmp_path)
    report = summarize_comparison(tmp_path)
    assert report["shared_inputs_verified"]
    assert report["fedavg_seconds_over_splitfleet_seconds"] == 2
    with (tmp_path / "convergence_comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and {r["method"] for r in rows} == {"splitfleet", "fedavg"}


@pytest.mark.parametrize("file,path", [
    ("config.json", ["seed"]),
    ("validation.json", ["initial_model_hash"]),
    ("validation.json", ["status"]),
    ("partition_manifest.json", ["clients", "0", "hash"]),
    ("partition_manifest.json", ["data", "data_content_hash"]),
    ("runtime_manifest.json", ["source_identity"]),
])
def test_mismatched_pair_cannot_produce_comparison_outputs(tmp_path, file, path):
    prepare_pair(tmp_path)
    target = tmp_path / "fedavg" / file
    data = json.loads(target.read_text())
    owner = data
    for part in path[:-1]:
        owner = owner[part]
    owner[path[-1]] = "different"
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        summarize_comparison(tmp_path)
    assert not (tmp_path / "convergence_comparison.csv").exists()
