"""Configured edge identities must keep partition and profile indices aligned."""
import copy
import json

import pytest

from experiments.common.physical_workers import physical_workers
from experiments.orchestrate_physical_multitask import validate_result
from tests.test_physical_multitask_barrier import _complete_short_round
from tests.test_physical_multitask_full_epoch import patch_unit_workload
from experiments.physical_multitask import load_bundle, prepare_bundle
from experiments.summarize_physical_multitask import summarize


def test_mixed_host_configuration_uses_contiguous_private_partition_indices():
    hosts = [{"id": "small", "workers": ["gpu"]}, {"id": "large1"}, {"id": "large2"}]
    workers = physical_workers(hosts)
    assert [w["identity"] for w in workers] == ["small-gpu", "large1-cpu", "large1-gpu", "large2-cpu", "large2-gpu"]
    assert [w["index"] for w in workers] == list(range(5))
    assert [w["device"] for w in workers] == ["cuda:0", "cpu", "cuda:0", "cpu", "cuda:0"]


@pytest.mark.parametrize("kinds", [[], ["gpu", "gpu"], ["tpu"]])
def test_invalid_host_devices_are_rejected(kinds):
    with pytest.raises(ValueError):
        physical_workers([{"id": "device", "workers": kinds}])


@pytest.mark.parametrize("gpu_only", [False, True])
def test_configured_bundle_owns_every_example_without_an_extra_client(tmp_path, monkeypatch, gpu_only):
    checkpoint = patch_unit_workload(monkeypatch, train_size=61, test_size=17, seed=19, tmp_path=tmp_path)
    identities = ["small-gpu", "large1-cpu", "large1-gpu", "large2-cpu", "large2-gpu"]
    if gpu_only:
        identities = ["small-gpu", "large1-gpu", "large2-gpu"]
    count = len(identities)
    path = tmp_path / "server.pt"
    record = prepare_bundle(task="image_classification", data_root="unused", output=path, seed=19,
        train_samples=61, test_samples=17, batch_size=8, model_name="resnet50_pretrained",
        pretrain_weights=str(checkpoint), worker_ids=identities)
    assert record["worker_ids"] == identities and record["num_clients"] == count
    positions = []
    for index in range(count):
        _, bundle = load_bundle(path.with_name(f"server.client_{index}.pt"))
        positions.extend(bundle["partition_positions"])
    assert sorted(positions) == list(range(61))
    assert not path.with_name(f"server.client_{count}.pt").exists()
    assert record["edge_client_partition_sizes"]["small"] == record["partition_sizes"]["0"]


@pytest.mark.parametrize("gpu_only", [False, True])
def test_configured_validation_requires_all_updates_and_barrier(gpu_only):
    hosts, bundle, result, receipts = _complete_short_round()
    hosts[0]["workers"] = ["gpu"]
    if gpu_only:
        for host in hosts:
            host["workers"] = ["gpu"]
    identities = [w["identity"] for w in physical_workers(hosts)]
    count = len(identities)
    result["fit_records"] = [r for r in result["fit_records"] if r["metrics"]["logical_client_id"] in identities]
    result["expected_clients"] = count
    bundle.update(train_size=count, partition_sizes={str(i): 1 for i in range(count)},
                  assignments={str(i): [i] for i in range(count)}, worker_ids=identities)
    receipts[0]["ready_ids"] = identities
    receipts[0]["ready_monotonic_ns"] = {i: t for i, t in receipts[0]["ready_monotonic_ns"].items() if i in identities}
    report = validate_result(result, task="image_classification", method="fedavg", rounds=1,
        hosts=hosts, batch_size=1, bundle=bundle, barrier_records=receipts)
    assert report["valid"] and report["barrier_verified"] and report["workers"] == count
    missing = copy.deepcopy(result)
    missing["fit_records"].pop()
    assert not validate_result(missing, task="image_classification", method="fedavg", rounds=1,
        hosts=hosts, batch_size=1, bundle=bundle, barrier_records=receipts)["valid"]


def test_five_worker_summary_retains_six_matched_schemes(comparison_root):
    for folder in (p for p in comparison_root.iterdir() if p.is_dir()):
        result_path = folder / "result.json"
        result = json.loads(result_path.read_text())
        result.update(expected_clients=5, train_size=20)
        result["fit_records"].pop()
        result_path.write_text(json.dumps(result))
        validation_path = folder / "validation_report.json"
        validation = json.loads(validation_path.read_text())
        validation["workers"] = 5
        validation_path.write_text(json.dumps(validation))
        process_path = folder / "process_manifest.json"
        processes = json.loads(process_path.read_text())
        processes["worker_exit_codes"].pop("5")
        process_path.write_text(json.dumps(processes))
    summary = summarize(comparison_root)
    assert summary["workers"] == 5 and summary["scheme_count"] == 6
    assert all(r["train_examples"] == 20 for r in summary["learning_curves"])


# Reuse a protocol receipt fixture; no model-performance experiment is added.
from tests.test_physical_multitask_summary import comparison_root
