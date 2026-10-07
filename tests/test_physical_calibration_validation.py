"""Physical result admission accepts restored Adam calibration, never tampered receipts."""
import copy
import json

import pytest

from experiments.orchestrate_physical_multitask import validate_result
from tests.test_physical_multitask_barrier import _complete_short_round
from tests.test_physical_multitask_full_epoch import patch_unit_workload
from experiments.physical_multitask import load_bundle, prepare_bundle


def calibrated_round():
    hosts, bundle, result, barriers = _complete_short_round()
    anchors = [f"cut-{i}" for i in range(4)]
    bundle.update(online_calibration_boundaries=anchors)
    result.update(method="splitfleet", owned_state_exchange=True, online_cost_learning=True,
                  cosplit_config={}, online_calibration_boundaries=anchors,
                  candidate_catalog=[{"boundary": cut, "prefix_trainable_parameter_count": 1} for cut in anchors])
    server = {"schema": "splitfleet.cosplit-calibration", "source": "server_shape_matched_current_deployment",
              "device": "cuda:1", "model_hash_before": "initial", "model_hash_after": "initial",
              "state_and_torch_rng_preserved": True, "persistent_optimizer_steps": 0,
              "optimizer_steps": 8, "warmup_batches": 1, "elapsed_sec": 1,
              "records": [{"boundary": cut, "graph_signature": "graph", "feature_abi_id": cut,
                           "client_forward_ms": 1, "client_backward_ms": 2, "local_tail_service_ms": 3,
                           "loss": .5, "measured_batches": 1, "optimizer_steps": 2, "optimizer_stage": "suffix"}
                          for cut in anchors]}
    contexts, samples, durations = {}, [], {}
    for fit in result["fit_records"]:
        identity = fit["metrics"]["logical_client_id"]
        fit["metrics"]["prefix_compute_sec"] = .1
        receipt = copy.deepcopy(server)
        receipt.update(source="client_private_sample_current_deployment", device=fit["metrics"]["device"])
        for row in receipt["records"]:
            row["optimizer_stage"] = "prefix"
            samples.append({"logical_client_id": identity, **row})
        durations[identity] = receipt["elapsed_sec"]
        contexts[fit["cid"]] = {"logical_client_id": identity, "online_calibration_receipt": json.dumps(receipt)}
    result["online_worker_contexts"] = contexts
    result["server_fit_records"] = [{"round_id": 1, "cid": fit["cid"]} for fit in result["fit_records"]]
    result["round_diagnostics"] = {"1": {"assignment": {cid: anchors[0] for cid in contexts},
        "learner_update_counts": {cid: {"server_update_count": 1, "group_update_count": 1} for cid in contexts}}}
    result["cost_initialization_receipt"] = {
        "source": "current_deployment_restored_training_cost_calibration", "initial_model_hash": "initial",
        "temporary_optimizer_updates": True, "discounted_online_updates_retained": True,
        "network_initialized": True, "prediction_override": False, "server_calibration": server,
        "client_samples": samples, "client_calibration_elapsed_sec": durations}
    workers = result["online_worker_contexts"]
    initialization = result["cost_initialization_receipt"]
    for row in result["candidate_catalog"]:
        row.update(optimizer_prefix_parameter_bytes=4, optimizer_suffix_parameter_bytes=8,
                   boundary_forward_bytes_by_batch_size={1: 12})
    initialization["transport_bootstrap"] = {
        "schema": "splitfleet.current-transport-probe", "source": "current_deployment_flower_echo",
        "elapsed_sec": 1, "samples": [{"logical_client_id": context["logical_client_id"], "payload_bytes": size,
                                       "repeat": repeat, "roundtrip_ms": 2, "echo_verified": True}
                                      for context in workers.values() for size in (65536, 2097152) for repeat in range(2)]}
    initialization["transport_bootstrap"]["warmup_samples"] = [
        {"logical_client_id": context["logical_client_id"], "payload_bytes": 65536, "repeat": repeat,
         "roundtrip_ms": 2, "echo_verified": True} for context in workers.values() for repeat in range(2)]
    initialization["state_exchange_bootstrap"] = {
        "schema": "splitfleet.current-state-exchange-probe",
        "source": "current_deployment_controlled_request_reply_sizes",
        "samples": [{"logical_client_id": context["logical_client_id"], "request_bytes": down,
                     "reply_bytes": up, "repeat": repeat, "roundtrip_ms": 2, "reply_verified": True}
                    for context in workers.values()
                    for down, up in ((65536, 65536), (2097152, 65536), (65536, 2097152)) for repeat in range(2)]}
    for row in result["fit_records"]:
        row["metrics"].update(coordinator_fit_rpc_ms=100, client_fit_handler_ms=80)
    for counts in result["round_diagnostics"]["1"]["learner_update_counts"].values():
        counts["state_exchange_update_count"] = 7
    return hosts, bundle, result, barriers


def validate(case):
    hosts, bundle, result, barriers = case
    return validate_result(result, task="image_classification", method="splitfleet", rounds=1,
                           hosts=hosts, batch_size=1, bundle=bundle, barrier_records=barriers)


def test_preparation_and_manifest_use_the_same_canonical_calibration(tmp_path, monkeypatch):
    checkpoint = patch_unit_workload(monkeypatch, train_size=61, test_size=17, seed=19, tmp_path=tmp_path)
    path = tmp_path / "server.pt"
    metadata = prepare_bundle(task="image_classification", data_root="unused", output=path,
        seed=19, train_samples=61, test_samples=17, batch_size=1, model_name="resnet50_pretrained",
        pretrain_weights=str(checkpoint))
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    _, server = load_bundle(path)
    anchors = tuple(metadata["online_calibration_boundaries"])
    assert anchors == tuple(manifest["online_calibration_boundaries"]) == tuple(server["online_calibration_boundaries"])
    assert "cosplit_cost_model" not in metadata
    for index in range(6):
        _, client = load_bundle(path.with_name(f"server.client_{index}.pt"))
        assert tuple(client["online_calibration_boundaries"]) == anchors
    case = calibrated_round()
    case[1]["online_calibration_boundaries"] = tuple(case[1]["online_calibration_boundaries"])
    assert validate(case)["valid"]


@pytest.mark.parametrize("tuple_bundle", [False, True])
def test_physical_admission_accepts_restored_optimizer_calibration(tuple_bundle):
    case = calibrated_round()
    if tuple_bundle:
        case[1]["online_calibration_boundaries"] = tuple(case[1]["online_calibration_boundaries"])
    report = validate(case)
    assert report["valid"] and report["barrier_verified"]


@pytest.mark.parametrize("mutation", ["wrong_source", "schema", "model_hash", "persistent_steps",
    "temporary_steps", "rng", "stage", "coverage", "abi", "nonfinite_cost", "worker_missing",
    "sample_changed", "duration_changed"])
def test_physical_admission_rejects_tampered_training_calibration(mutation):
    case = calibrated_round()
    _, bundle, result, _ = case
    initialization = result["cost_initialization_receipt"]
    server = initialization["server_calibration"]
    contexts = result["online_worker_contexts"]
    first = next(iter(contexts.values()))
    client = json.loads(first["online_calibration_receipt"])
    if mutation == "wrong_source":
        initialization["source"] = "current_deployment_no_update_split_calibration"
    elif mutation == "schema":
        server["schema"] = "unsupported-calibration"
    elif mutation == "model_hash":
        server["model_hash_after"] = "changed"
    elif mutation == "persistent_steps":
        server["persistent_optimizer_steps"] = 15
    elif mutation == "temporary_steps":
        server["optimizer_steps"] = 0
    elif mutation == "rng":
        client["state_and_torch_rng_preserved"] = False
    elif mutation == "stage":
        client["records"][0]["optimizer_stage"] = "suffix"
    elif mutation == "coverage":
        client["records"].pop()
    elif mutation == "abi":
        client["records"][0]["feature_abi_id"] = "wrong"
    elif mutation == "nonfinite_cost":
        server["records"][0]["local_tail_service_ms"] = float("nan")
    elif mutation == "worker_missing":
        contexts.pop(next(iter(contexts)))
    elif mutation == "sample_changed":
        initialization["client_samples"][0]["client_backward_ms"] += 1
    else:
        initialization["client_calibration_elapsed_sec"][first["logical_client_id"]] += 1
    first["online_calibration_receipt"] = json.dumps(client)
    report = validate(case)
    assert not report["valid"] and report["errors"]


@pytest.mark.parametrize("mutation", ["missing_probe", "duplicate_probe", "nonfinite_probe", "wrong_echo",
                                     "unknown_parameter_bytes", "schema", "persistent_optimizer", "missing_warmup", "unknown_batch_payload",
                                     "missing_exchange", "duplicate_exchange", "nonfinite_exchange", "wrong_reply",
                                     "missing_actual_rpc", "handler_exceeds_rpc", "wrong_exchange_count"])
def test_online_physical_validation_rejects_tampering(mutation):
    case = calibrated_round()
    result = case[2]
    initialization = result["cost_initialization_receipt"]
    samples = initialization["transport_bootstrap"]["samples"]
    if mutation == "missing_probe":
        samples.pop()
    elif mutation == "duplicate_probe":
        samples.append(samples[0])
    elif mutation == "nonfinite_probe":
        samples[0]["roundtrip_ms"] = float("nan")
    elif mutation == "wrong_echo":
        samples[0]["echo_verified"] = False
    elif mutation == "unknown_parameter_bytes":
        result["candidate_catalog"][0]["optimizer_prefix_parameter_bytes"] = None
    elif mutation == "schema":
        initialization["server_calibration"]["schema"] = "unsupported-calibration"
    elif mutation == "missing_warmup":
        initialization["transport_bootstrap"]["warmup_samples"].pop()
    elif mutation == "unknown_batch_payload":
        result["candidate_catalog"][0]["boundary_forward_bytes_by_batch_size"] = {}
    elif mutation == "missing_exchange":
        initialization["state_exchange_bootstrap"]["samples"].pop()
    elif mutation == "duplicate_exchange":
        initialization["state_exchange_bootstrap"]["samples"].append(initialization["state_exchange_bootstrap"]["samples"][0])
    elif mutation == "nonfinite_exchange":
        initialization["state_exchange_bootstrap"]["samples"][0]["roundtrip_ms"] = float("inf")
    elif mutation == "wrong_reply":
        initialization["state_exchange_bootstrap"]["samples"][0]["reply_verified"] = False
    elif mutation == "missing_actual_rpc":
        result["fit_records"][0]["metrics"].pop("coordinator_fit_rpc_ms")
    elif mutation == "handler_exceeds_rpc":
        result["fit_records"][0]["metrics"]["client_fit_handler_ms"] = 200
    elif mutation == "wrong_exchange_count":
        next(iter(result["round_diagnostics"]["1"]["learner_update_counts"].values()))["state_exchange_update_count"] = 6
    else:
        initialization["server_calibration"]["persistent_optimizer_steps"] = 1
    assert not validate(case)["valid"]
