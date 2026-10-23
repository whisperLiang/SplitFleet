"""Protocol integrity checks for the physical three-Orin experiment."""

import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.resnet50_backends.common import LOGICAL_CLIENTS, Images, learning_rate, validate_round_budget
from experiments.resnet50_backends.summarize import read_pair


def config():
    return {"seed": 233, "rounds": 50, "batch_size": 4, "train_samples": 30,
            "learning_rate": .05, "minimum_learning_rate": .0005,
            "mean": [.4, .5, .6], "std": [.2, .3, .4]}


def rows():
    return [{"client_index": i, "num_examples": 10,
             "metrics": {"num_batches": 3, "data_batch_hash": "a" * 64}} for i in range(3)]


def test_physical_budget_requires_all_clients_and_final_short_batches():
    assert validate_round_budget(rows(), config()) == {"examples": 30, "optimization_steps": 9}
    for broken in (rows()[:2], [rows()[0], rows()[0], rows()[2]]):
        with pytest.raises(RuntimeError, match="3 distinct"):
            validate_round_budget(broken, config())
    for name, value in (("num_batches", 2), ("skipped_examples", 2), ("skipped_batches", 1)):
        broken = rows()
        broken[1]["metrics"][name] = value
        with pytest.raises(RuntimeError, match="omitted, repeated or skipped"):
            validate_round_budget(broken, config())


def six_client_config():
    return {**config(), "clients": 6, "logical_clients": list(LOGICAL_CLIENTS),
            "train_samples": 45000, "batch_size": 64}


def six_client_rows():
    return [{"client_index": spec["index"], "num_examples": 7500,
             "metrics": {"num_batches": 118, "data_batch_hash": "a" * 64,
                         "logical_client_id": spec["id"], "device": spec["device"],
                         "physical_host": spec["host_id"], "machine": "aarch64"}}
            for spec in LOGICAL_CLIENTS]


def test_six_clients_require_each_physical_cpu_and_gpu_and_all_local_examples():
    assert validate_round_budget(six_client_rows(), six_client_config()) == {
        "examples": 45000, "optimization_steps": 708}
    for broken in (six_client_rows()[:-1], [six_client_rows()[0]] + six_client_rows()[:-1]):
        with pytest.raises(RuntimeError, match="6 distinct"):
            validate_round_budget(broken, six_client_config())
    for name, value in (("device", "cpu"), ("physical_host", "orin118"),
                        ("logical_client_id", "orin140_cpu"), ("machine", "x86_64")):
        broken = six_client_rows()
        broken[1]["metrics"][name] = value
        with pytest.raises(RuntimeError, match="placement"):
            validate_round_budget(broken, six_client_config())
    broken = six_client_rows()
    broken[1]["metrics"]["num_batches"] = 117
    with pytest.raises(RuntimeError, match="omitted"):
        validate_round_budget(broken, six_client_config())


def test_launch_hides_gpu_from_cpu_clients_and_preserves_unique_receipt_paths(tmp_path):
    from experiments.resnet50_backends.campaign import command
    import shlex
    manifest = {"workspace": str(tmp_path), "server_devices": {"jax": "cpu"}}
    clients = [{**spec, "root": "/isolated/run", "ssh": "nvidia@192.168.66.140"}
               for spec in LOGICAL_CLIENTS[:2]]
    commands = [shlex.split(command(tmp_path, manifest, "jax", "client", host=client,
                output=f"/isolated/run/{client['id']}.json", address="127.0.0.1:12345")[-1])
                for client in clients]
    assert "CUDA_VISIBLE_DEVICES=-1" in commands[0] and "JAX_PLATFORMS=cpu" in commands[0]
    assert "CUDA_VISIBLE_DEVICES=0" in commands[1] and "JAX_PLATFORMS=cuda,cpu" in commands[1]
    assert "JAX_DISABLE_JIT=1" in commands[1] and "JAX_DISABLE_JIT=1" not in commands[0]
    assert commands[0][commands[0].index("--admission") + 1] != commands[1][commands[1].index("--admission") + 1]
    assert commands[0][commands[0].index("--index") + 1] == "0"
    assert commands[1][commands[1].index("--index") + 1] == "1"


def test_figure3_server_uses_a6000_gpu_for_every_backend(tmp_path):
    from experiments.resnet50_backends.campaign import SERVER_DEVICES, command

    manifest = {"workspace": str(tmp_path), "server_devices": dict(SERVER_DEVICES),
                "server": {"host": "fy205"}}
    for backend in ("tensorflow", "jax", "paddle"):
        args = command(tmp_path, manifest, backend, "server", output=tmp_path / "server.json",
                       address="127.0.0.1:12345")
        assert args[args.index("--device") + 1] == "cuda:0"
        assert "CUDA_VISIBLE_DEVICES=0" in args
        assert "SPLITFLEET_CLIENT_ID=server" in args
        assert "SPLITFLEET_PHYSICAL_HOST=fy205" in args
        assert "JAX_PLATFORMS=cpu" not in args
        assert "JAX_DISABLE_JIT=1" not in args


def test_measured_placement_reaches_each_cpu_gpu_and_server_process(tmp_path):
    from experiments.resnet50_backends.campaign import command
    import shlex

    manifest = {"workspace": str(tmp_path), "server_devices": {"paddle": "cuda:0"},
                "placement_policy": "greedy"}
    for spec in LOGICAL_CLIENTS:
        host = {**spec, "root": "/isolated/run", "ssh": "nvidia@192.168.66.140"}
        args = shlex.split(command(tmp_path, manifest, "paddle", "client", host=host,
            output="/isolated/run/client.json", address="127.0.0.1:12345")[-1])
        assert args[args.index("--placement") + 1] == "greedy"
    args = command(tmp_path, manifest, "paddle", "server", output=tmp_path / "server.json")
    assert args[args.index("--placement") + 1] == "greedy"
    manifest.pop("placement_policy")
    args = command(tmp_path, manifest, "paddle", "server", output=tmp_path / "server.json")
    assert args[args.index("--placement") + 1] == "fixed_stage2"


def selected_admission_receipt(boundary, size):
    return {"status": "passed", "batch_size": size,
            "cuts": [{"boundary": boundary, "status": "passed",
                "checks": [{"passed": True}] * 644, "gradient_tensor_count": 161,
                "state_tensor_count": 320, "bn_counters": {str(i): 1 for i in range(53)}}]}


def test_selected_cut_evidence_requires_both_batch_shapes_and_exact_bn_counts(tmp_path):
    from experiments.resnet50_backends.summarize import validate_selected_admission

    protocol = {**six_client_config(), "batch_size": 16}
    boundary, identity = "after:native_pool", "orin140_cpu"
    digest = hashlib.sha256(boundary.encode()).hexdigest()[:16]
    paths = {}
    for size in (16, 12):
        paths[size] = tmp_path / f"{identity}_{digest}_b{size}.json"
        paths[size].write_text(json.dumps(selected_admission_receipt(boundary, size)))
    validate_selected_admission(tmp_path, identity, boundary, protocol)
    valid = json.loads(paths[12].read_text())
    for broken in (dict(valid, batch_size=16), dict(valid, status="running")):
        paths[12].write_text(json.dumps(broken))
        with pytest.raises(ValueError, match="B=16 and B=12"):
            validate_selected_admission(tmp_path, identity, boundary, protocol)
    valid["cuts"][0]["bn_counters"]["0"] = 2
    paths[12].write_text(json.dumps(valid))
    with pytest.raises(ValueError, match="B=16 and B=12"):
        validate_selected_admission(tmp_path, identity, boundary, protocol)


@pytest.mark.parametrize("samples,batch_size", [(60, 16), (60, 8), (60, 12),
                                              (10, 16), (60, 64), (60, 59)])
def test_selected_cut_admission_covers_every_actual_epoch_shape(tmp_path, monkeypatch, samples, batch_size):
    from experiments.resnet50_backends import worker
    from experiments.resnet50_backends.summarize import validate_selected_admission

    protocol = {**six_client_config(), "train_samples": samples * 6, "batch_size": batch_size}
    source = tmp_path / "client_0.npz"
    np.savez(source, ids=np.arange(samples), images=np.zeros((samples, 32, 32, 3), np.uint8),
             labels=np.arange(samples) % 10)
    actual_sizes = list(dict.fromkeys(len(ids) for ids, _, _ in Images(source, protocol).batches(
        client=0, round_id=1)))
    boundary, calls = "after:native_pool", []

    def admit(args):
        calls.append(args.admission_batch_size)
        args.output.write_text(json.dumps(selected_admission_receipt(boundary, args.admission_batch_size)))

    monkeypatch.setattr(worker, "admit", admit)
    args = SimpleNamespace(output=tmp_path / "server.json", identity="")
    worker.admit_selected_boundaries(args, [boundary, boundary], protocol)
    assert calls == actual_sizes
    directory = tmp_path / "selected_admission"
    validate_selected_admission(directory, "server", boundary, protocol)
    calls.clear()
    worker.admit_selected_boundaries(args, [boundary], protocol)
    assert not calls
    digest = hashlib.sha256(boundary.encode()).hexdigest()[:16]
    cached = directory / f"server_{digest}_b{actual_sizes[0]}.json"
    cached.write_text(json.dumps(selected_admission_receipt(boundary, actual_sizes[0] + 1)))
    with pytest.raises(RuntimeError, match="retain its receipt"):
        worker.admit_selected_boundaries(args, [boundary], protocol)


@pytest.mark.parametrize("role,samples", [("server", 60), ("client", 60), ("server", 10)])
def test_calibration_uses_a_real_training_batch_with_a_tiny_validation_set(tmp_path, role, samples):
    from experiments.resnet50_backends.common import file_hash
    from experiments.resnet50_backends.worker import calibration_batch

    protocol = {**six_client_config(), "train_samples": samples * 6, "batch_size": 16}
    index = 1 if role == "client" else 0
    source = tmp_path / f"client_{index}.npz"
    np.savez(source, ids=np.arange(samples), images=np.zeros((samples, 32, 32, 3), np.uint8),
             labels=np.arange(samples) % 10)
    np.savez(tmp_path / "validation.npz", ids=np.arange(10),
             images=np.full((10, 32, 32, 3), 255, np.uint8), labels=np.arange(10))
    protocol["files"] = {source.name: file_hash(source)}
    args = SimpleNamespace(command=role, index=index, bundle=tmp_path, backend="pytorch", device="cpu")
    batch = calibration_batch(args, object(), protocol)
    ids, images, labels = next(Images(source, protocol).batches(client=index, round_id=1))
    assert batch.num_examples == len(ids) == min(samples, 16)
    np.testing.assert_array_equal(batch.inputs.args[0].numpy(), images)
    np.testing.assert_array_equal(batch.targets.numpy(), labels)


def test_admission_cli_accepts_protocol_specific_batch_sizes(tmp_path, monkeypatch):
    from experiments.resnet50_backends import worker

    calls = []
    monkeypatch.setattr(worker, "admit", lambda args: calls.append(args.admission_batch_size))
    monkeypatch.setattr("sys.argv", ["worker", "admit", "--backend", "tensorflow", "--device", "cpu",
        "--bundle", str(tmp_path), "--output", str(tmp_path / "admit.json"), "--admission-batch-size", "8"])
    worker.main()
    assert calls == [8]


def test_executed_placement_must_match_every_independent_client_decision(tmp_path, monkeypatch):
    from experiments.resnet50_backends import summarize

    decisions = [{"logical_client_id": spec["id"], "boundary": "after:pool", "algorithm": "Greedy",
                  "candidate_count": 1411} for spec in LOGICAL_CLIENTS]
    (tmp_path / "placement_round_001.json").write_text(json.dumps({"decisions": decisions}))
    clients = [{"client_index": spec["index"], "metrics": {
        "boundary": "after:pool", "selected_boundary": "after:pool", "placement_policy": "greedy"}}
        for spec in LOGICAL_CLIENTS]
    checked = []
    monkeypatch.setattr(summarize, "validate_selected_admission", lambda *args: checked.append(args))
    summarize.validate_placement_round(tmp_path, {"round": 1, "clients": clients}, six_client_config())
    assert len(checked) == 12
    clients[0]["metrics"]["selected_boundary"] = "after:fixed_stage2"
    with pytest.raises(ValueError, match="independent measured Greedy"):
        summarize.validate_placement_round(tmp_path, {"round": 1, "clients": clients}, six_client_config())


def test_batches_reproduce_augmentations_without_global_rng_or_dropped_examples(tmp_path):
    raw = np.random.default_rng(12).integers(0, 256, (10, 32, 32, 3), dtype=np.uint8)
    path = tmp_path / "data.npz"
    np.savez(path, images=raw, labels=np.arange(10, dtype=np.int64), ids=np.arange(100, 110, dtype=np.int64))
    data = Images(path, config())

    def collect(round_id):
        result, batches = hashlib.sha256(), list(data.batches(client=1, round_id=round_id))
        for ids, images, labels in batches:
            result.update(ids.tobytes() + images.tobytes() + labels.tobytes())
        assert sorted(np.concatenate([ids for ids, _, _ in batches])) == list(range(100, 110))
        assert [len(ids) for ids, _, _ in batches] == [4, 4, 2]
        return result.hexdigest()

    first = collect(1)
    np.random.seed(777)
    np.random.random(100)
    assert collect(1) == first
    assert collect(2) != first


def test_cosine_schedule_has_fixed_endpoints_and_rejects_extra_rounds():
    assert learning_rate(1, config()) == .05
    assert learning_rate(50, config()) == .0005
    values = [learning_rate(i, config()) for i in range(1, 51)]
    assert all(a > b for a, b in zip(values, values[1:]))
    for index in (0, 51):
        with pytest.raises(ValueError, match="outside"):
            learning_rate(index, config())


def test_partial_or_failed_admission_cannot_be_presented_as_convergence(tmp_path):
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle/protocol.json").write_text(json.dumps(config()))
    path = tmp_path / "admission/jax"
    path.mkdir(parents=True)
    (path / "server.json").write_text(json.dumps({"status": "failed", "cuts": [{}, {}, {}]}))
    with pytest.raises(ValueError, match="must pass"):
        read_pair(tmp_path, "jax")


def test_progress_detects_host_reboot_instead_of_reporting_stale_running(tmp_path):
    from experiments.resnet50_backends.status import status
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle/protocol.json").write_text(json.dumps({"formal": True, "rounds": 50}))
    (tmp_path / "status.json").write_text(json.dumps({"status": "running", "boot_id": "previous-boot",
                                                     "backends": {}, "pid": 123}))
    assert status(tmp_path)["status"] == "interrupted_host_reboot"


@pytest.mark.parametrize("root_flag", ["--root", "--root="])
def test_campaign_process_identity_handles_relative_roots_and_process_exit(tmp_path, root_flag):
    from experiments.resnet50_backends.status import campaign_process_alive

    root = tmp_path / "campaign"
    root.mkdir()
    arguments = [root_flag, root.name] if root_flag == "--root" else [root_flag + root.name]
    process = subprocess.Popen([sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)",
        "experiments.resnet50_backends.campaign", "execute", *arguments], cwd=tmp_path,
        stdout=subprocess.PIPE, text=True)
    state = {"pid": process.pid}
    try:
        assert process.stdout.readline().strip() == "ready"
        assert campaign_process_alive(root, state)
        assert not campaign_process_alive(tmp_path / "another-campaign", state)
    finally:
        process.terminate()
        process.wait(timeout=5)
        process.stdout.close()
    assert not campaign_process_alive(root, state)


@pytest.mark.parametrize("previous_status", ["running", "waiting_for_preflight", "waiting_for_campaign"])
@pytest.mark.parametrize("reused_pid", [False, True])
def test_progress_detects_dead_or_reused_coordinator_pid(tmp_path, previous_status, reused_pid):
    from experiments.resnet50_backends.campaign import boot_id
    from experiments.resnet50_backends.status import status

    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle/protocol.json").write_text(json.dumps({"formal": True, "rounds": 50}))
    (tmp_path / "status.json").write_text(json.dumps({"status": previous_status,
        "boot_id": boot_id(), "backends": {}, "pid": os.getpid() if reused_pid else 2147483647}))
    assert status(tmp_path)["status"] == "interrupted_process_exit"


@pytest.mark.parametrize("wait_name", ["wait_for_preflight", "wait_for_campaign"])
@pytest.mark.parametrize("previous_status", ["running", "waiting_for_preflight", "waiting_for_campaign"])
@pytest.mark.parametrize("reused_pid", [False, True])
def test_queued_campaign_stops_when_prerequisite_process_disappears(
        tmp_path, wait_name, previous_status, reused_pid):
    from experiments.resnet50_backends import campaign

    root, previous = tmp_path / "queued", tmp_path / "previous"
    root.mkdir()
    previous.mkdir()
    (previous / "status.json").write_text(json.dumps({"status": previous_status,
        "boot_id": campaign.boot_id(), "pid": os.getpid() if reused_pid else 2147483647}))
    with pytest.raises(RuntimeError, match="process exited or no longer owns"):
        getattr(campaign, wait_name)(root, previous)
    state = json.loads((root / "status.json").read_text())
    assert state["status"] == "not_started" and "process exited" in state["error"]
    assert not (root / "preflight_release.json").exists()
    assert not (root / "device_queue_release.json").exists()
    assert not (root / "execution_attempt.json").exists()


@pytest.mark.parametrize("wait_name", ["wait_for_preflight", "wait_for_campaign"])
def test_prerequisite_completion_during_process_check_is_not_reported_as_failure(tmp_path, monkeypatch, wait_name):
    from experiments.resnet50_backends import campaign

    root, previous = tmp_path / "queued", tmp_path / "previous"
    root.mkdir()
    previous.mkdir()
    state = {"status": "running", "boot_id": campaign.boot_id(), "pid": 2147483647}
    campaign.save_json(previous / "status.json", state)

    def finish(root, state):
        campaign.save_json(root / "status.json", {**state, "status": "completed"})
        return False

    monkeypatch.setattr(campaign, "campaign_process_alive", finish)
    getattr(campaign, wait_name)(root, previous)
    release = "preflight_release.json" if wait_name == "wait_for_preflight" else "device_queue_release.json"
    assert (root / release).exists()
    assert not (root / "execution_attempt.json").exists()


@pytest.mark.parametrize("previous_status", ["completed", "failed", "stopped_by_user"])
def test_device_queue_waits_for_completion_without_treating_failure_as_admission(tmp_path, previous_status):
    from experiments.resnet50_backends.campaign import boot_id, wait_for_campaign

    root, previous = tmp_path / "queued", tmp_path / "previous"
    root.mkdir()
    previous.mkdir()
    (previous / "status.json").write_text(json.dumps({
        "status": previous_status, "pid": -1, "boot_id": boot_id()}))
    if previous_status == "stopped_by_user":
        with pytest.raises(RuntimeError, match="stopped"):
            wait_for_campaign(root, previous)
        assert json.loads((root / "status.json").read_text())["status"] == "not_started"
    else:
        wait_for_campaign(root, previous)
        release = json.loads((root / "device_queue_release.json").read_text())
        assert release["previous_status"] == previous_status
        assert "not numerical admission" in release["scope"]
        assert not (root / "preflight_release.json").exists()


@pytest.mark.parametrize("initial_status", [None, "waiting_for_campaign", "running"])
def test_preflight_wait_handles_startup_and_a_queued_device_cohort(tmp_path, monkeypatch, initial_status):
    from experiments.resnet50_backends import campaign
    monkeypatch.setattr(campaign, "campaign_process_alive", lambda *_: True)

    root, previous = tmp_path / "formal", tmp_path / "smoke"
    root.mkdir()
    previous.mkdir()
    status = previous / "status.json"
    if initial_status is not None:
        status.write_text(json.dumps({"status": initial_status, "boot_id": campaign.boot_id()}))

    def finish(_):
        status.write_text(json.dumps({"status": "completed", "boot_id": campaign.boot_id()}))

    monkeypatch.setattr(campaign.time, "sleep", finish)
    campaign.wait_for_preflight(root, previous)
    assert json.loads((root / "preflight_release.json").read_text())["status"] == "passed"
    assert not (root / "execution_attempt.json").exists()


@pytest.mark.parametrize("initial_status", [None, "waiting_for_campaign", "waiting_for_preflight"])
def test_device_queue_waits_for_predecessor_status_creation(tmp_path, monkeypatch, initial_status):
    from experiments.resnet50_backends import campaign
    monkeypatch.setattr(campaign, "campaign_process_alive",
                        lambda _, state: state["status"] not in {"completed", "failed"})

    root, previous = tmp_path / "next", tmp_path / "previous"
    root.mkdir()
    previous.mkdir()
    calls = []
    if initial_status is not None:
        campaign.save_json(previous / "status.json", {
            "status": initial_status, "boot_id": campaign.boot_id(), "pid": -1})

    def publish(_):
        calls.append(True)
        campaign.save_json(previous / "status.json", {
            "status": "completed", "boot_id": campaign.boot_id(), "pid": -1})

    monkeypatch.setattr(campaign.time, "sleep", publish)
    campaign.wait_for_campaign(root, previous)
    assert calls == [True]
    assert json.loads((root / "device_queue_release.json").read_text())["previous_status"] == "completed"
    assert not (root / "execution_attempt.json").exists()


def write_completed_methods(tmp_path, six_clients, methods=("fedavg", "splitfleet"), *, round_count=2, formal=False):
    settings = {**(six_client_config() if six_clients else config()), "rounds": round_count,
                "formal": formal, "convergence": {"target_val_top1": .7, "sustained_rounds": 5}}
    identities = [spec["id"] for spec in LOGICAL_CLIENTS] if six_clients else ["orin140", "orin118", "orin238"]
    client_rows = six_client_rows if six_clients else rows
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle/protocol.json").write_text(json.dumps(settings))
    admission = tmp_path / "admission/tensorflow"
    admission.mkdir(parents=True)
    for identity in ("server", *identities):
        (admission / (identity + ".json")).write_text(json.dumps({"status": "passed", "cuts": [
            {"stage": stage, "status": "passed", "checks": [{"passed": True}]} for stage in (1, 2, 3)]}))
    for method in methods:
        directory = tmp_path / "runs/tensorflow" / method
        directory.mkdir(parents=True)
        (directory / "execution.json").write_text(json.dumps({"status": "completed"}))
        observations = [{"round": index, "clients": client_rows(), "learning_rate": .01,
                         "val_top1": .4, "val_ce": 1.5, "train_online_ce": 1.6, "elapsed_wall_sec": index * 10.,
                         "suffixes": [{} for _ in identities]} for index in range(1, round_count + 1)]
        (directory / "server.json").write_text(json.dumps({"status": "completed", "protocol": settings,
            "rounds": ([{"round": 0}] if method == "fedavg" else []) + observations,
            "duration_sec": round_count * 10. + 1., "test": {"final": {"top1": .4}, "best_validation": {"top1": .4}}}))
        for index, identity in enumerate(identities):
            environment = {"machine": "aarch64"}
            if six_clients:
                spec = LOGICAL_CLIENTS[index]
                environment.update(logical_client_id=identity, device=spec["device"], physical_host=spec["host_id"])
            (directory / (identity + ".json")).write_text(json.dumps({"status": "completed",
                "environment": environment, "rounds": [
                    {"round": round_id, "num_examples": 7500 if six_clients else 10,
                     "metrics": {"data_batch_hash": "a" * 64}} for round_id in range(1, round_count + 1)]}))
    return identities


@pytest.mark.parametrize("six_clients", [False, True])
def test_pairing_handles_native_round_zero_and_requires_matching_client_hashes(tmp_path, six_clients):
    identities = write_completed_methods(tmp_path, six_clients)
    assert set(read_pair(tmp_path, "tensorflow")) == {"fedavg", "splitfleet"}
    path = tmp_path / f"runs/tensorflow/splitfleet/{identities[1]}.json"
    client = json.loads(path.read_text())
    client["rounds"][0]["metrics"]["data_batch_hash"] = "b" * 64
    path.write_text(json.dumps(client))
    with pytest.raises(ValueError, match="disagree"):
        read_pair(tmp_path, "tensorflow")
    if six_clients:
        client["rounds"][0]["metrics"]["data_batch_hash"] = "a" * 64
        client["environment"]["device"] = "cpu"
        path.write_text(json.dumps(client))
        with pytest.raises(ValueError, match="placement"):
            read_pair(tmp_path, "tensorflow")


@pytest.mark.parametrize("queued_backends", [[], ["jax"]])
def test_splitfleet_summary_ignores_stopped_fedavg_and_never_claims_pairing(tmp_path, queued_backends):
    from experiments.resnet50_backends.summarize import summarize

    write_completed_methods(tmp_path, True, methods=("splitfleet",))
    (tmp_path / "campaign.json").write_text(json.dumps({
        "method_order": ["splitfleet"], "source_identity": "unit-test-source"}))
    (tmp_path / "execution_attempt.json").write_text(json.dumps({"backends": ["tensorflow", *queued_backends]}))
    (tmp_path / "status.json").write_text(json.dumps({"backends": {"tensorflow": {"phase": "completed"}}}))
    stopped = tmp_path / "runs/tensorflow/fedavg"
    stopped.mkdir()
    (stopped / "execution.json").write_text(json.dumps({"status": "stopped_by_user"}))
    (stopped / "server.json").write_text(json.dumps({"status": "stopped_by_user"}))
    result = summarize(tmp_path, ["tensorflow"])["tensorflow"]
    assert result["complete_requested_training"] and not result["complete_paired_training"]
    assert set(result["methods"]) == {"splitfleet"} and "final_test_top1_difference" not in result
    output = json.loads((tmp_path / "summary/summary.json").read_text())
    assert output["requested_methods"] == ["splitfleet"]
    assert output["all_requested_backends_completed"] == (not queued_backends)
    assert output["excluded_backends"] == {name: {"phase": "not_started"} for name in queued_backends}
    assert "fedavg" not in (tmp_path / "summary/curves.csv").read_text()
    with pytest.raises(ValueError, match="Incomplete"):
        read_pair(tmp_path, "tensorflow")


@pytest.mark.parametrize("formal,round_count", [(False, 10), (False, 50), (True, 50)])
def test_convergence_assessment_requires_a_formal_budget(tmp_path, formal, round_count):
    from experiments.resnet50_backends.summarize import summarize

    write_completed_methods(tmp_path, True, methods=("splitfleet",), round_count=round_count, formal=formal)
    (tmp_path / "campaign.json").write_text(json.dumps({
        "method_order": ["splitfleet"], "source_identity": "unit-test-source"}))
    (tmp_path / "execution_attempt.json").write_text(json.dumps({"backends": ["tensorflow"]}))
    (tmp_path / "status.json").write_text(json.dumps({"backends": {"tensorflow": {"phase": "completed"}}}))
    path = tmp_path / "runs/tensorflow/splitfleet/server.json"
    run = json.loads(path.read_text())
    for row in run["rounds"]:
        row.update(val_top1=.8, val_ce=2. - .01 * row["round"])
    path.write_text(json.dumps(run))

    result = summarize(tmp_path, ["tensorflow"])
    convergence = result["tensorflow"]["methods"]["splitfleet"]["convergence"]
    assert convergence["converged"] is (True if formal else None)
    saved = json.loads((tmp_path / "summary/summary.json").read_text())
    assert saved["formal"] is formal
    assert saved["backends"]["tensorflow"]["methods"]["splitfleet"]["convergence"] == convergence
    assert ("SMOKE CHECK; convergence not assessed" in (tmp_path / "summary/convergence.svg").read_text()) is (not formal)


@pytest.mark.parametrize("retention", ["all", "latest_and_best"])
@pytest.mark.parametrize("failure", ["metrics", "pointer", None])
def test_checkpoint_publication_preserves_a_complete_previous_round_on_failure(tmp_path, monkeypatch, retention, failure):
    from experiments.resnet50_backends import worker

    output = tmp_path / "server.json"
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    history = []
    for round_id in (1, 2):
        weights = np.asarray([round_id], dtype=np.float32)
        np.savez(checkpoint / f"round_{round_id:03d}.npz", weights=weights)
        history.append({"round": round_id, "checkpoint_state_hash": hashlib.sha256(weights.tobytes()).hexdigest()})
        record = {"rounds": list(history), "protocol": {"checkpoint_retention": retention}}
        if round_id == 1:
            worker._publish_server_round(output, record)

    original_save = worker.save_json

    def fail_publication(path, value):
        if (failure == "metrics" and path == output
                or failure == "pointer" and path.name == "latest_checkpoint.json"):
            raise OSError("injected persistence failure")
        original_save(path, value)

    monkeypatch.setattr(worker, "save_json", fail_publication)
    if failure:
        with pytest.raises(OSError, match="injected persistence failure"):
            worker._publish_server_round(output, record)
    else:
        worker._publish_server_round(output, record)

    pointer = json.loads((tmp_path / "latest_checkpoint.json").read_text())
    metrics = json.loads(output.read_text())
    committed = next(row for row in metrics["rounds"] if row["round"] == pointer["round"])
    with np.load(tmp_path / pointer["path"]) as saved:
        assert hashlib.sha256(saved["weights"].tobytes()).hexdigest() == pointer["state_hash"] == committed["checkpoint_state_hash"]
    assert pointer["round"] == (1 if failure else 2)
    assert len(metrics["rounds"]) == (1 if failure == "metrics" else 2)
    assert (checkpoint / "round_001.npz").exists() is bool(failure or retention == "all")


def test_round_zero_publishes_metrics_without_a_checkpoint_pointer(tmp_path):
    from experiments.resnet50_backends.worker import _publish_server_round

    output = tmp_path / "server.json"
    record = {"rounds": [{"round": 0}], "protocol": {"checkpoint_retention": "latest_and_best"}}
    _publish_server_round(output, record)
    assert json.loads(output.read_text()) == record
    assert not (tmp_path / "latest_checkpoint.json").exists()
