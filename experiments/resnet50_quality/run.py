"""Run complete paired native ResNet50 training with one explicit configuration."""

from __future__ import annotations

import argparse
import codecs
import csv
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from splitfleet.virtual.config import ClientPopulation, Profile, Resources, Transport, VirtualConfig

from .model_contract import NAMES
from .native import BACKENDS
from .protocol import SEEDS, hash_file, validate_main_protocol, validate_worker_protocol

RUN_LIMITS = {"rpc_timeout_sec": 3600, "task_timeout_sec": 86_400,
              "round_timeout_sec": None, "run_timeout_sec": 14 * 86_400}


def save_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)



def make_config(*, method, backend="pytorch", seed, protocol, data_root, hardware_profile):
    if backend not in BACKENDS:
        raise ValueError(f"Unknown native backend {backend!r}")
    if method not in {"fedavg", "splitfleet"}:
        raise ValueError("The method must be fedavg or splitfleet")
    frozen = json.loads((Path(protocol) / "protocol.json").read_text())
    validate_worker_protocol(frozen)
    rounds, _ = validate_main_protocol(frozen)
    if seed not in SEEDS:
        raise ValueError("Seed differs from the frozen protocol")
    from .hardware import apply_hardware_profile
    hardware = json.loads(Path(hardware_profile).read_text())
    clients = BACKENDS[backend]
    workers = hardware["worker_counts"][backend]
    if type(workers) is not int or not 3 <= workers <= clients:
        raise ValueError("Declare a fixed worker count between three and the client population")
    policy = "one_per_client" if workers == clients else "pooled"
    options = {
        "backend": backend, "method": method, "stage": "main",
        "protocol": str(Path(protocol).resolve()), "data_root": str(Path(data_root).resolve()),
        "worker_count_policy": policy, "worker_count": workers, "worker_target_count": clients,
        "isolate_unused_frameworks": True, "run_limits": dict(RUN_LIMITS),
        "round_timeout_sec": None,
    }
    config = VirtualConfig(
        rounds=rounds, seed=seed, batch_size=16, local_epochs=1, learning_rate=.05,
        server_device="cuda:1", placement="cosplit_ucb",
        virtual_clients=ClientPopulation(total=clients, training_timeout_sec=RUN_LIMITS["task_timeout_sec"]),
        resources=Resources(client_cpus_per_worker=1, max_concurrent_clients=workers,
            worker_count_policy=policy, max_suffix_replicas=clients, per_client_suffix_models=True,
            max_resident_suffix_replicas=workers, server_concurrency=workers,
            client_gpus_per_worker=1 / workers, client_gpu_pool_size=1, object_store_memory_mb=512,
            client_runtime_slots=2 if policy == "pooled" else 1),
        profiles={"gpu": Profile(device="cuda:0", torch_threads=1, gpu_fraction=1 / workers)},
        transport=Transport(rpc_timeout_sec=RUN_LIMITS["rpc_timeout_sec"]), workload=options,
    )
    return apply_hardware_profile(config, hardware_profile)


def validate_quality_budget(output, config, spec):
    output = Path(output)
    rows = [json.loads(line) for line in (output / "client_metrics.jsonl").read_text().splitlines()]
    successes = [row for row in rows if row["status"] == "success"]
    expected_tasks = config.rounds * config.virtual_clients.total
    expected_examples = config.rounds * sum(map(len, spec.frozen_partitions.values()))
    expected_steps = config.rounds * sum((len(ids) + 15) // 16 for ids in spec.frozen_partitions.values())
    if len(successes) != expected_tasks or sum(row["num_examples"] for row in successes) != expected_examples:
        raise RuntimeError("Quality client/sample budget is incomplete")
    if sum(row["metrics"]["num_batches"] for row in successes) != expected_steps:
        raise RuntimeError("Quality SGD step budget differs from the shared batch manifest")
    by_identity = {(row["partition_id"], row["round_id"]): row for row in successes}
    if len(by_identity) != expected_tasks:
        raise RuntimeError("Quality logical client/round identity is duplicated")
    workers = {(row["metrics"]["worker_host"], row["metrics"]["worker_pid"]) for row in successes}
    expected_workers = config.resources.max_concurrent_clients
    if len(workers) != expected_workers:
        raise RuntimeError(f"Quality protocol requires {expected_workers} workers, observed {len(workers)}")
    for row in successes:
        pid, round_id, metrics = row["partition_id"], row["round_id"], row["metrics"]
        batches = (len(spec.frozen_partitions[str(pid)]) + 15) // 16
        if (row["num_examples"] != len(spec.frozen_partitions[str(pid)])
                or metrics["completed_batches"] != round_id * batches
                or len(metrics["data_batch_hash"]) != 64):
            raise RuntimeError("Worker reuse changed logical data identity or progress")
        if config.workload.get("hardware_profile"):
            profile = config.profile(pid)
            if (metrics.get("device") != profile.device or not metrics.get("worker_model_device_verified")
                    or metrics.get("worker_cuda_visible_devices") != "0"):
                raise RuntimeError("Heterogeneous worker failed actual model-device or client-only GPU isolation")
            if profile.affinity is not None and json.loads(metrics["worker_affinity"]) != profile.affinity:
                raise RuntimeError("Heterogeneous worker did not apply the frozen CPU affinity")
    if config.workload["method"] == "splitfleet":
        resources = json.loads((output / "resource_metrics.json").read_text())
        if (resources["peak_suffix_replicas"] > config.resources.suffix_capacity
                or resources["peak_server_executions"] > config.resources.server_concurrency):
            raise RuntimeError("Quality suffix capacity/execution limit was violated")
        if (resources.get("observed_suffix_model_count") != config.virtual_clients.total
                or resources.get("per_client_suffix_models") is not True
                or resources.get("suffix_replicas_reused") != 0):
            raise RuntimeError("Quality execution did not retain one distinct suffix model per logical client")
        for round_id in range(1, config.rounds + 1):
            identities = [row["metrics"].get("suffix_model_instance_id")
                          for row in successes if row["round_id"] == round_id]
            if len(set(identities)) != config.virtual_clients.total or None in identities:
                raise RuntimeError("A suffix model was shared across logical clients")
        if any(row["metrics"]["prefix_changed_tensors"] <= 0 for row in successes):
            raise RuntimeError("A quality prefix failed to update trainable model state")
    report = {"status": "passed", "examples_seen": expected_examples, "optimization_steps": expected_steps,
              "training_tasks": expected_tasks, "workers": [list(v) for v in sorted(workers)],
              "worker_count": len(workers), "worker_count_policy": config.resources.worker_count_policy,
              "requested_worker_count": config.workload.get("worker_target_count", config.virtual_clients.total),
              "all_clients_completed": True, "data_source": "real CIFAR-10"}
    save_json(output / "quality_validation.json", report)
    return report, successes



def round_csv(output, successes):
    output = Path(output)
    with (output / "round_metrics.csv").open(newline="") as stream:
        clocks = {int(row["round"]): float(row["elapsed_wall_sec"]) for row in csv.DictReader(stream)}
    evaluations = {row["round_id"]: row["metrics"] for row in json.loads(
        (output / "model_metrics.json").read_text())["evaluations"]}
    cumulative, rows = 0, []
    from .protocol import learning_rate
    for round_id in sorted({row["round_id"] for row in successes}):
        clients = [row for row in successes if row["round_id"] == round_id]
        examples = sum(row["num_examples"] for row in clients)
        cumulative += examples
        evaluation = evaluations.get(round_id, {})
        rows.append({"round": round_id,
                     "train_online_ce": sum(r["num_examples"] * r["metrics"]["loss"] for r in clients) / examples,
                     "val_ce": evaluation.get("loss"), "val_top1": evaluation.get("top1"),
                     "examples_seen": cumulative, "successful_clients": len(clients),
                     "optimization_steps": sum(r["metrics"]["num_batches"] for r in clients),
                     "elapsed_wall_sec": clocks[round_id],
                     "learning_rate": float(learning_rate(round_id))})
    with (output / "round_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def execute_frozen(output):
    output = Path(output)
    values = json.loads((output / "config.json").read_text())
    values.pop("config_sha256")
    config = VirtualConfig.from_dict(values)
    if config.workload.get('hardware_profile'):
        from .hardware import apply_execution_affinity
        apply_execution_affinity(config.workload['hardware_profile']['path'])
    from experiments.common.evidence import finish_run
    from splitfleet.virtual.backend_ops import FrameworkOps
    from splitfleet.virtual.fedavg import run_fedavg
    from splitfleet.virtual.runtime import run_virtual

    from .native import build_model, export_state
    from .workloads import QualityEvaluation, quality_spec

    # Evidence claims and validates the frozen run once inside the selected
    # runtime.  The wrapper reads configuration without consuming that claim.
    try:
        spec = quality_spec(config)
        backend = config.workload["backend"]
        ops = FrameworkOps.for_spec(spec)
        runtime = run_fedavg if config.workload["method"] == "fedavg" else run_virtual
        started = time.perf_counter()
        spec.round_callback.start(started)
        result = runtime(config, spec, output=output)
        result["wall_sec"] = time.perf_counter() - started
        from .efficiency import TRAINING_WALL_SCOPE
        result["wall_scope"] = TRAINING_WALL_SCOPE
        validation, successes = validate_quality_budget(output, config, spec)
        round_csv(output, successes)
        model = ops.build_model(spec.model_factory, config.server_device)
        ops.load_checkpoint(model, output / "final_model")
        np.savez(output / "canonical_final_model.npz", **export_state(backend, model))
        evaluate = QualityEvaluation(config.workload["data_root"], test=True, backend=backend)
        final_test = evaluate(model, config.server_device)
        with np.load(output / "best_model.npz") as best:
            model = build_model(backend, {name: best[name].copy() for name in NAMES}, config.server_device)
        best_test = evaluate(model, config.server_device)
        save_json(output / "test_metrics.json", {"final": final_test, "best_validation": best_test})
        save_json(output / "run_manifest.json", {
            "backend": backend, "method": config.workload["method"], "stage": config.workload["stage"],
            "clients": config.virtual_clients.total, "seed": config.seed, "status": "passed",
            "rounds": config.rounds,
            "protocol_sha256": hash_file(Path(config.workload["protocol"]) / "protocol.json"),
            "config_sha256": hash_file(output / "config.json"), "initial_hash": spec.data_manifest["initial_hash"],
            "initial_hash_layout": "canonical_resnet50_oihw",
            "precision": "FP32; AMP/TF32 disabled", "device": config.server_device,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "canonical_initial_hash": spec.data_manifest["canonical_initial_hash"],
            "canonical_initial_hash_layout": "canonical_resnet50_oihw",
            "native_initial_hash": result.get("initial_model_hash") or json.loads(
                (output / "model_metrics.json").read_text())["initial_hash"],
            "native_initial_hash_layout": "native_adapter_schema",
            "model_hash_layout": "native_adapter_schema",
            "native_environment": json.loads((output / "native_backend.json").read_text()),
            "aggregation": {"floating": "sample_weighted_mean", "integer": "maximum"},
            "source_identity": json.loads((output / "runtime_manifest.json").read_text())["source_identity"],
            "hardware_profile": config.workload.get("hardware_profile"),
        })
        finish_run(output, {**result, **validation}, validation=validation)
        return 0
    except BaseException as error:  # noqa: BLE001 - persist native child failure
        traceback.print_exc()
        failed = {"status": "failed", "reason": repr(error), "traceback": traceback.format_exc()}
        save_json(output / "run_manifest.json", failed)
        finish_run(output, failed, validation=failed)
        return 1



def frozen_child(config, output, *, prepared=False):
    if config.workload.get('hardware_profile'):
        from .hardware import apply_execution_affinity
        apply_execution_affinity(config.workload['hardware_profile']['path'])
    from experiments.common.evidence import begin_run
    output = Path(output).resolve()
    if not prepared:
        begin_run(output, config.as_dict())
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    environment.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",
                       NVIDIA_TF32_OVERRIDE="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONFAULTHANDLER="1")
    if config.workload.get("hardware_profile"):
        hardware = config.workload["hardware_profile"]
        if hash_file(hardware["path"]) != hardware["sha256"]:
            raise ValueError("Frozen native hardware settings changed before child launch")
    environment["PYTHONPATH"] = str(output / "runtime_snapshot")
    command = [sys.executable, "-m", "experiments.resnet50_quality.run", "--frozen", "--output", str(output)]
    print(json.dumps({"event": "quality_run_started", "method": config.workload["method"],
                      "clients": config.virtual_clients.total, "output": str(output)}, sort_keys=True), flush=True)
    started = time.perf_counter()
    with (output / "stdout.log").open("a", buffering=1) as stream:
        child = subprocess.Popen(command, cwd=output / "runtime_snapshot", env=environment,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 bufsize=0, start_new_session=True)
        hard_deadline = started + config.workload["run_limits"]["run_timeout_sec"]
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def forward(data, *, final=False):
            text = decoder.decode(data, final=final)
            stream.write(text)
            sys.stdout.write(text)
            sys.stdout.flush()

        interrupted = None
        while child.poll() is None:
            if hard_deadline is not None and time.perf_counter() >= hard_deadline:
                interrupted = "The frozen complete-run hard timeout was exceeded"
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            for key, _ in selector.select(timeout=1):
                data = os.read(key.fd, 65536)
                if data:
                    forward(data)
                else:
                    selector.unregister(key.fileobj)
        # Drain available bytes after exit without waiting on stdout handles
        # inherited by an orphaned Ray process. Binary reads also avoid a text
        # buffer hiding progress from select() while the native child runs.
        while ready := selector.select(timeout=0):
            for key, _ in ready:
                data = os.read(key.fd, 65536)
                if data:
                    forward(data)
                else:
                    selector.unregister(key.fileobj)
        forward(b"", final=True)
        child.stdout.close()
        selector.close()
        code = child.wait()
    result = json.loads((output / "result.json").read_text())
    if code or result["status"] != "passed":
        if result["status"] == "running" or interrupted:
            result = {"status": "failed", "exit_code": code,
                      "reason": interrupted or "The native child exited before completing its receipt"}
            save_json(output / "result.json", result)
            save_json(output / "validation_report.json", result)
        reason = result.get('reason') or 'training validation failed'
        with (output / "stdout.log").open("rb") as log:
            log.seek(max(0, log.seek(0, os.SEEK_END) - 262144))
            diagnostic = log.read().decode("utf-8", errors="replace")
        errors = list(re.finditer(r"ServerApp thread raised an exception:([^\n]*)", diagnostic))
        if errors:
            native = errors[-1].group(1).strip()
            if not native:
                context = diagnostic[errors[-1].end():errors[-1].end() + 6000]
                root = re.search(r"(?m)^[\w.]+(?:Error|Exception)(?::[^\n]*)?$", context)
                native = root.group() if root else context[:1000]
            reason += "; ServerApp: " + native
        if "Fatal Python error:" in diagnostic:
            from .concurrency import native_failure_detail
            reason += "; " + native_failure_detail(diagnostic)
        raise RuntimeError(f"Native quality child failed: {reason}")
    return {**result, "output": str(output), "outer_wall_seconds": time.perf_counter() - started}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="pytorch")
    parser.add_argument("--method", choices=("fedavg", "splitfleet"))
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--hardware-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen", action="store_true")
    args = parser.parse_args()
    if args.frozen:
        return execute_frozen(args.output)
    if not all((args.backend, args.method, args.protocol, args.data_root, args.hardware_profile)):
        parser.error("Training requires backend, method, protocol, data root and hardware profile")
    from .hardware import apply_execution_affinity
    apply_execution_affinity(args.hardware_profile)
    config = make_config(method=args.method, backend=args.backend, seed=args.seed,
        protocol=args.protocol, data_root=args.data_root, hardware_profile=args.hardware_profile)
    frozen_child(config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
