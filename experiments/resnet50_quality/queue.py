"""Freeze and execute complete paired training with fixed resources."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from .native import BACKENDS
from .protocol import SEEDS, hash_file, validate_main_protocol, validate_worker_protocol

BACKEND_ORDER = ("pytorch",)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def timestamp():
    return datetime.now(UTC).isoformat()


def freeze_queue_runtime(workspace, output):
    from experiments.www2027_study import freeze_runtime
    _, manifest = freeze_runtime(workspace, output)
    for name in ('pyproject.toml', 'uv.lock', 'torchlens-2.34.1-py3-none-any.whl'):
        if (workspace / name).is_file():
            shutil.copyfile(workspace / name, output / 'runtime_snapshot' / name)
    return manifest


def prepare_queue(*, protocol, hardware, data_root, output, backends=None, runtime_bundle=None):
    from .hardware import gpu_snapshot
    from .run import RUN_LIMITS, make_config
    order = tuple(BACKEND_ORDER if backends is None else backends)
    if not order or len(set(order)) != len(order) or any(b not in BACKENDS for b in order):
        raise ValueError("Select distinct supported experiment backends: " + ", ".join(BACKEND_ORDER))
    protocol, hardware, data_root = (Path(p).resolve(strict=True) for p in (protocol, hardware, data_root))
    frozen = read_json(protocol / "protocol.json")
    validate_worker_protocol(frozen)
    rounds, examples = validate_main_protocol(frozen)
    worker_counts = read_json(hardware)["worker_counts"]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    workspace = Path(__file__).resolve().parents[2]
    for index, seed in enumerate(SEEDS):
        for backend in order:
            workers, clients = worker_counts[backend], BACKENDS[backend]
            for method in (("fedavg", "splitfleet") if index % 2 == 0 else ("splitfleet", "fedavg")):
                make_config(method=method, backend=backend, seed=seed, protocol=protocol,
                            data_root=data_root, hardware_profile=hardware)
                python = Path(sys.executable).absolute()
                destination = output / "runs" / backend / f"seed_{seed}" / method
                argv = [str(python), "-m", "experiments.resnet50_quality.run", "--backend", backend,
                        "--method", method, "--seed", str(seed), "--protocol", str(protocol),
                        "--data-root", str(data_root), "--hardware-profile", str(hardware),
                        "--output", str(destination)]
                jobs.append({"backend": backend, "clients": clients, "seed": seed, "method": method,
                             "rounds": rounds, "examples_per_run": examples, "worker_count": workers,
                             "argv": argv, "output": str(destination), "status": "pending",
                             "hard_timeout_sec": RUN_LIMITS["run_timeout_sec"],
                             "protocol": {"file": str(protocol / "protocol.json"),
                                          "sha256": hash_file(protocol / "protocol.json")},
                             "hardware": {"file": str(hardware), "sha256": hash_file(hardware)}})
    if runtime_bundle is None:
        manifest = freeze_queue_runtime(workspace, output)
    else:
        bundle = Path(runtime_bundle).resolve(strict=True)
        manifest = read_json(bundle / "runtime_manifest.json")
        for name in ("runtime_snapshot", "runtime_manifest.json"):
            (output / name).symlink_to(bundle / name)
    queue = {"status": "prepared_not_launched", "execution_mode": "training",
             "backends": list(order), "created_utc": timestamp(), "gpu": "0,1",
             "physical_gpus": gpu_snapshot(), "planned_runs": len(jobs),
             "planned_examples": len(jobs) * examples, "all_rounds": rounds,
             "examples_per_run": examples, "jobs": jobs, "failure_policy": "stop",
             "host_memory_reserve_mb": 12288,
             "worker_plan": {b: {"requested_workers": BACKENDS[b], "workers": worker_counts[b],
                                  "policy": "explicit fixed hardware configuration"} for b in order},
             "source_identity": manifest["source_identity"]}
    save_json(output / "queue.json", queue)
    return queue


def progress(job):
    output = Path(job["output"])
    path = output / "client_metrics.jsonl"
    tasks, examples, round_id = 0, 0, 0
    if path.exists():
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A currently-written final line is not a completed task.
                if row.get("status") == "success":
                    tasks += 1
                    examples += row["num_examples"]
                    round_id = max(round_id, row["round_id"])
    return {"successful_tasks": tasks, "examples_seen": examples, "current_round": round_id}


def execute_queue(path):
    path = Path(path).resolve()
    root, queue = path.parent, read_json(path)
    if queue["status"] != "prepared_not_launched":
        raise FileExistsError("Do not overwrite or blindly retry a queue attempt")
    if queue.get("execution_mode") != "training":
        raise ValueError("Prepare a current native training queue before execution")
    hardware_files = {job['hardware']['file'] for job in queue['jobs'] if job.get('hardware')}
    if hardware_files:
        if len(hardware_files) != 1:
            raise ValueError('One queue must use one frozen hardware profile')
        from .hardware import apply_execution_affinity
        hardware_path = next(iter(hardware_files))
        for job in queue['jobs']:
            if job.get('hardware') and hash_file(hardware_path) != job['hardware']['sha256']:
                raise ValueError('Frozen hardware changed before controller startup')
        queue['execution_affinity_applied'] = apply_execution_affinity(hardware_path)
    import hashlib
    manifest = read_json(root / "runtime_manifest.json")
    snapshot = root / "runtime_snapshot"
    hashes = {str(source.relative_to(snapshot)): hashlib.sha256(source.read_bytes()).hexdigest()
              for folder in ("splitfleet", "experiments")
              for source in sorted((snapshot / folder).rglob("*.py"))}
    if hashes != manifest["source_hashes"] or queue["source_identity"] != manifest["source_identity"]:
        raise ValueError("The source frozen for the complete queue changed before execution")
    from .hardware import gpu_snapshot
    current = gpu_snapshot()
    if any(current[key]['compute_processes'] or current[key]['uuid'] != queue['physical_gpus'][key]['uuid']
           for key in ('0', '1')):
        raise RuntimeError('The full queue needs its two idle physical GPUs')
    with (root / "execution_claim.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "started_utc": timestamp()}, stream)
    queue.update(status="running", controller_pid=os.getpid(), started_utc=timestamp())
    save_json(path, queue)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=queue["gpu"], NVIDIA_TF32_OVERRIDE="0",
                       OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONFAULTHANDLER="1",
                       PYTHONPATH=str(root / "runtime_snapshot"), PYTHONDONTWRITEBYTECODE="1")
    queue["gpu_environment"] = {name: environment[name] for name in (
        "CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE")}
    save_json(path, queue)
    while True:
        pending = [(index, job) for index, job in enumerate(queue['jobs']) if job['status'] == 'pending']
        if not pending:
            break
        index, job = pending[0]
        job_environment = environment.copy()
        for key in ("protocol",):
            if hash_file(job[key]["file"]) != job[key]["sha256"]:
                queue.update(status="failed", reason=f"Frozen {key} changed before job {index}")
                save_json(path, queue)
                return 1
        if job.get("hardware") and hash_file(job["hardware"]["file"]) != job["hardware"]["sha256"]:
            queue.update(status="failed", reason=f"Frozen hardware changed before job {index}")
            save_json(path, queue)
            return 1
        frozen_protocol = read_json(job["protocol"]["file"])
        validate_worker_protocol(frozen_protocol)
        main_rounds, expected_examples = validate_main_protocol(frozen_protocol)
        if (job.get('rounds', main_rounds) != main_rounds
                or queue.get('all_rounds', main_rounds) != main_rounds
                or job.get('examples_per_run', expected_examples) != expected_examples
                or queue.get('examples_per_run', expected_examples) != expected_examples):
            raise ValueError('Queued main budget differs from its frozen protocol')
        # A short path keeps Ray's AF_UNIX socket names within their limit.
        # All abandoned Ray/state temporaries belong to this one run.
        scratch = Path(tempfile.mkdtemp(prefix='sfq-'))
        job_environment.update(TMPDIR=str(scratch), RAY_TMPDIR=str(scratch))
        job['temporary_root'] = str(scratch)
        job.update(status="running", started_utc=timestamp())
        suffix = f"_w{job['worker_count']}"
        log_path = root / f"job_{index:02d}_{job['backend']}_{job['seed']}_{job['method']}{suffix}.log"
        with log_path.open("x") as log:
            child = subprocess.Popen(job["argv"], cwd=root / "runtime_snapshot", env=job_environment,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            import psutil
            job["controller_pid"] = child.pid
            job['controller_birth_time'] = psutil.Process(child.pid).create_time()
            save_json(path, queue)
            print(json.dumps({"event": "main_started", "job": index, "backend": job["backend"],
                              "seed": job["seed"], "method": job["method"], "pid": child.pid}), flush=True)
            from .concurrency import native_crash_report, run_bounded
            last_update = 0

            def record_progress():
                nonlocal last_update
                if time.monotonic() - last_update >= 10:
                    job['progress'] = progress(job)
                    queue['updated_utc'] = timestamp()
                    save_json(path, queue)
                    last_update = time.monotonic()

            def native_failure():
                report = native_crash_report(output=job['output'], scratch=scratch)
                return report['reason'] if report else None

            code, stopped = run_bounded(child, deadline=time.monotonic() + job['hard_timeout_sec'],
                                        floor_mb=queue.get('host_memory_reserve_mb', 4096),
                                        on_poll=record_progress, failure_check=native_failure)
        diagnostics = native_crash_report(output=job['output'], scratch=scratch)
        if diagnostics:
            job['failure_diagnostics'] = diagnostics
        from .concurrency import discard_output
        discard_output(scratch, root='/tmp')
        job['temporary_output_deleted'] = True
        report = Path(job["output"]) / "quality_validation.json"
        passed = (code == 0 and stopped is None and diagnostics is None
                  and report.is_file() and read_json(report).get("status") == "passed"
                  and read_json(report).get("examples_seen") == expected_examples
                  and (Path(job["output"]) / "test_metrics.json").is_file())
        job.update(status="passed" if passed else "failed", exit_code=code, finished_utc=timestamp(),
                   progress=progress(job))
        queue["completed_runs"] = sum(row["status"] == "passed" for row in queue["jobs"])
        queue["failed_runs"] = sum(row["status"] == "failed" for row in queue["jobs"])
        if not passed:
            from .concurrency import discard_output, native_failure_detail

            with log_path.open('rb') as stream:
                stream.seek(max(0, stream.seek(0, os.SEEK_END) - 262144))
                detail = native_failure_detail(stream.read().decode('utf-8', errors='replace'))
            job['reason'] = (diagnostics['reason'] if diagnostics else stopped or detail
                             or f'Native main child exited with code {code}')

            # A failure never enters the complete comparison. The user retains
            # its concise queue status, rather than logs, partial states or data.
            discard_output(job["output"], root=root.parent)
            log_path.unlink(missing_ok=True)
            job["failed_output_deleted"] = True
            queue.update(status="failed", finished_utc=timestamp(),
                         reason="Training failed; fixed resources are unchanged and later jobs remain pending")
            save_json(path, queue)
            return 1
        save_json(path, queue)
    queue.update(status="validating_results", training_finished_utc=timestamp())
    save_json(path, queue)
    from .summarize import summarize_queue
    report = summarize_queue(path, root / "summary")
    expected_backends = {job["backend"] for job in queue["jobs"]}
    complete = set(report['comparisons']) == expected_backends
    queue.update(status="passed" if complete else "incomplete", finished_utc=timestamp(),
                 result_validation="passed" if complete else "incomplete",
                 failed_runs=sum(job['status'] == 'failed' for job in queue['jobs']))
    save_json(path, queue)
    return 0 if complete else 1


def run_controller(path):
    """Retain controller errors independently from a child's native evidence."""
    import traceback
    try:
        return execute_queue(path)
    except BaseException as error:  # noqa: BLE001 - preserve controller/interruption evidence
        state = read_json(path)
        # Refusing an existing attempt must not rewrite its historical outcome.
        if state.get("status") in {"running", "validating_results"} and state.get("controller_pid") == os.getpid():
            state.update(status="controller_failed", finished_utc=timestamp(), reason=repr(error),
                         partial_comparison=False)
            save_json(path, state)
        traceback.print_exc()
        return 1
