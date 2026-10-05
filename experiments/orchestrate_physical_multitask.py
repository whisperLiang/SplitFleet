"""Run paired real-data methods on CPU and GPU of three Jetson hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch

from experiments.common.identity import tensor_state_hash
from experiments.common.physical_workers import physical_workers
from experiments.physical_multitask import (
    FIXED_BOUNDARIES, METHODS, TASKS, prepare_bundle, scheme_name,
)


class RoundBarrier:
    """Release each Flower round only after every configured worker is ready."""

    def __init__(self, address: str, identities: set[str], rounds: int) -> None:
        host, port = address.rsplit(":", 1)
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((host, int(port)))
        self.listener.listen(len(identities))
        self.listener.settimeout(1)
        self.identities = identities
        self.rounds = rounds
        self.stopped = threading.Event()
        self.error: Exception | None = None
        self.records: list[dict[str, Any]] = []
        self._diagnostic_lock = threading.Lock()
        self._pending: dict[str, Any] = {}
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        self.listener.close()
        self.thread.join(timeout=2)

    def diagnostic(self) -> dict[str, Any]:
        """Preserve readiness of an unfinished round as well as a thread error."""
        with self._diagnostic_lock:
            pending = json.loads(json.dumps(self._pending))
        return {"pending_round": pending, "completed_rounds": len(self.records),
                "thread_alive": self.thread.is_alive(), "stopped": self.stopped.is_set(),
                "error": None if self.error is None else {
                    "type": type(self.error).__name__, "message": str(self.error)}}

    def _serve(self) -> None:
        connections: dict[str, socket.socket] = {}
        try:
            for round_id in range(1, self.rounds + 1):
                connections = {}
                arrivals: dict[str, int] = {}
                with self._diagnostic_lock:
                    self._pending = {"round_id": round_id, "ready_ids": [],
                                     "missing_ids": sorted(self.identities),
                                     "ready_monotonic_ns": {}, "last_message": None}
                while len(connections) < len(self.identities) and not self.stopped.is_set():
                    try:
                        connection, _ = self.listener.accept()
                    except socket.timeout:
                        continue
                    try:
                        connection.settimeout(10)
                        payload = b""
                        while not payload.endswith(b"\n") and len(payload) < 512:
                            chunk = connection.recv(512 - len(payload))
                            if not chunk:
                                break
                            payload += chunk
                        if not payload.endswith(b"\n"):
                            raise RuntimeError("incomplete round-barrier message")
                        message = json.loads(payload)
                        with self._diagnostic_lock:
                            self._pending["last_message"] = message
                        identity = str(message["id"])
                        if int(message["round_id"]) != round_id or identity not in self.identities or identity in connections:
                            raise RuntimeError(f"unexpected worker at round barrier {round_id}: {message}")
                    except Exception:
                        connection.close()
                        raise
                    connections[identity] = connection
                    arrivals[identity] = time.monotonic_ns()
                    with self._diagnostic_lock:
                        self._pending.update(ready_ids=sorted(arrivals),
                                             missing_ids=sorted(self.identities - arrivals.keys()),
                                             ready_monotonic_ns=dict(arrivals))
                if self.stopped.is_set():
                    break
                release_started = time.monotonic_ns()
                for connection in connections.values():
                    connection.sendall(b"1")
                    connection.close()
                self.records.append({
                    "round_id": round_id, "ready_ids": sorted(arrivals),
                    "ready_monotonic_ns": arrivals,
                    "release_started_monotonic_ns": release_started,
                    "release_finished_monotonic_ns": time.monotonic_ns(),
                })
        except OSError as exc:
            if not self.stopped.is_set():
                self.error = exc
        except Exception as exc:
            self.error = exc
        finally:
            for connection in connections.values():
                connection.close()


def validate_result(result: dict[str, Any], *, task: str, method: str, rounds: int,
                    hosts: list[dict[str, str]], batch_size: int,
                    bundle: dict[str, Any],
                    barrier_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    workers = physical_workers(hosts)
    identities = {worker["identity"]: worker["kind"] for worker in workers}
    indices_by_id = {worker["identity"]: worker["index"] for worker in workers}
    num_workers = len(workers)
    if bundle.get("worker_ids") is not None and bundle["worker_ids"] != list(identities):
        errors.append("configured workers differ from the frozen bundle")
    for key, value in (("task", task), ("method", method), ("rounds", rounds),
                       ("image_model", bundle.get("image_model")),
                       ("pretrain_checkpoint_sha256", bundle.get("pretrain_checkpoint_sha256")),
                       ("data_content_hash", bundle["data_content_hash"]),
                       ("partition_hash", bundle["partition_hash"]),
                       ("initial_model_hash", bundle["initial_model_hash"])):
        if result.get(key) != value:
            errors.append(f"{key} differs from the frozen task bundle")
    if result.get("final_model_hash") == result.get("initial_model_hash"):
        errors.append("global model did not change during training")
    for key in ("model_id", "model_metadata", "num_parameters", "trainable_parameters"):
        if key in bundle and result.get(key) != bundle[key]:
            errors.append(f"{key} differs from the frozen task bundle")
    indices = [index for positions in bundle["assignments"].values() for index in positions]
    if sorted(indices) != list(range(bundle["train_size"])):
        errors.append("training partitions are not disjoint and exhaustive")
    if sum(bundle["partition_sizes"].values()) != bundle["train_size"]:
        errors.append("partition sizes do not sum to the training size")
    if method == "splitfleet":
        if not result.get("owned_state_exchange"):
            errors.append("SplitFleet did not use owned state exchange")
        catalog = result.get("candidate_catalog", [])
        if not catalog or any(
            int(candidate.get("prefix_trainable_parameter_count") or 0) < 1
            for candidate in catalog
        ):
            errors.append("adaptive candidate catalog includes a nontrainable client prefix")
        allowed = {candidate.get("boundary") for candidate in catalog}
        for diagnostic in result.get("round_diagnostics", {}).values():
            if not set(diagnostic.get("assignment", {}).values()) <= allowed:
                errors.append("adaptive assignment is absent from the audited candidate catalog")
        if result.get("online_cost_learning") is not True:
            errors.append("SplitFleet online cost learning receipt is missing")
        initialization = result.get("cost_initialization_receipt") or {}
        if initialization.get("source") != "current_deployment_no_update_split_calibration":
            errors.append("current-deployment calibration receipt is missing")
        if initialization.get("initial_model_hash") != bundle.get("initial_model_hash"):
            errors.append("calibration initial model differs from the frozen bundle")
        if initialization.get("prediction_override") is not False:
            errors.append("calibration must seed learners without replacing predictions")
        for round_id in range(1, rounds + 1):
            diagnostic = result.get("round_diagnostics", {}).get(str(round_id), {})
            counts = diagnostic.get("learner_update_counts", {})
            if len(counts) != num_workers or any(
                int(value.get("server_update_count", 0)) < 1
                or int(value.get("group_update_count", 0)) < 1
                for value in counts.values()
            ):
                errors.append(f"round {round_id}: online learner update receipt is missing")
    if result.get("fit_failures") or result.get("expected_clients") != num_workers:
        errors.append("all configured failure-free workers were not reported")
    all_fits = result.get("fit_records", [])
    all_tails = result.get("server_fit_records", [])
    all_evaluations = result.get("evaluation_records", [])
    overlap_by_round: dict[str, float] = {}
    for round_id in range(1, rounds + 1):
        if barrier_records is not None:
            receipts = [row for row in barrier_records if row.get("round_id") == round_id]
            if len(receipts) != 1 or set(receipts[0].get("ready_ids", [])) != set(identities):
                errors.append(f"round {round_id}: configured-worker readiness receipt is missing")
            else:
                receipt = receipts[0]
                arrivals = receipt.get("ready_monotonic_ns", {})
                release_start = receipt.get("release_started_monotonic_ns", 0)
                release_end = receipt.get("release_finished_monotonic_ns", 0)
                if (set(arrivals) != set(identities) or not arrivals
                        or min(arrivals.values()) <= 0
                        or max(arrivals.values()) > release_start
                        or release_end < release_start):
                    errors.append(f"round {round_id}: barrier released before every worker was ready")
        fits = [row for row in all_fits if row.get("round_id") == round_id]
        tails = [row for row in all_tails if row.get("round_id") == round_id]
        evaluations = [row for row in all_evaluations if row.get("round_id") == round_id]
        ids = [row.get("metrics", {}).get("logical_client_id") for row in fits]
        cids = [row.get("cid") for row in fits]
        if len(fits) != num_workers or set(ids) != set(identities) or len(set(cids)) != num_workers:
            errors.append(f"round {round_id}: distinct configured worker updates are missing")
        if len(evaluations) != 1 or not evaluations[0].get("metrics"):
            errors.append(f"round {round_id}: task evaluation is missing")
        else:
            for name, value in evaluations[0]["metrics"].items():
                try:
                    valid = math.isfinite(float(value)) and 0 <= float(value) <= 1
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    errors.append(f"round {round_id}: invalid {name}")
        if method in ("splitfed_fixed", "splitfleet"):
            if len(tails) != num_workers or {row.get("cid") for row in tails} != set(cids):
                errors.append(f"round {round_id}: configured server suffixes are missing")
        elif tails:
            errors.append(f"round {round_id}: full-local method produced server suffixes")
        starts: list[int] = []
        finishes: list[int] = []
        by_id = {row.get("metrics", {}).get("logical_client_id"): row for row in fits}
        for host in hosts:
            if len(host.get("workers", ["cpu", "gpu"])) != 2:
                continue
            cpu = by_id.get(f"{host['id']}-cpu", {}).get("metrics", {})
            gpu = by_id.get(f"{host['id']}-gpu", {}).get("metrics", {})
            if cpu.get("pid") is None or gpu.get("pid") is None or cpu["pid"] == gpu["pid"]:
                errors.append(f"round {round_id}: {host['id']} has no distinct CPU/GPU processes")
        for row in fits:
            metrics = row.get("metrics", {})
            identity = metrics.get("logical_client_id")
            kind = identities.get(identity)
            device = str(metrics.get("device", ""))
            if kind == "cpu" and device != "cpu" or kind == "gpu" and not device.startswith("cuda"):
                errors.append(f"round {round_id}: {identity} used {device}")
            worker_index = indices_by_id.get(identity)
            expected = bundle["partition_sizes"].get(str(worker_index))
            expected_batches = math.ceil(expected / batch_size) if expected is not None else None
            if row.get("num_examples") != expected or metrics.get("num_batches") != expected_batches:
                errors.append(f"round {round_id}: {identity} did not complete its full partition")
            if metrics.get("partition_hash") != bundle["partition_hash"]:
                errors.append(f"round {round_id}: {identity} used a different partition")
            loss = metrics.get("task_loss", metrics.get("loss"))
            try:
                valid_loss = math.isfinite(float(loss)) and float(loss) >= 0
            except (TypeError, ValueError):
                valid_loss = False
            if not valid_loss:
                errors.append(f"round {round_id}: {identity} has no finite loss")
            if method in ("splitfed_fixed", "splitfleet") and float(metrics.get("prefix_compute_sec", 0)) <= 0:
                errors.append(f"round {round_id}: {identity} has no prefix computation")
            start, finish = int(metrics.get("fit_started_unix_ns", 0)), int(metrics.get("fit_finished_unix_ns", 0))
            if start <= 0 or finish <= start:
                errors.append(f"round {round_id}: {identity} has invalid fit timestamps")
            else:
                starts.append(start)
                finishes.append(finish)
        if len(starts) == num_workers:
            overlap = (min(finishes) - max(starts)) / 1e9
            overlap_by_round[str(round_id)] = overlap
            if overlap <= 0:
                warnings.append(f"round {round_id}: no common configured-worker wall-clock fit intersection; "
                                "clock offsets and scheduling can exceed short fit durations")
    return {"schema": "splitfleet.physical-multitask-validation.v3",
            "valid": not errors, "errors": errors, "warnings": warnings,
            "barrier_verified": barrier_records is not None and not any("barrier" in e or "readiness" in e for e in errors),
            "barrier_records": barrier_records,
            "task": task, "method": method,
            "rounds": rounds, "workers": num_workers, "worker_ids": list(identities),
            "fit_interval_overlap_sec": overlap_by_round,
            "examples_per_round": sum(bundle["partition_sizes"].values()),
            "partition_sizes": bundle["partition_sizes"]}


def _ssh(host: dict[str, str], script: str, *, timeout: float = 30) -> str:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                           host["ssh"], script], text=True, capture_output=True,
                          check=True, timeout=timeout).stdout.strip()


def _wait_listen(address: str, process: subprocess.Popen[Any], timeout: float) -> None:
    host, port = address.rsplit(":", 1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before listening: {process.returncode}")
        try:
            with socket.create_connection((host, int(port)), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError("server did not listen")


def _run_one(config: dict[str, Any], *, task: str, method: str, bundle_path: Path,
             bundle: dict[str, Any], remote_root: str, run_dir: Path, rounds: int,
             timeout: float, learning_rate: float, optimizer: str,
             fixed_boundary: str,
             split_state_exchange: str = "full",
             min_residence_rounds: int = 2,
             log_dir: Path | None = None,
             local_source_root: Path | None = None) -> dict[str, Any]:
    server = config["server"]
    hosts = config["hosts"]
    run_dir.mkdir(parents=True, exist_ok=False)
    if log_dir is None:
        log_dir = Path("logs/physical_multitask") / run_dir.parent.name / run_dir.name
    log_dir.mkdir(parents=True, exist_ok=True)
    remote_server = bool(server.get("ssh"))
    server_bundle = f"{remote_root}/bundles/{task}.pt" if remote_server else str(bundle_path)
    server_output = f"{remote_root}/result_{task}_{method}.json" if remote_server else str(run_dir / "result.json")
    server_cmd = [server["python"], "-m", "experiments.physical_multitask", "server",
                  "--bundle", server_bundle, "--method", method,
                  "--device", server.get("device", "cpu"), "--bind", server["bind"],
                  "--rounds", str(rounds), "--output", server_output,
                  "--learning-rate", str(learning_rate), "--optimizer", optimizer,
                  "--fixed-boundary", fixed_boundary]
    server_cmd.extend(["--min-residence-rounds", str(min_residence_rounds)])
    server_cmd.extend(["--split-state-exchange", split_state_exchange])
    server_process: subprocess.Popen[Any] | None = None
    workers: list[tuple[str, subprocess.Popen[Any], Any]] = []
    failure: dict[str, str] | None = None
    started = time.time()
    host_name, port_text = server["connect"].rsplit(":", 1)
    try:
        with socket.create_connection((host_name, int(port_text)), timeout=1):
            raise RuntimeError(f"server port {server['connect']} is already occupied")
    except (ConnectionRefusedError, TimeoutError, OSError):
        pass
    barrier = RoundBarrier(
        config["barrier"]["bind"],
        {worker["identity"] for worker in physical_workers(hosts)},
        rounds,
    )
    barrier.start()
    with (log_dir / "server.log").open("w", encoding="utf-8") as server_log:
        try:
            if remote_server:
                command = "cd " + shlex.quote(remote_root) \
                    + " && echo $$ > server.pid && exec env PYTHONPATH=" \
                    + shlex.quote(remote_root) + " " + shlex.join(server_cmd)
                server_launch = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                 server["ssh"], command]
                server_process = subprocess.Popen(server_launch, stdout=server_log,
                                                  stderr=subprocess.STDOUT)
            else:
                local_cmd = (["env", "PYTHONPATH=" + str(local_source_root), *server_cmd]
                             if local_source_root is not None else server_cmd)
                server_process = subprocess.Popen(local_cmd, cwd=local_source_root or server["workdir"],
                                                  stdout=server_log, stderr=subprocess.STDOUT)
            for worker in physical_workers(hosts):
                host, kind, device = worker["host"], worker["kind"], worker["device"]
                identity, index = worker["identity"], worker["index"]
                remote_cmd = [host["python"], "-m", "experiments.physical_multitask", "client",
                              "--bundle", f"{remote_root}/bundles/{task}.client_{index}.pt",
                              "--method", method, "--device", device,
                              "--server", server["connect"], "--client-id", identity,
                              "--client-index", str(index),
                              "--barrier", config["barrier"]["connect"],
                              "--learning-rate", str(learning_rate),
                              "--optimizer", optimizer,
                              "--fixed-boundary", fixed_boundary]
                command = "cd " + shlex.quote(remote_root) + " && PYTHONPATH=" \
                    + shlex.quote(remote_root) + " " + shlex.join(remote_cmd)
                log = (log_dir / f"{identity}.log").open("w", encoding="utf-8")
                process = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                            host["ssh"], command], stdout=log,
                                           stderr=subprocess.STDOUT)
                workers.append((identity, process, log))
            # Clients prewarm their local TorchLens graph before connecting;
            # start them immediately so tracing overlaps server startup and
            # candidate preparation.
            _wait_listen(server["connect"], server_process, max(180, min(timeout, 600)))
            deadline = time.monotonic() + timeout
            while server_process.poll() is None:
                if barrier.error is not None:
                    raise RuntimeError(f"round barrier failed: {barrier.error}")
                failed = [(name, process.returncode) for name, process, _ in workers
                          if process.poll() is not None and process.returncode != 0]
                if failed:
                    raise RuntimeError(f"physical workers failed: {failed}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{task}/{method} exceeded {timeout} seconds")
                time.sleep(1)
            if server_process.returncode != 0:
                raise RuntimeError(f"server exited with {server_process.returncode}")
            if barrier.error is not None:
                raise RuntimeError(f"round barrier failed: {barrier.error}")
            for name, process, _ in workers:
                process.wait(timeout=30)
                if process.returncode != 0:
                    raise RuntimeError(f"{name} exited with {process.returncode}")
            if remote_server:
                subprocess.run(["scp", "-q", "-o", "BatchMode=yes",
                                f"{server['ssh']}:{server_output}", str(run_dir / "result.json")],
                               check=True, timeout=30)
                remote_checkpoint = str(Path(server_output).with_suffix(".model.pt"))
                subprocess.run(["scp", "-q", "-o", "BatchMode=yes",
                                f"{server['ssh']}:{remote_checkpoint}",
                                str(run_dir / "final_model.pt")], check=True, timeout=30)
                if method == "splitfleet":
                    remote_bandit = str(Path(server_output).with_suffix(".bandit.json"))
                    subprocess.run(["scp", "-q", "-o", "BatchMode=yes",
                                    f"{server['ssh']}:{remote_bandit}",
                                    str(run_dir / "bandit_state.json")], check=True, timeout=30)
            else:
                Path(str(Path(server_output).with_suffix(".model.pt"))).rename(
                    run_dir / "final_model.pt"
                )
                if method == "splitfleet":
                    Path(server_output).with_suffix(".bandit.json").rename(
                        run_dir / "bandit_state.json"
                    )
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            if result.get("final_model_hash"):
                checkpoint = torch.load(run_dir / "final_model.pt", map_location="cpu",
                                        weights_only=False)
                if tensor_state_hash(checkpoint["state_dict"]) != result["final_model_hash"]:
                    raise RuntimeError("downloaded final model differs from the server result")
                invalid = [name for name, value in checkpoint["state_dict"].items()
                           if value.is_floating_point() and not torch.isfinite(value).all()]
                if invalid:
                    raise FloatingPointError(f"Final model has nonfinite tensors: {invalid[:3]}")
            report = validate_result(result, task=task, method=method, rounds=rounds,
                                     hosts=hosts, batch_size=int(bundle["batch_size"]),
                                     bundle=bundle, barrier_records=barrier.records)
            (run_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
            if not report["valid"]:
                raise RuntimeError(f"{task}/{method} failed validation: {report['errors']}")
            final = result["evaluation_records"][-1]["metrics"]
            return {"task": task, "method": method, "fixed_boundary": result["fixed_boundary"],
                    "valid": True, "final_metrics": final,
                    "elapsed_sec": result.get("server_duration_sec", result["finished_unix"] - result["started_unix"]),
                    "elapsed_time_basis": "server_monotonic" if "server_duration_sec" in result else "server_wall_clock",
                    "fit_interval_overlap_sec": report["fit_interval_overlap_sec"],
                    "data_content_hash": result["data_content_hash"],
                    "initial_model_hash": result["initial_model_hash"],
                    "final_model_hash": result.get("final_model_hash"),
                    "partition_hash": result["partition_hash"],
                    "run_dir": str(run_dir)}
        except Exception as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            barrier.stop()
            (run_dir / "barrier_receipts.json").write_text(json.dumps(barrier.records, indent=2) + "\n")
            barrier_diagnostic = barrier.diagnostic()
            (run_dir / "barrier_diagnostics.json").write_text(json.dumps(barrier_diagnostic, indent=2) + "\n")
            for _, process, log in workers:
                if process.poll() is None:
                    process.terminate()
                log.close()
            # Terminating a local SSH process does not necessarily stop its
            # remote Python child. Stale clients can reconnect during the next
            # experiment with the same logical identity and corrupt the six
            # worker barrier. Match only this run's private bundle path.
            cleanup_code = (
                "import os,signal; root=" + repr(remote_root.encode()) + "; "
                "[(os.kill(int(pid),signal.SIGTERM)) for pid in os.listdir('/proc') "
                "if pid.isdigit() and int(pid)!=os.getpid() and "
                "(lambda a: b'experiments.physical_multitask' in a and b'client' in a "
                "and any(v.startswith(root+b'/bundles/') for v in a))"
                "(open('/proc/'+pid+'/cmdline','rb').read().split(b'\\0'))]"
            )
            for host in hosts:
                try:
                    subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                                    host["ssh"], "python3 -c " + shlex.quote(cleanup_code)],
                                   check=False, timeout=10, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if server_process is not None and server_process.poll() is None:
                if remote_server:
                    try:
                        pid = int(_ssh(server, "cat " + shlex.quote(remote_root + "/server.pid")))
                        command_line = _ssh(server, "tr '\\0' ' ' < /proc/" + str(pid) + "/cmdline")
                        if remote_root in command_line and "experiments.physical_multitask" in command_line:
                            _ssh(server, "kill -TERM " + str(pid))
                    except (subprocess.SubprocessError, OSError, ValueError):
                        pass
                server_process.terminate()
            (run_dir / "process_manifest.json").write_text(json.dumps({
                "started_unix": started, "finished_unix": time.time(),
                "server_exit_code": server_process.poll() if server_process else None,
                "worker_exit_codes": {name: process.poll() for name, process, _ in workers},
                "failure": failure,
                "barrier_diagnostic": barrier_diagnostic,
            }, indent=2) + "\n")


def run_matrix(config: dict[str, Any], *, run_id: str, output_root: Path,
               tasks: list[str], methods: list[str], data_root: str, seed: int,
               train_samples: int | None, test_samples: int | None, batch_size: int,
               rounds: int, timeout: float, alpha: float = 0.5,
               learning_rate: float = 0.01, optimizer: str = "sgd",
               fixed_boundaries: tuple[str, ...] = FIXED_BOUNDARIES, image_model: str | None = None,
               pretrain_weights: str | None = None,
               model_name: str | None = None, tokenizer_path: str | None = None,
                     split_state_exchange: str = "full",
               log_root: Path = Path("logs/physical_multitask")) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("unsafe run ID")
    if (not fixed_boundaries or len(set(fixed_boundaries)) != len(fixed_boundaries)
            or any(boundary not in FIXED_BOUNDARIES for boundary in fixed_boundaries)):
        raise ValueError("fixed_boundaries must contain distinct supported cuts")
    hosts = config["hosts"]
    worker_ids = [worker["identity"] for worker in physical_workers(hosts)]
    if len(hosts) != 3 or len({host["ssh"] for host in hosts}) != 3:
        raise ValueError("three distinct SSH hosts are required")
    output = (output_root / run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    remote_root = f"/tmp/splitfleet_physical_multitask_{run_id}"
    server_ssh = config["server"].get("ssh")
    remote_hosts = list(hosts)
    if server_ssh and server_ssh not in {host["ssh"] for host in hosts}:
        remote_hosts.append(config["server"])
    inventory = {}
    for host in hosts:
        lines = _ssh(host, "cd " + shlex.quote(host["workdir"])
                     + " && nvidia-smi -L && timedatectl show -p NTPSynchronized && git rev-parse HEAD").splitlines()
        if not any("GPU 0:" in line for line in lines) or "NTPSynchronized=yes" not in lines:
            raise RuntimeError(f"{host['id']} lacks GPU or synchronized clock")
        inventory[host["id"]] = {"gpu": next(line for line in lines if "GPU 0:" in line),
                                  "git_commit": lines[-1], "ntp_synchronized": True}
    (output / "host_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    source_hashes = {}
    local_source_root = output / "source_snapshot"
    local_source_root.mkdir()
    for folder in (Path("splitfleet"), Path("experiments")):
        shutil.copytree(folder, local_source_root / folder.name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for path in sorted(folder.rglob("*.py")):
            source_hashes[str(path)] = hashlib.sha256((local_source_root / path).read_bytes()).hexdigest()
    (output / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    prepared: dict[str, tuple[Path, dict[str, Any]]] = {}
    for task in tasks:
        bundle_path = output / "bundles" / f"{task}.pt"
        metadata = prepare_bundle(task=task, data_root=data_root, output=bundle_path,
                                  seed=seed, train_samples=train_samples,
                                  test_samples=test_samples, batch_size=batch_size,
                                  alpha=alpha, image_model=image_model,
                                  pretrain_weights=pretrain_weights,
                                  model_name=model_name, tokenizer_path=tokenizer_path,
                                  worker_ids=worker_ids)
        prepared[task] = (bundle_path, metadata)
    try:
        for host_index, host in enumerate(remote_hosts):
            _ssh(host, "mkdir -p " + shlex.quote(remote_root + "/bundles"))
            subprocess.run(["rsync", "-az", "--exclude=__pycache__",
                            str(local_source_root / "splitfleet"), str(local_source_root / "experiments"),
                            f"{host['ssh']}:{remote_root}/"], check=True, timeout=120)
            audit_code = (
                "import pathlib,hashlib,json;root=pathlib.Path(" + repr(remote_root) + ");"
                "print(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() "
                "for f in ('splitfleet','experiments') for p in (root/f).rglob('*.py')},sort_keys=True))"
            )
            staged_hashes = json.loads(_ssh(host, "python3 -c " + shlex.quote(audit_code)))
            (output / (host.get("id", "server") + ".staged_source_hashes.json")).write_text(
                json.dumps(staged_hashes, indent=2) + "\n")
            if staged_hashes != source_hashes:
                raise RuntimeError(f"{host.get('id', 'server')} staged sources differ from the frozen coordinator snapshot")
            for task, (bundle_path, _) in prepared.items():
                paths = [bundle_path.with_name(f"{task}.client_{worker['index']}.pt")
                         for worker in physical_workers(hosts) if worker["host"]["ssh"] == host["ssh"]]
                if host["ssh"] == server_ssh:
                    paths.append(bundle_path)
                for path in paths:
                    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(path),
                                    f"{host['ssh']}:{remote_root}/bundles/{path.name}"],
                                   check=True, timeout=300)
        rows = []
        attempts: list[dict[str, Any]] = []
        for task in tasks:
            bundle_path, bundle = prepared[task]
            for method in methods:
                boundaries = fixed_boundaries if method == "splitfed_fixed" else (None,)
                for boundary in boundaries:
                    scheme = scheme_name(method, boundary)
                    print(f"start {task}/{scheme}", flush=True)
                    attempt = {"task": task, "method": method, "scheme": scheme,
                               "status": "running", "started_unix": time.time()}
                    attempts.append(attempt)
                    (output / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
                    try:
                        row = _run_one(
                            config, task=task, method=method, bundle_path=bundle_path,
                            bundle=bundle, remote_root=remote_root,
                            run_dir=output / f"{task}_{scheme}", rounds=rounds,
                            log_dir=log_root / run_id / f"{task}_{scheme}",
                            timeout=timeout, learning_rate=learning_rate,
                            optimizer=optimizer, fixed_boundary=boundary or "50%",
                            split_state_exchange=split_state_exchange,
                            local_source_root=local_source_root,
                        )
                    except Exception as exc:
                        attempt.update(status="failed", finished_unix=time.time(),
                                       error_type=type(exc).__name__, error=str(exc))
                        (output / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
                        raise
                    attempt.update(status="completed", finished_unix=time.time())
                    (output / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
                    rows.append(row)
                    (output / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
                    print(f"complete {task}/{scheme}: {row['final_metrics']}", flush=True)
    finally:
        for host in remote_hosts:
            try:
                _ssh(host, "rm -rf " + shlex.quote(remote_root), timeout=30)
            except (subprocess.SubprocessError, OSError):
                pass
    return output / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results/physical_multitask"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/physical_multitask"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--fixed-boundaries", nargs="+", choices=FIXED_BOUNDARIES,
                        default=list(FIXED_BOUNDARIES))
    from experiments.unified_multitask.edge_models import EDGE_MODELS
    parser.add_argument("--image-model", choices=tuple(EDGE_MODELS), default=None)
    parser.add_argument("--model", choices=tuple(EDGE_MODELS))
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--pretrain-weights")
    parser.add_argument("--split-state-exchange", choices=("full", "owned"), default="full")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    config = json.loads(args.deployment.read_text(encoding="utf-8"))
    print(run_matrix(config, run_id=args.run_id, output_root=args.output_root,
                     tasks=args.tasks, methods=args.methods, data_root=args.data_root,
                     seed=args.seed, train_samples=args.train_samples,
                     test_samples=args.test_samples, batch_size=args.batch_size,
                     rounds=args.rounds, timeout=args.timeout,
                     alpha=args.dirichlet_alpha,
                     learning_rate=args.learning_rate,
                     optimizer=args.optimizer,
                     fixed_boundaries=tuple(args.fixed_boundaries),
                     image_model=args.image_model,
                     pretrain_weights=args.pretrain_weights,
                     model_name=args.model, tokenizer_path=args.tokenizer_path,
                     split_state_exchange=args.split_state_exchange,
                     log_root=args.log_root))


if __name__ == "__main__":
    main()
