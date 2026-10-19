"""Matched physical placement comparisons across capacity and actual TCP regimes.

Input preparation is separate: every arm reads the same frozen server/client
bundles. Results are prospective development measurements, never automatically
substituted into a publication cohort.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import time

from experiments.common.network_schedule import RoundNetworkSchedule


CONTROLLERS = ("Greedy", "L2S")


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def network_phases(name, rounds):
    if rounds < 3:
        raise ValueError("Stress comparisons require at least three rounds")
    if name == "unshaped":
        return [{"start_round": 1, "upload_mbps": None, "download_mbps": None, "rtt_ms": 0}]
    phases = [
        {"start_round": 1, "upload_mbps": 1024., "download_mbps": 1024., "rtt_ms": 2.},
        {"start_round": rounds // 3 + 1, "upload_mbps": 128., "download_mbps": 256., "rtt_ms": 15.},
        {"start_round": 2 * rounds // 3 + 1, "upload_mbps": 1024., "download_mbps": 1024., "rtt_ms": 2.}]
    if name == "dynamic":
        return phases
    if name != "stationary":
        raise ValueError("Unknown network profile")
    counts = [((phases[i+1]["start_round"] if i+1 < len(phases) else rounds+1) - row["start_round"])
              for i, row in enumerate(phases)]
    # Equal bytes per round would have the same total serialization time.
    # This is a protocol control, not a claim about realized transferred bytes.
    return [{"start_round": 1,
             **{name: rounds / sum(n / row[name] for n, row in zip(counts, phases))
                for name in ("upload_mbps", "download_mbps")},
             "rtt_ms": sum(n * row["rtt_ms"] for n, row in zip(counts, phases)) / rounds}]


def source_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for name in ("experiments", "splitfleet") for path in sorted((root / name).rglob("*.py"))}


def validate_network_execution(result, proxy, phases, rounds):
    """Reject scheduled experiments when the real training path skipped shaping."""
    schedule = result.get("network_schedule", {})
    if schedule.get("phases") != phases or schedule.get("future_phases_visible_to_controller") is not False:
        raise ValueError("The executed network schedule differs from the plan")
    applied = schedule.get("applied", [])
    if [row.get("round_id") for row in applied] != list(range(rounds + 1)):
        raise ValueError("The network schedule was not applied before every training round")
    if proxy.get("physical_transfer") is not True:
        raise ValueError("Missing physical TCP transfer evidence")
    for rid in range(rounds + 1):
        phase = next(row for row in reversed(phases) if row["start_round"] <= max(rid, 1))
        if any(applied[rid].get(key) != phase[key] for key in ("upload_mbps", "download_mbps", "rtt_ms")):
            raise ValueError(f"Network control disagrees with the planned phase at round {rid}")
        if rid == 0:
            continue
        counter = proxy.get("round_counters", {}).get(str(rid), {})
        if any(counter.get(key) != phase[key] for key in ("upload_mbps", "download_mbps", "rtt_ms")):
            raise ValueError(f"Actual TCP shaping disagrees with the planned phase at round {rid}")
        if not all(counter.get(key, 0) > 0 for key in ("upload_bytes", "download_bytes")):
            raise ValueError(f"Missing bidirectional physical TCP traffic at round {rid}")


def run_matrix(*, deployment, bundle_path, output, source_root, rounds, controllers,
               capacities, profiles, timeout=3600):
    import torch
    from experiments.common.bundle_storage import bundle_dependencies
    from experiments.common.physical_workers import physical_workers
    from experiments.common.ssh_transport import SshReverseTransport
    from experiments.orchestrate_physical_multitask import _run_one, _ssh

    output, bundle_path, source_root = (Path(p).resolve() for p in (output, bundle_path, source_root))
    frozen_driver = source_root / "experiments/placement_stress.py"
    if not frozen_driver.is_file() or frozen_driver.read_bytes() != Path(__file__).read_bytes():
        raise ValueError("The matrix driver must match the frozen runtime; use its matching driver")
    output.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(deployment)
    if config["server"].get("ssh"):
        raise ValueError("This stress runner requires a local coordinator")
    workers = physical_workers(config["hosts"])
    bundle = json.loads(bundle_path.with_suffix(".manifest.json").read_text())
    if bundle["worker_ids"] != [worker["identity"] for worker in workers]:
        raise ValueError("The input partition does not match the configured worker population")
    if not controllers or not set(controllers) <= set(CONTROLLERS) or len(set(controllers)) != len(controllers):
        raise ValueError("Invalid placement controller population")
    if not capacities or any(capacity < 1 for capacity in capacities):
        raise ValueError("Execution capacities must be positive")
    phases = {name: network_phases(name, rounds) for name in profiles}
    digest = source_hashes(source_root)
    save(output / "runtime_manifest.json", {"source_hashes": digest})
    jobs = []
    rng = random.Random(bundle["seed"])
    for capacity in capacities:
        for profile in profiles:
            order = list(controllers)
            rng.shuffle(order)
            jobs.extend({"capacity": capacity, "network": profile, "controller": controller, "status": "pending"}
                        for controller in order)
    progress = {"status": "preflight", "pid": os.getpid(), "started_unix": time.time(), "jobs": jobs}
    save(output / "plan.json", {"rounds": rounds, "seed": bundle["seed"], "jobs": jobs,
        "transport_protocol": "affine_bootstrap_with_actual_round_feedback",
        "network_phases": phases, "source_root": str(source_root), "paired_manifest": bundle,
        "order": "seeded controller permutation per capacity/network block", "automatic_retry": False,
        "scope": "development pilot; additional paired seeds required for publication estimates",
        "deployment": config})
    remote = "/tmp/splitfleet_stress_" + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    staged = []
    transport = None
    active = None
    try:
        for host in config["hosts"]:
            _ssh(host, "mkdir " + shlex.quote(remote) + " && mkdir " + shlex.quote(remote + "/bundles"))
            staged.append(host)
            subprocess.run(["rsync", "-az", "--exclude=__pycache__", str(source_root / "experiments"),
                            str(source_root / "splitfleet"), host["ssh"] + ":" + remote + "/"],
                           check=True, timeout=180)
            code = ("import pathlib,hashlib,json; r=pathlib.Path(" + repr(remote) + "); "
                    "print(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() "
                    "for n in ('experiments','splitfleet') for p in (r/n).rglob('*.py')}))")
            actual = json.loads(_ssh(host, "python3 -c " + shlex.quote(code)))
            if actual != digest:
                raise ValueError("Remote source differs from the coordinator")
            save(output / (host["id"] + ".staged_source_hashes.json"), actual)
            assets = set()
            for worker in workers:
                if worker["host"]["ssh"] != host["ssh"]:
                    continue
                client = bundle_path.with_name(bundle_path.stem + ".client_" + str(worker["index"]) + ".pt")
                for asset in [*bundle_dependencies(client), client]:
                    if asset in assets:
                        continue
                    target = remote + "/bundles/" + asset.relative_to(bundle_path.parent).as_posix()
                    _ssh(host, "mkdir -p " + shlex.quote(str(Path(target).parent)))
                    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(asset), host["ssh"] + ":" + target],
                                   check=True, timeout=600)
                    assets.add(asset)
        transport = SshReverseTransport(config, output / "transport_receipt.json")
        transport.start()
        for index, active in enumerate(jobs):
            if (output / "STOP").exists():
                raise RuntimeError("User stop marker encountered before next arm")
            if source_hashes(source_root) != digest:
                raise ValueError("Frozen execution source changed")
            arm = f"c{active['capacity']}_{active['network']}_{active['controller'].lower()}"
            control = output / "network_control.json"
            schedule_path = output / (arm + ".schedule.json")
            save(schedule_path, {"phases": phases[active["network"]]})
            RoundNetworkSchedule(phases[active["network"]], control).publish_initialization()
            proxy_receipt = output / (arm + ".tcp.json")
            active.update(status="running", started_unix=time.time())
            progress.update(status="running", active_job=arm, completed_jobs=index)
            save(output / "progress.json", progress)
            print("start", arm, flush=True)
            proxy_log = output / (arm + ".proxy.log")
            with proxy_log.open("w") as log:
                proxy = subprocess.Popen([config["server"]["python"], "-m", "experiments.tcp_pacing",
                    "--bind", config["client_server"]["connect"], "--target", config["server"]["connect"],
                    "--control", str(control), "--output", str(proxy_receipt), "--pipeline"],
                    cwd=source_root, env={**os.environ, "PYTHONPATH": str(source_root)}, stdout=log, stderr=subprocess.STDOUT)
                try:
                    # The proxy's target has not started yet. Connecting to its
                    # listener as a readiness probe would create a spurious
                    # target-refusal error in the transport evidence.
                    deadline = time.monotonic() + 30
                    while '"event": "proxy_listening"' not in proxy_log.read_text():
                        if proxy.poll() is not None:
                            raise RuntimeError("TCP proxy exited before listening")
                        if time.monotonic() >= deadline:
                            raise TimeoutError("TCP proxy did not become ready")
                        time.sleep(.05)
                    run_dir = output / "physical" / arm
                    _run_one(config, task=bundle["task"], method="splitfleet", bundle_path=bundle_path,
                        bundle=bundle, remote_root=remote, run_dir=run_dir, rounds=rounds, timeout=timeout,
                        learning_rate=deployment["learning_rate"], optimizer=deployment["optimizer"],
                        fixed_boundary="50%", split_state_exchange="owned", server_concurrency=active["capacity"],
                        placement_controller=active["controller"], network_schedule=schedule_path, network_control=control,
                        log_dir=output / "logs" / arm, local_source_root=source_root)
                finally:
                    if proxy.poll() is None:
                        proxy.terminate()
                    proxy.wait(timeout=20)
            result_path = run_dir / "result.json"
            result = json.loads(result_path.read_text())
            if result.get("placement_controller") != active["controller"]:
                raise ValueError("Wrong controller executed")
            for key in ("initial_model_hash", "partition_hash", "data_content_hash", "batch_size", "worker_ids"):
                if result[key] != bundle[key]:
                    raise ValueError("Paired condition differs: " + key)
            if result.get("fit_failures") or len(result["evaluation_records"]) != rounds:
                raise ValueError("Incomplete stress endpoint")
            validate_network_execution(result, json.loads(proxy_receipt.read_text()),
                                       phases[active["network"]], rounds)
            active.update(status="completed", job_sec=result["server_duration_sec"],
                          finished_unix=time.time(), final_metrics=result["evaluation_records"][-1]["metrics"],
                          result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(), result=str(result_path))
            save(output / "progress.json", progress)
            print("completed", arm, active["job_sec"], active["final_metrics"], flush=True)
            active = None
        progress.update(status="completed", completed_jobs=len(jobs), active_job=None, finished_unix=time.time())
    except Exception as exc:
        if active is not None:
            active.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        progress.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(output / "progress.json", progress)
        if transport is not None:
            transport.close()
        for host in staged:
            try:
                _ssh(host, "rm -rf " + shlex.quote(remote))
            except (subprocess.SubprocessError, OSError):
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True, help="Includes optimizer and learning_rate")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True, help="Immutable shared runtime snapshot")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS[:3]))
    parser.add_argument("--capacities", nargs="+", type=int, default=[1])
    parser.add_argument("--network-profiles", nargs="+", choices=("unshaped", "stationary", "dynamic"),
                        default=["unshaped", "stationary", "dynamic"])
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    run_matrix(deployment=json.loads(args.deployment.read_text()), bundle_path=args.bundle,
               output=args.output, source_root=args.source_root, rounds=args.rounds,
               controllers=args.controllers, capacities=args.capacities,
               profiles=args.network_profiles, timeout=args.timeout)


if __name__ == "__main__":
    main()
