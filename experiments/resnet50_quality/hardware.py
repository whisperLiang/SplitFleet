"""Freeze real CPU/GPU client placement with a separate local server GPU."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from collections import Counter
from dataclasses import replace
from pathlib import Path

from splitfleet.virtual.config import Profile

from .native import BACKENDS
from .protocol import hash_file
from .queue import read_json, save_json

HARDWARE_SCHEMA = "splitfleet.resnet50.pytorch50_hardware.v1"


def apply_execution_affinity(profile):
    """Bind the controller before it starts native libraries and Ray workers."""
    profile = read_json(profile) if isinstance(profile, (str, Path)) else profile
    selected = profile.get('execution_affinity')
    if selected is None:
        return None
    if (not isinstance(selected, list) or not selected
            or any(type(cpu) is not int or cpu < 0 for cpu in selected)
            or len(set(selected)) != len(selected)
            or not set(selected).issubset(os.sched_getaffinity(0))):
        raise ValueError('Execution affinity must be a nonempty subset of the allowed CPU pool')
    if any(value.get('affinity') and not set(value['affinity']).issubset(selected)
           for value in profile['profiles'].values()):
        raise ValueError('Logical CPU profiles must stay within the execution CPU pool')
    for entry in Path('/proc/self/task').iterdir():
        try:
            os.sched_setaffinity(int(entry.name), selected)
        except ProcessLookupError:
            pass
    return list(selected)


def hardware_profile(*, cpus, worker_counts=None, execution_cpus=None):
    """GPU20% / CPU80%, spreading one-thread CPU clients over actual cores."""
    cpus = list(cpus)
    if (len(cpus) < 2 or len(set(cpus)) != len(cpus)
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus)):
        raise ValueError("Heterogeneous emulation requires two distinct allowed physical CPUs")
    worker_counts = dict({"pytorch": 13} if worker_counts is None else worker_counts)
    if set(worker_counts) != set(BACKENDS) or any(
            type(value) is not int or not 3 <= value <= BACKENDS[backend]
            for backend, value in worker_counts.items()):
        raise ValueError("worker_counts must declare a fixed PyTorch pool of 3..50 workers")
    names = ["cpu_a", "cpu_b", *[f"cpu_{cpu}" for cpu in cpus[2:]]]
    assignments = {backend: {str(cid): "gpu" if cid % 5 == 0 else names[(cid - cid // 5 - 1) % len(names)]
                             for cid in range(clients)} for backend, clients in BACKENDS.items()}
    if execution_cpus is not None:
        execution_cpus = list(execution_cpus)
        if (not execution_cpus or len(set(execution_cpus)) != len(execution_cpus)
                or any(type(cpu) is not int or cpu < 0 for cpu in execution_cpus)
                or not set(cpus).issubset(execution_cpus)):
            raise ValueError('Execution CPU pool must contain every logical CPU profile')
    return {
        "schema": HARDWARE_SCHEMA,
        "name": "gpu20_cpu80_separate_server_gpu",
        "host": socket.gethostname(),
        "cuda_visible_devices": "0,1",
        "server_device": "cuda:1",
        "client_gpu_pool_size": 1,
        "resource_mode": "emulation",
        "worker_count_policy": "fixed_hardware",
        "worker_counts": worker_counts,
        **({'execution_affinity': execution_cpus} if execution_cpus is not None else {}),
        "profiles": {
            "gpu": {"device": "cuda:0", "torch_threads": 1},
            "cpu_a": {"device": "cpu", "torch_threads": 1, "cpu_limit": 1, "affinity": [cpus[0]]},
            "cpu_b": {"device": "cpu", "torch_threads": 1, "cpu_limit": 1, "affinity": [cpus[1]]},
            **{name: {"device": "cpu", "torch_threads": 1, "cpu_limit": 1, "affinity": [cpu]}
               for name, cpu in zip(names[2:], cpus[2:], strict=True)},
        },
        "client_profiles": assignments,
        "profile_counts": {backend: dict(Counter(values.values())) for backend, values in assignments.items()},
        "scope": (
            "The Ray pool uses the first visible GPU for GPU clients; the second visible GPU "
            "is reserved for the coordinator/private suffix. CPU affinity is actually applied "
            "and restored per logical task. Each of the 50 logical clients retains a private state; the "
            "explicit fixed worker pool executes every client. "
            "The uniform Ray GPU reservation is 1 / configured workers. "
            "GPU fractions are scheduling reservations."
        ),
    }


def apply_hardware_profile(config, path):
    """Apply the declared device assignment and exact worker population."""
    path = Path(path).resolve()
    profile = read_json(path)
    if profile.get("schema") != HARDWARE_SCHEMA:
        raise ValueError("Unknown heterogeneous hardware profile schema")
    backend = config.workload["backend"]
    population = BACKENDS[backend]
    assignments = profile["client_profiles"][backend]
    if (set(assignments) != {str(cid) for cid in range(population)}
            or config.virtual_clients.total != population
            or profile["server_device"] != "cuda:1"
            or profile["cuda_visible_devices"] != "0,1"
            or profile["client_gpu_pool_size"] != 1):
        raise ValueError("Hardware profile does not assign the declared population and two separate GPUs")
    if profile.get("worker_count_policy") != "fixed_hardware":
        raise ValueError("Hardware requires an explicit fixed worker policy")
    expected_workers = profile.get("worker_counts", {}).get(backend)
    if type(expected_workers) is not int or not 3 <= expected_workers <= population:
        raise ValueError("Frozen hardware and runtime worker population differ")
    profiles = {name: Profile(**value) for name, value in profile["profiles"].items()}
    profiles["gpu"] = replace(profiles["gpu"], gpu_fraction=config.resources.client_gpus_per_worker)
    if (profiles.get("gpu") is None or profiles["gpu"].device != "cuda:0"
            or any(profiles[name].device != "cpu" for name in ("cpu_a", "cpu_b"))):
        raise ValueError("Heterogeneous client profiles require genuine GPU0 and CPU execution")
    if any(name not in profiles for name in assignments.values()):
        raise ValueError("Hardware profile contains an unknown logical client group")
    selected = {str(cid): assignments[str(cid)] for cid in range(config.virtual_clients.total)}
    identity = {"path": str(path), "sha256": hash_file(path), "name": profile["name"]}
    return replace(
        config, server_device=profile["server_device"], profiles=profiles, client_profiles=selected,
        resources=replace(config.resources, client_gpu_pool_size=profile["client_gpu_pool_size"],
                          mode=profile["resource_mode"], worker_count_policy="fixed_hardware",
                          max_concurrent_clients=expected_workers,
                          client_cpus_per_worker=1.0),
        workload={**config.workload, "hardware_profile": identity},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-cores", type=int, nargs="+")
    parser.add_argument('--execution-cpu-cores', type=int, nargs='+')
    parser.add_argument("--workers", type=int, default=13,
                        help="Fixed physical worker pool for the 50 logical clients")
    args = parser.parse_args()
    allowed = sorted(os.sched_getaffinity(0))
    cpus = args.cpu_cores or allowed[:2]
    if not set(cpus).issubset(allowed):
        parser.error("CPU cores must be within this host's actual affinity")
    if args.execution_cpu_cores and not set(args.execution_cpu_cores).issubset(allowed):
        parser.error('Execution CPU pool must be within the host affinity')
    worker_counts = {"pytorch": args.workers}
    profile = hardware_profile(cpus=cpus, worker_counts=worker_counts, execution_cpus=args.execution_cpu_cores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x"):
        pass
    save_json(args.output, profile)
    print(json.dumps({"status": "hardware_frozen", "file": str(args.output.resolve()),
                      "sha256": hash_file(args.output), "profile_counts": profile["profile_counts"]}))


if __name__ == "__main__":
    main()


def gpu_snapshot():
    output = subprocess.check_output(["nvidia-smi", "--id=0,1", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True, timeout=10)
    devices = {}
    for line in output.splitlines():
        index, uuid, name, total, used, utilization = [value.strip() for value in line.split(",")]
        devices[index] = {"uuid": uuid, "name": name, "memory_total_mib": int(total), "memory_used_mib": int(used),
                          "utilization_percent": int(utilization), "compute_processes": []}
    if set(devices) != {"0", "1"} or len({row["uuid"] for row in devices.values()}) != 2:
        raise ValueError("Control requires both distinct physical GPUs 0 and 1")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True, timeout=10)
    for line in apps.splitlines():
        uuid, pid, used = [value.strip() for value in line.split(",")]
        for value in devices.values():
            if value["uuid"] == uuid:
                value["compute_processes"].append({"pid": int(pid), "used_memory_mib": int(used) if used.isdigit() else used})
    return devices
