"""Run synthetic split cost calibration on all six real Orin devices."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
from pathlib import Path
import shlex
import subprocess
from experiments.common.physical_workers import physical_workers


def run(args, *, log=None, timeout=600):
    return subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT,
                          timeout=timeout)


def profile_host(host, workers, bundles, output, log_dir, task="object_detection"):
    identity = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:12]
    remote = f"/tmp/splitfleet_device_profile_{output.name}_{identity}"
    ssh = host["ssh"]
    run(["ssh", ssh, "mkdir -p " + shlex.quote(remote + "/bundles")])
    run(["rsync", "-az", "--exclude=__pycache__", "splitfleet", "experiments",
         f"{ssh}:{remote}/"], timeout=120)
    for worker in workers:
        name, device = worker["identity"], worker["device"]
        bundle = bundles / f"{task}.client_{worker['index']}.pt"
        run(["scp", "-q", str(bundle), f"{ssh}:{remote}/bundles/{bundle.name}"], timeout=240)
        target = remote + "/" + name + ".json"
        command = ("cd " + shlex.quote(remote) + " && PYTHONPATH=" + shlex.quote(remote) + " "
                   + shlex.join([host["python"], "-m", "experiments.profile_split_execution",
                                   "--bundle", remote + "/bundles/" + bundle.name,
                                   "--device", device, "--output", target]))
        with (log_dir / (name + ".log")).open("w") as log:
            run(["ssh", ssh, command], log=log, timeout=600)
        run(["scp", "-q", f"{ssh}:{target}", str(output / (name + ".json"))], timeout=120)
        print(name, json.loads((output / (name + ".json")).read_text())["calibration_ms"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, default=Path("logs/device_split_profiles"))
    parser.add_argument("--task", default="object_detection")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_root / args.output.name
    log_dir.mkdir(parents=True, exist_ok=True)
    hosts = json.loads(args.deployment.read_text())["hosts"]
    workers = physical_workers(hosts)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(profile_host, host, [w for w in workers if w["host"] == host],
                              args.bundles, args.output, log_dir, args.task): host["id"]
                   for host in hosts}
        for future in as_completed(futures):
            future.result()
    profiles = {"server": json.loads((args.output / "server.json").read_text())}
    for worker in workers:
        name = worker["identity"]
        profiles[name] = json.loads((args.output / (name + ".json")).read_text())
    graph = profiles["server"]["graph_signature"]
    if any(profile["graph_signature"] != graph for profile in profiles.values()):
        raise RuntimeError("Different physical hosts captured different graphs")
    data = {"schema": "splitfleet.device-cost-profiles.v1", "graph_signature": graph,
            "profiles": profiles}
    (args.output / "device_profiles.json").write_text(json.dumps(data) + "\n")
    print(args.output / "device_profiles.json")


if __name__ == "__main__":
    main()
