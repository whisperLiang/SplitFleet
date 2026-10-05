"""Run coordinator/device admission and serial four-task training.

Every run uses a new output directory and preserves failed attempts.
"""

from __future__ import annotations

import argparse
import copy
from importlib import metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from experiments.www2027_study import save_json
from experiments.unified_multitask.edge_models import file_sha256
from experiments.common.physical_workers import physical_workers
from experiments.common.ssh_transport import SshReverseTransport


def remote(host, command, *, timeout=900):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host["ssh"], command],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout).stdout


def environment_setup_code(edge_venv: str) -> str:
    # The script runs with the original remote interpreter. chr(10) avoids
    # putting a literal backslash-n into the executable .pth import line.
    return (
        "from pathlib import Path;import subprocess,sys,sysconfig;"
        f"target=Path({edge_venv!r});"
        "subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(target)],check=True);"
        "site=target/'lib'/('python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor))/'site-packages';"
        "(site/'splitfleet_base.pth').write_text('import site; site.addsitedir('+repr(sysconfig.get_paths()['purelib'])+')'+chr(10));"
        "subprocess.run([str(target/'bin/python'),'-m','pip','install','--disable-pip-version-check','--no-deps','transformers==5.8.1'],check=True)"
    )


def run(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    admission = args.admission.resolve()
    runtime = admission / "runtime_snapshot"
    manifest = json.loads((admission / "runtime_manifest.json").read_text())
    plan = json.loads(args.plan.read_text())
    required_coordinator_attempts = 3 * len(plan.get("resources", {}).get("models", {})) or 12
    config = json.loads(Path(plan["deployment"]).read_text())
    status = {"schema": "splitfleet.edge-study-queue.v1", "status": "waiting",
        "models": list(plan.get("resources", {}).get("models", {})),
        "worker_ids_by_model": {stage["model_id"]: stage.get("worker_ids") for stage in plan.get("stages", [])},
        "source_identity": manifest["source_identity"], "admission_root": str(admission),
        "plan_file": str(args.plan.resolve()),
        "wait_for_ledger": str(args.wait_for.resolve()), "pid": os.getpid(), "started_unix": time.time(),
        "queue_driver_sha256": file_sha256(Path(__file__)),
        "attempts": [], "failure_policy": "record failure and stop; no automatic retry"}
    save = lambda: save_json(root / "queue.json", status)
    save()
    transport = None
    env = dict(os.environ, PYTHONPATH=str(runtime), OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")

    def logged(name, command, *, cwd=runtime, timeout=14400):
        attempt = {"name": name, "status": "running", "command": command, "started_unix": time.time()}
        status["attempts"].append(attempt)
        save()
        path = root / "logs" / (name + ".log")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("w") as log:
                result = subprocess.run(command, env=env, cwd=cwd, stdout=log,
                                          stderr=subprocess.STDOUT, timeout=timeout)
            attempt.update(exit_code=result.returncode, status="passed" if result.returncode == 0 else "failed")
        except Exception as exc:
            attempt.update(status="failed", error=str(exc))
            raise
        finally:
            attempt["finished_unix"] = time.time()
            save()
        if attempt["status"] != "passed":
            raise RuntimeError(f"{name} failed; see {path}")

    try:
        # Admission can still be running locally when this queue is created.
        incomplete_gate_since = None
        while True:
            local = admission / "coordinator_attempts.json"
            try:
                previous = json.loads(args.wait_for.read_text())
                attempts = json.loads(local.read_text()) if local.is_file() else []
            except json.JSONDecodeError as exc:
                # Older admission runners truncate before writing their next
                # snapshot. Wait for valid metadata; never repeat a model job
                # or treat an incomplete gate as passed. Persistent corruption
                # still stops the queue instead of waiting without a bound.
                now = time.monotonic()
                if incomplete_gate_since is None:
                    incomplete_gate_since = now
                status["incomplete_gate_reads"] = status.get("incomplete_gate_reads", 0) + 1
                status["last_gate_read_error"] = str(exc)
                save()
                if now - incomplete_gate_since > 60:
                    raise RuntimeError("Admission metadata remains invalid for 60 seconds") from exc
                time.sleep(1)
                continue
            incomplete_gate_since = None
            if len(attempts) == required_coordinator_attempts and all(attempt["status"] != "running" for attempt in attempts):
                if any(attempt["status"] != "passed" for attempt in attempts):
                    raise RuntimeError("Coordinator admission failed; primary comparison is not started")
                if previous["status"] in ("completed", "failed"):
                    break
            # Failed preparation can stop before all model gates are recorded.
            elif attempts and attempts[-1]["status"] == "failed":
                raise RuntimeError("Coordinator admission failed; primary comparison is not started")
            time.sleep(10)
        status["status"] = "device_admission"
        save()
        # Isolated environments pin the same capture dependencies on all hosts.
        for host in config["hosts"]:
            original_python = host["python"]
            edge_venv = host["workdir"] + "/.venv-edge-standard"
            code = environment_setup_code(edge_venv)
            output = remote(host, shlex.join([original_python, "-c", code]))
            (root / (host["id"] + ".environment.log")).write_text(output)
            host["python"] = edge_venv + "/bin/python"
            version_code = "import importlib.metadata as m,json;print(json.dumps({p:m.version(p) for p in ['torch','torchvision','torchlens','transformers','tokenizers','rfdetr']}))"
            versions = json.loads(remote(host, shlex.join([host["python"], "-c", version_code])))
            if versions["transformers"] != "5.8.1" or versions["torchlens"] != "2.34.1":
                raise RuntimeError("A primary device does not match the pinned capture dependencies")
            save_json(root / (host["id"] + ".packages.json"), versions)
        deployment = root / "deployment.json"
        save_json(deployment, config)
        if config.get("transport", {}).get("kind") == "ssh_reverse_tunnel":
            transport = SshReverseTransport(config, root / "transport_receipt.json")
            transport.start()
            status["transport_receipt"] = str(root / "transport_receipt.json")
            save()
        plan["deployment"] = str(deployment)
        plan["source_root"] = str(runtime)
        plan["source_identity"] = manifest["source_identity"]
        model_order = plan.get("model_execution_order", list(plan["resources"]["models"]))
        if len(model_order) != len(plan["resources"]["models"]) or set(model_order) != set(plan["resources"]["models"]):
            raise ValueError("Invalid frozen model execution order")
        for name in model_order:
            resource = plan["resources"]["models"][name]
            task = resource["task"]
            bundle_root = admission / name / "bundles"
            model_config = copy.deepcopy(config)
            if name in plan.get("model_worker_devices", {}):
                for host in model_config["hosts"]:
                    host["workers"] = list(plan["model_worker_devices"][name])
            workers = physical_workers(model_config["hosts"])
            model_deployment = root / "deployments" / (name + ".json")
            save_json(model_deployment, model_config)
            for stage in plan["stages"]:
                if stage["model_id"] == name:
                    if stage.get("worker_ids") and stage["worker_ids"] != [w["identity"] for w in workers]:
                        raise ValueError("Frozen model plan and deployment worker identities differ")
                    stage["deployment"] = str(model_deployment)
            for host in model_config["hosts"]:
                remote_root = f"/tmp/splitfleet_edge_admission_{root.name}_{name}"
                remote(host, "mkdir -p " + shlex.quote(remote_root + "/bundles"))
                logged(name + "_" + host["id"] + "_sources", ["rsync", "-az", "--exclude=__pycache__",
                    str(runtime / "splitfleet"), str(runtime / "experiments"), f"{host['ssh']}:{remote_root}/"])
                for worker in [w for w in workers if w["host"] == host]:
                    identity, kind, device = worker["identity"], worker["kind"], worker["device"]
                    bundle = bundle_root / f"{task}.client_{worker['index']}.pt"
                    logged(name + "_" + identity + "_bundle", ["scp", "-q", str(bundle),
                        f"{host['ssh']}:{remote_root}/bundles/{bundle.name}"])
                    target = remote_root + "/" + identity + ".json"
                    arguments = [host["python"], "-m", "experiments.edge_model_admission",
                        "--bundle", remote_root + "/bundles/" + bundle.name, "--device", device, "--output", target]
                    if kind == "gpu":
                        # Controlled allocator failure protects the Jetson OS.
                        # It is reported as a budget rejection, not a measured
                        # FL/SFL failure or evidence of physical infeasibility.
                        arguments += ["--cuda-memory-fraction", "0.5"]
                    command = "cd " + shlex.quote(remote_root) + " && " + shlex.join(arguments)
                    destination = root / "admission" / name / (identity + ".json")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        logged(name + "_" + identity + "_admit", ["ssh", "-o", "BatchMode=yes", host["ssh"], command])
                    finally:
                        # Preserve failed model receipts as well as their log.
                        receipt = subprocess.run(["scp", "-q", f"{host['ssh']}:{target}", str(destination)],
                                                  check=False, timeout=120)
                        status["attempts"].append({"name": name + "_" + identity + "_receipt",
                            "status": "passed" if receipt.returncode == 0 else "failed",
                            "exit_code": receipt.returncode, "finished_unix": time.time()})
                        save()
        status["status"] = "training"
        save()
        frozen_plan = root / "primary_plan.json"
        save_json(frozen_plan, plan)
        for name in model_order:
            logged(name + "_timing_execution", [sys.executable, "-m", "experiments.www2027_study", "run",
                "--plan", str(frozen_plan), "--root", str(root / "study" / name / "timing"),
                "--stages", name + "_timing"], timeout=604800)
        status.update(status="timing_completed", quality_status="reported at fixed work budget; no convergence claim")
    except Exception as exc:
        status.update(status="failed", error=str(exc))
        raise
    finally:
        if transport is not None:
            transport.close()
        status["updated_unix"] = time.time()
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
