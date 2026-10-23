"""Freeze, admit and execute the three backends on the three existing Orins."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

from .common import BACKENDS, HOST_SUFFIXES, LOGICAL_CLIENTS, METHODS, file_hash, prepare, save_json, validate_methods
from .status import campaign_process_alive

SERVER_PYTHON = {"tensorflow": ".venv/bin/python", "jax": ".venv-jax-gpu/bin/python",
                 "paddle": ".venv-paddle-gpu/bin/python"}
SERVER_DEVICES = {backend: "cuda:0" for backend in BACKENDS}
ENVIRONMENT = {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "2",
               "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1",
               "TF_CPP_MIN_LOG_LEVEL": "2", "TF_ENABLE_ONEDNN_OPTS": "0",
               "TF_FORCE_GPU_ALLOW_GROWTH": "true", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
               "NVIDIA_TF32_OVERRIDE": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
               "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def ssh(host, command, *, timeout=120, **kwargs):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
                           shlex.join(command)], check=True, timeout=timeout, **kwargs)


def prepare_campaign(root, *, methods=("splitfleet",), placement="greedy", **settings):
    methods = validate_methods(methods)
    if placement not in {"greedy", "fixed_stage2"}:
        raise ValueError("Choose measured per-client Greedy or the explicit fixed-stage control")
    workspace = Path(__file__).resolve().parents[2]
    reference = workspace / "results/figure3_splitfleet_optimized_20261017/inputs/resnet50_pretrained.deployment.json"
    reference_deployment = json.loads(reference.read_text())
    if reference_deployment["server"]["device"] != "cuda:0":
        raise ValueError("Figure 3 hardware reference no longer specifies the A6000 GPU0 server")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    prepare(root / "bundle", **settings)
    protocol_path = root / "bundle" / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["placement_policy"] = placement
    save_json(protocol_path, protocol)
    snapshot = root / "runtime_snapshot"
    snapshot.mkdir()
    for folder in ("splitfleet", "experiments"):
        shutil.copytree(workspace / folder, snapshot / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in ("tests/__init__.py", "tests/integration/__init__.py"):
        destination = snapshot / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / path, destination)
    shutil.copytree(workspace / "tests/integration/resnet50_reference",
                    snapshot / "tests/integration/resnet50_reference",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    wheel = "torchlens-2.34.1-py3-none-any.whl"
    shutil.copyfile(workspace / wheel, snapshot / wheel)
    hashes = {str(p.relative_to(snapshot)): file_hash(p) for p in sorted(snapshot.rglob("*.py"))}
    source_identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    remote_root = "/home/nvidia/SplitFleet/runs/" + root.name
    hosts = [{"id": "orin" + suffix, "ssh": f"nvidia@192.168.66.{suffix}",
              "root": remote_root} for suffix in HOST_SUFFIXES]
    clients = [{**specification, **{key: next(host for host in hosts if host["id"] == specification["host_id"])[key]
                                   for key in ("ssh", "root")}} for specification in LOGICAL_CLIENTS]
    manifest = {"schema": "splitfleet.resnet50.physical-campaign.v2", "source_identity": source_identity,
                "source_hashes": hashes, "wheel_sha256": file_hash(snapshot / wheel),
                "hosts": hosts, "clients": clients, "workspace": str(workspace), "backends": list(BACKENDS),
                "client_devices": {name: [client["device"] for client in clients] for name in BACKENDS},
                "server_devices": dict(SERVER_DEVICES),
                "server": {"host": socket.gethostname(), "identity": reference_deployment["server"]["identity"],
                           "model": "NVIDIA RTX A6000", "device": "cuda:0", "cuda_visible_devices": "0",
                           "server_concurrency": 1},
                "figure3_hardware_reference": {"path": str(reference), "sha256": file_hash(reference),
                    "scope": "Client/server hardware and C=1 only; this experiment retains its own 50-round training protocol",
                    "server": reference_deployment["server"], "hosts": reference_deployment["hosts"]},
                "seed_count": 1, "method_order": methods,
                "backend_runtime": {
                    "jax": {"cpu_version": "0.6.2", "orin_gpu_version": "0.5.1", "orin_dependency_overlay": "/home/nvidia/SplitFleet/.jax-051-py311-20261022",
                            "xla_gpu_autotune_level": 0, "orin_gpu_jit": False,
                            "training_programs": "cached_native_jaxpr_eager_without_fusion",
                            "cuda_libraries": "JetPack system libraries"},
                    "paddle": {"orin_version": "3.4.0.dev20260119", "orin_dependency_overlay": "/home/nvidia/SplitFleet/.paddle-jetson-py310-20261022",
                               "wheel_source": "https://github.com/SidaWang12/paddle3.0-jetson-jetpack6-build/releases/download/0.0.0/paddlepaddle_gpu-3.4.0.dev20260119-cp310-cp310-linux_aarch64.whl",
                               "coordinator_version": "3.3.1", "build_scope": "community JetPack build; native full/split training admission required"}},
                "placement_policy": placement,
                "placement": ("SplitFleet measured Greedy; independent per-client operation cuts each round"
                              if placement == "greedy" else "fixed native residual stage2 control"),
                "claim_scope": "backend training correctness and descriptive convergence; no cross-device speed ranking"}
    save_json(root / "campaign.json", manifest)
    return manifest


def stage(root, manifest):
    snapshot = root / "runtime_snapshot"
    for name, digest in manifest["source_hashes"].items():
        if file_hash(snapshot / name) != digest:
            raise RuntimeError("Frozen campaign source changed")

    def one(host):
        remote = host["root"]
        ssh(host["ssh"], ["mkdir", "-p", remote + "/bundle", remote + "/admission"])
        for folder in ("splitfleet", "experiments", "tests"):
            subprocess.run(["rsync", "-a", str(snapshot / folder), host["ssh"] + ":" + remote + "/"],
                           check=True, timeout=180)
        files = [root / "bundle" / name for name in ("protocol.json", "initial.npz")]
        files += [root / "bundle" / f"client_{client['index']}.npz"
                  for client in manifest["clients"] if client["host_id"] == host["id"]]
        subprocess.run(["rsync", "-a", *map(str, files), host["ssh"] + ":" + remote + "/bundle/"],
                       check=True, timeout=300)
        wheel = snapshot / "torchlens-2.34.1-py3-none-any.whl"
        subprocess.run(["rsync", "-a", str(wheel), host["ssh"] + ":" + remote + "/"], check=True, timeout=120)
        ssh(host["ssh"], ["/home/nvidia/.local/bin/uv", "pip", "install", "--target", remote + "/dependencies",
                         "--no-deps", remote + "/" + wheel.name])
        code = ("import pathlib,hashlib,json; root=pathlib.Path(" + repr(remote) + "); "
                "print(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() "
                "for folder in ('splitfleet','experiments','tests') for p in sorted((root/folder).rglob('*.py'))}))")
        result = ssh(host["ssh"], ["python3", "-c", code], capture_output=True, text=True)
        actual = json.loads(result.stdout)
        if actual != manifest["source_hashes"]:
            raise RuntimeError(f"Frozen code verification failed on {host['id']}")
        save_json(root / (host["id"] + ".source_hashes.json"), actual)
        return {"host": host["id"], "status": "verified", "source_identity": manifest["source_identity"]}

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(one, manifest["hosts"]))
    save_json(root / "staging.json", results)


def command(root, manifest, backend, role, *, output, method="splitfleet", host=None, address=None):
    remote = host is not None
    base = host["root"] if remote else str(root / "runtime_snapshot")
    bundle = base + "/bundle" if remote else str(root / "bundle")
    if remote:
        python = ("/home/nvidia/SplitFleet/.venv-real-device/bin/python" if backend in {"tensorflow", "paddle"}
                  else "/home/nvidia/SplitFleet/.venv-backends-20261022/bin/python")
    else:
        python = str(Path(manifest["workspace"]) / SERVER_PYTHON[backend])
    environment = {**ENVIRONMENT, "PYTHONPATH": base + (":" + base + "/dependencies" if remote else "")}
    device = host["device"] if remote else manifest["server_devices"][backend]
    environment["CUDA_VISIBLE_DEVICES"] = "-1" if device == "cpu" else "0"
    if remote:
        environment.update(SPLITFLEET_CLIENT_ID=host["id"], SPLITFLEET_PHYSICAL_HOST=host["host_id"])
    elif "server" in manifest:
        environment.update(SPLITFLEET_CLIENT_ID="server", SPLITFLEET_PHYSICAL_HOST=manifest["server"]["host"])
    if backend == "jax":
        # The GPU is mandatory, while native debug callbacks also need a
        # host CPU device. Model/input state placement is checked separately.
        environment["JAX_PLATFORMS"] = "cpu" if device == "cpu" else "cuda,cpu"
        if remote and device.startswith("cuda"):
            environment["PYTHONPATH"] += ":/home/nvidia/SplitFleet/.jax-051-py311-20261022"
            # Native eager execution passes the fixed gradient gate on the
            # JetPack build; compiled JAX 0.5.1 replay exceeds that tolerance.
            environment["JAX_DISABLE_JIT"] = "1"
        # Wheel autotuning helper kernels omit Orin sm_87. Keep deterministic
        # operators and native JetPack cuDNN, with autotuning disabled.
        environment["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
        if remote:
            environment["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu"
            environment["PATH"] = "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
    if backend == "paddle" and remote:
        environment["PYTHONPATH"] += ":/home/nvidia/SplitFleet/.paddle-jetson-py310-20261022"
        environment["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu"
    args = [python, "-u", "-m", "experiments.resnet50_backends.worker", role,
            "--backend", backend, "--bundle", bundle, "--device", device, "--output", str(output),
            "--method", method, "--placement", manifest.get("placement_policy", "fixed_stage2")]
    if remote:
        args += ["--index", str(host["index"]), "--identity", host["id"], "--physical-host", host["host_id"]]
    if role != "admit":
        admission = (base + "/admission/" + backend + "_" + host["id"] + ".json" if remote else
                     str(root / "admission" / backend / "server.json"))
        args += ["--admission", admission, "--address", address]
    args = ["env", *(key + "=" + value for key, value in environment.items()), *args]
    if remote:
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
                host["ssh"], shlex.join(args)]
    return args


def admissions(root, manifest, backend):
    directory = root / "admission" / backend
    directory.mkdir(parents=True, exist_ok=False)

    def one(host):
        identity = host["id"] if host else "server"
        output = host["root"] + "/admission/" + backend + "_" + identity + ".json" if host else str(directory / "server.json")
        args = command(root, manifest, backend, "admit", output=output, host=host)
        with (directory / (identity + ".log")).open("w") as stream:
            result = subprocess.run(args, cwd=root / "runtime_snapshot", stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=3600)
        if host:
            subprocess.run(["rsync", "-a", host["ssh"] + ":" + output, str(directory / (identity + ".json"))],
                           check=True, timeout=60)
        receipt = json.loads((directory / (identity + ".json")).read_text())
        if result.returncode or receipt.get("status") != "passed":
            raise RuntimeError(f"{backend} native correctness admission failed on {identity}; inspect {directory}")
        return {"host": identity, "status": "passed"}

    # Orin shares 8 GB between CPU and GPU. Admit one worker per physical
    # host at a time; the training smoke checks their concurrent residency.
    results = [one(None)]
    for device in ("cpu", "cuda:0"):
        with ThreadPoolExecutor(max_workers=3) as pool:
            results += list(pool.map(one, [client for client in manifest["clients"] if client["device"] == device]))
    save_json(directory / "summary.json", results)
    return results


def free_ports():
    handles = []
    try:
        for _ in range(2):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            handles.append(sock)
        return [sock.getsockname()[1] for sock in handles]
    finally:
        for sock in handles:
            sock.close()


def run_method(root, manifest, backend, method):
    from experiments.common.ssh_transport import SshReverseTransport
    directory = root / "runs" / backend / method
    directory.mkdir(parents=True, exist_ok=False)
    port, barrier = free_ports()
    address = f"127.0.0.1:{port}"
    deployment = {"hosts": manifest["hosts"], "server": {"bind": address, "connect": address},
                  "barrier": {"bind": f"127.0.0.1:{barrier}", "connect": f"127.0.0.1:{barrier}"},
                  "transport": {"kind": "ssh_reverse_tunnel", "cipher": "aes128-gcm@openssh.com"}}
    transport = SshReverseTransport(deployment, directory / "transport.json")
    processes, streams = [], []
    result = {"status": "starting", "backend": backend, "method": method, "processes": [],
              "started_unix": time.time()}

    def launch(host=None):
        identity = host["id"] if host else "server"
        output = (host["root"] + f"/{backend}_{method}_{identity}.json" if host else str(directory / "server.json"))
        args = command(root, manifest, backend, "client" if host else "server",
                       output=output, host=host, method=method, address=address)
        stream = (directory / (identity + ".log")).open("w")
        process = subprocess.Popen(args, cwd=root / "runtime_snapshot", stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        streams.append(stream)
        result["processes"].append({"identity": identity, "command": args, "pid": process.pid, "output": output})
        save_json(directory / "execution.json", result)
        return process

    try:
        transport.start()
        server = launch()
        deadline = time.monotonic() + 1800
        while True:
            if server.poll() is not None:
                raise RuntimeError("Server failed during initialization")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server initialization deadline")
                time.sleep(.5)
        for host in manifest["clients"]:
            launch(host)
        result["status"] = "training"
        save_json(directory / "execution.json", result)
        deadline = time.monotonic() + 14 * 86400
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("A physical training process failed; partial results are retained")
            if time.monotonic() > deadline:
                raise TimeoutError("Campaign method deadline exceeded")
            time.sleep(2)
        for host in manifest["clients"]:
            output = host["root"] + f"/{backend}_{method}_{host['id']}.json"
            subprocess.run(["rsync", "-a", host["ssh"] + ":" + output,
                            str(directory / (host["id"] + ".json"))], check=True, timeout=120)
        if manifest.get("placement_policy") == "greedy" and method == "splitfleet":
            for host in manifest["hosts"]:
                destination = directory / "selected_admission" / host["id"]
                destination.mkdir(parents=True, exist_ok=True)
                subprocess.run(["rsync", "-a", host["ssh"] + ":" + host["root"] + "/selected_admission/",
                                str(destination) + "/"], check=True, timeout=120)
        receipts = [json.loads((directory / (name + ".json")).read_text())
                    for name in ("server", *(client["id"] for client in manifest["clients"]))]
        if any(row.get("status") != "completed" for row in receipts):
            raise RuntimeError("The server and all six CPU/GPU clients must confirm full completion")
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        if result["status"] != "completed":
            for host in manifest["clients"]:
                code = ("import pathlib,os,signal; p=pathlib.Path(" + repr(host["root"] + f"/{backend}_{method}_{host['id']}.pid")
                        + "); pid=int(p.read_text()) if p.exists() else None; "
                        "c=pathlib.Path('/proc/'+str(pid)+'/cmdline') if pid else None; "
                        "os.kill(pid,signal.SIGTERM) if c and c.exists() and " + repr(host["root"]) +
                        " in c.read_bytes().decode().replace('\\x00',' ') else None")
                try:
                    ssh(host["ssh"], ["python3", "-c", code], timeout=20, capture_output=True)
                except subprocess.SubprocessError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        for stream in streams:
            stream.close()
        transport.close()
        result.update(finished_unix=time.time(), exit_codes=[p.poll() for p in processes])
        save_json(directory / "execution.json", result)
    return result


def execute(root, backends):
    root = Path(root).resolve()
    manifest = json.loads((root / "campaign.json").read_text())
    methods = validate_methods(manifest["method_order"])
    if "figure3_hardware_reference" in manifest:
        if manifest["server_devices"] != SERVER_DEVICES or manifest["server"]["host"] != socket.gethostname():
            raise RuntimeError("Figure 3 requires every backend's server on this host's A6000 GPU0")
        inventory = subprocess.check_output(["nvidia-smi", "--id=0",
            "--query-gpu=index,name,uuid,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
        if inventory.split(",")[1].strip() != "NVIDIA RTX A6000":
            raise RuntimeError("Physical GPU0 differs from the Figure 3 RTX A6000 server")
        save_json(root / "server_hardware.json", {"host": socket.gethostname(), "device": "cuda:0",
                  "cuda_visible_devices": "0", "gpu_inventory": inventory, "server_concurrency": 1,
                  "observed_unix": time.time(), "figure3_reference": manifest["figure3_hardware_reference"]})
    with (root / "execution_attempt.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "started_unix": time.time(), "backends": backends,
                   "boot_id": boot_id()}, stream)
    status = {"status": "running", "backends": {}, "pid": os.getpid(), "boot_id": boot_id()}
    save_json(root / "status.json", status)
    try:
        stage(root, manifest)
        for backend in backends:
            try:
                status["backends"][backend] = {"phase": "native_admission"}
                save_json(root / "status.json", status)
                admissions(root, manifest, backend)
                for method in methods:
                    status["backends"][backend] = {"phase": method}
                    save_json(root / "status.json", status)
                    run_method(root, manifest, backend, method)
                status["backends"][backend] = {"phase": "completed"}
            except Exception as error:
                # Each backend is an independent physical experiment. Preserve
                # its failure without silently rerunning it or discarding the
                # other requested backend experiments.
                status["backends"][backend].update(phase="failed", error=repr(error))
            save_json(root / "status.json", status)
        from .summarize import summarize
        completed = [name for name in backends if status["backends"][name]["phase"] == "completed"]
        if completed:
            summarize(root, completed, methods=methods)
        status["status"] = "completed" if len(completed) == len(backends) else "failed"
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        save_json(root / "status.json", status)


def _require_prerequisite_process(root, state):
    if campaign_process_alive(root, state):
        return True
    # A coordinator may have committed completion between the status read and
    # its exit. Let the caller handle that final state on its next iteration.
    if json.loads((Path(root) / "status.json").read_text()) != state:
        return False
    raise RuntimeError("Prerequisite campaign process exited or no longer owns the recorded task")


def wait_for_preflight(root, prerequisite):
    """Release a new formal attempt only after its physical smoke completes."""
    state = {"status": "waiting_for_preflight", "pid": os.getpid(), "boot_id": boot_id(),
             "prerequisite": str(Path(prerequisite).resolve()), "backends": {}}
    save_json(Path(root) / "status.json", state)
    try:
        startup_deadline = time.monotonic() + 120
        while True:
            try:
                previous = json.loads((Path(prerequisite) / "status.json").read_text())
            except FileNotFoundError:
                if time.monotonic() >= startup_deadline:
                    raise TimeoutError("Prerequisite preflight did not publish its initial status")
                time.sleep(.5)
                continue
            if previous.get("boot_id") and previous["boot_id"] != boot_id():
                raise RuntimeError("Prerequisite was interrupted by a coordinator reboot")
            if previous["status"] == "completed":
                save_json(Path(root) / "preflight_release.json", {"status": "passed",
                          "prerequisite": str(Path(prerequisite).resolve()), "released_unix": time.time()})
                return
            if previous["status"] not in {"running", "waiting_for_campaign", "waiting_for_preflight"}:
                raise RuntimeError("Prerequisite physical smoke failed or was interrupted")
            if not _require_prerequisite_process(prerequisite, previous):
                continue
            time.sleep(5)
    except BaseException as error:
        state.update(status="not_started", error=repr(error))
        save_json(Path(root) / "status.json", state)
        raise


def wait_for_campaign(root, prerequisite):
    """Wait for an existing cohort to release the physical devices, without treating it as admission."""
    root, prerequisite = Path(root), Path(prerequisite).resolve()
    state = {"status": "waiting_for_campaign", "pid": os.getpid(), "boot_id": boot_id(),
             "prerequisite": str(prerequisite), "backends": {}}
    save_json(root / "status.json", state)
    try:
        startup_deadline = time.monotonic() + 120
        while True:
            try:
                previous = json.loads((prerequisite / "status.json").read_text())
            except FileNotFoundError:
                if time.monotonic() >= startup_deadline:
                    raise TimeoutError("Previous campaign did not publish its initial status")
                time.sleep(.5)
                continue
            if previous.get("boot_id") and previous["boot_id"] != boot_id():
                raise RuntimeError("Previous campaign was interrupted by a host reboot")
            if previous["status"] in {"completed", "failed"}:
                if campaign_process_alive(prerequisite, previous):
                    time.sleep(2)
                    continue
                save_json(root / "device_queue_release.json", {
                    "prerequisite": str(prerequisite), "previous_status": previous["status"],
                    "released_unix": time.time(), "scope": "device availability only; not numerical admission"})
                return
            if previous["status"] not in {"running", "waiting_for_campaign", "waiting_for_preflight"}:
                raise RuntimeError("Previous campaign was stopped; queued training was not dispatched")
            if not _require_prerequisite_process(prerequisite, previous):
                continue
            time.sleep(5)
    except BaseException as error:
        state.update(status="not_started", error=repr(error))
        save_json(root / "status.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", default="data/cifar-10-batches-py")
    p.add_argument("--train-samples", type=int, default=45000)
    p.add_argument("--validation-samples", type=int, default=5000)
    p.add_argument("--test-samples", type=int, default=10000)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=233)
    p.add_argument("--checkpoint-retention", choices=("all", "latest_and_best"), default="all")
    p.add_argument("--methods", choices=METHODS, nargs="+", default=["splitfleet"],
                   help="Methods to execute; defaults to SplitFleet only")
    p.add_argument("--placement", choices=("greedy", "fixed_stage2"), default="greedy",
                   help="SplitFleet measured per-client placement; fixed_stage2 is an explicit control")
    p = sub.add_parser("execute")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--backends", choices=BACKENDS, nargs="+", default=list(BACKENDS))
    p.add_argument("--wait-for", type=Path, help="An existing physical preflight that must complete before dispatch")
    p.add_argument("--after-campaign", type=Path, help="Wait until an existing cohort has finished and released devices")
    args = parser.parse_args()
    if args.command == "prepare":
        values = vars(args).copy()
        values.pop("command")
        output = values.pop("output")
        print(json.dumps(prepare_campaign(output, **values)), flush=True)
    else:
        if args.after_campaign:
            wait_for_campaign(args.root, args.after_campaign)
        if args.wait_for:
            wait_for_preflight(args.root, args.wait_for)
        execute(args.root, args.backends)
        if json.loads((args.root / "status.json").read_text())["status"] != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
