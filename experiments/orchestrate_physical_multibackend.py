"""Execute one real multi-backend CIFAR deployment with retained process receipts."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

from experiments.multibackend_cifar import save_json


def run(deployment, bundle, output, *, timeout=1800):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(deployment).read_text())
    bundle = Path(bundle).resolve()
    experiment = json.loads((bundle / "config.json").read_text())
    if len(config["hosts"]) != experiment["clients"]:
        raise ValueError("Deployment must contain exactly the configured client count")
    source = Path(__file__).resolve().parents[1]
    hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
              for folder in ("splitfleet", "experiments") for path in sorted((source / folder).rglob("*.py"))}
    save_json(output / "source_manifest.json", hashes)
    save_json(output / "deployment.json", config)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "TF_FORCE_GPU_ALLOW_GROWTH": "true",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "JAX_PLATFORMS": "cpu"}
    processes, streams, remote_outputs = [], [], []
    receipt = {"backend": config["backend"], "status": "starting", "started_unix": time.time(),
               "commands": [], "output": str(output), "bundle_config": experiment}
    save_json(output / "execution.json", receipt)
    transport = nullcontext()
    if any(host.get("ssh") for host in config["hosts"]):
        from experiments.common.ssh_transport import SshReverseTransport
        transport = SshReverseTransport(config, output / "transport.json")

    def launch(role, worker, index=None):
        remote = bool(worker.get("ssh"))
        result_path = worker.get("output", str(output / f"{role}_{index if index is not None else 'global'}.json"))
        data = worker.get("bundle", str(bundle))
        command = [worker.get("python", sys.executable), "-u", "-m", "experiments.physical_multibackend", role,
                   "--backend", config["backend"], "--bundle", data, "--device", worker["device"],
                   "--admission", worker["admission"], "--address", config["server"]["bind" if role == "server" else "connect"],
                   "--output", result_path]
        if index is not None:
            command += ["--index", str(index)]
        worker_env = {**env, **worker.get("environment", {})}
        if remote:
            exports = {key:worker_env[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                       "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "TF_FORCE_GPU_ALLOW_GROWTH",
                       "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_PLATFORMS")}
            exports.update(worker.get("environment", {}))
            exports["PYTHONPATH"] = worker["workdir"]
            remote_command = ["env", *(f"{key}={value}" for key,value in exports.items()), *command]
            command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", worker["ssh"], shlex.join(remote_command)]
            remote_outputs.append((worker, result_path, index))
        stream = (output / f"{role}_{index if index is not None else 'global'}.log").open("w")
        process = subprocess.Popen(command, cwd=source, env=worker_env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        processes.append(process)
        streams.append(stream)
        receipt["commands"].append({"role": role, "index": index, "command": command, "pid": process.pid,
                                    "remote": remote, "result": result_path})
        save_json(output / "execution.json", receipt)
        return process

    try:
        if hasattr(transport, "start"):
            transport.start()
        server = launch("server", config["server"])
        address, port = config["server"]["connect"].rsplit(":", 1)
        startup = time.monotonic() + 180
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited during startup ({server.returncode}); see server_global.log")
            try:
                with socket.create_connection((address, int(port)), timeout=1):
                    break
            except OSError:
                if time.monotonic() > startup:
                    raise TimeoutError("Server gRPC port did not become ready")
                time.sleep(.5)
        for index, host in enumerate(config["hosts"]):
            launch("client", host, index)
        deadline = time.monotonic() + timeout
        receipt["status"] = "training"
        save_json(output / "execution.json", receipt)
        print(f"PHYSICAL_TRAINING backend={config['backend']} clients={len(config['hosts'])}", flush=True)
        while any(process.poll() is None for process in processes):
            bad = [process.returncode for process in processes if process.poll() not in (None, 0)]
            if bad:
                raise RuntimeError(f"Physical worker failed with exit code(s) {bad}; see process logs")
            if time.monotonic() > deadline:
                raise TimeoutError("Physical training exceeded its configured timeout")
            time.sleep(1)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError("A physical worker did not exit successfully")
        for worker, path, index in remote_outputs:
            subprocess.run(["rsync", "-a", worker["ssh"] + ":" + path, str(output / f"client_{index}.json")], check=True, timeout=30)
        server_result = json.loads((output / "server_global.json").read_text())
        if server_result.get("status") != "completed":
            raise RuntimeError("Server result does not confirm training completion")
        clients = [json.loads((output / f"client_{i}.json").read_text()) for i in range(experiment["clients"])]
        if any(len(c["rounds"]) != experiment["rounds"] for c in clients):
            raise RuntimeError("Client did not complete all rounds")
        receipt.update(status="completed", final_accuracy=server_result["rounds"][-1]["accuracy"],
                       initial_accuracy=server_result["rounds"][0]["accuracy"],
                       initial_test_loss=server_result["rounds"][0]["test_loss"],
                       final_test_loss=server_result["rounds"][-1]["test_loss"])
        print(json.dumps({key:receipt[key] for key in ("backend", "status", "final_accuracy", "final_test_loss")}), flush=True)
    except Exception as exc:
        receipt.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for worker, path, index in remote_outputs:
            if receipt["status"] != "completed":
                code = "import os,signal; from pathlib import Path; p=Path(" + repr(str(Path(path).with_suffix('.pid'))) + "); os.kill(int(p.read_text()),signal.SIGTERM) if p.exists() else None"
                subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", worker["ssh"],
                                shlex.join([worker["python"], "-c", code])], capture_output=True, timeout=20)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for stream in streams:
            stream.close()
        if hasattr(transport, "close"):
            transport.close()
        receipt.update(finished_unix=time.time(), exit_codes=[process.returncode for process in processes])
        save_json(output / "execution.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    run(args.deployment, args.bundle, args.output, timeout=args.timeout)


if __name__ == "__main__":
    main()
