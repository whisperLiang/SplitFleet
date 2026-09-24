"""Start and validate one CoSplit-UCB run across configured physical hosts."""

from __future__ import annotations

import argparse
import json
import shlex
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from .validate_run import validate_run


MODULE = "experiments.physical_cosplit_ucb.run"


def _powershell_quote(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _common_args(config: Mapping[str, Any], num_clients: int) -> list[str]:
    args = ["--num-clients", str(num_clients)]
    for name in (
        "model", "seed", "batch_size", "local_epochs", "learning_rate",
        "momentum", "weight_decay", "dirichlet_alpha", "data_root",
    ):
        if name in config:
            args.extend(("--" + name.replace("_", "-"), str(config[name])))
    if config.get("download", False):
        args.append("--download")
    return args


def build_commands(config: Mapping[str, Any], *, run_id: str, output: Path) -> tuple[list[str], list[list[str]]]:
    """Return local server and SSH client commands without starting processes."""

    server = config["server"]
    clients = config["clients"]
    if not isinstance(clients, list) or len(clients) < 2:
        raise ValueError("deployment must list at least two physical clients")
    ids = [str(client["id"]) for client in clients]
    if len(set(ids)) != len(ids):
        raise ValueError("physical client IDs must be unique")
    experiment = config.get("experiment", {})
    common = _common_args(experiment, len(clients))
    server_command = [
        str(server["python"]), "-m", MODULE, "server", *common,
        "--bind", str(server["bind"]),
        "--device", str(server.get("device", "auto")),
        "--run-id", run_id,
        "--rounds", str(experiment.get("rounds", 3)),
        "--server-concurrency", str(experiment.get("server_concurrency", 1)),
        "--max-candidates", str(experiment.get("max_candidates", 8)),
        "--evaluation-batch-size", str(experiment.get("evaluation_batch_size", 256)),
        "--output", str(output),
    ]
    client_commands = []
    for index, client in enumerate(clients):
        role = [
            str(client["python"]), "-m", MODULE, "client", *common,
            "--server", str(server["connect"]),
            "--device", str(client.get("device", "auto")),
            "--client-id", str(client["id"]),
            "--client-index", str(index),
        ]
        shell = client.get("shell", "posix")
        if shell == "posix":
            remote = "cd " + shlex.quote(str(client["workdir"])) + " && " + shlex.join(role)
        elif shell == "powershell":
            script = (
                "Set-Location -LiteralPath " + _powershell_quote(client["workdir"])
                + "; & " + _powershell_quote(role[0]) + " "
                + " ".join(_powershell_quote(value) for value in role[1:])
                + "; exit $LASTEXITCODE"
            )
            remote = 'powershell -NoProfile -NonInteractive -Command "' + script + '"'
        else:
            raise ValueError(f"unsupported remote shell: {shell!r}")
        client_commands.append([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            str(client["ssh"]), remote,
        ])
    return server_command, client_commands


def _wait_for_server(address: str, process: subprocess.Popen, timeout: float) -> None:
    host, port_text = address.rsplit(":", 1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before listening (code {process.returncode})")
        try:
            with socket.create_connection((host, int(port_text)), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"server did not listen at {address} within {timeout:g}s")


def _validate_client_starts(config: Mapping[str, Any], *, run_id: str, output: Path) -> None:
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    expected_hash = result["partition_hash"]
    for client in config["clients"]:
        log_path = output.parent / f"{run_id}.{client['id']}.log"
        starts = []
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "client_start":
                starts.append(row)
        if len(starts) != 1 or starts[0].get("client_id") != client["id"]:
            raise RuntimeError(f"client {client['id']} has no matching startup record")
        if starts[0].get("partition_hash") != expected_hash:
            raise RuntimeError(f"client {client['id']} used a different data partition")


def run_deployment(config: Mapping[str, Any], *, run_id: str, output: Path, startup_timeout: float, timeout: float) -> dict[str, Any]:
    output = output.resolve()
    server_command, client_commands = build_commands(config, run_id=run_id, output=output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # The server owns output/result.json and requires the run directory to be new.
    if output.exists():
        raise FileExistsError(output)
    server_host = config["server"]
    client_log_files = []
    client_processes: list[subprocess.Popen] = []
    started = time.time()
    server_process: subprocess.Popen | None = None
    try:
        server_log = output.parent / f"{run_id}.server.log"
        with server_log.open("w", encoding="utf-8") as stream:
            server_process = subprocess.Popen(
                server_command, cwd=str(server_host["workdir"]),
                stdout=stream, stderr=subprocess.STDOUT,
            )
            _wait_for_server(str(server_host["connect"]), server_process, startup_timeout)
            for client, command in zip(config["clients"], client_commands):
                log_path = output.parent / f"{run_id}.{client['id']}.log"
                log_stream = log_path.open("w", encoding="utf-8")
                client_log_files.append(log_stream)
                client_processes.append(subprocess.Popen(command, stdout=log_stream, stderr=subprocess.STDOUT))
            deadline = time.monotonic() + timeout if timeout > 0 else None
            while server_process.poll() is None:
                failed = [
                    (config["clients"][index]["id"], process.returncode)
                    for index, process in enumerate(client_processes)
                    if process.poll() is not None and process.returncode != 0
                ]
                if failed:
                    raise RuntimeError(f"physical client exited with error: {failed}")
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("physical run exceeded its time limit")
                time.sleep(2)
            if server_process.returncode != 0:
                raise RuntimeError(f"server exited with code {server_process.returncode}")
            for process in client_processes:
                process.wait(timeout=30)
            if any(process.returncode != 0 for process in client_processes):
                raise RuntimeError("one or more physical clients exited with an error")
        _validate_client_starts(config, run_id=run_id, output=output)
        report = validate_run(output)
        if not report["valid"]:
            raise RuntimeError("physical result failed validation: " + "; ".join(report["errors"]))
        return {"run_id": run_id, "valid": True, "result": str(output / "result.json")}
    finally:
        for process in client_processes:
            if process.poll() is None:
                process.terminate()
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
        for stream in client_log_files:
            stream.close()
        manifest = {
            "schema": "splitfleet.physical-cosplit-ucb-orchestration.v1",
            "run_id": run_id,
            "started_unix": started,
            "finished_unix": time.time(),
            "server_command": server_command,
            "client_commands": client_commands,
            "server_exit_code": server_process.poll() if server_process is not None else None,
            "client_exit_codes": [process.poll() for process in client_processes],
        }
        (output.parent / f"{run_id}.orchestration.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/physical_cosplit_ucb"))
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--timeout", type=float, default=0, help="whole-run seconds; 0 has no limit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.deployment.read_text(encoding="utf-8"))
    output = (args.output_root / args.run_id).resolve()
    if args.dry_run:
        server, clients = build_commands(config, run_id=args.run_id, output=output)
        print(json.dumps({"server": server, "clients": clients}, indent=2))
        return
    print(json.dumps(run_deployment(
        config, run_id=args.run_id, output=output,
        startup_timeout=args.startup_timeout, timeout=args.timeout,
    ), indent=2))


if __name__ == "__main__":
    main()
