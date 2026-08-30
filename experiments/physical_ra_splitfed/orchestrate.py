"""Orchestrate one four-client physical RA-SplitFed training run from fy205."""

from __future__ import annotations

import argparse
import json
import shlex
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Sequence

from .run import FEDAVG_METHOD, ProfileGuidedPlacementPolicy
from .validate_run import validate_run


LINUX_CLIENTS = (
    ("orin140", "192.168.66.140", 1),
    ("orin118", "192.168.66.118", 2),
    ("orin238", "192.168.66.238", 3),
)


def _quoted(values: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(value)) for value in values)


def _windows_powershell(command: str) -> str:
    """Return one remote command string for the Windows OpenSSH cmd shell."""
    if '"' in command:
        raise ValueError("Windows PowerShell payload must not contain double quotes")
    return f'powershell -NoProfile -NonInteractive -Command "{command}"'


def _cleanup_remote_clients(port: int, log_dir: Path) -> None:
    """Best-effort cleanup scoped to client commands targeting one run port."""
    target = f"192.168.66.205:{int(port)}"
    with (log_dir / "cleanup.log").open("a", encoding="utf-8") as cleanup_log:
        for _logical_id, address, _index in LINUX_CLIENTS:
            pattern = (
                "[e]xperiments.physical_ra_splitfed.run client --server " + target
            )
            subprocess.run(
                [
                    "ssh", "-o", "BatchMode=yes", f"nvidia@{address}",
                    f"pkill -f {shlex.quote(pattern)} || true",
                ],
                stdout=cleanup_log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
                check=False,
            )
        windows_cleanup = (
            "$targets = Get-CimInstance Win32_Process | Where-Object { "
            "$_.Name -eq 'python.exe' -and "
            "$_.CommandLine -like '*experiments.physical_ra_splitfed.run client*"
            f"{target}*' "
            "}; $targets | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
        )
        subprocess.run(
            [
                "ssh", "-o", "BatchMode=yes", "18514@192.168.66.136",
                _windows_powershell(windows_cleanup),
            ],
            stdout=cleanup_log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )


def _wait_for_port(process: subprocess.Popen, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited before listening with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Server did not listen on port {port} within {timeout}s")


def _common_client_args(args: argparse.Namespace) -> list[str]:
    return [
        "--method", str(args.method),
        "--num-clients", "4",
        "--model", str(args.model),
        "--seed", str(args.seed),
        "--batch-size", str(args.batch_size),
        "--local-epochs", str(args.local_epochs),
        "--learning-rate", str(args.learning_rate),
        "--momentum", str(args.momentum),
        "--weight-decay", str(args.weight_decay),
        "--paced-client-id", "win136",
        "--paced-round-start", str(args.paced_round_start),
        "--paced-round-end", str(args.paced_round_end),
        "--paced-uplink-mbps", str(args.paced_uplink_mbps),
        "--paced-downlink-mbps", str(args.paced_downlink_mbps),
        "--nominal-uplink-mbps", str(args.nominal_uplink_mbps),
        "--nominal-downlink-mbps", str(args.nominal_downlink_mbps),
        "--split-runtime", str(args.split_runtime),
        "--prefix-executor", str(args.prefix_executor),
        "--compile-backend", str(args.compile_backend),
    ]


def run(args: argparse.Namespace) -> Path:
    repo = Path(args.repo).resolve()
    if args.method in ProfileGuidedPlacementPolicy.METHODS and not args.profile_report:
        raise ValueError(f"Method {args.method!r} requires a fresh --profile-report")
    profile_report = (
        (repo / args.profile_report).resolve() if args.profile_report else None
    )
    results_root = (repo / args.results_root).resolve()
    run_dir = results_root / args.run_id
    log_dir = results_root / "logs" / args.run_id
    if run_dir.exists() or log_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run/log directory for {args.run_id}")
    log_dir.mkdir(parents=True)
    module = "experiments.physical_ra_splitfed.run"
    server_command = [
        sys.executable, "-m", module, "server",
        "--bind", f"0.0.0.0:{args.port}",
        "--run-id", args.run_id,
        "--output", str(run_dir),
        "--num-clients", "4",
        "--rounds", str(args.rounds),
        "--method", args.method,
        "--model", args.model,
        "--protocol-file", args.protocol_file,
        "--calibration-rounds", "3",
        "--server-concurrency", str(args.server_concurrency),
        "--seed", str(args.seed),
        "--batch-size", str(args.batch_size),
        "--local-epochs", str(args.local_epochs),
        "--learning-rate", str(args.learning_rate),
        "--momentum", str(args.momentum),
        "--weight-decay", str(args.weight_decay),
        "--paced-client-id", "win136",
        "--paced-round-start", str(args.paced_round_start),
        "--paced-round-end", str(args.paced_round_end),
        "--paced-uplink-mbps", str(args.paced_uplink_mbps),
        "--paced-downlink-mbps", str(args.paced_downlink_mbps),
        "--nominal-uplink-mbps", str(args.nominal_uplink_mbps),
        "--nominal-downlink-mbps", str(args.nominal_downlink_mbps),
        "--device", "cuda:0",
        "--evaluation-batch-size", str(args.evaluation_batch_size),
        "--split-runtime", str(args.split_runtime),
        "--prefix-executor", str(args.prefix_executor),
        "--compile-backend", str(args.compile_backend),
    ]
    if profile_report is not None:
        server_command.extend(["--profile-report", str(profile_report)])
    common = _common_client_args(args)
    processes: list[tuple[str, subprocess.Popen]] = []
    started = time.time()
    try:
        with ExitStack() as stack:
            server_log = stack.enter_context((log_dir / "server.log").open("w", encoding="utf-8"))
            server = subprocess.Popen(
                server_command,
                cwd=repo,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(("server", server))
            _wait_for_port(server, args.port, args.listen_timeout_sec)
            for logical_id, address, index in LINUX_CLIENTS:
                remote = [
                    "cd", "/home/nvidia/SplitFleet", "&&",
                    "./.venv-real-device/bin/python", "-m", module, "client",
                    "--server", f"192.168.66.205:{args.port}",
                    "--client-id", logical_id,
                    "--client-index", str(index),
                    *common,
                    "--device", "cuda:0",
                ]
                # Keep the shell operator literal while quoting every argument.
                remote_command = (
                    "cd /home/nvidia/SplitFleet && "
                    + _quoted(remote[3:])
                )
                log = stack.enter_context((log_dir / f"{logical_id}.log").open("w", encoding="utf-8"))
                process = subprocess.Popen(
                    ["ssh", "-o", "BatchMode=yes", f"nvidia@{address}", remote_command],
                    cwd=repo,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                processes.append((logical_id, process))
            windows_args = [
                "&", "'.\\.venv\\Scripts\\python.exe'", "-m", module, "client",
                "--server", f"192.168.66.205:{args.port}",
                "--client-id", "win136",
                "--client-index", "0",
                *common,
                "--device", "cpu",
            ]
            windows_command = (
                "Set-Location 'D:\\ProgramCode\\SplitFleet'; "
                + " ".join(windows_args)
            )
            windows_log = stack.enter_context((log_dir / "win136.log").open("w", encoding="utf-8"))
            windows = subprocess.Popen(
                [
                    "ssh", "-o", "BatchMode=yes", "18514@192.168.66.136",
                    _windows_powershell(windows_command),
                ],
                cwd=repo,
                stdout=windows_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(("win136", windows))
            server.wait(timeout=args.run_timeout_sec)
            for _name, process in processes[1:]:
                process.wait(timeout=120)
    except BaseException:
        for _name, process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for _name, process in reversed(processes):
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        _cleanup_remote_clients(args.port, log_dir)
        raise
    statuses = {name: process.returncode for name, process in processes}
    status_record = {
        "schema": "splitfleet.physical-orchestration.v1",
        "run_id": args.run_id,
        "method": args.method,
        "seed": args.seed,
        "started_unix": started,
        "finished_unix": time.time(),
        "process_exit_codes": statuses,
    }
    (run_dir / "orchestration.json").write_text(
        json.dumps(status_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bad = {name: code for name, code in statuses.items() if code != 0}
    if bad:
        raise RuntimeError(f"Physical processes failed: {bad}")
    report = validate_run(run_dir)
    (run_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report["valid"]:
        raise RuntimeError(f"Run validation failed: {report['errors']}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--method",
        choices=(FEDAVG_METHOD, *ProfileGuidedPlacementPolicy.METHODS),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--model",
        choices=("resnet18", "resnet50", "resnet101", "wide_resnet50_2"),
        default="resnet18",
    )
    parser.add_argument(
        "--protocol-file",
        default="experiments/physical_ra_splitfed/protocol.yaml",
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--server-concurrency", type=int, default=4)
    parser.add_argument("--paced-round-start", type=int, default=26)
    parser.add_argument("--paced-round-end", type=int, default=60)
    parser.add_argument("--paced-uplink-mbps", type=float, default=10.0)
    parser.add_argument("--paced-downlink-mbps", type=float, default=50.0)
    parser.add_argument("--nominal-uplink-mbps", type=float, default=1000.0)
    parser.add_argument("--nominal-downlink-mbps", type=float, default=1000.0)
    parser.add_argument(
        "--split-runtime",
        choices=("torchlens", "native_prefix"),
        default="torchlens",
    )
    parser.add_argument(
        "--prefix-executor",
        choices=("persistent_eager", "torch_compile"),
        default="persistent_eager",
    )
    parser.add_argument("--compile-backend", default="aot_eager")
    parser.add_argument("--profile-report")
    parser.add_argument("--results-root", default="results/physical_ra_splitfed")
    parser.add_argument("--listen-timeout-sec", type=float, default=300.0)
    parser.add_argument("--run-timeout-sec", type=float, default=7200.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
