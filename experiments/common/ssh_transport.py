"""Managed SSH forwarding for a whole physical experiment cohort.

All methods and workers use the same encrypted transport. Forwarding setup is
deployment work; encryption and transfer during training remain in the measured
server interval. A failed connection stops the cohort instead of restarting it.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


class SshReverseTransport:
    def __init__(self, deployment: dict, receipt: Path):
        options = deployment["transport"]
        if options["kind"] != "ssh_reverse_tunnel":
            raise ValueError("Unsupported managed transport")
        if deployment["server"].get("ssh"):
            raise ValueError("SSH reverse transport requires a local coordinator")
        self.deployment = deployment
        self.receipt_path = receipt
        self.directory: Path | None = None
        self.started: list[tuple[str, str]] = []
        self.record = {
            "schema": "splitfleet.ssh-reverse-transport.v1",
            "policy": options,
            "status": "not_started",
            "hosts": [],
            "restart_policy": "No automatic tunnel or model-job restart",
        }

    def _save(self):
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.record, indent=2) + "\n")
        temporary.replace(self.receipt_path)

    def start(self):
        forwards = []
        for service in ("server", "barrier"):
            host, port = self.deployment[service]["connect"].rsplit(":", 1)
            bind_port = self.deployment[service]["bind"].rsplit(":", 1)[1]
            if host != "127.0.0.1" or port != bind_port or not 0 < int(port) < 65536:
                raise ValueError("Forwarded services must use matching loopback ports")
            forwards += ["-R", f"127.0.0.1:{port}:127.0.0.1:{port}"]
        self.directory = Path(tempfile.mkdtemp(prefix="splitfleet-ssh-"))
        self.record.update(status="starting", started_unix=time.time(),
                           control_directory=str(self.directory))
        self._save()
        try:
            for index, host in enumerate(self.deployment["hosts"]):
                control = str(self.directory / str(index))
                command = ["ssh", "-fNT", "-M", "-S", control,
                    "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    "-o", "ExitOnForwardFailure=yes", "-o", "Compression=no",
                    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                    "-c", self.deployment["transport"]["cipher"],
                    *forwards, host["ssh"]]
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                entry = {"id": host["id"], "ssh": host["ssh"], "command": command,
                         "exit_code": result.returncode, "stderr": result.stderr,
                         "finished_unix": time.time()}
                self.record["hosts"].append(entry)
                self._save()
                result.check_returncode()
                self.started.append((host["ssh"], control))
                check = subprocess.run(["ssh", "-S", control, "-O", "check", host["ssh"]],
                                       capture_output=True, text=True, timeout=10, check=True)
                entry["control_check"] = check.stderr.strip()
                self._save()
            self.record.update(status="active", ready_unix=time.time())
            self._save()
        except Exception as exc:
            self.record["startup_error"] = f"{type(exc).__name__}: {exc}"
            self.close()
            raise

    def close(self):
        cleanup = []
        for ssh_host, control in reversed(self.started):
            try:
                result = subprocess.run(["ssh", "-S", control, "-O", "exit", ssh_host],
                                        capture_output=True, text=True, timeout=10)
                cleanup.append({"ssh": ssh_host, "exit_code": result.returncode,
                                "stderr": result.stderr.strip()})
            except (OSError, subprocess.TimeoutExpired) as exc:
                cleanup.append({"ssh": ssh_host, "error": str(exc)})
        self.started.clear()
        if self.directory is not None:
            shutil.rmtree(self.directory)
            self.directory = None
        self.record.update(status="closed", finished_unix=time.time(), cleanup=cleanup)
        self._save()
