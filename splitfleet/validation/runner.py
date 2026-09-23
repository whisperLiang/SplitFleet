"""Isolated, machine-readable task correctness validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from importlib.metadata import version
from typing import Any, Sequence

from splitfleet.validation.matrix import TASK_BACKENDS, TASKS, build_case, validate_case


def _failed_cases(backend: str, tasks: Sequence[str], reason: str) -> list[dict[str, Any]]:
    return [
        {
            "backend": backend,
            "task": task,
            "scope": "synthetic correctness",
            "status": "failed",
            "reason": reason,
            "nodes": [],
            "counts": {"passed": 0, "failed": 0, "unsupported": 0},
        }
        for task in tasks
    ]


def _run_backend(backend, tasks, seed, all_nodes, max_nodes):
    results = []
    for task in tasks:
        try:
            case = build_case(task, backend=backend, seed=seed)
            results.append(validate_case(case, all_nodes=all_nodes, max_nodes=max_nodes))
        except Exception as exc:
            results.extend(_failed_cases(backend, [task], f"{type(exc).__name__}: {exc}"))
    return results


def run_matrix(
    *,
    backends: Sequence[str] = ("torch",),
    tasks: Sequence[str] = TASKS,
    seed: int = 2026,
    all_nodes: bool = False,
    max_nodes: int = 4,
    isolate_backends: bool = True,
    timeout: float = 600,
) -> dict[str, Any]:
    """Validate all selected tasks, with separate processes for native backends.

    The fixtures exercise real task losses using synthetic inputs. They do not
    measure dataset accuracy, distributed convergence, or task coverage for
    TensorFlow, PaddlePaddle, and tinygrad. Those backends have separate native
    replay/training protocol checks in ``tests/integration``.
    """
    backends = tuple(dict.fromkeys(backends))
    tasks = tuple(dict.fromkeys(tasks))
    if not backends or any(backend not in TASK_BACKENDS for backend in backends):
        raise ValueError(f"backends must contain supported entries from {TASK_BACKENDS}")
    if not tasks or any(task not in TASKS for task in tasks):
        raise ValueError(f"tasks must contain supported entries from {TASKS}")
    if max_nodes < 1 or timeout <= 0:
        raise ValueError("max_nodes and timeout must be positive")
    results = []
    for backend in backends:
        if not isolate_backends:
            results.extend(_run_backend(backend, tasks, seed, all_nodes, max_nodes))
            continue
        env = os.environ.copy()
        for variable in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
        ):
            env[variable] = "1"
        env.setdefault("JAX_PLATFORMS", "cpu")
        with tempfile.TemporaryDirectory(prefix=f"splitfleet-validation-{backend}-") as directory:
            output = Path(directory) / "report.json"
            command = [
                sys.executable, "-m", "splitfleet.validation", "--backends", backend,
                "--tasks", *tasks, "--seed", str(seed), "--max-nodes", str(max_nodes),
                "--output", str(output), "--no-isolation",
            ]
            if all_nodes:
                command.append("--all-nodes")
            try:
                child = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                results.extend(_failed_cases(backend, tasks, f"Backend process exceeded {timeout:g} seconds"))
                continue
            if not output.exists():
                details = (child.stderr or child.stdout).strip()[-4000:]
                results.extend(_failed_cases(backend, tasks, f"Backend process exited {child.returncode}: {details}"))
                continue
            child_report = json.loads(output.read_text(encoding="utf-8"))
            results.extend(child_report["results"])
    counts = {
        status: sum(result["status"] == status for result in results)
        for status in ("passed", "failed", "partial", "unsupported")
    }
    status = "failed" if counts["failed"] else "partial" if counts["partial"] or counts["unsupported"] else "passed"
    return {
        "schema_version": 1,
        "torchlens_version": version("torchlens"),
        "scope": "synthetic task correctness; not dataset accuracy or federated convergence",
        "backends": list(backends),
        "tasks": list(tasks),
        "seed": seed,
        "selection": "all" if all_nodes else "sampled",
        "backend_isolation": isolate_backends,
        "checks": ["output", "task_loss", "all_parameter_gradients", "one_sgd_update"],
        "wire_roundtrips": ["activations", "boundary_gradients"],
        "status": status,
        "case_counts": counts,
        "node_counts": {
            status: sum(result["counts"][status] for result in results)
            for status in ("passed", "failed", "unsupported")
        },
        "results": results,
    }
