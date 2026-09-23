"""Run a bounded task/method matrix one subprocess at a time.

Each benchmark process exits before the next starts, releasing its model,
dataset, CPU threads, and any accelerator allocations.  This launcher never
starts a concurrent worker or an implicit dataset download.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .run import METHODS


TASKS = (
    "image_classification",
    "text_classification",
    "object_detection",
    "semantic_segmentation",
)


def run_matrix(
    *,
    tasks: tuple[str, ...],
    methods: tuple[str, ...],
    output_root: str | Path,
    run_prefix: str,
    source: str,
    data_root: str | Path,
    seed: int,
    rounds: int,
    max_train_samples: int,
    max_test_samples: int,
    batch_size: int,
    max_batches_per_client: int,
    device: str = "cpu",
) -> list[Path]:
    if not tasks or any(task not in TASKS for task in tasks):
        raise ValueError(f"tasks must be non-empty and drawn from {TASKS}.")
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be non-empty and drawn from {METHODS}.")
    if not run_prefix or any(character in run_prefix for character in ("/", "\\", "..")):
        raise ValueError("run_prefix must be a non-empty directory-name fragment.")
    if min(rounds, max_train_samples, max_test_samples, batch_size, max_batches_per_client) < 1:
        raise ValueError("rounds and resource budgets must be positive.")
    if source not in ("real", "fixture"):
        raise ValueError("source must be 'real' or 'fixture'.")
    root = Path(output_root)
    destinations = [root / f"{run_prefix}_{task}_{method}_seed{seed}" for task in tasks for method in methods]
    existing = [str(destination) for destination in destinations if destination.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing runs: {existing}")
    root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = "1"
    completed: list[Path] = []
    for task in tasks:
        for method in methods:
            destination = root / f"{run_prefix}_{task}_{method}_seed{seed}"
            command = [
                sys.executable, "-m", "experiments.unified_multitask.run",
                "--task", task, "--method", method, "--source", source,
                "--data-root", str(data_root), "--seed", str(seed),
                "--rounds", str(rounds),
                "--max-train-samples", str(max_train_samples),
                "--max-test-samples", str(max_test_samples),
                "--batch-size", str(batch_size),
                "--max-batches-per-client", str(max_batches_per_client),
                "--device", device, "--output", str(destination),
            ]
            print(f"[serial matrix] start {task}/{method}: {destination}", flush=True)
            subprocess.run(command, check=True, env=environment)
            subprocess.run(
                [sys.executable, "-m", "experiments.unified_multitask.validate", "--run-dir", str(destination)],
                check=True,
                env=environment,
            )
            completed.append(destination)
            print(f"[serial matrix] complete {task}/{method}", flush=True)
    return completed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output-root", default="results/unified_multitask")
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--source", choices=("real", "fixture"), default="real")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches-per-client", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    run_matrix(**vars(args))


if __name__ == "__main__":
    main()
