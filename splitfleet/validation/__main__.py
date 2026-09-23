"""Run with ``python -m splitfleet.validation --backends torch jax --all-nodes``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from splitfleet.validation.matrix import TASK_BACKENDS, TASKS
from splitfleet.validation.runner import run_matrix


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate split outputs and training against native execution.")
    parser.add_argument("--backends", nargs="+", choices=TASK_BACKENDS, default=["torch"])
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--all-nodes", action="store_true", help="Check every enumerated before/after boundary.")
    parser.add_argument("--max-nodes", type=int, default=4, help="Maximum sampled boundaries per case (default: 4).")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout", type=float, default=600, help="Timeout per backend process, in seconds.")
    parser.add_argument("--output", type=Path, help="Write full JSON report to this path (default: stdout).")
    parser.add_argument("--no-isolation", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.max_nodes < 1 or args.timeout <= 0:
        parser.error("--max-nodes and --timeout must be positive")
    report = run_matrix(
        backends=args.backends, tasks=args.tasks, seed=args.seed,
        all_nodes=args.all_nodes, max_nodes=args.max_nodes,
        isolate_backends=not args.no_isolation, timeout=args.timeout,
    )
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"{report['status']}: {report['case_counts']}; nodes={report['node_counts']}; report={args.output}")
    else:
        print(text, end="")
    return 1 if report["status"] == "failed" else 2 if report["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
