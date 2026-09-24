"""Plot CoSplit-UCB smoke makespan and dynamic regret."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(results_dir: str | Path, output: str | Path | None = None) -> Path:
    import matplotlib.pyplot as plt

    root = Path(results_dir)
    rows = [
        json.loads(line)
        for line in (root / "round_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    destination = Path(output) if output is not None else root / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    rounds = [row["round_id"] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(rounds, [row["selected_makespan_ms"] for row in rows], label="selected")
    axes[0].plot(rounds, [row["oracle_makespan_ms"] for row in rows], label="offline oracle")
    axes[0].set_ylabel("makespan (ms)")
    axes[0].legend()
    axes[1].plot(rounds, [row["cumulative_dynamic_regret_ms"] for row in rows])
    axes[1].set_ylabel("cumulative regret (ms)")
    axes[1].set_xlabel("round")
    figure.tight_layout()
    path = destination / "cosplit_ucb_smoke.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    print(plot(args.results_dir, args.output))


if __name__ == "__main__":
    main()
