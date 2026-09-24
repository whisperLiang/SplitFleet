"""Validate one CoSplit-UCB smoke output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    errors: list[str] = []
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"invalid metadata: {exc}"]}
    rows = []
    try:
        rows = [
            json.loads(line)
            for line in (root / "round_metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid round metrics: {exc}")
    expected = int(metadata.get("rounds", 0))
    if [row.get("round_id") for row in rows] != list(range(1, expected + 1)):
        errors.append("round ids are incomplete, duplicated, or unordered")
    for row in rows:
        clients = row.get("clients") or []
        if len(clients) != 3 or len({item.get("client_id") for item in clients}) != 3:
            errors.append(f"round {row.get('round_id')} does not contain three unique clients")
        selected = float(row.get("selected_makespan_ms", -1))
        oracle = float(row.get("oracle_makespan_ms", -1))
        regret = float(row.get("instantaneous_regret_ms", -1))
        if selected < 0 or oracle < 0 or regret < -1e-7:
            errors.append(f"round {row.get('round_id')} has invalid makespan/regret")
    if metadata.get("method") == "cosplit_ucb":
        changes = metadata.get("distinct_boundaries_per_client", {})
        if max((int(value) for value in changes.values()), default=0) < 2:
            errors.append("CoSplit-UCB smoke contains no boundary change")
    return {"valid": not errors, "errors": errors, "rounds": len(rows)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args(argv)
    report = validate_run(args.results_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
