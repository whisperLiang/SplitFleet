"""Run configured methods, seeds, and optional scaling sweeps sequentially."""

from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
from typing import Any, Mapping

from .config_utils import load_config
from .experiment_runner import run_experiment


def _scaled_profiles(base: Mapping[str, int], total: int) -> dict[str, int]:
    names = list(base)
    weights = [int(base[name]) for name in names]
    denominator = sum(weights)
    raw = [total * weight / denominator for weight in weights]
    counts = [int(value) for value in raw]
    for index in sorted(range(len(names)), key=lambda idx: raw[idx] - counts[idx], reverse=True)[: total - sum(counts)]:
        counts[index] += 1
    return {name: count for name, count in zip(names, counts)}


def run_suite(
    config: dict[str, Any],
    *,
    run_prefix: str,
    resume: bool = False,
) -> list[Path]:
    methods = list(config.get("methods") or ["resource_adaptive_splitfed"])
    seeds = [int(value) for value in config.get("seeds", [1, 2, 3, 4, 5])]
    sweep = dict(config.get("sweep") or {})
    dimensions = sorted(sweep)
    combinations = list(itertools.product(*(sweep[name] for name in dimensions))) if dimensions else [()]
    outputs = []
    base_profiles = dict(config.get("client_profiles") or {})
    for values in combinations:
        resolved = copy.deepcopy(config)
        suffix_parts = []
        for name, value in zip(dimensions, values):
            resolved[name] = value
            suffix_parts.append(f"{name}-{value}")
        if "num_clients" in dimensions and base_profiles:
            resolved["client_profiles"] = _scaled_profiles(base_profiles, int(resolved["num_clients"]))
        sweep_suffix = "_".join(suffix_parts)
        for method in methods:
            for seed in seeds:
                run_id = "_".join(part for part in (run_prefix, sweep_suffix, method, f"seed-{seed}") if part)
                existing = Path(resolved.get("results_root", "results/resource_adaptive_splitfed")) / run_id
                if resume and existing.exists():
                    from .validate_results import validate_run

                    report = validate_run(existing)
                    if not report["valid"]:
                        raise RuntimeError(
                            f"Cannot resume past invalid existing run {run_id!r}: {report['errors']}"
                        )
                    outputs.append(existing)
                    continue
                outputs.append(run_experiment(resolved, method, seed, run_id))
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip only existing runs that pass strict validation; invalid runs stop the suite.",
    )
    args = parser.parse_args(argv)
    for output in run_suite(
        load_config(args.config), run_prefix=args.run_prefix, resume=args.resume
    ):
        print(output)


if __name__ == "__main__":
    main()
