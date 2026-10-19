"""Run actual split training, never a placement-cost simulation."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from splitfleet.virtual import VirtualConfig, run_virtual

from .workloads import build_spec


def resolve_paths(config):
    workload = dict(config.workload)
    for key in ("data_root", "tokenizer_path", "pretrain_weights"):
        if workload.get(key):
            workload[key] = str(Path(workload[key]).resolve())
    from dataclasses import replace

    transport = config.transport
    credentials = {
        key: str(Path(getattr(transport, key)).resolve())
        for key in ("ca_cert", "server_cert", "server_key", "client_cert", "client_key")
        if getattr(transport, key)
    }
    return replace(
        config, workload=workload, transport=replace(transport, **credentials)
    )


def frozen_run(config, output):
    """Use the repository's source freeze/attempt contract and execute that copy."""
    from experiments.common.evidence import begin_run
    from splitfleet.virtual.runtime import check_simulation_dependencies

    check_simulation_dependencies()
    output = output.resolve()
    begin_run(output, config.as_dict())
    snapshot = output / "runtime_snapshot"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(snapshot)
    command = [
        sys.executable,
        "-m",
        "experiments.virtual_sfl",
        "run",
        "--frozen",
        "--config",
        str(output / "config.json"),
        "--output",
        str(output),
    ]
    with (output / "stdout.log").open("a") as log:
        result = subprocess.run(
            command,
            cwd=snapshot,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"Frozen virtual run failed (exit {result.returncode}); inspect {output / 'stdout.log'}"
        )
    return json.loads((output / "summary.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run native Flower/Ray virtual split learning")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    value = yaml.safe_load(args.config.read_text())
    value.pop("config_sha256", None)
    if "runs" in value:
        args.output.mkdir(parents=True, exist_ok=False)
        summaries = []
        for name, run_config in value["runs"].items():
            config = resolve_paths(VirtualConfig.from_dict(run_config))
            summaries.append(frozen_run(config, args.output / name))
        print(json.dumps(summaries, indent=2))
    else:
        config = resolve_paths(VirtualConfig.from_dict(value))
        if args.frozen:
            from experiments.common.evidence import finish_run

            if (
                Path(__file__).resolve().parents[2]
                != args.output.resolve() / "runtime_snapshot"
            ):
                raise ValueError(
                    "Execute --frozen only from the recorded runtime_snapshot"
                )
            if (args.output / "execution_attempt.json").exists():
                raise FileExistsError(
                    "Frozen run already attempted; preserve it and choose a new output"
                )
            if (
                json.loads((args.output / "result.json").read_text()).get("status")
                != "running"
            ):
                raise FileExistsError(
                    "Frozen run already finished; preserve it and choose a new output"
                )
            try:
                summary = run_virtual(config, build_spec(config), output=args.output)
                validation = json.loads((args.output / "validation.json").read_text())
                finish_run(
                    args.output,
                    summary,
                    validation=validation,
                    metrics=[
                        json.loads(row)
                        for row in (args.output / "round_metrics.jsonl")
                        .read_text()
                        .splitlines()
                    ],
                )
            except BaseException as exc:
                finish_run(
                    args.output,
                    {"status": "failed", "reason": str(exc)},
                    validation={"status": "failed", "reason": str(exc)},
                )
                raise
        else:
            summary = frozen_run(config, args.output)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
