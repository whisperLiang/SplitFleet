"""Run actual split training, never a placement-cost simulation."""

import argparse
import json
import os
import subprocess
import sys
import time
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
    began = time.perf_counter()
    with (output / "stdout.log").open("a") as log:
        result = subprocess.run(
            command,
            cwd=snapshot,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    (output / "invocation_timing.json").write_text(json.dumps({
        "wall_sec": time.perf_counter() - began, "child_exit_code": result.returncode,
        "scope": "Frozen child invocation, including imports, data verification, initialization, training, evaluation, checkpoints and cleanup; excludes queue wait",
    }, indent=2) + "\n")
    if result.returncode:
        reason = f"Frozen virtual run failed (exit {result.returncode}); inspect {output / 'stdout.log'}"
        # Ray can terminate the interpreter before the child writes its final
        # status. Keep completed measurements and close the stale running flag.
        if json.loads((output / "result.json").read_text()).get("status") == "running":
            from experiments.common.evidence import finish_run

            rows = []
            metrics_file = output / "round_metrics.jsonl"
            if metrics_file.exists():
                for line in metrics_file.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        # A hard exit may leave the final append unfinished.
                        break
            report = {"status": "failed", "reason": reason,
                      "rounds_completed": len(rows), "child_exit_code": result.returncode}
            finish_run(output, report, validation=report, metrics=rows)
        raise RuntimeError(reason)
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
        campaign = {"status": "running", "run_order": list(value["runs"]),
                    "jobs": {name: {"status": "pending"} for name in value["runs"]}}

        def record_campaign():
            pending = args.output / ".campaign_status.pending"
            pending.write_text(json.dumps(campaign, indent=2) + "\n")
            pending.replace(args.output / "campaign_status.json")

        record_campaign()
        errors = {}
        for name, run_config in value["runs"].items():
            config = resolve_paths(VirtualConfig.from_dict(run_config))
            campaign["jobs"][name]["status"] = "running"
            record_campaign()
            try:
                summaries.append(frozen_run(config, args.output / name))
            except Exception as exc:
                errors[name] = str(exc)
                campaign["jobs"][name].update(status="failed", reason=str(exc))
            else:
                campaign["jobs"][name]["status"] = "passed"
            record_campaign()
        if errors:
            campaign["status"] = "failed"
            record_campaign()
            raise RuntimeError(f"Campaign contains failed runs: {errors}")
        if set(value["runs"]) == {"splitfleet", "fedavg"}:
            from .comparison import summarize_comparison

            try:
                summarize_comparison(args.output)
            except Exception as exc:
                campaign.update(status="failed", reason=str(exc))
                record_campaign()
                raise
        campaign["status"] = "passed"
        record_campaign()
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
                method = config.workload.get("training_method", "splitfleet")
                spec = build_spec(config)
                if method == "fedavg":
                    from splitfleet.virtual.fedavg import run_fedavg

                    summary = run_fedavg(config, spec, output=args.output)
                else:
                    summary = run_virtual(config, spec, output=args.output)
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
