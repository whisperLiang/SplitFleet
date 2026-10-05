"""Record the existing gated residual-model suite with its actual evidence scope."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET

from experiments.www2027_study import freeze_runtime, save_json
from experiments.analysis.partition_coverage import _torchlens_source_identity


BACKENDS = ("torch", "tf", "jax", "paddle", "tinygrad")
TEST = "tests/integration/test_resnet18_all_backends_all_nodes.py"


def run(output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    runtime, manifest = freeze_runtime(source, output)
    shutil.copy2(source / "pyproject.toml", runtime / "pyproject.toml")
    shutil.copytree(source / "tests", runtime / "tests", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    test_hashes = {str(path.relative_to(runtime)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted((runtime / "tests").rglob("*.py"))}
    environment = {**os.environ, "PYTHONPATH": str(runtime), "CUDA_VISIBLE_DEVICES": "",
                   "JAX_PLATFORMS": "cpu", "SPLITFLEET_RUN_RESNET18_ALL_NODES": "1",
                   "SPLITFLEET_RESNET18_THREADS": "1"}
    command = [sys.executable, "-m", "pytest", TEST, "-q", "-s", "--junitxml", str(output / "tests.xml")]
    save_json(output / "config.json", {"command": command, "device": "cpu", "threads": 1,
        "source_identity": manifest["source_identity"], "test_sha256": test_hashes,
        "pyproject_sha256": hashlib.sha256((runtime / "pyproject.toml").read_bytes()).hexdigest(),
        "torchlens_source_sha256": _torchlens_source_identity(),
        "scope": "Existing per-backend residual-model replay/training/state/wire checks; not one canonical five-backend numerical equivalence model"})
    timed_out = False
    with (output / "stdout.log").open("w") as stream:
        process = subprocess.Popen(command, cwd=runtime, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            exit_code = process.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            timed_out, exit_code = True, None
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    log = (output / "stdout.log").read_text()
    cases = {}
    if (output / "tests.xml").exists():
        for case in ET.parse(output / "tests.xml").getroot().iter("testcase"):
            match = re.search(r"\[(torch|tf|jax|paddle|tinygrad)\]", case.attrib.get("name", ""))
            if match:
                cases[match[1]] = case
    rows = []
    for backend in BACKENDS:
        case = cases.get(backend)
        status, reason = "not_run", "No completed test case receipt"
        if case is not None:
            failure, error, skipped = case.find("failure"), case.find("error"), case.find("skipped")
            if failure is not None or error is not None:
                detail = failure if failure is not None else error
                status, reason = "failed", detail.attrib.get("message", "See stdout.log")[:1500]
            elif skipped is not None:
                status, reason = "skipped", skipped.attrib.get("message", "See stdout.log")
            else:
                status, reason = "passed", None
        match = re.search(rf"RESNET18_ALL_NODES backend={backend} graph_nodes=(\d+) viable=(\d+) trained=(\d+) "
                          r"terminal=(\d+) non_differentiable=(\d+) unsupported=(\d+)", log)
        counts = {name: int(value) for name, value in zip(
            ("graph_nodes", "viable", "trained", "terminal", "non_differentiable", "unsupported"), match.groups())} if match else {}
        rows.append({"backend": backend, "status": status, "reason": reason,
            "architecture": "scalar affine residual topology" if backend == "tinygrad" else
                            "residual model without BatchNorm" if backend == "jax" else "backend-specific ResNet18",
            "output_equivalence_checked": True if status == "passed" else None,
            "finite_loss_and_parameter_change_checked": True if status == "passed" else None,
            "boundary_roundtrip_checked": True if status == "passed" else None,
            "loss_equivalence_checked": None, "parameter_gradient_equivalence_checked": None,
            "one_step_update_equivalence_checked": None, "gradient_roundtrip_checked": None,
            **{name: counts.get(name) for name in ("graph_nodes", "viable", "trained", "terminal", "non_differentiable", "unsupported")}})
    result = {"schema": "splitfleet.gated-backend-validation.v1", "exit_code": exit_code,
              "timed_out": timed_out, "rows": rows, "canonical_five_backend_equivalence": False,
              "source_identity": manifest["source_identity"], "test_sha256": test_hashes,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(output / "backend_correctness.json", result)
    save_json(output / "result.json", result)
    save_json(output / "validation_report.json", {"all_backends_passed": all(row["status"] == "passed" for row in rows),
        "canonical_five_backend_equivalence": False, "exit_code": exit_code, "timed_out": timed_out,
        "counts_not_inferred_for_failed_or_unfinished_backends": True})
    with (output / "backend_correctness.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Gated residual-model backend validation", "",
             "This is the existing per-backend suite. JAX omits BatchNorm and tinygrad uses scalar affine residual blocks; it does not establish one canonical ResNet18 across five backends.", "",
             "| Backend | Status | Graph nodes | Trained cuts | Terminal cuts |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['backend']} | {row['status']} | {row['graph_nodes']} | {row['trained']} | {row['terminal']} |")
    lines += ["", "Passed checks cover replay output equivalence, finite training loss, parameter changes, boundary transport and federated state loading. Complete native/split loss, gradient and update equivalence and gradient wire parity remain unestablished by this suite.", "",
              "Failed, skipped and unfinished backends retain their receipts; unknown candidate counts are not reported as zero. See stdout.log and tests.xml.", ""]
    (output / "backend_correctness.md").write_text("\n".join(lines))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"exit_code": result["exit_code"], "timed_out": result["timed_out"],
                      "backends": {row["backend"]: row["status"] for row in result["rows"]}}))
    if result["timed_out"] or result["exit_code"] != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
