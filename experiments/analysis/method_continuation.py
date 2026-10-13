"""Audit an explicit same-source method continuation with retained successes."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retained_row(result_path):
    result_path = Path(result_path)
    result = json.loads(result_path.read_text())
    validation = json.loads(result_path.with_name("validation_report.json").read_text())
    if not validation.get("valid") or not validation.get("barrier_verified"):
        raise ValueError("The retained method must already have passed physical validation")
    return {"task": result["task"], "method": result["method"],
        "fixed_boundary": result["fixed_boundary"], "valid": True,
        "final_metrics": result["evaluation_records"][-1]["metrics"],
        "elapsed_sec": result["server_duration_sec"], "elapsed_time_basis": "server_monotonic",
        "fit_interval_overlap_sec": validation["fit_interval_overlap_sec"],
        **{k: result.get(k) for k in ("data_content_hash", "initial_model_hash", "final_model_hash", "partition_hash")},
        "run_dir": str(result_path.parent)}


def expected_remaining_command(original, execution_root):
    """Only change output locations and omit the already-completed Fixed50."""
    command = list(original)
    for flag, folder in (("--output-root", "physical"), ("--log-root", "logs")):
        command[command.index(flag) + 1] = str(Path(execution_root) / folder)
    index = command.index("--fixed-boundaries") + 1
    end = next(i for i in range(index, len(command)) if command[i].startswith("--"))
    if command[index:end] != ["50%", "75%", "25%"]:
        raise ValueError("Unexpected original seed's fixed-cut order")
    command[index:end] = ["75%", "25%"]
    return command


def validate_phase_continuations(root, plan, ledger):
    root = Path(root).resolve()
    artifacts, continuations = {}, []
    for job in ledger["jobs"]:
        name = job.get("method_continuation_receipt")
        if not name:
            continue
        if sha(name) != job.get("method_continuation_receipt_sha256"):
            raise ValueError("Method-continuation authorization changed")
        receipt = json.loads(Path(name).read_text())
        if (receipt.get("schema") != "splitfleet.user-directed-method-continuation.v1"
                or receipt.get("user_instruction") != "原失败记录不需要保留，直接从RF-DETR 种子 10505 的 Fixed75开始重跑后续实验"
                or Path(receipt["original_phase_root"]).resolve() != root
                or receipt["run_id"] != job["run_id"] or job["seed"] != 10505
                or job["stage"] != "rfdetr_nano_budget8"
                or receipt["runtime_source_identity"] != plan["source_identity"]
                or receipt["discarded_failed_method_attempts"] != 1
                or receipt["retained_schemes"] != ["splitfed_fixed50"]
                or receipt["remaining_schemes"] != ["splitfed_fixed75", "splitfed_fixed25", "splitfleet", "fedavg", "fedasync", "fedbuff", "fedcompass"]
                or job["command"] != receipt["original_command"]):
            raise ValueError("Method continuation changes the authorized original block")
        for path, expected in receipt["bound_artifacts_sha256"].items():
            if sha(path) != expected:
                raise ValueError("Retained success or frozen continuation input changed: " + path)
        artifacts.update(receipt["bound_artifacts_sha256"])
        artifacts[str(Path(name))] = sha(name)
        execution = Path(receipt["execution_root"])
        expected = expected_remaining_command(receipt["original_command"], execution)
        if receipt["remaining_command"] != expected:
            raise ValueError("Method continuation changes a training argument")
        execution_ledger = json.loads((execution / "ledger.json").read_text())
        if (not receipt["created_unix"] < execution_ledger["started_unix"]
                or execution_ledger["runtime_source_identity"] != plan["source_identity"]
                or len(execution_ledger["jobs"]) != 1
                or execution_ledger["jobs"][0]["command"] != expected):
            raise ValueError("Method selection was not fixed before its outcomes")
        runtime = json.loads((execution / "runtime_manifest.json").read_text())
        original_runtime = json.loads((root / "runtime_manifest.json").read_text())
        if runtime["source_hashes"] != original_runtime["source_hashes"]:
            raise ValueError("Continuation changes executed training source")
        for path, expected_hash in runtime["source_hashes"].items():
            target = execution / "runtime_snapshot" / path
            if sha(target) != expected_hash:
                raise ValueError("Continuation's training snapshot changed")
            artifacts[str(target)] = expected_hash
        original_attempts = json.loads((root / "physical" / job["run_id"] / "attempts.json").read_text())
        if original_attempts[:1] != receipt["retained_completed_attempts"]:
            raise ValueError("The completed Fixed50 attempt was replaced")
        attempt_path = execution / "physical" / job["run_id"] / "attempts.json"
        resumed = json.loads(attempt_path.read_text()) if attempt_path.exists() else []
        if ([a["scheme"] for a in resumed] != receipt["remaining_schemes"][:len(resumed)]
                or len(resumed) > 7):
            raise ValueError("Repeated or substituted continuation methods")
        if job["status"] == "completed":
            if (len(resumed) != 7 or any(a["status"] != "completed" for a in resumed)
                    or original_attempts != receipt["retained_completed_attempts"] + resumed
                    or execution_ledger["status"] != "completed"):
                raise ValueError("A resumed paired block lacks all seven remaining successes")
            new_folder = execution / "physical" / job["run_id"]
            if json.loads((new_folder / "source_hashes.json").read_text()) != runtime["source_hashes"]:
                raise ValueError("Executed method sources differ from their continuation snapshot")
            for path, expected_hash in runtime["source_hashes"].items():
                target = new_folder / "source_snapshot" / path
                if sha(target) != expected_hash:
                    raise ValueError("Executed method snapshot changed")
                artifacts[str(target)] = expected_hash
            expected_rows = [retained_row(receipt["retained_result_path"])] + json.loads((new_folder / "summary.json").read_text())
            if json.loads((root / "physical" / job["run_id"] / "summary.json").read_text()) != expected_rows:
                raise ValueError("Joined results differ from the actual retained and resumed runs")
            retained = json.loads(Path(receipt["retained_result_path"]).read_text())
            for row in expected_rows[1:]:
                value = json.loads((Path(row["run_dir"]) / "result.json").read_text())
                for key in ("seed", "data_content_hash", "partition_hash", "initial_model_hash", "pretrain_checkpoint_sha256"):
                    if value[key] != retained[key]:
                        raise ValueError("Continuation changes the seed, model initialization or data partition")
        for path in (execution / "ledger.json", execution / "runtime_manifest.json", attempt_path):
            if path.exists(): artifacts[str(path)] = sha(path)
        continuations.append(receipt)
    return continuations, artifacts
