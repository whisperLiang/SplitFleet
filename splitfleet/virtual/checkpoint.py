"""Commit aggregated round models before beginning the next round."""

import json
from pathlib import Path
import shutil
import tempfile


def save_round_checkpoint(ops, model, arrays, directory, *, round_id, model_hash):
    directory = Path(directory)
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    stem = f"round_{round_id:04d}"
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=checkpoints) as pending:
        pending = Path(pending)
        model_path = ops.save_checkpoint(model, arrays, pending / stem)
        record = {
            "round_id": round_id,
            "checkpoint": str(Path("checkpoints") / model_path.name),
            "model_hash": model_hash,
            "framework_backend": ops.backend,
            "contents": "aggregated global model; optimizers are rebuilt each round",
        }
        controller = directory / "greedy_state.json"
        if controller.is_file():
            controller_copy = pending / f"{stem}.greedy.json"
            shutil.copyfile(controller, controller_copy)
            record["controller_state"] = str(Path("checkpoints") / controller_copy.name)
        (pending / f"{stem}.checkpoint.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
        files = list(pending.iterdir())
        for source in files:
            if (checkpoints / source.name).exists():
                raise FileExistsError(f"Round checkpoint already exists: {source.name}")
        for source in files:
            source.replace(checkpoints / source.name)
        # Publishing the pointer last leaves the previous checkpoint usable
        # if a process dies while writing the next model or its metadata.
        pointer = pending / "latest_checkpoint.json"
        pointer.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        pointer.replace(directory / "latest_checkpoint.json")
    return record
