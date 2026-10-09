"""Distinguish live file checks from recorded verification of pruned weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "paper/evidence/storage_cleanup_receipt.json"


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pruned_checkpoint(path: Path) -> dict:
    """Return a pre-deletion receipt, never evidence of a fresh tensor check."""
    path = path.resolve()
    assert not path.exists(), f"Use the retained file for verification: {path}"
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["schema"] == "splitfleet.storage-pruning.v1"
    assert receipt["status"] == "passed", "Storage cleanup has not been verified"
    name = str(path.relative_to(ROOT))
    assert name in receipt["removed"], f"Unrecorded missing artifact: {path}"
    row = receipt["removed"][name]
    assert path.suffix == ".pt" and row["kind"] == "training_checkpoint"
    assert row["deleted"] and row["size_bytes"] > 0
    verification = row["checkpoint_verification"]
    assert verification["status"] == "passed" and verification["finite"]
    return row


def verify_retained_or_pruned(path: Path, expected_sha256: str) -> str:
    """Require live bytes, or an explicit passed receipt for removed weights."""
    if path.is_file():
        assert file_digest(path) == expected_sha256, f"File changed: {path}"
        return "retained_hash_checked"
    row = pruned_checkpoint(path)
    assert row["sha256"] == expected_sha256, f"Pruning receipt hash differs: {path}"
    return "checkpoint_verified_before_pruning"


def write_training_result(output: Path, result: dict, state: dict, *, save_model: bool = False) -> None:
    """Verify final tensors in memory and persist full weights only on request."""
    import torch
    from splitfleet.common.model_state import tensor_state_hash

    invalid = [name for name, value in state.items()
               if value.is_floating_point() and not bool(torch.isfinite(value).all())]
    if invalid:
        raise FloatingPointError(f"Final model has nonfinite tensors: {invalid[:3]}")
    state_hash = tensor_state_hash(state)
    if result.get("final_model_hash") != state_hash:
        raise ValueError("Final model differs from the result hash")
    result["final_state_verification"] = {
        "schema": "splitfleet.final-state-verification.v1", "finite": True,
        "state_hash": state_hash, "tensor_count": len(state),
        "tensor_bytes": sum(value.numel() * value.element_size() for value in state.values()),
        "basis": "in_memory_before_result_write", "weights_saved": save_model}
    output.parent.mkdir(parents=True, exist_ok=True)
    if save_model:
        torch.save({"task": result["task"], "method": result["method"],
                    "state_dict": {name: value.detach().cpu().clone() for name, value in state.items()}},
                   output.with_suffix(".model.pt"))
    output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")


def verify_final_state_receipt(result: dict) -> None:
    receipt = result.get("final_state_verification", {})
    if (receipt.get("schema") != "splitfleet.final-state-verification.v1"
            or receipt.get("finite") is not True
            or receipt.get("state_hash") != result.get("final_model_hash")
            or receipt.get("basis") != "in_memory_before_result_write"
            or receipt.get("tensor_count", 0) < 1):
        raise ValueError("Final in-memory state verification is missing or invalid")


def cleanup_input_bundles(directory: Path, receipt_path: Path) -> None:
    """Delete inputs owned by a finished matrix, retaining manifests and hashes."""
    rows = []
    for path in sorted(directory.rglob("*.pt")):
        if path.is_symlink():
            raise ValueError(f"Input cleanup must not follow shared external assets: {path}")
        rows.append({"path": str(path.relative_to(directory)), "sha256": file_digest(path),
                     "size_bytes": path.stat().st_size})
    receipt = {"status": "pruning", "files": rows,
        "removed_bytes": sum(row["size_bytes"] for row in rows),
        "scope": "finished matrix's owned inputs; raw results and manifests retained"}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    for row in rows:
        (directory / row["path"]).unlink()
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            path.rmdir()
    receipt["status"] = "pruned"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
