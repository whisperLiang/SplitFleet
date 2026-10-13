"""Verify the FedBuff buffer repair while retaining every actual source identity."""
from pathlib import Path
import hashlib
import json

EDITS = {
    "experiments/baselines/async_fl.py": (
        (b"        self.model, self.config = model, config\n",
         b"        self.model, self.config = model, config\n        self.parameter_names = set(dict(model.named_parameters(remove_duplicate=False)))\n"),
        (b"                else:\n                    for update, weight in zip(updates, weights, strict=True):\n                        candidate[name] += weight * (update.trained[name] - update.base[name])\n",
         b"                elif name in self.parameter_names:\n                    for update, weight in zip(updates, weights, strict=True):\n                        candidate[name] += weight * (update.trained[name] - update.base[name])\n                else:\n                    # Running statistics are states, not optimizer deltas.\n                    candidate[name] = torch.stack([u.trained[name] for u in updates]).mean(dim=0)\n"),
        (b'                                else "mean_model_delta", "buffer_size": len(updates)})\n',
         b'                                else "mean_model_delta", "buffer_size": len(updates),\n            "floating_buffer_rule": "stale_model_interpolation" if self.config.method == "fedasync"\n                                    else "uniform_completed_state_average"})\n'),
    ),
    "experiments/common/workload_training.py": (
        (b"            all_targets.extend(targets.detach().cpu().numpy())\n",
         b'            if not torch.isfinite(outputs).all():\n                raise FloatingPointError("Native evaluation predictions are nonfinite")\n            all_targets.extend(targets.detach().cpu().numpy())\n'),
    ),
    "experiments/rfdetr_nano_physical.py": (
        (b"            targets.extend(records)\n",
         b'            if not all(torch.isfinite(outputs[key]).all() for key in ("pred_logits", "pred_boxes")):\n                raise FloatingPointError("Native detection predictions are nonfinite")\n            targets.extend(records)\n'),
    ),
}
FILES = tuple(EDITS)
GUARD_FILES = FILES[1:]
ASYNC_MODULE = FILES[0]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_after(content, name):
    for before, after in EDITS[name]:
        if content.count(before) != 1:
            raise ValueError("Baseline correction source anchor changed: " + name)
        content = content.replace(before, after, 1)
    return content


def baseline_source(root, runtime, plan, ledger):
    path = Path(root) / "baseline_source_amendment.json"
    if not path.exists():
        return None, {}, {}
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != "splitfleet.fedbuff-buffer-source-amendment.v1"
            or receipt.get("allowed_files") != list(FILES)
            or receipt.get("affected_aggregation_method") != "fedbuff"
            or receipt.get("quality_or_timing_selection_gate") is not False
            or receipt["created_unix"] >= ledger["started_unix"]):
        raise ValueError("Invalid or retrospective FedBuff source amendment")
    artifacts = {str(path): sha(path)}
    for name, expected in receipt["bound_artifacts_sha256"].items():
        if sha(name) != expected:
            raise ValueError("Baseline amendment artifact changed: " + name)
        artifacts[name] = expected
    paths = [Path(receipt[k]) for k in ("original_runtime_manifest_path", "amended_runtime_manifest_path")]
    if any(str(p) not in artifacts for p in paths):
        raise ValueError("Both baseline source manifests must be bound")
    before, after = [json.loads(p.read_text()) for p in paths]
    for key in ("python", "executable", "platform", "packages", "torchlens_source_sha256", "dependency_artifacts_sha256"):
        if before.get(key) != after.get(key):
            raise ValueError("FedBuff repair changes the dependency environment")
    identity = lambda h: hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest()
    bh, ah = before["source_hashes"], after["source_hashes"]
    if (identity(bh) != before["source_identity"] or identity(ah) != after["source_identity"]
            or before["source_identity"] != receipt["original_source_identity"]
            or after["source_identity"] != receipt["amended_source_identity"]
            or runtime["source_identity"] != after["source_identity"] or runtime["source_hashes"] != ah
            or plan["source_identity"] != after["source_identity"]
            or Path(plan["source_root"]).resolve() != Path(receipt["amended_source_root"]).resolve()
            or set(bh) != set(ah) or {n for n in bh if bh[n] != ah[n]} != set(FILES)):
        raise ValueError("FedBuff correction changes its declared source or file population")
    original = Path(receipt["original_source_root"])
    actual = Path(root) / "runtime_snapshot"
    if {str(p.relative_to(actual)) for p in actual.rglob("*.py")} != set(ah):
        raise ValueError("Corrected runtime contains unbound source files")
    guard_hashes = {}
    for name, expected in bh.items():
        previous, current = original / name, actual / name
        if sha(previous) != expected or sha(current) != ah[name]:
            raise ValueError("Baseline source file changed: " + name)
        predicted = expected_after(previous.read_bytes(), name) if name in EDITS else previous.read_bytes()
        if current.read_bytes() != predicted:
            raise ValueError("FedBuff repair changes code beyond the declared buffer rule and evaluation guards")
        artifacts[str(previous)] = expected
        if name in GUARD_FILES:
            guard_hashes[name] = expected
    return receipt, artifacts, guard_hashes


def shared_execution_identity(root, runtime, guard_hashes=None):
    """Compare shared execution code; the asynchronous adapter retains its own hash.

    Only the bound evaluation guards and the established acknowledgment-grace
    constants are normalized. No parameter/buffer repair is hidden in a source
    identity: the adapter is explicitly outside this shared-code fingerprint.
    An unamended run retains evaluation guards as part of its original source.
    """
    from experiments.analysis.transport_source_amendment import FILES as TRANSPORT, OLD, NEW
    hashes = {n: h for n, h in runtime["source_hashes"].items() if n != ASYNC_MODULE}
    for name in TRANSPORT:
        content = (Path(root) / "runtime_snapshot" / name).read_bytes()
        if content.count(NEW) == 1:
            content = content.replace(NEW, OLD)
        elif content.count(OLD) != 1:
            raise ValueError("Shared execution has an undeclared transport grace")
        hashes[name] = hashlib.sha256(content).hexdigest()
    if (Path(root) / "baseline_source_amendment.json").exists():
        for name in GUARD_FILES:
            content = (Path(root) / "runtime_snapshot" / name).read_bytes()
            old, new = EDITS[name][0]
            if new in content and name not in (guard_hashes or {}):
                raise ValueError("Evaluation guard source is not bound")
    hashes.update(guard_hashes or {})
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
