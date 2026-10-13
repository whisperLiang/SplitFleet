"""Verify a narrowly bound transport grace change without hiding source identity.

The actual runtime identity is never rewritten. A separate comparison identity
normalizes only the declared 10-to-60-second acknowledgment-grace constants.
"""
from pathlib import Path
import hashlib
import json

FILES = ("splitfleet/client/grpc/connection.py", "splitfleet/server/app.py")
OLD = b'("grpc.keepalive_timeout_ms", 10_000)'
NEW = b'("grpc.keepalive_timeout_ms", 60_000)'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparison_source(root, runtime, plan, ledger):
    path = Path(root) / "source_amendment.json"
    if not path.exists():
        return runtime["source_identity"], None, {}
    amendment = json.loads(path.read_text())
    if (amendment.get("schema") != "splitfleet.transport-grace-source-amendment.v1"
            or amendment.get("allowed_files") != list(FILES)
            or amendment.get("original_grace_ms") != 10000
            or amendment.get("amended_grace_ms") != 60000
            or amendment["created_unix"] >= ledger["started_unix"]):
        raise ValueError("Invalid or retrospectively bound transport source amendment")
    before_path = Path(amendment["original_runtime_manifest_path"])
    after_path = Path(amendment["amended_runtime_manifest_path"])
    artifacts = {str(path): sha(path)}
    for name, expected in amendment["bound_artifacts_sha256"].items():
        if sha(name) != expected:
            raise ValueError("Transport amendment artifact changed: " + name)
        artifacts[name] = expected
    for manifest in (before_path, after_path):
        if str(manifest) not in artifacts:
            raise ValueError("Both source manifests must be prospectively bound")
    before, after = [json.loads(p.read_text()) for p in (before_path, after_path)]
    for key in ("python", "executable", "platform", "packages", "torchlens_source_sha256"):
        if before.get(key) != after.get(key):
            raise ValueError("Transport-only amendment changes the dependency environment")
    before_hashes, after_hashes = [p["source_hashes"] for p in (before, after)]
    identity = lambda hashes: hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if (identity(before_hashes) != before["source_identity"]
            or identity(after_hashes) != after["source_identity"]
            or before["source_identity"] != amendment["original_source_identity"]
            or after["source_identity"] != amendment["amended_source_identity"]
            or runtime["source_hashes"] != after_hashes
            or runtime["source_identity"] != after["source_identity"]
            or plan["source_identity"] != after["source_identity"]
            or Path(plan["source_root"]).resolve() != Path(amendment["amended_source_root"]).resolve()
            or set(before_hashes) != set(after_hashes)
            or set(name for name in before_hashes if before_hashes[name] != after_hashes[name]) != set(FILES)):
        raise ValueError("Transport amendment changes the declared runtime or file population")
    original_root = Path(amendment["original_source_root"])
    actual_root = Path(root) / "runtime_snapshot"
    if {str(p.relative_to(actual_root)) for p in actual_root.rglob("*.py")} != set(after_hashes):
        raise ValueError("Amended runtime contains unbound source files")
    normalized = {}
    for name, expected in before_hashes.items():
        before_file, actual_file = original_root / name, actual_root / name
        if sha(before_file) != expected or sha(actual_file) != after_hashes[name]:
            raise ValueError("Original or actual source file changed: " + name)
        content = actual_file.read_bytes()
        if name in FILES:
            original = before_file.read_bytes()
            if original.count(OLD) != 1 or content.count(NEW) != 1 or original.replace(OLD, NEW) != content:
                raise ValueError("Transport amendment changes code beyond acknowledgment grace")
            content = content.replace(NEW, OLD)
        normalized[name] = hashlib.sha256(content).hexdigest()
        artifacts[str(before_file)] = expected
    if normalized != before_hashes:
        raise ValueError("Normalized source differs from the original complete implementation")
    return before["source_identity"], amendment, artifacts
