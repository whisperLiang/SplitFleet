"""Stable identifiers for paired experiments and frozen model states."""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any

from splitfleet.common.model_state import tensor_state_hash


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def torchlens_source_identity() -> str:
    """Hash installed TorchLens Python sources for frozen execution receipts."""
    import importlib.util
    from pathlib import Path

    package = Path(importlib.util.find_spec("torchlens").origin).parent
    hashes = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(package.rglob("*.py"))}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
