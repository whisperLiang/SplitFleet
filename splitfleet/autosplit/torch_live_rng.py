"""Version-pinned PyTorch replay extension for stochastic training.

TorchLens 2.34.1 restores capture-time RNG around stochastic operations, which
is appropriate for reproducing a capture but freezes dropout during training.
Bind only this runtime's segments to use the caller's advancing RNG stream.
The installed package and other runtimes are never monkey-patched globally.
This private extension deliberately fails on an incompatible runtime.
"""

from __future__ import annotations

import inspect
from types import MethodType
from typing import Any


def _invoke_live_rng(self, node, args, kwargs):
    from torchlens.utils.rng import AutocastRestore

    if node.target is None:
        raise RuntimeError(f"Missing replay target at {node.canonical_id}")
    autocast = getattr(node.op, "func_autocast_state", None)
    # Retain the pure-op fast path and captured autocast semantics. Random
    # targets consume the current generator rather than resetting to capture.
    if self._can_skip_autocast_restore(autocast):
        return node.target(*args, **kwargs)
    with AutocastRestore(autocast or {}):
        return node.target(*args, **kwargs)


def enable_live_torch_rng(runtime: Any) -> Any:
    """Install live RNG on PyTorch segments; other backends keep their API."""
    from splitfleet.autosplit.torchlens_runtime import require_torchlens_version

    if runtime.request.backend != "torch":
        return runtime
    require_torchlens_version()
    segments = runtime.segments
    for segment in (segments.prefix, segments.training_prefix, segments.suffix):
        if segment is None:
            continue
        if getattr(segment, "_splitfleet_live_rng", False):
            continue
        invoke = getattr(segment, "_invoke_func", None)
        if invoke is None or list(inspect.signature(invoke).parameters) != ["node", "args", "kwargs"]:
            raise RuntimeError("TorchLens live-RNG extension does not match the pinned segment API")
        if not callable(getattr(segment, "_can_skip_autocast_restore", None)):
            raise RuntimeError("TorchLens live-RNG extension lacks its captured autocast guard")
        segment._invoke_func = MethodType(_invoke_live_rng, segment)
        segment._splitfleet_live_rng = True
    return runtime
