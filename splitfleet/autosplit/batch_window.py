"""Dynamic batch window contract shared by the prefix and the suffix.

A TorchLens split runtime is traced once with a sample batch and replays any
batch size inside its ``dynamic_batch`` window.  Cross-device split federated
learning depends on that window: every device runs its own local batches, and
the negotiated window is the only thing that guarantees the same prefix and
suffix runtime accept them.  The window also feeds the feature ABI id, so a
client that silently picks its own window ends up with an incompatible runtime.
"""

from __future__ import annotations

import json
from typing import Any


class BatchWindowError(ValueError):
    """Raised when a batch falls outside the negotiated dynamic batch window."""


def normalize_batch_window(value: Any) -> tuple[int, int] | None:
    """Coerce a configured/serialized dynamic batch value into ``(min, max)``."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "null":
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BatchWindowError(f"Invalid dynamic batch window {value!r}.") from exc
        if value is None:
            return None
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise BatchWindowError(
                f"A dynamic batch window needs exactly two bounds, got {value!r}."
            )
        low, high = (int(value[0]), int(value[1]))
    else:
        raise BatchWindowError(f"Unsupported dynamic batch window {value!r}.")
    if low < 1:
        raise BatchWindowError(f"Dynamic batch window lower bound must be >= 1, got {low}.")
    if high < low:
        raise BatchWindowError(
            f"Dynamic batch window upper bound {high} is smaller than lower bound {low}."
        )
    return (low, high)


def batch_window_contains(window: tuple[int, int] | None, batch_size: int) -> bool:
    """Return whether ``batch_size`` can be replayed by a runtime with ``window``."""

    if window is None:
        return True
    if batch_size <= 0:
        return False
    return window[0] <= int(batch_size) <= window[1]


def describe_batch_window(window: tuple[int, int] | None) -> str:
    return "unbounded" if window is None else f"[{window[0]}, {window[1]}]"


def require_batch_in_window(
    batch_size: int,
    window: tuple[int, int] | None,
    *,
    stage: str,
    trace_batch_mode: str = "",
) -> None:
    """Reject a batch the runtime cannot replay, naming the fix in the message."""

    if batch_window_contains(window, batch_size):
        return
    assert window is not None  # ``None`` accepts every positive batch size
    traced = f" (traced in {trace_batch_mode} mode)" if trace_batch_mode else ""
    hints = [
        f"widen `dynamic_batch`{traced} on AutoSplitStrategy so every device's "
        "batch size fits",
    ]
    if batch_size < window[0]:
        hints.append(
            "or drop short trailing batches (DataLoader(drop_last=True), "
            "or partial_batch_policy='skip' on the split client)"
        )
    else:
        hints.append("or lower the client batch size")
    raise BatchWindowError(
        f"{stage} received batch size {batch_size}, outside the negotiated dynamic batch "
        f"window {describe_batch_window(window)}: " + "; ".join(hints) + "."
    )


__all__ = [
    "BatchWindowError",
    "batch_window_contains",
    "describe_batch_window",
    "normalize_batch_window",
    "require_batch_in_window",
]
