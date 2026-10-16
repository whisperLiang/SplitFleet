"""Stable digest of model parameters and buffers."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np
import torch


def tensor_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def native_array_state_hash(backend, names, arrays):
    """Hash native array layouts without changing dtype or scalar shape."""
    if len(names) != len(arrays):
        raise ValueError("Native state tensor count differs from the model")
    digest = hashlib.sha256(f"splitfleet.native-state.v1:{backend}".encode())
    for name, value in sorted(zip(names, arrays, strict=True)):
        value = np.asarray(value)
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def model_state_hash(model, adapter=None):
    """Use the same state identity on coordinators and native clients."""
    if adapter is None:
        from splitfleet.backends.utils import adapter_for
        adapter = adapter_for(model)
    if adapter.backend_name == "torch":
        return tensor_state_hash(model.state_dict())
    return native_array_state_hash(adapter.backend_name, list(adapter._state(model)), adapter.export_ndarrays(model))
