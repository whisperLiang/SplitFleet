"""PyTorch implementation of the framework state adapter."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import torch

from splitfleet.backends.base import ParameterEnvelope, StateEntry, StateManifest
from splitfleet.transport import TensorEnvelope, decode_tensor, encode_tensor


class TorchBackendAdapter:
    backend_name = "torch"

    def clone_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return copy.deepcopy(model)

    def move_model(self, model: torch.nn.Module, device: Any) -> torch.nn.Module:
        return model.to(device)

    def set_training(self, model: torch.nn.Module, training: bool) -> None:
        model.train(bool(training))

    def state_manifest(self, model: torch.nn.Module) -> StateManifest:
        entries = tuple(
            StateEntry(str(name), tuple(int(d) for d in value.shape), str(value.dtype).removeprefix("torch."))
            for name, value in model.state_dict().items()
        )
        payload = json.dumps(
            [entry.__dict__ for entry in entries], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        schema_hash = hashlib.sha256(payload).hexdigest()
        return StateManifest(self.backend_name, entries, schema_hash)

    def export_state(self, model: torch.nn.Module) -> ParameterEnvelope:
        manifest = self.state_manifest(model)
        tensors = tuple(encode_tensor(name, value) for name, value in model.state_dict().items())
        return ParameterEnvelope(self.backend_name, manifest.schema_hash, tensors, {})

    def load_state(self, model: torch.nn.Module, state: ParameterEnvelope) -> None:
        if state.backend != self.backend_name:
            raise ValueError(f"Cannot load {state.backend!r} state into torch model")
        expected = self.state_manifest(model)
        if state.schema_hash != expected.schema_hash:
            raise ValueError("Model state schema hash mismatch")
        current = model.state_dict()
        decoded = {item.tensor_id: decode_tensor(item, current[item.tensor_id].device) for item in state.tensors}
        if set(decoded) != set(current):
            raise ValueError("Model state tensor names do not match manifest")
        model.load_state_dict(decoded, strict=True)

    def encode_tensor(self, tensor_id: str, tensor: torch.Tensor) -> TensorEnvelope:
        return encode_tensor(tensor_id, tensor)

    def decode_tensor(self, envelope: TensorEnvelope, device: Any = "cpu") -> torch.Tensor:
        return decode_tensor(envelope, device)

    def build_optimizer(self, model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
        name = str(config.get("name", "sgd")).lower()
        kwargs = {key: value for key, value in config.items() if key != "name"}
        if name == "sgd":
            return torch.optim.SGD(model.parameters(), **kwargs)
        if name == "adam":
            return torch.optim.Adam(model.parameters(), **kwargs)
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), **kwargs)
        raise ValueError(f"Unsupported torch optimizer {name!r}")

    def scalar_value(self, value: Any) -> float:
        return float(value.detach().item() if isinstance(value, torch.Tensor) else value)

    def export_ndarrays(self, model: torch.nn.Module) -> list[np.ndarray]:
        return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]

    def load_ndarrays(self, model: torch.nn.Module, values: list[np.ndarray]) -> None:
        if not values:
            return
        state = model.state_dict()
        if len(state) != len(values):
            raise ValueError(f"Expected {len(state)} state tensors, received {len(values)}")
        decoded = {
            name: torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
            for (name, reference), array in zip(state.items(), values)
        }
        model.load_state_dict(decoded, strict=True)
