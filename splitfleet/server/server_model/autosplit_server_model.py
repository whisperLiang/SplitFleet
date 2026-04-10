"""Server-model bridge that drives autosplit execution inside Flower rounds."""

from __future__ import annotations

import copy
import threading
import uuid
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import torch

from splitfleet.server.server_model.numpy_server_model import NumPyServerModel


def _to_torch(value: Any, *, device: str) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device)
    if isinstance(value, dict):
        return {key: _to_torch(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_torch(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_torch(item, device=device) for item in value)
    return value


def _batch_size(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.shape[0]) if value.ndim > 0 else 1
    if isinstance(value, np.ndarray):
        return int(value.shape[0]) if value.ndim > 0 else 1
    if isinstance(value, dict):
        for item in value.values():
            size = _batch_size(item)
            if size > 0:
                return size
    if isinstance(value, (list, tuple)) and value:
        return _batch_size(value[0])
    return 1


def _model_to_ndarrays(model: torch.nn.Module) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy() for tensor in model.state_dict().values()]


def _load_model_from_ndarrays(model: torch.nn.Module, ndarrays: list[np.ndarray]) -> None:
    if not ndarrays:
        return
    state_dict = model.state_dict()
    if len(state_dict) != len(ndarrays):
        raise ValueError(
            "Autosplit server-model parameter mismatch: "
            f"expected {len(state_dict)} tensors, received {len(ndarrays)}."
        )
    loaded_state = OrderedDict()
    for (name, reference), array in zip(state_dict.items(), ndarrays):
        loaded_state[name] = torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
    model.load_state_dict(loaded_state, strict=True)


class AutoSplitServerModel(NumPyServerModel):
    """Expose autosplit training/eval as server-model methods consumable by clients."""

    def __init__(
        self,
        *,
        runtime_manager,
        model: torch.nn.Module,
        optimizer_fn=None,
        loss_fn=None,
        device: str = "cpu",
    ) -> None:
        self.runtime_manager = runtime_manager
        self.base_model = model
        self.optimizer_fn = optimizer_fn
        self.loss_fn = loss_fn
        self.device = device
        self.model = copy.deepcopy(model).to(device)
        self.optimizer = None
        self.placement_plan = None
        self.sid = ""
        self.num_examples = 0
        self.loss_total = 0.0
        self._lock = threading.Lock()

    def get_parameters(self):
        return _model_to_ndarrays(self.model)

    def configure_fit(self, parameters, config) -> None:
        _ = config
        self._configure(parameters=parameters, sid=config.get("sid", ""))
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        if trainable:
            if self.optimizer_fn is not None:
                self.optimizer = self.optimizer_fn(self.model)
            else:
                self.optimizer = torch.optim.SGD(trainable, lr=0.01)
        else:
            self.optimizer = None
        self.model.train()

    def configure_evaluate(self, parameters, config) -> None:
        _ = config
        self._configure(parameters=parameters, sid=config.get("sid", ""))
        self.optimizer = None
        self.model.eval()

    def get_fit_result(self):
        average_loss = self.loss_total / max(self.num_examples, 1)
        return self.get_parameters(), {
            "num_examples": self.num_examples,
            "avg_loss": average_loss,
        }

    def train_batch(self, inputs, targets=None):
        with self._lock:
            target_batches = targets if targets is not None else [None] * len(inputs)
            responses = []
            for batch_inputs, batch_targets in zip(inputs, target_batches):
                torch_inputs = _to_torch(batch_inputs, device=self.device)
                torch_targets = _to_torch(batch_targets, device=self.device)
                result = self.runtime_manager.run_train_plan(
                    self.placement_plan,
                    torch_inputs,
                    targets=torch_targets,
                    loss_fn=self.loss_fn,
                    optimizer=self.optimizer,
                )
                examples = _batch_size(torch_inputs)
                loss_value = float(result["loss"].detach().cpu())
                self.num_examples += examples
                self.loss_total += loss_value * examples
                responses.append(
                    {
                        "loss": np.asarray([loss_value], dtype=np.float32),
                        "num_examples": np.asarray([examples], dtype=np.int64),
                    }
                )
            return responses

    def evaluate_batch(self, inputs, targets=None):
        with self._lock:
            target_batches = targets if targets is not None else [None] * len(inputs)
            responses = []
            for batch_inputs, batch_targets in zip(inputs, target_batches):
                torch_inputs = _to_torch(batch_inputs, device=self.device)
                torch_targets = _to_torch(batch_targets, device=self.device)
                outputs = self.runtime_manager.run_eval_plan(
                    self.placement_plan,
                    torch_inputs,
                )
                loss = self.runtime_manager.autosplit_session.compute_loss(
                    outputs,
                    torch_targets,
                    self.loss_fn,
                )
                responses.append(
                    {
                        "loss": np.asarray([float(loss.detach().cpu())], dtype=np.float32),
                        "num_examples": np.asarray([_batch_size(torch_inputs)], dtype=np.int64),
                    }
                )
            return responses

    def _configure(self, *, parameters, sid: str) -> None:
        self.sid = sid
        self.num_examples = 0
        self.loss_total = 0.0
        _load_model_from_ndarrays(self.model, parameters)
        plan_suffix = f"{sid or 'shared'}_{uuid.uuid4().hex[:8]}"
        self.placement_plan = self.runtime_manager.clone_placement_plan(
            model=self.model,
            plan_id_suffix=plan_suffix,
        )
