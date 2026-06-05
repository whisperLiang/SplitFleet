"""TorchLens suffix server model for client-prefix split learning."""

from __future__ import annotations

import copy
import uuid
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from splitfleet.autosplit import SplitRuntimeHandle
from splitfleet.autosplit.serde import dumps_torch_object, loads_torch_object
from splitfleet.common import BatchData, ControlCode, ServerModelEvaluateIns, ServerModelFitIns, ServerModelFitRes
from splitfleet.common.constants import AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY
from splitfleet.server.server_model.server_model import ServerModel


def _load_model_from_ndarrays(model: torch.nn.Module, ndarrays: list[np.ndarray]) -> None:
    if not ndarrays:
        return
    state_dict = model.state_dict()
    if len(state_dict) != len(ndarrays):
        raise ValueError(
            "TorchLens tail server-model parameter mismatch: "
            f"expected {len(state_dict)} tensors, received {len(ndarrays)}."
        )
    loaded_state = OrderedDict()
    for (name, reference), array in zip(state_dict.items(), ndarrays):
        loaded_state[name] = torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
    model.load_state_dict(loaded_state, strict=True)


def _model_to_ndarrays(model: torch.nn.Module) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy() for tensor in model.state_dict().values()]


def _batch_size_from_boundary(boundary: Any) -> int:
    batch_size = getattr(boundary, "batch_size", None)
    if batch_size is not None:
        return int(batch_size)
    for tensor in getattr(boundary, "tensors", {}).values():
        if isinstance(tensor, torch.Tensor) and tensor.ndim > 0:
            return int(tensor.shape[0])
    return 1


class AutoSplitTailServerModel(ServerModel):
    """Execute the TorchLens suffix for split-learning rounds."""

    def __init__(
        self,
        *,
        runtime_manager,
        model: torch.nn.Module,
        optimizer_fn=None,
        loss_fn=None,
        boundary: str = "50%",
        mode: str = "generated_eager",
        device: str = "cpu",
    ) -> None:
        self.runtime_manager = runtime_manager
        self.optimizer_fn = optimizer_fn
        self.loss_fn = loss_fn
        self.boundary = boundary
        self.mode = mode
        self.device = device
        self.model = copy.deepcopy(model).to(device)
        self.optimizer = None
        self.runtime_handle: SplitRuntimeHandle | None = None
        self.num_examples = 0
        self.loss_total = 0.0
        self.sid = ""

    def get_parameters(self):
        return _model_to_ndarrays(self.model)

    def configure_fit(self, ins: ServerModelFitIns) -> None:
        self._configure_common(ins.parameters, ins.config, sid=ins.sid, training=True)
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = (
            self.optimizer_fn(self.model)
            if self.optimizer_fn is not None and trainable
            else torch.optim.SGD(trainable, lr=0.01)
            if trainable
            else None
        )

    def get_fit_result(self) -> ServerModelFitRes:
        average_loss = self.loss_total / max(self.num_examples, 1)
        return ServerModelFitRes(
            parameters=self.get_parameters(),
            config={
                "num_examples": self.num_examples,
                "avg_loss": average_loss,
            },
        )

    def configure_evaluate(self, ins: ServerModelEvaluateIns) -> None:
        self._configure_common(ins.parameters, ins.config, sid=ins.sid, training=False)
        self.optimizer = None

    def train_tail(self, batches: list[BatchData]) -> list[BatchData]:
        runtime_handle = self._require_runtime_handle()
        responses = []
        for batch in batches:
            payload = loads_torch_object(batch.data["payload"], map_location=self.device)
            boundary = payload["boundary"]
            targets = payload.get("targets")
            result = self.runtime_manager.run_train_tail_plan(
                runtime_handle,
                boundary,
                targets=targets,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
            )
            examples = int(payload.get("num_examples") or _batch_size_from_boundary(boundary))
            loss_value = float(result["loss"].detach().cpu())
            self.num_examples += examples
            self.loss_total += loss_value * examples
            responses.append(
                BatchData(
                    data={
                        "payload": dumps_torch_object(
                            {
                                "boundary_grads": result["boundary_grads"],
                                "loss": loss_value,
                                "num_examples": examples,
                            }
                        )
                    },
                    control_code=ControlCode.OK,
                    metadata={"sid": self.sid},
                )
            )
        return responses

    def evaluate_tail(self, batches: list[BatchData]) -> list[BatchData]:
        runtime_handle = self._require_runtime_handle()
        responses = []
        for batch in batches:
            payload = loads_torch_object(batch.data["payload"], map_location=self.device)
            boundary = payload["boundary"]
            targets = payload.get("targets")
            outputs = self.runtime_manager.run_eval_tail_plan(runtime_handle, boundary)
            loss = self.runtime_manager.autosplit_session.compute_loss(outputs, targets, self.loss_fn)
            examples = int(payload.get("num_examples") or _batch_size_from_boundary(boundary))
            responses.append(
                BatchData(
                    data={
                        "payload": dumps_torch_object(
                            {
                                "loss": float(loss.detach().cpu()),
                                "num_examples": examples,
                            }
                        )
                    },
                    control_code=ControlCode.OK,
                    metadata={"sid": self.sid},
                )
            )
        return responses

    def _configure_common(self, parameters, config, *, sid: str, training: bool) -> None:
        client_stage_count = int(config.get(AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY, 1))
        if client_stage_count != 1:
            raise ValueError(
                "TorchLens autosplit backend supports exactly one client-local prefix stage."
            )
        self.sid = sid
        self.num_examples = 0
        self.loss_total = 0.0
        _load_model_from_ndarrays(self.model, parameters)
        if training:
            self.model.train()
        else:
            self.model.eval()
        plan_suffix = f"{sid or 'shared'}_{uuid.uuid4().hex[:8]}"
        self.runtime_handle = self.runtime_manager.clone_runtime_for_model(
            self.model,
            suffix=plan_suffix,
        )

    def _require_runtime_handle(self) -> SplitRuntimeHandle:
        if self.runtime_handle is None:
            raise RuntimeError("AutoSplitTailServerModel has not been configured.")
        return self.runtime_handle
