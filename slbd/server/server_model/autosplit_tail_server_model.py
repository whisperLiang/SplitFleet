"""Server-model bridge for client-prefix split learning with autosplit tails."""

from __future__ import annotations

import copy
import uuid
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from slbd.autosplit.serde import dumps_torch_object, loads_torch_object
from slbd.common import BatchData, ControlCode, ServerModelEvaluateIns, ServerModelFitIns, ServerModelFitRes
from slbd.common.constants import AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY
from slbd.server.server_model.server_model import ServerModel


def _load_model_from_ndarrays(model: torch.nn.Module, ndarrays: list[np.ndarray]) -> None:
    if not ndarrays:
        return
    state_dict = model.state_dict()
    if len(state_dict) != len(ndarrays):
        raise ValueError(
            "Autosplit tail server-model parameter mismatch: "
            f"expected {len(state_dict)} tensors, received {len(ndarrays)}."
        )
    loaded_state = OrderedDict()
    for (name, reference), array in zip(state_dict.items(), ndarrays):
        loaded_state[name] = torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
    model.load_state_dict(loaded_state, strict=True)


def _model_to_ndarrays(model: torch.nn.Module) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy() for tensor in model.state_dict().values()]


def _batch_size_from_seeded_values(seeded_values: dict[int, Any]) -> int:
    for value in seeded_values.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[0]) if value.ndim > 0 else 1
        if isinstance(value, np.ndarray):
            return int(value.shape[0]) if value.ndim > 0 else 1
    return 1


class AutoSplitTailServerModel(ServerModel):
    """Execute only the remote autosplit tail for split learning rounds."""

    def __init__(
        self,
        *,
        runtime_manager,
        model: torch.nn.Module,
        optimizer_fn=None,
        loss_fn=None,
        client_stage_count: int = 1,
        device: str = "cpu",
    ) -> None:
        self.runtime_manager = runtime_manager
        self.optimizer_fn = optimizer_fn
        self.loss_fn = loss_fn
        self.default_client_stage_count = client_stage_count
        self.device = device
        self.model = copy.deepcopy(model).to(device)
        self.optimizer = None
        self.placement_plan = None
        self.client_stage_count = client_stage_count
        self.num_examples = 0
        self.loss_total = 0.0
        self.sid = ""

    def get_parameters(self):
        return _model_to_ndarrays(self.model)

    def configure_fit(self, ins: ServerModelFitIns) -> None:
        self._configure_common(ins.parameters, ins.config, sid=ins.sid)
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        if trainable:
            if self.optimizer_fn is not None:
                self.optimizer = self.optimizer_fn(self.model)
            else:
                self.optimizer = torch.optim.SGD(trainable, lr=0.01)
        else:
            self.optimizer = None
        self.model.train()

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
        self._configure_common(ins.parameters, ins.config, sid=ins.sid)
        self.optimizer = None
        self.model.eval()

    def train_tail(self, batches: list[BatchData]) -> list[BatchData]:
        responses = []
        for batch in batches:
            payload = loads_torch_object(batch.data["payload"], map_location=self.device)
            seeded_values = payload["seeded_values"]
            targets = payload.get("targets")
            result = self.runtime_manager.run_train_tail_plan(
                self.placement_plan,
                seeded_values,
                start_stage_index=self.client_stage_count,
                targets=targets,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
            )
            examples = int(payload.get("num_examples") or _batch_size_from_seeded_values(seeded_values))
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
        responses = []
        for batch in batches:
            payload = loads_torch_object(batch.data["payload"], map_location=self.device)
            seeded_values = payload["seeded_values"]
            targets = payload.get("targets")
            outputs = self.runtime_manager.run_eval_tail_plan(
                self.placement_plan,
                seeded_values,
                start_stage_index=self.client_stage_count,
            )
            loss = self.runtime_manager.autosplit_session.compute_loss(outputs, targets, self.loss_fn)
            examples = int(payload.get("num_examples") or _batch_size_from_seeded_values(seeded_values))
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

    def _configure_common(self, parameters, config, *, sid: str) -> None:
        self.sid = sid
        self.num_examples = 0
        self.loss_total = 0.0
        self.client_stage_count = int(
            config.get(
                AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY,
                self.default_client_stage_count,
            )
        )
        _load_model_from_ndarrays(self.model, parameters)
        plan_suffix = f"{sid or 'shared'}_{uuid.uuid4().hex[:8]}"
        self.placement_plan = self.runtime_manager.clone_placement_plan(
            model=self.model,
            plan_id_suffix=plan_suffix,
        )
