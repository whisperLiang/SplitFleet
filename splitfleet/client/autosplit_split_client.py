"""Client-side Ariadne split-learning adapter with local prefix execution."""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch

from splitfleet.autosplit import AutoSplitSession, AriadneRuntimeHandle, normalize_inputs
from splitfleet.autosplit.serde import dumps_torch_object, loads_torch_object
from splitfleet.client.numpy_client import NumPyClient
from splitfleet.common import BatchData, ControlCode
from splitfleet.common.constants import (
    AUTOSPLIT_BACKEND_CONFIG_KEY,
    AUTOSPLIT_BACKEND_VALUE_ARIADNE,
    AUTOSPLIT_BOUNDARY_CONFIG_KEY,
    AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY,
    AUTOSPLIT_MODE_CONFIG_KEY,
    AUTOSPLIT_PLAN_ID_CONFIG_KEY,
    AUTOSPLIT_SPLIT_ID_CONFIG_KEY,
)


def _model_to_ndarrays(model: torch.nn.Module) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy() for tensor in model.state_dict().values()]


def _load_model_from_ndarrays(model: torch.nn.Module, ndarrays: list[np.ndarray]) -> None:
    if not ndarrays:
        return
    state_dict = model.state_dict()
    if len(state_dict) != len(ndarrays):
        raise ValueError(
            "Ariadne split client parameter mismatch: "
            f"expected {len(state_dict)} tensors, received {len(ndarrays)}."
        )
    loaded_state = OrderedDict()
    for (name, reference), array in zip(state_dict.items(), ndarrays):
        loaded_state[name] = torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
    model.load_state_dict(loaded_state, strict=True)


def _move_to_device(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
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


class AutoSplitSplitLearningClient(NumPyClient):
    """Run an Ariadne prefix locally and delegate suffix work to the server model."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_data: Iterable[Any],
        sample_inputs: Any,
        evaluate_data: Optional[Iterable[Any]] = None,
        sample_kwargs: Optional[dict] = None,
        batch_adapter: Optional[Callable[[Any], tuple[Any, Any]]] = None,
        optimizer_fn=None,
        autosplit_session: Optional[AutoSplitSession] = None,
        device: str = "cpu",
    ) -> None:
        if sample_kwargs:
            raise ValueError("Ariadne backend currently accepts positional model inputs only.")
        self.model = copy.deepcopy(model).to(device)
        self.train_data = train_data
        self.evaluate_data = evaluate_data if evaluate_data is not None else train_data
        self.sample_inputs = sample_inputs
        self.batch_adapter = batch_adapter or self._default_batch_adapter
        self.optimizer_fn = optimizer_fn
        self.autosplit_session = autosplit_session or AutoSplitSession(device=device)
        self.device = device
        self._runtime_cache: dict[str, AriadneRuntimeHandle] = {}

    def get_parameters(self, config):
        _ = config
        return _model_to_ndarrays(self.model)

    def fit(self, parameters, config):
        fit_start = time.perf_counter()
        runtime_handle = self._prepare_round(parameters, config, training=True)
        prefix_optimizer = self._build_optimizer()

        num_examples = 0
        weighted_loss = 0.0
        for batch in self.train_data:
            inputs, targets = self.batch_adapter(batch)
            torch_inputs = _move_to_device(inputs, self.device)
            torch_targets = _move_to_device(targets, self.device)

            self.model.zero_grad(set_to_none=True)
            if prefix_optimizer is not None:
                prefix_optimizer.zero_grad(set_to_none=True)

            boundary = runtime_handle.runtime.run_training_prefix(*normalize_inputs(torch_inputs))
            response = self._call_tail(
                method_name="train_tail",
                boundary=boundary,
                targets=torch_targets,
                num_examples=_batch_size(torch_inputs),
            )
            runtime_handle.runtime.backward_prefix(
                boundary,
                boundary_grads=response["boundary_grads"],
                optimizer=prefix_optimizer,
            )

            batch_examples = int(response["num_examples"])
            batch_loss = float(response["loss"])
            num_examples += batch_examples
            weighted_loss += batch_loss * batch_examples

        average_loss = weighted_loss / max(num_examples, 1)
        metrics = {
            "loss": average_loss,
            "fit_duration_sec": time.perf_counter() - fit_start,
            "num_examples": num_examples,
        }
        return _model_to_ndarrays(self.model), num_examples, metrics

    def evaluate(self, parameters, config):
        runtime_handle = self._prepare_round(parameters, config, training=False)

        num_examples = 0
        weighted_loss = 0.0
        with torch.no_grad():
            for batch in self.evaluate_data:
                inputs, targets = self.batch_adapter(batch)
                torch_inputs = _move_to_device(inputs, self.device)
                torch_targets = _move_to_device(targets, self.device)
                boundary = runtime_handle.runtime.run_prefix(*normalize_inputs(torch_inputs))
                response = self._call_tail(
                    method_name="evaluate_tail",
                    boundary=boundary,
                    targets=torch_targets,
                    num_examples=_batch_size(torch_inputs),
                )
                batch_examples = int(response["num_examples"])
                batch_loss = float(response["loss"])
                num_examples += batch_examples
                weighted_loss += batch_loss * batch_examples

        average_loss = weighted_loss / max(num_examples, 1)
        return float(average_loss), num_examples, {"loss": average_loss}

    def _prepare_round(self, parameters, config, *, training: bool) -> AriadneRuntimeHandle:
        if config.get(AUTOSPLIT_BACKEND_CONFIG_KEY) not in (None, AUTOSPLIT_BACKEND_VALUE_ARIADNE):
            raise ValueError("AutoSplitSplitLearningClient only supports the Ariadne backend.")
        client_stage_count = int(config.get(AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY, 1))
        if client_stage_count != 1:
            raise ValueError(
                "Ariadne backend currently supports exactly one client-local prefix stage."
            )
        _load_model_from_ndarrays(self.model, parameters)
        if training:
            self.model.train()
        else:
            self.model.eval()
        return self._ensure_runtime_handle(config)

    def _ensure_runtime_handle(self, config) -> AriadneRuntimeHandle:
        plan_id = str(config[AUTOSPLIT_PLAN_ID_CONFIG_KEY])
        split_id = str(config.get(AUTOSPLIT_SPLIT_ID_CONFIG_KEY, ""))
        graph_signature = str(config.get(AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY, ""))
        module_mode = "train" if self.model.training else "eval"
        cache_key = "|".join([plan_id, split_id, graph_signature, module_mode])
        cached = self._runtime_cache.get(cache_key)
        if cached is not None:
            return cached

        handle = self.autosplit_session.prepare_runtime(
            self.model,
            self.sample_inputs,
            boundary=str(config.get(AUTOSPLIT_BOUNDARY_CONFIG_KEY, "50%")),
            mode=str(config.get(AUTOSPLIT_MODE_CONFIG_KEY, "generated_eager")),
            trainable=True,
        )
        if split_id and handle.plan.split_id != split_id:
            raise RuntimeError(
                f"Ariadne split id mismatch: prepared {handle.plan.split_id}, expected {split_id}."
            )
        if graph_signature and handle.plan.graph_signature != graph_signature:
            raise RuntimeError(
                "Ariadne graph signature mismatch: "
                f"prepared {handle.plan.graph_signature}, expected {graph_signature}."
            )
        self._runtime_cache[cache_key] = handle
        return handle

    def _call_tail(
        self,
        *,
        method_name: str,
        boundary,
        targets,
        num_examples: int,
    ):
        payload = {
            "boundary": boundary,
            "targets": targets,
            "num_examples": num_examples,
        }
        request = BatchData(
            data={"payload": dumps_torch_object(payload)},
            control_code=ControlCode.OK,
        )
        response = getattr(self._require_server_model_proxy(), method_name)(
            request,
            _streams_=False,
        )
        return loads_torch_object(response.data["payload"], map_location=self.device)

    def _build_optimizer(self):
        if self.optimizer_fn is not None:
            return self.optimizer_fn(self.model)
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        if not trainable:
            return None
        return torch.optim.SGD(trainable, lr=0.01)

    def _require_server_model_proxy(self):
        proxy = getattr(self, "server_model_proxy", None)
        if proxy is None:
            raise RuntimeError("AutoSplitSplitLearningClient requires a server_model_proxy.")
        return proxy

    @staticmethod
    def _default_batch_adapter(batch: Any) -> tuple[Any, Any]:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        raise ValueError(
            "Expected each batch to be a `(inputs, targets)` pair. "
            "Provide `batch_adapter` to customize parsing."
        )
