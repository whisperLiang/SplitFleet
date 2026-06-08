"""Client-side TorchLens split-learning adapter with local prefix execution."""

from __future__ import annotations

import copy
import json
import time
from collections import OrderedDict
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch

from splitfleet.autosplit import AutoSplitSession, SplitRuntimeHandle, normalize_inputs
from splitfleet.autosplit.serde import dumps_torch_object, loads_torch_object
from splitfleet.autosplit.torchlens_contract import (
    classify_contract_compatibility,
    runtime_contract_digest,
    stable_json,
)
from splitfleet.client.numpy_client import NumPyClient
from splitfleet.common import BatchData, ControlCode
from splitfleet.common.constants import (
    AUTOSPLIT_BACKEND_CONFIG_KEY,
    AUTOSPLIT_BACKEND_VALUE_TORCHLENS,
    AUTOSPLIT_BOUNDARY_CONFIG_KEY,
    AUTOSPLIT_BOUNDARY_TENSOR_LABELS_CONFIG_KEY,
    AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY,
    AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY,
    AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY,
    AUTOSPLIT_MODE_CONFIG_KEY,
    AUTOSPLIT_PLAN_ID_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_BACKEND_VALUE_TORCHLENS_NATIVE,
    AUTOSPLIT_RUNTIME_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_CONTRACT_DIGEST_CONFIG_KEY,
    AUTOSPLIT_SPLIT_ID_CONFIG_KEY,
    AUTOSPLIT_TORCHLENS_VERSION_CONFIG_KEY,
    AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY,
)


def _model_to_ndarrays(model: torch.nn.Module) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy() for tensor in model.state_dict().values()]


def _load_model_from_ndarrays(model: torch.nn.Module, ndarrays: list[np.ndarray]) -> None:
    if not ndarrays:
        return
    state_dict = model.state_dict()
    if len(state_dict) != len(ndarrays):
        raise ValueError(
            "TorchLens split client parameter mismatch: "
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


def _decode_json_value(value: Any, default: Any = None, *, field_name: str = "autosplit config") -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON for {field_name}.") from exc
    return value


def _decode_dynamic_batch(value: Any) -> tuple[int, int] | None:
    decoded = _decode_json_value(value, field_name=AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY)
    if decoded is None:
        return None
    low, high = list(decoded)
    return int(low), int(high)


def _decode_runtime_contract(value: Any) -> dict[str, Any]:
    decoded = _decode_json_value(value, default={}, field_name=AUTOSPLIT_RUNTIME_CONTRACT_CONFIG_KEY)
    return dict(decoded) if isinstance(decoded, dict) else {}


class AutoSplitSplitLearningClient(NumPyClient):
    """Run a TorchLens prefix locally and delegate suffix work to the server model."""

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
            raise ValueError("TorchLens autosplit backend accepts positional model inputs only.")
        self.model = copy.deepcopy(model).to(device)
        self.train_data = train_data
        self.evaluate_data = evaluate_data if evaluate_data is not None else train_data
        self.sample_inputs = sample_inputs
        self.batch_adapter = batch_adapter or self._default_batch_adapter
        self.optimizer_fn = optimizer_fn
        self.autosplit_session = autosplit_session or AutoSplitSession(device=device)
        self.device = device
        self._runtime_cache: dict[str, SplitRuntimeHandle] = {}

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

            boundary = runtime_handle.backend.run_prefix(
                *normalize_inputs(torch_inputs),
                training=True,
            )
            response = self._call_tail(
                method_name="train_tail",
                runtime_handle=runtime_handle,
                boundary=boundary,
                targets=torch_targets,
                num_examples=_batch_size(torch_inputs),
            )
            runtime_handle.backend.backward_prefix(
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
                boundary = runtime_handle.backend.run_prefix(*normalize_inputs(torch_inputs))
                response = self._call_tail(
                    method_name="evaluate_tail",
                    runtime_handle=runtime_handle,
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

    def _prepare_round(self, parameters, config, *, training: bool) -> SplitRuntimeHandle:
        if config.get(AUTOSPLIT_BACKEND_CONFIG_KEY) not in (None, AUTOSPLIT_BACKEND_VALUE_TORCHLENS):
            raise ValueError("AutoSplitSplitLearningClient only supports the TorchLens backend.")
        client_stage_count = int(config.get(AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY, 1))
        if client_stage_count != 1:
            raise ValueError(
                "TorchLens autosplit backend supports exactly one client-local prefix stage."
            )
        _load_model_from_ndarrays(self.model, parameters)
        if training:
            self.model.train()
        else:
            self.model.eval()
        return self._ensure_runtime_handle(config)

    def _ensure_runtime_handle(self, config) -> SplitRuntimeHandle:
        plan_id = str(config[AUTOSPLIT_PLAN_ID_CONFIG_KEY])
        split_id = str(config.get(AUTOSPLIT_SPLIT_ID_CONFIG_KEY, ""))
        graph_signature = str(config.get(AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY, ""))
        boundary = str(config.get(AUTOSPLIT_BOUNDARY_CONFIG_KEY, "50%"))
        runtime_backend = str(
            config.get(
                AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY,
                AUTOSPLIT_RUNTIME_BACKEND_VALUE_TORCHLENS_NATIVE,
            )
        )
        torchlens_version = str(config.get(AUTOSPLIT_TORCHLENS_VERSION_CONFIG_KEY, ""))
        feature_abi_id = str(config.get(AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY, ""))
        trace_batch_mode = str(config.get(AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY, "") or "")
        dynamic_batch = _decode_dynamic_batch(config.get(AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY))
        server_contract = _decode_runtime_contract(config.get(AUTOSPLIT_RUNTIME_CONTRACT_CONFIG_KEY))
        server_contract_digest = str(config.get(AUTOSPLIT_RUNTIME_CONTRACT_DIGEST_CONFIG_KEY, "") or "")
        if runtime_backend != AUTOSPLIT_RUNTIME_BACKEND_VALUE_TORCHLENS_NATIVE:
            raise ValueError(
                "AutoSplitSplitLearningClient only supports TorchLens native runtime backend, "
                f"got {runtime_backend!r}."
            )
        if torchlens_version and torchlens_version != "2.18.0":
            raise ValueError(
                f"AutoSplitSplitLearningClient requires torchlens_version='2.18.0', got {torchlens_version!r}."
            )
        if server_contract and server_contract_digest:
            actual_digest = runtime_contract_digest(server_contract)
            if actual_digest != server_contract_digest:
                raise RuntimeError(
                    "TorchLens runtime contract digest mismatch in server config: "
                    f"computed {actual_digest}, expected {server_contract_digest}."
                )
        module_mode = "train" if self.model.training else "eval"
        cache_key = "|".join(
            [
                plan_id,
                split_id,
                boundary,
                graph_signature,
                feature_abi_id,
                runtime_backend,
                torchlens_version,
                trace_batch_mode,
                stable_json(dynamic_batch),
                module_mode,
            ]
        )
        cached = self._runtime_cache.get(cache_key)
        if cached is not None:
            return cached

        handle = self.autosplit_session.prepare_runtime(
            self.model,
            self.sample_inputs,
            boundary=boundary,
            mode=str(config.get(AUTOSPLIT_MODE_CONFIG_KEY, "generated_eager")),
            trainable=True,
            dynamic_batch=dynamic_batch,
            trace_batch_mode=trace_batch_mode or None,
        )
        if split_id and handle.plan.split_id != split_id:
            raise RuntimeError(
                f"TorchLens split id mismatch: prepared {handle.plan.split_id}, expected {split_id}."
            )
        if graph_signature and handle.plan.graph_signature != graph_signature:
            raise RuntimeError(
                "TorchLens graph signature mismatch: "
                f"prepared {handle.plan.graph_signature}, expected {graph_signature}."
            )
        if feature_abi_id and handle.feature_abi_id != feature_abi_id:
            raise RuntimeError(
                "TorchLens feature ABI mismatch: "
                f"prepared {handle.feature_abi_id}, expected {feature_abi_id}."
            )
        if server_contract:
            compatibility = classify_contract_compatibility(handle.runtime_contract, server_contract)
            if not bool(compatibility.get("compatible")):
                raise RuntimeError(
                    "TorchLens runtime contract mismatch: "
                    f"{compatibility}."
                )
        self._runtime_cache[cache_key] = handle
        return handle

    def _call_tail(
        self,
        *,
        method_name: str,
        runtime_handle: SplitRuntimeHandle,
        boundary,
        targets,
        num_examples: int,
    ):
        self._stamp_boundary_payload(boundary, runtime_handle)
        dynamic_batch = (
            list(runtime_handle.plan.dynamic_batch)
            if runtime_handle.plan.dynamic_batch is not None
            else None
        )
        contract_digest = runtime_contract_digest(runtime_handle.runtime_contract or {})
        payload = {
            "boundary": boundary,
            "targets": targets,
            "num_examples": num_examples,
            "feature_abi_id": runtime_handle.feature_abi_id,
            "runtime_contract_digest": contract_digest,
            "boundary_tensor_labels": list(runtime_handle.plan.boundary_tensor_labels),
            "batch_size": getattr(boundary, "batch_size", None),
            "torchlens_version": runtime_handle.torchlens_version,
            "runtime_backend": runtime_handle.plan.runtime_backend,
            "trace_batch_mode": runtime_handle.plan.trace_batch_mode,
            "dynamic_batch": dynamic_batch,
            AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY: runtime_handle.plan.runtime_backend,
            AUTOSPLIT_TORCHLENS_VERSION_CONFIG_KEY: runtime_handle.torchlens_version,
            AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY: runtime_handle.feature_abi_id,
            AUTOSPLIT_RUNTIME_CONTRACT_DIGEST_CONFIG_KEY: contract_digest,
            AUTOSPLIT_BOUNDARY_TENSOR_LABELS_CONFIG_KEY: list(runtime_handle.plan.boundary_tensor_labels),
            AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY: runtime_handle.plan.trace_batch_mode,
            AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY: dynamic_batch,
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

    def _stamp_boundary_payload(self, boundary, runtime_handle: SplitRuntimeHandle) -> None:
        if not hasattr(boundary, "metadata"):
            return
        metadata = dict(getattr(boundary, "metadata", {}) or {})
        dynamic_batch = (
            list(runtime_handle.plan.dynamic_batch)
            if runtime_handle.plan.dynamic_batch is not None
            else None
        )
        contract_digest = runtime_contract_digest(runtime_handle.runtime_contract or {})
        metadata["feature_abi_id"] = runtime_handle.feature_abi_id
        metadata["runtime_contract_digest"] = contract_digest
        metadata["boundary_tensor_labels"] = list(runtime_handle.plan.boundary_tensor_labels)
        metadata["batch_size"] = getattr(boundary, "batch_size", None)
        metadata["torchlens_version"] = runtime_handle.torchlens_version
        metadata["runtime_backend"] = runtime_handle.plan.runtime_backend
        metadata["trace_batch_mode"] = runtime_handle.plan.trace_batch_mode
        metadata["dynamic_batch"] = dynamic_batch
        metadata[AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY] = runtime_handle.plan.runtime_backend
        metadata[AUTOSPLIT_TORCHLENS_VERSION_CONFIG_KEY] = runtime_handle.torchlens_version
        metadata[AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY] = runtime_handle.feature_abi_id
        metadata[AUTOSPLIT_RUNTIME_CONTRACT_DIGEST_CONFIG_KEY] = contract_digest
        metadata[AUTOSPLIT_BOUNDARY_TENSOR_LABELS_CONFIG_KEY] = list(runtime_handle.plan.boundary_tensor_labels)
        metadata[AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY] = runtime_handle.plan.trace_batch_mode
        metadata[AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY] = dynamic_batch
        boundary.metadata = metadata
        spec = getattr(boundary, "spec", None)
        if spec is not None:
            spec.feature_abi_id = runtime_handle.feature_abi_id

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
