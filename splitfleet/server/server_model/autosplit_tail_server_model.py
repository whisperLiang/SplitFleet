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
from splitfleet.autosplit.torchlens_contract import runtime_contract_digest
from splitfleet.common import BatchData, ControlCode, ServerModelEvaluateIns, ServerModelFitIns, ServerModelFitRes
from splitfleet.common.constants import (
    AUTOSPLIT_BACKEND_CONFIG_KEY,
    AUTOSPLIT_BACKEND_VALUE_TORCHLENS,
    AUTOSPLIT_BOUNDARY_TENSOR_LABELS_CONFIG_KEY,
    AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY,
    AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_BACKEND_VALUE_TORCHLENS_NATIVE,
    AUTOSPLIT_RUNTIME_CONTRACT_DIGEST_CONFIG_KEY,
    AUTOSPLIT_TORCHLENS_VERSION_CONFIG_KEY,
    AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY,
)
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


def _normalise_list(value: Any) -> list[Any] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        import json

        value = json.loads(value)
    return list(value)


def _normalise_dynamic_batch(value: Any) -> list[int] | None:
    decoded = _normalise_list(value)
    if decoded is None:
        return None
    low, high = decoded
    return [int(low), int(high)]


def _payload_value(payload: dict[str, Any], boundary: Any, primary: str, fallback: str | None = None) -> Any:
    metadata = getattr(boundary, "metadata", {}) or {}
    if primary in payload:
        return payload.get(primary)
    if primary in metadata:
        return metadata.get(primary)
    if fallback and fallback in payload:
        return payload.get(fallback)
    if fallback and fallback in metadata:
        return metadata.get(fallback)
    return None


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
            self._validate_boundary_payload(payload, boundary)
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
            self._validate_boundary_payload(payload, boundary)
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
        if config.get(AUTOSPLIT_BACKEND_CONFIG_KEY) not in (None, AUTOSPLIT_BACKEND_VALUE_TORCHLENS):
            raise ValueError("AutoSplitTailServerModel only supports autosplit_backend='torchlens'.")
        runtime_backend = str(
            config.get(
                AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY,
                AUTOSPLIT_RUNTIME_BACKEND_VALUE_TORCHLENS_NATIVE,
            )
        )
        if runtime_backend != AUTOSPLIT_RUNTIME_BACKEND_VALUE_TORCHLENS_NATIVE:
            raise ValueError(
                "AutoSplitTailServerModel only supports autosplit_runtime_backend='torchlens_native'."
            )
        torchlens_version = str(config.get(AUTOSPLIT_TORCHLENS_VERSION_CONFIG_KEY, "") or "")
        if torchlens_version and torchlens_version != "2.18.0":
            raise ValueError(
                f"AutoSplitTailServerModel requires torchlens version 2.18.0, got {torchlens_version!r}."
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
        expected_abi = str(config.get(AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY, "") or "")
        if expected_abi and self.runtime_handle.feature_abi_id != expected_abi:
            raise RuntimeError(
                "AutoSplitTailServerModel feature ABI mismatch after runtime clone: "
                f"prepared {self.runtime_handle.feature_abi_id}, expected {expected_abi}."
            )

    def _require_runtime_handle(self) -> SplitRuntimeHandle:
        if self.runtime_handle is None:
            raise RuntimeError("AutoSplitTailServerModel has not been configured.")
        return self.runtime_handle

    def _validate_boundary_payload(self, payload: dict[str, Any], boundary: Any) -> None:
        runtime_handle = self._require_runtime_handle()
        expected = runtime_handle.feature_abi_id
        actual = (
            _payload_value(payload, boundary, AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY, "feature_abi_id")
            or getattr(getattr(boundary, "spec", None), "feature_abi_id", "")
        )
        if not actual:
            raise RuntimeError(
                "BoundaryPayload is missing feature_abi_id; refusing TorchLens suffix execution."
            )
        if str(actual) != str(expected):
            raise RuntimeError(
                "BoundaryPayload feature ABI mismatch: "
                f"received {actual}, expected {expected}."
            )
        runtime_backend = _payload_value(
            payload,
            boundary,
            AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY,
            "runtime_backend",
        )
        if not runtime_backend:
            raise RuntimeError(
                "BoundaryPayload is missing runtime_backend; refusing TorchLens suffix execution."
            )
        if str(runtime_backend) != runtime_handle.plan.runtime_backend:
            raise RuntimeError(
                "BoundaryPayload runtime backend mismatch: "
                f"received {runtime_backend}, expected {runtime_handle.plan.runtime_backend}."
            )
        version = (
            _payload_value(payload, boundary, AUTOSPLIT_TORCHLENS_VERSION_CONFIG_KEY, "torchlens_version")
        )
        if not version:
            raise RuntimeError(
                "BoundaryPayload is missing torchlens_version; refusing TorchLens suffix execution."
            )
        if str(version) != runtime_handle.torchlens_version:
            raise RuntimeError(
                "BoundaryPayload TorchLens version mismatch: "
                f"received {version}, expected {runtime_handle.torchlens_version}."
            )
        actual_labels = _normalise_list(
            _payload_value(
                payload,
                boundary,
                AUTOSPLIT_BOUNDARY_TENSOR_LABELS_CONFIG_KEY,
                "boundary_tensor_labels",
            )
        )
        expected_labels = list(runtime_handle.plan.boundary_tensor_labels)
        if actual_labels is None:
            raise RuntimeError(
                "BoundaryPayload is missing boundary tensor labels; refusing TorchLens suffix execution."
            )
        if [str(label) for label in actual_labels] != [str(label) for label in expected_labels]:
            raise RuntimeError(
                "BoundaryPayload boundary tensor labels mismatch: "
                f"received {actual_labels}, expected {expected_labels}."
            )
        actual_batch_size = _payload_value(payload, boundary, "batch_size")
        if actual_batch_size is None:
            raise RuntimeError(
                "BoundaryPayload is missing batch_size; refusing TorchLens suffix execution."
            )
        boundary_batch_size = _batch_size_from_boundary(boundary)
        if int(actual_batch_size) != int(boundary_batch_size):
            raise RuntimeError(
                "BoundaryPayload batch size mismatch: "
                f"received {actual_batch_size}, inferred {boundary_batch_size}."
            )
        expected_dynamic_batch = _normalise_dynamic_batch(runtime_handle.plan.dynamic_batch)
        actual_dynamic_batch = _normalise_dynamic_batch(
            _payload_value(
                payload,
                boundary,
                AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY,
                "dynamic_batch",
            )
        )
        if expected_dynamic_batch is not None:
            if actual_dynamic_batch is None:
                raise RuntimeError(
                    "BoundaryPayload is missing dynamic_batch; refusing TorchLens suffix execution."
                )
            if actual_dynamic_batch != expected_dynamic_batch:
                raise RuntimeError(
                    "BoundaryPayload dynamic batch mismatch: "
                    f"received {actual_dynamic_batch}, expected {expected_dynamic_batch}."
                )
            low, high = expected_dynamic_batch
            if not (low <= int(boundary_batch_size) <= high):
                raise RuntimeError(
                    "BoundaryPayload batch size is outside the dynamic batch range: "
                    f"batch_size={boundary_batch_size}, dynamic_batch={expected_dynamic_batch}."
                )
        trace_batch_mode = _payload_value(
            payload,
            boundary,
            AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY,
            "trace_batch_mode",
        )
        if not trace_batch_mode:
            raise RuntimeError(
                "BoundaryPayload is missing trace_batch_mode; refusing TorchLens suffix execution."
            )
        if str(trace_batch_mode) != runtime_handle.plan.trace_batch_mode:
            raise RuntimeError(
                "BoundaryPayload trace batch mode mismatch: "
                f"received {trace_batch_mode}, expected {runtime_handle.plan.trace_batch_mode}."
            )
        expected_digest = runtime_contract_digest(runtime_handle.runtime_contract or {})
        actual_digest = (
            _payload_value(
                payload,
                boundary,
                AUTOSPLIT_RUNTIME_CONTRACT_DIGEST_CONFIG_KEY,
                "runtime_contract_digest",
            )
        )
        if not actual_digest:
            raise RuntimeError(
                "BoundaryPayload is missing runtime_contract_digest; refusing TorchLens suffix execution."
            )
        if str(actual_digest) != str(expected_digest):
            raise RuntimeError(
                "BoundaryPayload runtime contract digest mismatch: "
                f"received {actual_digest}, expected {expected_digest}."
            )
