"""Client-side TorchLens split-learning adapter with local prefix execution."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch

from splitfleet.autosplit import AutoSplitSession, SplitRuntimeHandle, normalize_inputs
from splitfleet.backends import TorchBackendAdapter
from splitfleet.runtime import PrefixContextStore
from splitfleet.split_engine.contracts import GraphContract, ModelVersionContract, validate_contract
from splitfleet.split_engine import graph_contract_for_runtime_handle
from splitfleet.autosplit.torchlens_runtime import torchlens_runtime_version
from splitfleet.transport import decode_gradients, encode_boundary, encode_bundle_wire
from splitfleet.transport.split_wire import boundary_to_envelope, envelope_to_gradients
from splitfleet.client.numpy_client import NumPyClient
from splitfleet.common import BatchData, ControlCode
from splitfleet.common.constants import (
    AUTOSPLIT_BACKEND_CONFIG_KEY,
    AUTOSPLIT_BACKEND_VALUE_TORCHLENS,
    AUTOSPLIT_BOUNDARY_CONFIG_KEY,
    AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY,
    AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_GRAPH_CONTRACT_DIGEST_CONFIG_KEY,
    AUTOSPLIT_MODEL_VERSION_CONFIG_KEY,
    AUTOSPLIT_MODEL_VERSION_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_MODE_CONFIG_KEY,
    AUTOSPLIT_PLAN_ID_CONFIG_KEY,
    AUTOSPLIT_SPLIT_ID_CONFIG_KEY,
)
from splitfleet.common.constants import CLIENT_ID_CONFIG_KEY


def _model_to_ndarrays(model: torch.nn.Module) -> list[np.ndarray]:
    return TorchBackendAdapter().export_ndarrays(model)


def _load_model_from_ndarrays(model: torch.nn.Module, ndarrays: list[np.ndarray]) -> None:
    TorchBackendAdapter().load_ndarrays(model, ndarrays)


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
        self.backend_adapter = TorchBackendAdapter()
        self.model = self.backend_adapter.move_model(self.backend_adapter.clone_model(model), device)
        self.train_data = train_data
        self.evaluate_data = evaluate_data if evaluate_data is not None else train_data
        self.sample_inputs = sample_inputs
        self.batch_adapter = batch_adapter or self._default_batch_adapter
        self.optimizer_fn = optimizer_fn
        self.autosplit_session = autosplit_session or AutoSplitSession(device=device)
        self.device = device
        self._runtime_cache: dict[str, SplitRuntimeHandle] = {}
        self._context_store = PrefixContextStore()
        self._round_config: dict[str, Any] = {}

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
            step_id = uuid.uuid4().hex
            round_id = int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0))
            client_id = str(config.get(CLIENT_ID_CONFIG_KEY, ""))
            self._context_store.put(round_id, client_id, step_id, boundary)
            contract = graph_contract_for_runtime_handle(runtime_handle)
            wire_boundary = boundary_to_envelope(
                boundary,
                round_id=round_id,
                client_id=client_id,
                step_id=step_id,
                plan_id=str(config[AUTOSPLIT_PLAN_ID_CONFIG_KEY]),
                split_id=contract.split_id,
                canonical_graph_hash=contract.canonical_graph_hash,
                boundary_schema_hash=contract.boundary_schema_hash,
                model_version=round_id,
            )
            response = self._call_tail(
                method_name="train_tail",
                boundary=wire_boundary,
                targets=torch_targets,
                num_examples=_batch_size(torch_inputs),
            )
            gradient_envelope = response["gradients"]
            if (
                gradient_envelope.round_id,
                gradient_envelope.client_id,
                gradient_envelope.step_id,
            ) != (round_id, client_id, step_id):
                raise RuntimeError("Gradient response step identity mismatch")
            if gradient_envelope.model_version != round_id:
                raise RuntimeError("Gradient response model version mismatch")
            local_boundary = self._context_store.pop(round_id, client_id, step_id)
            runtime_handle.backend.backward_prefix(
                local_boundary,
                boundary_grads=envelope_to_gradients(gradient_envelope, self.device),
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
                contract = graph_contract_for_runtime_handle(runtime_handle)
                wire_boundary = boundary_to_envelope(
                    boundary,
                    round_id=int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0)),
                    client_id=str(config.get(CLIENT_ID_CONFIG_KEY, "")),
                    step_id=uuid.uuid4().hex,
                    plan_id=str(config[AUTOSPLIT_PLAN_ID_CONFIG_KEY]),
                    split_id=contract.split_id,
                    canonical_graph_hash=contract.canonical_graph_hash,
                    boundary_schema_hash=contract.boundary_schema_hash,
                    model_version=int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0)),
                )
                response = self._call_tail(
                    method_name="evaluate_tail",
                    boundary=wire_boundary,
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
        if config.get(AUTOSPLIT_BACKEND_CONFIG_KEY) != AUTOSPLIT_BACKEND_VALUE_TORCHLENS:
            raise ValueError("Split config must explicitly declare backend='torchlens'.")
        client_stage_count = int(config.get(AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY, 1))
        if client_stage_count != 1:
            raise ValueError(
                "TorchLens autosplit backend supports exactly one client-local prefix stage."
            )
        _load_model_from_ndarrays(self.model, parameters)
        previous_round = int(self._round_config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, -1))
        current_round = int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0))
        if previous_round >= 0 and previous_round != current_round:
            self._context_store.discard_round(previous_round)
        self._round_config = dict(config)
        raw_version_contract = config.get(AUTOSPLIT_MODEL_VERSION_CONTRACT_CONFIG_KEY)
        if raw_version_contract:
            version_contract = ModelVersionContract.from_json(raw_version_contract)
            if version_contract.round_model_version != current_round:
                raise RuntimeError("Round model version contract mismatch")
            if version_contract.state_schema_hash != self.backend_adapter.state_manifest(self.model).schema_hash:
                raise RuntimeError("Client model state schema mismatch")
        if training:
            self.model.train()
        else:
            self.model.eval()
        return self._ensure_runtime_handle(config)

    def _ensure_runtime_handle(self, config) -> SplitRuntimeHandle:
        plan_id = str(config[AUTOSPLIT_PLAN_ID_CONFIG_KEY])
        split_id = str(config.get(AUTOSPLIT_SPLIT_ID_CONFIG_KEY, ""))
        graph_signature = str(config.get(AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY, ""))
        module_mode = "train" if self.model.training else "eval"
        state_schema = self.backend_adapter.state_manifest(self.model).schema_hash
        cache_key = "|".join([
            "torch", torchlens_runtime_version(), self.model.__class__.__qualname__,
            state_schema, plan_id, split_id, graph_signature, str(self.device), module_mode,
        ])
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
                f"TorchLens split id mismatch: prepared {handle.plan.split_id}, expected {split_id}."
            )
        if graph_signature and handle.plan.graph_signature != graph_signature:
            raise RuntimeError(
                "TorchLens graph signature mismatch: "
                f"prepared {handle.plan.graph_signature}, expected {graph_signature}."
            )
        raw_contract = config.get(AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY)
        if raw_contract:
            expected = GraphContract.from_json(raw_contract)
            if config.get(AUTOSPLIT_GRAPH_CONTRACT_DIGEST_CONFIG_KEY) != expected.digest:
                raise RuntimeError("Configured graph contract digest is invalid")
            validate_contract(expected, graph_contract_for_runtime_handle(handle))
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
        request = BatchData(
            data={
                "boundary": encode_boundary(boundary),
                "targets": encode_bundle_wire(targets),
                "metadata": json.dumps({"num_examples": num_examples}).encode("utf-8"),
            },
            control_code=ControlCode.OK,
        )
        response = getattr(self._require_server_model_proxy(), method_name)(
            request,
            _streams_=False,
        )
        metadata = json.loads(response.data["metadata"].decode("utf-8"))
        if "gradients" in response.data:
            metadata["gradients"] = decode_gradients(response.data["gradients"])
        return metadata

    def _build_optimizer(self):
        if self.optimizer_fn is not None:
            return self.optimizer_fn(self.model)
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        if not trainable:
            return None
        return self.backend_adapter.build_optimizer(self.model, {"name": "sgd", "lr": 0.01})

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
