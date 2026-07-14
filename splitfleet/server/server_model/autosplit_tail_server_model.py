"""TorchLens suffix server model for client-prefix split learning."""

from __future__ import annotations

import json
import uuid
from typing import Any
from threading import RLock

import numpy as np

from splitfleet.autosplit import SplitRuntimeHandle
from splitfleet.backends.utils import adapter_for
from splitfleet.split_engine.contracts import GraphContract, ModelVersionContract, validate_contract
from splitfleet.split_engine import graph_contract_for_runtime_handle
from splitfleet.transport import (
    decode_boundary,
    decode_bundle_wire,
    encode_gradients,
)
from splitfleet.transport.split_wire import envelope_to_boundary, gradients_to_envelope
from splitfleet.common import BatchData, ControlCode, ServerModelEvaluateIns, ServerModelFitIns, ServerModelFitRes
from splitfleet.common.constants import (
    AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_GRAPH_CONTRACT_DIGEST_CONFIG_KEY,
    AUTOSPLIT_MODEL_VERSION_CONFIG_KEY,
    AUTOSPLIT_MODEL_VERSION_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_PLAN_ID_CONFIG_KEY,
)
from splitfleet.server.server_model.server_model import ServerModel


def _load_model_from_ndarrays(model: Any, ndarrays: list[np.ndarray], adapter=None) -> None:
    (adapter or adapter_for(model, ())).load_ndarrays(model, ndarrays)


def _model_to_ndarrays(model: Any, adapter=None) -> list[np.ndarray]:
    adapter = adapter or adapter_for(model, ())
    return adapter.export_ndarrays(model)


def _batch_size_from_boundary(boundary: Any) -> int:
    batch_size = getattr(boundary, "batch_size", None)
    if batch_size is not None:
        return int(batch_size)
    for tensor in getattr(boundary, "tensors", {}).values():
        shape = tuple(getattr(tensor, "shape", ()) or ())
        if shape:
            return int(shape[0])
    return 1


class AutoSplitTailServerModel(ServerModel):
    """Execute the TorchLens suffix for split-learning rounds."""

    def __init__(
        self,
        *,
        runtime_manager,
        model: Any,
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
        try:
            sample_inputs = runtime_manager._require_runtime_handle().plan.metadata.get("_example_inputs")
        except AttributeError:
            sample_inputs = ()
        self.backend_adapter = adapter_for(model, sample_inputs)
        self.model = self.backend_adapter.move_model(self.backend_adapter.clone_model(model), device)
        self.optimizer = None
        self.runtime_handle: SplitRuntimeHandle | None = None
        self.num_examples = 0
        self.loss_total = 0.0
        self.sid = ""
        self.model_version = 0
        self.plan_id = ""
        self._processed_steps: set[tuple[int, str, str]] = set()
        self._inflight_steps: set[tuple[int, str, str]] = set()
        self._step_lock = RLock()
        self._runtime_cache: dict[tuple[str, str, bool, str], SplitRuntimeHandle] = {}

    def get_parameters(self):
        return _model_to_ndarrays(self.model, self.backend_adapter)

    def configure_fit(self, ins: ServerModelFitIns) -> None:
        self._configure_common(ins.parameters, ins.config, sid=ins.sid, training=True)
        if self.optimizer_fn is not None:
            self.optimizer = self.optimizer_fn(self.model)
        else:
            try: self.optimizer = self.backend_adapter.build_optimizer(self.model, {"name": "sgd", "lr": 0.01})
            except (NotImplementedError, ValueError): self.optimizer = None

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
            wire_boundary = decode_boundary(batch.data["boundary"])
            self._validate_wire_identity(wire_boundary)
            step_key = (wire_boundary.round_id, wire_boundary.client_id, wire_boundary.step_id)
            boundary = envelope_to_boundary(wire_boundary, runtime_handle.runtime, self.device)
            targets = decode_bundle_wire(batch.data["targets"], self.device)
            request_metadata = json.loads(batch.data["metadata"].decode("utf-8"))
            self._begin_step(step_key)
            try:
                result = self.runtime_manager.run_train_tail_plan(
                    runtime_handle,
                    boundary,
                    targets=targets,
                    loss_fn=self.loss_fn,
                    optimizer=self.optimizer,
                )
            except Exception:
                self._abort_step(step_key)
                raise
            self._complete_step(step_key)
            examples = int(request_metadata.get("num_examples") or _batch_size_from_boundary(boundary))
            loss_value = self.backend_adapter.scalar_value(result["loss"])
            self.num_examples += examples
            self.loss_total += loss_value * examples
            gradients = gradients_to_envelope(wire_boundary, result["boundary_grads"])
            responses.append(
                BatchData(
                    data={
                        "gradients": encode_gradients(gradients),
                        "metadata": json.dumps(
                            {"loss": loss_value, "num_examples": examples},
                            sort_keys=True,
                        ).encode("utf-8"),
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
            wire_boundary = decode_boundary(batch.data["boundary"])
            self._validate_wire_identity(wire_boundary)
            boundary = envelope_to_boundary(wire_boundary, runtime_handle.runtime, self.device)
            targets = decode_bundle_wire(batch.data["targets"], self.device)
            outputs = self.runtime_manager.run_eval_tail_plan(runtime_handle, boundary)
            loss = self.runtime_manager.autosplit_session.compute_loss(outputs, targets, self.loss_fn)
            request_metadata = json.loads(batch.data["metadata"].decode("utf-8"))
            examples = int(request_metadata.get("num_examples") or _batch_size_from_boundary(boundary))
            responses.append(
                BatchData(
                    data={
                        "metadata": json.dumps(
                            {"loss": self.backend_adapter.scalar_value(loss), "num_examples": examples},
                            sort_keys=True,
                        ).encode("utf-8")
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
        self._processed_steps.clear()
        self._inflight_steps.clear()
        self.model_version = int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0))
        self.plan_id = str(config.get(AUTOSPLIT_PLAN_ID_CONFIG_KEY, ""))
        _load_model_from_ndarrays(self.model, parameters, self.backend_adapter)
        raw_version_contract = config.get(AUTOSPLIT_MODEL_VERSION_CONTRACT_CONFIG_KEY)
        if raw_version_contract:
            version_contract = ModelVersionContract.from_json(raw_version_contract)
            if version_contract.round_model_version != self.model_version:
                raise RuntimeError("Round model version contract mismatch")
            if version_contract.state_schema_hash != self.backend_adapter.state_manifest(self.model).schema_hash:
                raise RuntimeError("Suffix model state schema mismatch")
        self.backend_adapter.set_training(self.model, training)
        runtime_key = (
            self.plan_id,
            self.backend_adapter.state_manifest(self.model).schema_hash,
            bool(training),
            str(self.device),
        )
        self.runtime_handle = self._runtime_cache.get(runtime_key)
        if self.runtime_handle is None:
            plan_suffix = f"{sid or 'shared'}_{uuid.uuid4().hex[:8]}"
            self.runtime_handle = self.runtime_manager.clone_runtime_for_model(
                self.model,
                suffix=plan_suffix,
            )
            self._runtime_cache[runtime_key] = self.runtime_handle
        raw_contract = config.get(AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY)
        if raw_contract:
            expected = GraphContract.from_json(raw_contract)
            if config.get(AUTOSPLIT_GRAPH_CONTRACT_DIGEST_CONFIG_KEY) != expected.digest:
                raise RuntimeError("Configured graph contract digest is invalid")
            validate_contract(expected, graph_contract_for_runtime_handle(self.runtime_handle))

    def _validate_wire_identity(self, boundary) -> None:
        runtime_handle = self._require_runtime_handle()
        contract = graph_contract_for_runtime_handle(runtime_handle)
        if boundary.engine != "torchlens" or boundary.backend != self.backend_adapter.backend_name:
            raise ValueError("Unsupported split engine/backend boundary")
        if boundary.plan_id != self.plan_id:
            raise ValueError("Boundary placement plan mismatch")
        if boundary.model_version != self.model_version:
            raise ValueError("Boundary model version is stale")
        if boundary.split_id != contract.split_id:
            raise ValueError("Boundary split id mismatch")
        if boundary.canonical_graph_hash != contract.canonical_graph_hash:
            raise ValueError("Boundary graph contract mismatch")
        if boundary.boundary_schema_hash != contract.boundary_schema_hash:
            raise ValueError("Boundary ABI mismatch")

    def _begin_step(self, step_key: tuple[int, str, str]) -> None:
        with self._step_lock:
            if step_key in self._processed_steps or step_key in self._inflight_steps:
                raise ValueError(f"Duplicate split step {step_key}")
            self._inflight_steps.add(step_key)

    def _complete_step(self, step_key: tuple[int, str, str]) -> None:
        with self._step_lock:
            self._inflight_steps.discard(step_key)
            self._processed_steps.add(step_key)

    def _abort_step(self, step_key: tuple[int, str, str]) -> None:
        with self._step_lock:
            self._inflight_steps.discard(step_key)

    def _require_runtime_handle(self) -> SplitRuntimeHandle:
        if self.runtime_handle is None:
            raise RuntimeError("AutoSplitTailServerModel has not been configured.")
        return self.runtime_handle
