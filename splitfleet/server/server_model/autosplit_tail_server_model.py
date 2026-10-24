"""TorchLens suffix server model for client-prefix split learning."""

from __future__ import annotations

import json
import hashlib
import time
import uuid
from functools import wraps
from threading import RLock
from typing import Any

from splitfleet.autosplit import (
    TorchLensRuntimeHandle,
    normalize_batch_window,
    require_batch_in_window,
)
from splitfleet.backends.utils import adapter_for, native_device_method
from splitfleet.common import (
    BatchData,
    ControlCode,
    ServerModelEvaluateIns,
    ServerModelFitIns,
    ServerModelFitRes,
)
from splitfleet.common.constants import (
    AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_GRAPH_CONTRACT_DIGEST_CONFIG_KEY,
    AUTOSPLIT_MODEL_VERSION_CONFIG_KEY,
    AUTOSPLIT_MODEL_VERSION_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_PLAN_ID_CONFIG_KEY,
    AUTOSPLIT_REPLICA_RNG_SEED_CONFIG_KEY,
)
from splitfleet.server.server_model.server_model import ServerModel
from splitfleet.split_engine import graph_contract_for_runtime_handle
from splitfleet.split_engine.contracts import (
    GraphContract,
    ModelVersionContract,
    validate_contract,
)
from splitfleet.tasks import TaskAdapter
from splitfleet.transport import (
    decode_boundary,
    decode_bundle_wire,
    encode_bundle_wire,
    encode_gradients,
)
from splitfleet.transport.split_wire import envelope_to_boundary, gradients_to_envelope


def _request_num_examples(payload: bytes) -> int:
    """Read the example count the prefix measured for this step."""

    metadata = json.loads(payload.decode("utf-8"))
    return int(metadata["num_examples"])


def _configuration_scope(method):
    """TF preparation scopes native phases separately from optional capture."""
    captured = native_device_method(capture=True)(method)

    @wraps(method)
    def configure(self, *args, **kwargs):
        if self.backend_adapter.backend_name == 'tf':
            return method(self, *args, **kwargs)
        return captured(self, *args, **kwargs)

    return configure


class AutoSplitTailServerModel(ServerModel):
    """Execute the TorchLens suffix for split-learning rounds."""

    def __init__(
        self,
        *,
        runtime_manager,
        model: Any,
        sample_inputs: Any = (),
        optimizer_fn=None,
        loss_fn=None,
        task: TaskAdapter | None = None,
        boundary: str = "50%",
        mode: str = "generated_eager",
        device: str = "cpu",
        defer_model: bool = False,
    ) -> None:
        self.runtime_manager = runtime_manager
        self.optimizer_fn = optimizer_fn
        self.task = task
        self.loss_fn = loss_fn if loss_fn is not None else (task.loss if task is not None else None)
        self.boundary = boundary
        self.mode = mode
        self.device = device
        self.backend_adapter = adapter_for(model, sample_inputs)
        self._replica_rng_seed_base = 0
        if self.backend_adapter.backend_name == "torch":
            import torch
            self._replica_rng_seed_base = torch.initial_seed()
        self._model_template = model if defer_model else None
        self.model = None if defer_model else self._clone_native_model(model)
        self.offloaded_parameters = None
        self.optimizer = None
        self.runtime_handle: TorchLensRuntimeHandle | None = None
        self.graph_contract: GraphContract | None = None
        self.batch_window: tuple[int, int] | None = None
        self.state_ownership = None
        self.num_examples = 0
        self.loss_total = 0.0
        self.sid = ""
        self.model_version = 0
        self.plan_id = ""
        self._processed_steps: set[tuple[int, str, str]] = set()
        self._inflight_steps: set[tuple[int, str, str]] = set()
        self._step_lock = RLock()
        self._runtime_cache: dict[tuple[str, str, bool, str], TorchLensRuntimeHandle] = {}
        self._replica_rng_state = None
        self._replica_rng_seed = None
        self.runtime_prepare_sec = 0.0
        self.boundary_deserialize_sec = 0.0
        self.gradient_serialize_sec = 0.0

    def runtime_handles(self):
        """Return owned runtime handles for lifecycle management."""
        return tuple(self._runtime_cache.values())

    def reset_task_state(self) -> None:
        """Clear task-local optimizer/step state before a physical slot is reused."""
        if self._inflight_steps:
            raise RuntimeError("Cannot reset a suffix with an in-flight step")
        self.optimizer = None
        self.sid = ""
        self.model_version = 0
        self.plan_id = ""
        self.num_examples = 0
        self.loss_total = 0.0
        self._processed_steps.clear()
        self.state_ownership = None
        self.graph_contract = None
        self.batch_window = None
        self.runtime_handle = None
        self._replica_rng_state = None
        self._replica_rng_seed = None

    def close_runtime(self) -> None:
        """Drop all per-replica runtime references without touching model state."""
        if self._inflight_steps:
            raise RuntimeError("Cannot close a suffix with an in-flight step")
        self.capture_replica_rng_state()
        self._runtime_cache.clear()
        self.runtime_handle = None
        self.optimizer = None

    def capture_replica_rng_state(self) -> None:
        """Persist RNG before the shared runtime handle is released."""
        if self.runtime_handle is None:
            return
        capture = getattr(self.runtime_handle.backend, "capture_replica_rng_state", None)
        if capture is not None:
            self._replica_rng_state = capture(self.device)

    reset_execution = reset_task_state
    close = close_runtime

    def get_parameters(self):
        if self.model is None:
            raise RuntimeError("The private suffix is offloaded; acquire its task before native execution")
        return self.backend_adapter.export_ndarrays(self.model)

    def offload_model(self):
        """Keep this client's object/state binding while dropping native tensors."""
        if self._inflight_steps:
            raise RuntimeError("Cannot offload a suffix with an in-flight step")
        if self._model_template is None:
            raise RuntimeError("Only a deferred per-client suffix may be offloaded")
        self.model = None
        self.optimizer = None

    @native_device_method
    def _clone_native_model(self, template):
        return self.backend_adapter.move_model(self.backend_adapter.clone_model(template), self.device)

    @_configuration_scope
    def configure_fit(self, ins: ServerModelFitIns) -> None:
        self._configure_common(ins.parameters, ins.config, sid=ins.sid, training=True)
        self._configure_optimizer()

    @native_device_method
    def _configure_optimizer(self):
        if self.optimizer_fn is not None:
            self.optimizer = self.optimizer_fn(self.model)
        else:
            self.optimizer = self.backend_adapter.build_optimizer(self.model, {"name": "sgd", "lr": 0.01})

    @native_device_method
    def get_fit_result(self) -> ServerModelFitRes:
        average_loss = self.loss_total / max(self.num_examples, 1)
        from splitfleet.autosplit.state_ownership import DIGEST_KEY, export_owned
        ownership = getattr(self, "state_ownership", None)
        started = time.perf_counter()
        values = export_owned(self.model, ownership, "suffix") if ownership else self.get_parameters()
        state_metrics = {"state_export_sec": time.perf_counter() - started,
                         "state_tensor_count": len(values), "state_bytes": sum(v.nbytes for v in values)}
        runtime = self._require_runtime_handle().runtime
        pool = getattr(runtime, '_splitfleet_shared_suffix_functions', None)
        if pool is not None:
            from splitfleet.runtime.tensorflow_forward import _ReboundSuffixTraining
            functions = getattr(runtime, '_splitfleet_native_training_functions', {})
            state_metrics.update(shared_native_suffix_graphs=pool.compile_count,
                shared_native_suffix_trace_sec=pool.compile_sec,
                suffix_native_function_rebound=bool(functions) and all(
                    isinstance(value, _ReboundSuffixTraining) for value in functions.values()))
        if ownership:
            state_metrics[DIGEST_KEY] = ownership[DIGEST_KEY]
        return ServerModelFitRes(
            parameters=values,
            config={
                **state_metrics,
                "num_examples": self.num_examples,
                "avg_loss": average_loss,
                "runtime_prepare_sec": self.runtime_prepare_sec,
                "runtime_capture_reused": bool(self._require_runtime_handle().plan.metadata.get("_reused_capture")),
                "boundary_deserialize_sec": self.boundary_deserialize_sec,
                "gradient_serialize_sec": self.gradient_serialize_sec,
            },
        )

    @_configuration_scope
    def configure_evaluate(self, ins: ServerModelEvaluateIns) -> None:
        self._configure_common(ins.parameters, ins.config, sid=ins.sid, training=False)
        self.optimizer = None

    @native_device_method
    def train_tail(self, batches: list[BatchData]) -> list[BatchData]:
        runtime_handle = self._require_runtime_handle()
        responses = []
        for batch in batches:
            deserialize_started = time.perf_counter()
            wire_boundary = decode_boundary(batch.data["boundary"])
            self._validate_wire_identity(wire_boundary)
            step_key = (wire_boundary.round_id, wire_boundary.client_id, wire_boundary.step_id)
            boundary = envelope_to_boundary(wire_boundary, runtime_handle.runtime, self.device)
            self.boundary_deserialize_sec += time.perf_counter() - deserialize_started
            targets = decode_bundle_wire(batch.data["targets"], self.device)
            examples = _request_num_examples(batch.data["metadata"])
            self._begin_step(step_key)
            measurements: dict[str, float] = {}
            try:
                result = self.runtime_manager.autosplit_session.run_suffix_train(
                    runtime_handle,
                    boundary,
                    targets=targets,
                    loss_fn=self.loss_fn,
                    optimizer=self.optimizer,
                    measurements=measurements,
                )
                auxiliary = getattr(runtime_handle.backend, "train_suffix_auxiliary", None)
                prefix_statistics = None
                if auxiliary is not None:
                    from splitfleet.autosplit.state_ownership import state_ownership
                    from splitfleet.runtime.auxiliary_state import (
                        commit_auxiliary_state,
                    )
                    from splitfleet.runtime.owned_auxiliary_state import (
                        expected_statistics_names,
                        statistics_for_stage,
                    )
                    ownership = self.state_ownership or state_ownership(
                        runtime_handle, self.backend_adapter.state_manifest(self.model).schema_hash
                    )
                    suffix_statistics = statistics_for_stage(self.model, auxiliary, ownership, "suffix")
                    prefix_statistics = statistics_for_stage(self.model, auxiliary, ownership, "prefix")
                    if set(auxiliary) != (expected_statistics_names(self.model, ownership, "prefix")
                                          | expected_statistics_names(self.model, ownership, "suffix")):
                        raise RuntimeError("Native suffix output is missing executed BN statistics")
                    commit_auxiliary_state(self.model, suffix_statistics)
            except Exception:
                self._abort_step(step_key)
                raise
            self._complete_step(step_key)
            if "server_total_ms" not in measurements:
                raise RuntimeError(
                    "Split suffix runtime did not report server_total_ms"
                )
            loss_value = self.backend_adapter.scalar_value(result["loss"])
            self.num_examples += examples
            self.loss_total += loss_value * examples
            serialize_started = time.perf_counter()
            gradients = gradients_to_envelope(wire_boundary, result["boundary_grads"])
            gradient_payload = encode_gradients(gradients)
            self.gradient_serialize_sec += time.perf_counter() - serialize_started
            responses.append(
                BatchData(
                    data={
                        "gradients": gradient_payload,
                        **({"auxiliary_state": encode_bundle_wire(
                            prefix_statistics, backend=self.backend_adapter.backend_name
                        )} if prefix_statistics is not None else {}),
                        "metadata": json.dumps(
                            {
                                "loss": loss_value,
                                "num_examples": examples,
                                "server_service_ms": float(measurements["server_total_ms"]),
                                "server_forward_ms": measurements.get("server_forward_ms"),
                                "server_loss_ms": measurements.get("server_loss_ms"),
                                "server_backward_ms": measurements.get("server_backward_ms"),
                            },
                            sort_keys=True,
                        ).encode("utf-8"),
                    },
                    control_code=ControlCode.OK,
                    metadata={"sid": self.sid},
                )
            )
        return responses

    @native_device_method
    def evaluate_tail(self, batches: list[BatchData]) -> list[BatchData]:
        runtime_handle = self._require_runtime_handle()
        responses = []
        for batch in batches:
            wire_boundary = decode_boundary(batch.data["boundary"])
            self._validate_wire_identity(wire_boundary)
            boundary = envelope_to_boundary(wire_boundary, runtime_handle.runtime, self.device)
            targets = decode_bundle_wire(batch.data["targets"], self.device)
            outputs = self.runtime_manager.autosplit_session.run_suffix_eval(runtime_handle, boundary)
            loss = self.runtime_manager.autosplit_session.compute_loss(outputs, targets, self.loss_fn)
            examples = _request_num_examples(batch.data["metadata"])
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

    @_configuration_scope
    def _configure_common(self, parameters, config, *, sid: str, training: bool) -> None:
        client_stage_count = int(config.get(AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY, 1))
        if client_stage_count != 1:
            raise ValueError(
                "TorchLens autosplit backend supports exactly one client-local prefix stage."
            )
        self.sid = sid
        self.num_examples = 0
        self.loss_total = 0.0
        self.boundary_deserialize_sec = 0.0
        self.gradient_serialize_sec = 0.0
        self._processed_steps.clear()
        self._inflight_steps.clear()
        self.model_version = int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0))
        self.plan_id = str(config.get(AUTOSPLIT_PLAN_ID_CONFIG_KEY, ""))
        state_schema = self._configure_native_state(parameters, config, training)
        runtime_key = (self.plan_id, state_schema, bool(training), str(self.device))
        prepare_started = time.perf_counter()
        self.runtime_handle = self._runtime_cache.get(runtime_key)
        if self.runtime_handle is None:
            # Only a fallback native capture needs the exclusive hook gate.
            self.runtime_manager.release_model_runtime(self)
            plan_suffix = f"{sid or 'shared'}_{uuid.uuid4().hex[:8]}"
            self.runtime_handle = self.runtime_manager.clone_runtime_for_model(
                self.model, suffix=plan_suffix, plan_id=self.plan_id)
            self._runtime_cache[runtime_key] = self.runtime_handle
        seed_rng = getattr(self.runtime_handle.backend, "enable_replica_rng_isolation", None)
        if seed_rng is not None:
            # Virtual strategies supply a seed derived from the experiment and
            # stable partition, never Flower's random transport node ID. Native
            # callers default to their initial Torch seed and client identity.
            seed = config.get(AUTOSPLIT_REPLICA_RNG_SEED_CONFIG_KEY)
            if seed is None:
                seed = int.from_bytes(hashlib.sha256(
                    f"{self._replica_rng_seed_base}:{self.sid}:suffix".encode()
                ).digest()[:8], "little")
            if type(seed) is not int:
                raise ValueError("The suffix replica RNG seed must be an integer")
            seed_rng(getattr(self.runtime_manager, "replica_rng_isolation", False), seed=seed)
            if self._replica_rng_state is not None and self._replica_rng_seed == seed:
                restore = getattr(self.runtime_handle.backend, "restore_replica_rng_state", None)
                if restore is not None:
                    restore(self._replica_rng_state, self.device)
            self._replica_rng_state = None
            self._replica_rng_seed = seed
        self.runtime_prepare_sec = time.perf_counter() - prepare_started
        self._configure_native_contract(config, state_schema)

    @native_device_method
    def _configure_native_state(self, parameters, config, training):
        if self.model is None:
            self.model = self._clone_native_model(self._model_template)
        self.offloaded_parameters = None
        self.backend_adapter.load_ndarrays(self.model, parameters)
        # Hashing the model state is not free, so the round hashes it once and
        # both the contract check and the runtime cache key reuse the result.
        state_schema = self.backend_adapter.state_manifest(self.model).schema_hash
        raw_version_contract = config.get(AUTOSPLIT_MODEL_VERSION_CONTRACT_CONFIG_KEY)
        if raw_version_contract:
            version_contract = ModelVersionContract.from_json(raw_version_contract)
            if version_contract.round_model_version != self.model_version:
                raise RuntimeError("Round model version contract mismatch")
            if version_contract.state_schema_hash != state_schema:
                raise RuntimeError("Suffix model state schema mismatch")
        self.backend_adapter.set_training(self.model, training)
        return state_schema

    @native_device_method
    def _configure_native_contract(self, config, state_schema):
        from splitfleet.autosplit.state_ownership import read_ownership
        self.state_ownership = read_ownership(config, self.runtime_handle, state_schema)
        # The contract hashes the whole model state, so it is resolved once per
        # round here rather than once per uploaded boundary.
        self.graph_contract = graph_contract_for_runtime_handle(self.runtime_handle)
        self.batch_window = normalize_batch_window(self.runtime_handle.plan.dynamic_batch)
        raw_contract = config.get(AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY)
        if raw_contract:
            expected = GraphContract.from_json(raw_contract)
            if config.get(AUTOSPLIT_GRAPH_CONTRACT_DIGEST_CONFIG_KEY) != expected.digest:
                raise RuntimeError("Configured graph contract digest is invalid")
            validate_contract(expected, self.graph_contract)

    def _validate_wire_identity(self, boundary) -> None:
        runtime_handle = self._require_runtime_handle()
        contract = self.graph_contract
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
        batch_size = int(boundary.batch_size)
        if batch_size < 1:
            raise ValueError("Boundary batch size must be positive.")
        require_batch_in_window(
            batch_size,
            self.batch_window,
            stage=f"Split suffix (sid={self.sid or 'shared'})",
            trace_batch_mode=str(runtime_handle.plan.trace_batch_mode or ""),
        )

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

    def _require_runtime_handle(self) -> TorchLensRuntimeHandle:
        if self.runtime_handle is None:
            raise RuntimeError("AutoSplitTailServerModel has not been configured.")
        return self.runtime_handle
