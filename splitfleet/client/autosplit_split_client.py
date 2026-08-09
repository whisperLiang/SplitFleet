"""Client-side TorchLens split-learning adapter with local prefix execution."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np

from splitfleet.autosplit import (
    AutoSplitSession,
    SplitRuntimeHandle,
    batch_window_contains,
    classify_contract_compatibility,
    describe_batch_window,
    first_tensor_batch_size,
    normalize_batch_window,
    normalize_inputs,
    require_batch_in_window,
)
from splitfleet.backends.utils import (
    adapter_for,
    bind_model_inputs,
    inference_context,
    model_training,
    move_value,
    zero_grad,
)
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
    AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY,
    AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY,
    AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY,
    AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_GRAPH_CONTRACT_DIGEST_CONFIG_KEY,
    AUTOSPLIT_MODEL_VERSION_CONFIG_KEY,
    AUTOSPLIT_MODEL_VERSION_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_MODE_CONFIG_KEY,
    AUTOSPLIT_PLAN_ID_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_SPLIT_ID_CONFIG_KEY,
    AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY,
)
from splitfleet.common.constants import CLIENT_ID_CONFIG_KEY


def _model_to_ndarrays(model: Any, adapter=None) -> list[np.ndarray]:
    adapter = adapter or adapter_for(model, ())
    return adapter.export_ndarrays(model)


def _load_model_from_ndarrays(model: Any, ndarrays: list[np.ndarray], adapter=None) -> None:
    (adapter or adapter_for(model, ())).load_ndarrays(model, ndarrays)


def _batch_size(value: Any) -> int:
    size = first_tensor_batch_size(value)
    if size is None:
        raise ValueError(
            "Split training needs a batched tensor in the model inputs to size a step."
        )
    return int(size)


PARTIAL_BATCH_POLICIES = ("error", "skip")


@dataclass(frozen=True)
class _RoundRuntime:
    """Everything a round needs that does not change between its batches.

    The graph contract in particular hashes the whole model state, so it is
    resolved once per round instead of once per batch.
    """

    handle: SplitRuntimeHandle
    contract: GraphContract
    batch_window: tuple[int, int] | None
    trace_batch_mode: str


@dataclass
class _RoundTelemetry:
    """Per-round split-execution counters reported back to the server."""

    num_batches: int = 0
    skipped_batches: int = 0
    skipped_examples: int = 0
    prefix_compute_sec: float = 0.0
    tail_wait_sec: float = 0.0
    upload_bytes: int = 0
    download_bytes: int = 0
    min_batch_size: int = 0
    max_batch_size: int = 0

    def record_batch(self, batch_size: int) -> None:
        self.num_batches += 1
        self.max_batch_size = max(self.max_batch_size, int(batch_size))
        self.min_batch_size = (
            int(batch_size) if self.min_batch_size == 0 else min(self.min_batch_size, int(batch_size))
        )

    def record_skip(self, batch_size: int) -> None:
        self.skipped_batches += 1
        self.skipped_examples += max(int(batch_size), 0)

    def as_metrics(self) -> dict[str, Any]:
        return asdict(self)


class AutoSplitSplitLearningClient(NumPyClient):
    """Run a TorchLens prefix locally and delegate suffix work to the server model."""

    def __init__(
        self,
        *,
        model: Any,
        train_data: Iterable[Any],
        sample_inputs: Any,
        evaluate_data: Optional[Iterable[Any]] = None,
        sample_kwargs: Optional[dict] = None,
        batch_adapter: Optional[Callable[[Any], tuple[Any, Any]]] = None,
        optimizer_fn=None,
        functional_update_fn: Optional[Callable[[Any, Any], Any]] = None,
        autosplit_session: Optional[AutoSplitSession] = None,
        device: str = "cpu",
        partial_batch_policy: str = "error",
    ) -> None:
        if sample_kwargs:
            raise ValueError("TorchLens autosplit backend accepts positional model inputs only.")
        if partial_batch_policy not in PARTIAL_BATCH_POLICIES:
            raise ValueError(
                f"Unsupported partial_batch_policy {partial_batch_policy!r}; "
                f"expected one of {PARTIAL_BATCH_POLICIES}."
            )
        self.backend_adapter = adapter_for(model, sample_inputs)
        self.model = self.backend_adapter.move_model(self.backend_adapter.clone_model(model), device)
        self.train_data = train_data
        self.evaluate_data = evaluate_data if evaluate_data is not None else train_data
        self.sample_inputs = bind_model_inputs(sample_inputs, self.backend_adapter)
        self.batch_adapter = batch_adapter or self._default_batch_adapter
        self.optimizer_fn = optimizer_fn
        self.functional_update_fn = functional_update_fn
        self.autosplit_session = autosplit_session or AutoSplitSession(device=device)
        self.device = device
        self.partial_batch_policy = partial_batch_policy
        self._runtime_cache: dict[str, _RoundRuntime] = {}
        self._context_store = PrefixContextStore()
        self._round_config: dict[str, Any] = {}

    def get_parameters(self, config):
        _ = config
        return _model_to_ndarrays(self.model, self.backend_adapter)

    def fit(self, parameters, config):
        fit_start = time.perf_counter()
        round_runtime = self._prepare_round(parameters, config, training=True)
        runtime_handle = round_runtime.handle
        contract = round_runtime.contract
        prefix_optimizer = self._build_optimizer()
        telemetry = _RoundTelemetry()
        round_id = int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0))
        client_id = str(config.get(CLIENT_ID_CONFIG_KEY, ""))
        plan_id = str(config[AUTOSPLIT_PLAN_ID_CONFIG_KEY])

        num_examples = 0
        weighted_loss = 0.0
        for batch in self.train_data:
            inputs, targets = self.batch_adapter(batch)
            torch_inputs = move_value(inputs, self.backend_adapter, self.device)
            torch_inputs = bind_model_inputs(torch_inputs, self.backend_adapter)
            torch_targets = move_value(targets, self.backend_adapter, self.device)
            batch_size = _batch_size(torch_inputs)
            if not self._accept_batch(batch_size, round_runtime, telemetry):
                continue

            zero_grad(self.model, prefix_optimizer)

            prefix_started = time.perf_counter()
            boundary = runtime_handle.backend.run_prefix(
                *normalize_inputs(torch_inputs),
                training=True,
            )
            telemetry.prefix_compute_sec += time.perf_counter() - prefix_started
            step_id = uuid.uuid4().hex
            self._context_store.put(round_id, client_id, step_id, boundary)
            wire_boundary = boundary_to_envelope(
                boundary,
                round_id=round_id,
                client_id=client_id,
                step_id=step_id,
                plan_id=plan_id,
                split_id=contract.split_id,
                canonical_graph_hash=contract.canonical_graph_hash,
                boundary_schema_hash=contract.boundary_schema_hash,
                model_version=round_id,
            )
            response = self._call_tail(
                method_name="train_tail",
                boundary=wire_boundary,
                targets=torch_targets,
                num_examples=batch_size,
                telemetry=telemetry,
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
            backward_started = time.perf_counter()
            prefix_result = runtime_handle.backend.backward_prefix(
                local_boundary,
                boundary_grads=envelope_to_gradients(gradient_envelope, self.device),
                optimizer=prefix_optimizer,
            )
            telemetry.prefix_compute_sec += time.perf_counter() - backward_started
            if self.backend_adapter.backend_name == "jax" and self.functional_update_fn is not None:
                update_target = (
                    self.backend_adapter.external_params
                    if self.backend_adapter.has_external_params
                    else self.model
                )
                updated = self.functional_update_fn(update_target, prefix_result)
                if self.backend_adapter.has_external_params and updated is not None:
                    self.backend_adapter.bind_external_params(updated)

            batch_examples = int(response["num_examples"])
            batch_loss = float(response["loss"])
            num_examples += batch_examples
            weighted_loss += batch_loss * batch_examples
            telemetry.record_batch(batch_size)

        average_loss = weighted_loss / max(num_examples, 1)
        metrics = {
            "loss": average_loss,
            "fit_duration_sec": time.perf_counter() - fit_start,
            "num_examples": num_examples,
            "boundary": str(runtime_handle.plan.boundary),
            "plan_id": str(runtime_handle.plan.plan_id),
            "device": str(self.device),
            **telemetry.as_metrics(),
        }
        return _model_to_ndarrays(self.model, self.backend_adapter), num_examples, metrics

    def evaluate(self, parameters, config):
        round_runtime = self._prepare_round(parameters, config, training=False)
        runtime_handle = round_runtime.handle
        contract = round_runtime.contract
        telemetry = _RoundTelemetry()
        round_id = int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0))
        client_id = str(config.get(CLIENT_ID_CONFIG_KEY, ""))
        plan_id = str(config[AUTOSPLIT_PLAN_ID_CONFIG_KEY])

        num_examples = 0
        weighted_loss = 0.0
        with inference_context(self.backend_adapter.backend_name):
            for batch in self.evaluate_data:
                inputs, targets = self.batch_adapter(batch)
                torch_inputs = move_value(inputs, self.backend_adapter, self.device)
                torch_inputs = bind_model_inputs(torch_inputs, self.backend_adapter)
                torch_targets = move_value(targets, self.backend_adapter, self.device)
                batch_size = _batch_size(torch_inputs)
                if not self._accept_batch(batch_size, round_runtime, telemetry):
                    continue
                boundary = runtime_handle.backend.run_prefix(*normalize_inputs(torch_inputs))
                wire_boundary = boundary_to_envelope(
                    boundary,
                    round_id=round_id,
                    client_id=client_id,
                    step_id=uuid.uuid4().hex,
                    plan_id=plan_id,
                    split_id=contract.split_id,
                    canonical_graph_hash=contract.canonical_graph_hash,
                    boundary_schema_hash=contract.boundary_schema_hash,
                    model_version=round_id,
                )
                response = self._call_tail(
                    method_name="evaluate_tail",
                    boundary=wire_boundary,
                    targets=torch_targets,
                    num_examples=batch_size,
                    telemetry=telemetry,
                )
                batch_examples = int(response["num_examples"])
                batch_loss = float(response["loss"])
                num_examples += batch_examples
                weighted_loss += batch_loss * batch_examples
                telemetry.record_batch(batch_size)

        average_loss = weighted_loss / max(num_examples, 1)
        metrics = {"loss": average_loss, **telemetry.as_metrics()}
        return float(average_loss), num_examples, metrics

    def _prepare_round(self, parameters, config, *, training: bool) -> _RoundRuntime:
        if config.get(AUTOSPLIT_BACKEND_CONFIG_KEY) != AUTOSPLIT_BACKEND_VALUE_TORCHLENS:
            raise ValueError("Split config must explicitly declare backend='torchlens'.")
        client_stage_count = int(config.get(AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY, 1))
        if client_stage_count != 1:
            raise ValueError(
                "TorchLens autosplit backend supports exactly one client-local prefix stage."
            )
        _load_model_from_ndarrays(self.model, parameters, self.backend_adapter)
        previous_round = int(self._round_config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, -1))
        current_round = int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0))
        if previous_round >= 0 and previous_round != current_round:
            self._context_store.discard_round(previous_round)
        self._round_config = dict(config)
        # Hashing the model state is not free, so the round hashes it once and
        # both the contract check and the runtime cache key reuse the result.
        state_schema = self.backend_adapter.state_manifest(self.model).schema_hash
        raw_version_contract = config.get(AUTOSPLIT_MODEL_VERSION_CONTRACT_CONFIG_KEY)
        if raw_version_contract:
            version_contract = ModelVersionContract.from_json(raw_version_contract)
            if version_contract.round_model_version != current_round:
                raise RuntimeError("Round model version contract mismatch")
            if version_contract.state_schema_hash != state_schema:
                raise RuntimeError("Client model state schema mismatch")
        self.backend_adapter.set_training(self.model, training)
        return self._ensure_round_runtime(config, state_schema)

    def _ensure_round_runtime(self, config, state_schema: str) -> _RoundRuntime:
        plan_id = str(config[AUTOSPLIT_PLAN_ID_CONFIG_KEY])
        split_id = str(config.get(AUTOSPLIT_SPLIT_ID_CONFIG_KEY, ""))
        graph_signature = str(config.get(AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY, ""))
        module_mode = "train" if model_training(self.model) else "eval"
        # The server negotiates the dynamic batch window for the whole fleet.
        # A device that re-derives it from its own sample batch would prepare a
        # runtime with a different feature ABI and reject every other batch size.
        dynamic_batch = normalize_batch_window(config.get(AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY))
        trace_batch_mode = str(config.get(AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY, "")) or None
        cache_key = "|".join([
            self.backend_adapter.backend_name, torchlens_runtime_version(), self.model.__class__.__qualname__,
            state_schema, plan_id, split_id, graph_signature, str(self.device), module_mode,
            describe_batch_window(dynamic_batch), str(trace_batch_mode or ""),
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
            dynamic_batch=dynamic_batch,
            trace_batch_mode=trace_batch_mode,
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
        contract = graph_contract_for_runtime_handle(handle)
        raw_contract = config.get(AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY)
        if raw_contract:
            expected = GraphContract.from_json(raw_contract)
            if config.get(AUTOSPLIT_GRAPH_CONTRACT_DIGEST_CONFIG_KEY) != expected.digest:
                raise RuntimeError("Configured graph contract digest is invalid")
            validate_contract(expected, contract)
        self._validate_feature_abi(config, handle)
        round_runtime = _RoundRuntime(
            handle=handle,
            contract=contract,
            batch_window=normalize_batch_window(handle.plan.dynamic_batch),
            trace_batch_mode=str(handle.plan.trace_batch_mode or ""),
        )
        self._runtime_cache[cache_key] = round_runtime
        return round_runtime

    def _validate_feature_abi(self, config, handle: SplitRuntimeHandle) -> None:
        """Fail the round before any boundary upload when the ABIs disagree.

        The full runtime contract digest also covers the trace batch size and
        the train/eval graph, both of which legitimately differ per device.
        Feature ABI equality is the property the boundary payload depends on.
        """

        expected_abi = str(config.get(AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY, ""))
        if not expected_abi:
            return
        if expected_abi == str(handle.plan.feature_abi_id):
            return
        compatibility = classify_contract_compatibility(
            config.get(AUTOSPLIT_RUNTIME_CONTRACT_CONFIG_KEY),
            handle.plan.runtime_contract,
        )
        raise RuntimeError(
            "Prefix feature ABI does not match the ABI announced by the server: "
            f"prepared {handle.plan.feature_abi_id!r}, expected {expected_abi!r} "
            f"(reason={compatibility.get('reason')!r}, "
            f"dynamic_batch={describe_batch_window(handle.plan.dynamic_batch)}, "
            f"trace_batch_mode={handle.plan.trace_batch_mode!r}). "
            "The client sample inputs must produce the same boundary schema as the "
            "strategy sample inputs."
        )

    def _accept_batch(
        self,
        batch_size: int,
        round_runtime: _RoundRuntime,
        telemetry: _RoundTelemetry,
    ) -> bool:
        """Apply the partial-batch policy to a batch the runtime cannot replay."""

        if batch_window_contains(round_runtime.batch_window, batch_size):
            return True
        if self.partial_batch_policy == "skip":
            telemetry.record_skip(batch_size)
            return False
        require_batch_in_window(
            batch_size,
            round_runtime.batch_window,
            stage="Split prefix",
            trace_batch_mode=round_runtime.trace_batch_mode,
        )
        return True

    def _call_tail(
        self,
        *,
        method_name: str,
        boundary,
        targets,
        num_examples: int,
        telemetry: Optional[_RoundTelemetry] = None,
    ):
        boundary_payload = encode_boundary(boundary)
        target_payload = encode_bundle_wire(targets, backend=self.backend_adapter.backend_name)
        request = BatchData(
            data={
                "boundary": boundary_payload,
                "targets": target_payload,
                "metadata": json.dumps({"num_examples": num_examples}).encode("utf-8"),
            },
            control_code=ControlCode.OK,
        )
        started = time.perf_counter()
        response = getattr(self._require_server_model_proxy(), method_name)(
            request,
            _streams_=False,
        )
        if telemetry is not None:
            telemetry.tail_wait_sec += time.perf_counter() - started
            telemetry.upload_bytes += len(boundary_payload) + len(target_payload)
            telemetry.download_bytes += sum(
                len(value) for value in response.data.values() if isinstance(value, bytes)
            )
        metadata = json.loads(response.data["metadata"].decode("utf-8"))
        if "gradients" in response.data:
            metadata["gradients"] = decode_gradients(response.data["gradients"])
        return metadata

    def _build_optimizer(self):
        if self.optimizer_fn is not None:
            return self.optimizer_fn(self.model)
        if self.backend_adapter.backend_name == "jax":
            return None
        try:
            return self.backend_adapter.build_optimizer(self.model, {"name": "sgd", "lr": 0.01})
        except (NotImplementedError, ValueError):
            return None

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
