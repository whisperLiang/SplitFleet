"""Client-side TorchLens split-learning adapter with local prefix execution."""

from __future__ import annotations

import gc
import json
import math
import platform
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from statistics import fmean, median
from typing import Any

from splitfleet.autosplit import (
    AutoSplitSession,
    TorchLensRuntimeHandle,
    batch_window_contains,
    classify_contract_compatibility,
    describe_batch_window,
    normalize_batch_window,
    require_batch_in_window,
)
from splitfleet.autosplit.torchlens_runtime import (
    model_input_batch_size,
    runtime_input_batch_size,
    torchlens_runtime_version,
)
from splitfleet.backends.utils import (
    adapter_for,
    bind_model_inputs,
    inference_context,
    model_training,
    move_value,
    zero_grad,
)
from splitfleet.client.numpy_client import NumPyClient
from splitfleet.common import BatchData, ControlCode
from splitfleet.common.constants import (
    AUTOSPLIT_BACKEND_CONFIG_KEY,
    AUTOSPLIT_BACKEND_VALUE_TORCHLENS,
    AUTOSPLIT_BOUNDARY_CONFIG_KEY,
    AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY,
    AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY,
    AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_GRAPH_CONTRACT_DIGEST_CONFIG_KEY,
    AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY,
    AUTOSPLIT_MODE_CONFIG_KEY,
    AUTOSPLIT_MODEL_VERSION_CONFIG_KEY,
    AUTOSPLIT_MODEL_VERSION_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_PLAN_ID_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_SPLIT_ID_CONFIG_KEY,
    AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY,
    AUTOSPLIT_TRAINABLE_CONFIG_KEY,
    CLIENT_ID_CONFIG_KEY,
    PLACEMENT_CALIBRATION_HASH_HANDSHAKE_KEY,
    PLACEMENT_CALIBRATION_MODEL_HASH_CONFIG_KEY,
    PLACEMENT_CALIBRATION_PARAMETERS_CONFIG_KEY,
    PLACEMENT_CALIBRATION_PARAMETERS_REQUIRED_KEY,
    TRANSPORT_CLIENT_SEND_NS_METADATA_KEY,
    TRANSPORT_SERVER_ELAPSED_NS_METADATA_KEY,
    TRANSPORT_SERVER_RECEIVE_NS_METADATA_KEY,
    TRANSPORT_SERVER_SEND_NS_METADATA_KEY,
)
from splitfleet.runtime import PrefixContextStore
from splitfleet.split_engine import graph_contract_for_runtime_handle
from splitfleet.split_engine.contracts import (
    GraphContract,
    ModelVersionContract,
    validate_contract,
)
from splitfleet.tasks import (
    ModelInputs,
    TaskAdapter,
    TaskBatch,
    infer_num_examples,
    prepare_task_batch,
)
from splitfleet.transport import (
    decode_bundle_wire,
    decode_gradients,
    encode_boundary,
    encode_bundle_wire,
)
from splitfleet.transport.split_wire import boundary_to_envelope, envelope_to_gradients

PARTIAL_BATCH_POLICIES = ("error", "skip")


def _model_precision(model: Any) -> str:
    """Return a stable precision class when a backend exposes parameter dtypes."""

    parameters = getattr(model, "parameters", None)
    try:
        values = parameters() if callable(parameters) else getattr(model, "trainable_variables", ())
        first = next(iter(values))
        dtype = str(first.dtype).lower()
    except (AttributeError, StopIteration, TypeError, RuntimeError, ValueError):
        return "fp32"
    if "bfloat16" in dtype:
        return "bf16"
    if "float16" in dtype:
        return "fp16"
    if "float64" in dtype or "double" in dtype:
        return "fp64"
    return "fp32"


def _accelerator_name(device: str) -> str:
    kind = str(device).split(":", 1)[0].lower()
    if kind == "cpu":
        return platform.machine().lower() or "generic_cpu"
    if kind == "cuda":
        try:
            import torch

            return str(torch.cuda.get_device_name(device)).lower().replace("|", "/")
        except (RuntimeError, AssertionError, ValueError, OSError, ImportError, AttributeError):
            return "generic_cuda"
    return f"generic_{kind}"


def _synchronize_prefix_device(backend_name: str, device: str, values=None) -> None:
    """Include queued CUDA work in measured prefix phases."""

    if backend_name == "torch" and str(device).split(":", 1)[0].lower() == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _transport_phase_ms(
    metadata: dict[str, Any],
    *,
    client_send_ns: int,
    client_receive_ns: int,
) -> tuple[float | None, float | None]:
    """Return one-way phases only when cross-host clocks are causally ordered.

    Deployments must synchronize wall clocks (for example with PTP or NTP).
    Invalid ordering leaves both measurements missing; round-trip duration is
    never divided using an assumed upload/download ratio.
    """

    try:
        server_receive_ns = int(
            metadata[TRANSPORT_SERVER_RECEIVE_NS_METADATA_KEY]
        )
        server_send_ns = int(metadata[TRANSPORT_SERVER_SEND_NS_METADATA_KEY])
    except (KeyError, TypeError, ValueError):
        return None, None
    if not (
        int(client_send_ns)
        <= server_receive_ns
        <= server_send_ns
        <= int(client_receive_ns)
    ):
        return None, None
    return (
        (server_receive_ns - int(client_send_ns)) / 1_000_000.0,
        (int(client_receive_ns) - server_send_ns) / 1_000_000.0,
    )


def _transport_roundtrip_ms(metadata, *, rpc_elapsed_ms):
    """Combined transport overhead from local spans, independent of clock offset.

    The server span includes its queue and processing. No upload/download split
    is inferred, and impossible durations stay missing rather than becoming zero.
    """
    try:
        server_ms = int(metadata[TRANSPORT_SERVER_ELAPSED_NS_METADATA_KEY]) / 1_000_000.0
        elapsed = float(rpc_elapsed_ms)
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(elapsed) or elapsed < 0 or server_ms < 0 or server_ms > elapsed:
        return None
    return elapsed - server_ms


@dataclass(frozen=True)
class _RoundRuntime:
    """Everything a round needs that does not change between its batches.

    The graph contract in particular hashes the whole model state, so it is
    resolved once per round instead of once per batch.
    """

    handle: TorchLensRuntimeHandle
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
    suffix_server_queue_sec: float = 0.0
    upload_bytes: int = 0
    download_bytes: int = 0
    activation_upload_bytes: int = 0
    target_upload_bytes: int = 0
    request_metadata_upload_bytes: int = 0
    gradient_download_bytes: int = 0
    response_metadata_download_bytes: int = 0
    boundary_serialize_sec: float = 0.0
    gradient_deserialize_sec: float = 0.0
    min_batch_size: int = 0
    max_batch_size: int = 0
    _client_forward_samples_ms: list[float] = field(default_factory=list, repr=False)
    _client_backward_samples_ms: list[float] = field(default_factory=list, repr=False)
    _server_forward_samples_ms: list[float] = field(default_factory=list, repr=False)
    _server_loss_samples_ms: list[float] = field(default_factory=list, repr=False)
    _server_backward_samples_ms: list[float] = field(default_factory=list, repr=False)
    _server_service_samples_ms: list[float] = field(default_factory=list, repr=False)
    _server_resource_service_samples_ms: list[float] = field(default_factory=list, repr=False)
    _network_upload_samples_ms: list[float] = field(default_factory=list, repr=False)
    _network_download_samples_ms: list[float] = field(default_factory=list, repr=False)
    _network_roundtrip_samples_ms: list[float] = field(default_factory=list, repr=False)
    _server_load_samples: dict[str, list[float]] = field(default_factory=dict, repr=False)

    def record_batch(self, batch_size: int) -> None:
        self.num_batches += 1
        self.max_batch_size = max(self.max_batch_size, int(batch_size))
        self.min_batch_size = (
            int(batch_size) if self.min_batch_size == 0 else min(self.min_batch_size, int(batch_size))
        )

    def record_skip(self, batch_size: int) -> None:
        self.skipped_batches += 1
        self.skipped_examples += max(int(batch_size), 0)

    def record_component(self, name: str, milliseconds: float | None) -> None:
        if milliseconds is None:
            return
        value = float(milliseconds)
        if value < 0:
            return
        samples = getattr(self, f"_{name}_samples_ms")
        samples.append(value)

    def as_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "num_batches": self.num_batches,
            "skipped_batches": self.skipped_batches,
            "skipped_examples": self.skipped_examples,
            "prefix_compute_sec": self.prefix_compute_sec,
            "tail_wait_sec": self.tail_wait_sec,
            "suffix_server_queue_sec": self.suffix_server_queue_sec,
            "upload_bytes": self.upload_bytes,
            "download_bytes": self.download_bytes,
            "activation_upload_bytes": self.activation_upload_bytes,
            "target_upload_bytes": self.target_upload_bytes,
            "request_metadata_upload_bytes": self.request_metadata_upload_bytes,
            "gradient_download_bytes": self.gradient_download_bytes,
            "response_metadata_download_bytes": self.response_metadata_download_bytes,
            "boundary_serialize_sec": self.boundary_serialize_sec,
            "gradient_deserialize_sec": self.gradient_deserialize_sec,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
        }
        for name, samples in self._server_load_samples.items():
            if samples:
                metrics[f"server_load_{name}"] = float(fmean(samples))
        for name in (
            "client_forward",
            "client_backward",
            "server_forward",
            "server_loss",
            "server_backward",
            "server_service",
            "server_resource_service",
            "network_upload",
            "network_download",
            "network_roundtrip",
        ):
            samples = getattr(self, f"_{name}_samples_ms")
            if samples:
                metrics[f"{name}_sec"] = float(sum(samples)) / 1000.
                metrics[f"{name}_ms"] = float(median(samples))
                metrics[f"{name}_mean_ms"] = float(fmean(samples))
                metrics[f"{name}_samples"] = len(samples)
        return metrics


class AutoSplitSplitLearningClient(NumPyClient):
    """Run a TorchLens prefix locally and delegate suffix work to the server model."""

    def __init__(
        self,
        *,
        model: Any,
        train_data: Iterable[Any],
        sample_inputs: Any,
        evaluate_data: Iterable[Any] | None = None,
        sample_kwargs: dict | None = None,
        batch_axes: dict[str, int] | None = None,
        batch_adapter: Callable[[Any], tuple[Any, Any]] | None = None,
        task: TaskAdapter | None = None,
        optimizer_fn=None,
        autosplit_session: AutoSplitSession | None = None,
        device: str = "cpu",
        partial_batch_policy: str = "error",
        max_cached_runtimes: int | None = None,
        calibration_batch: TaskBatch | None = None,
    ) -> None:
        if partial_batch_policy not in PARTIAL_BATCH_POLICIES:
            raise ValueError(
                f"Unsupported partial_batch_policy {partial_batch_policy!r}; "
                f"expected one of {PARTIAL_BATCH_POLICIES}."
            )
        if max_cached_runtimes is not None and max_cached_runtimes < 1:
            raise ValueError("max_cached_runtimes must be positive")
        sample_call = ModelInputs.from_value(sample_inputs.inputs if isinstance(sample_inputs, TaskBatch) else sample_inputs)
        if sample_kwargs is not None:
            if sample_call.kwargs:
                raise ValueError("Provide keyword samples in ModelInputs or sample_kwargs, not both.")
            sample_call = ModelInputs(sample_call.args, sample_kwargs)
        self.backend_adapter = adapter_for(model, sample_call.args if sample_call.args else sample_call.kwargs)
        self.model = self.backend_adapter.move_model(self.backend_adapter.clone_model(model), device)
        self.train_data = train_data
        self.evaluate_data = evaluate_data if evaluate_data is not None else train_data
        sample_call = sample_call.map_values(lambda value: move_value(value, self.backend_adapter, device))
        self.sample_inputs = bind_model_inputs(sample_call.args, self.backend_adapter, model=self.model)
        self.sample_kwargs = dict(sample_call.kwargs)
        self.batch_axes = batch_axes
        self.task = task
        self._calibration_targets = (calibration_batch.targets if calibration_batch is not None
                                     else sample_inputs.targets if isinstance(sample_inputs, TaskBatch) else None)
        self._calibration_inputs = (calibration_batch.inputs.map_values(
            lambda value: move_value(value, self.backend_adapter, device)) if calibration_batch is not None else None)
        self._sample_num_examples = sample_inputs.num_examples if isinstance(sample_inputs, TaskBatch) else None
        self.online_calibration_receipt = None
        self.batch_adapter = batch_adapter
        self.optimizer_fn = optimizer_fn
        self.autosplit_session = autosplit_session or AutoSplitSession(device=device)
        self.device = device
        self.partial_batch_policy = partial_batch_policy
        self.max_cached_runtimes = max_cached_runtimes
        self._runtime_cache: dict[str, _RoundRuntime] = {}
        # A runtime prepared before Flower negotiates the first round.  The
        # graph, batch contract, and TorchLens ABI are independent of the
        # round's plan id, so the first server config can adopt this handle
        # (or repartition it) instead of tracing the model on the critical
        # path of the first fit call.
        self._prewarmed_runtime: TorchLensRuntimeHandle | None = None
        self._calibration_runtime_cache = {}
        self._calibration_runtime_source = None
        self._last_split_prepare_sec = 0.0
        self._context_store = PrefixContextStore()
        self._round_config: dict[str, Any] = {}

    # These methods are the public lifecycle boundary used by ClientApp
    # adapters and worker reuse.  Runtime caches, TorchLens metadata and the
    # prefix context store remain implementation details of this executor.
    def bind_execution_context(
        self,
        *,
        train_data,
        sample_inputs,
        sample_kwargs=None,
        sample_num_examples=None,
        calibration_inputs=None,
        calibration_targets=None,
        evaluate_data=None,
        optimizer_fn=None,
    ) -> None:
        """Bind one logical task to this reusable native executor."""
        if len(self._context_store):
            raise RuntimeError("Cannot bind a new task with an unconsumed prefix context")
        self.server_model_proxy = None
        self.train_data = train_data
        self.evaluate_data = evaluate_data if evaluate_data is not None else train_data
        self.sample_inputs = sample_inputs
        self.sample_kwargs = dict(sample_kwargs or {})
        if sample_num_examples is not None:
            self._sample_num_examples = int(sample_num_examples)
        self._calibration_inputs = calibration_inputs
        self._calibration_targets = calibration_targets
        self.optimizer_fn = optimizer_fn
        self._round_config = {}
        self.online_calibration_receipt = None
        self._quality_rpc_spans = []
        self._quality_native_execution_intervals = []
        self._call_tail = type(self)._call_tail.__get__(self, type(self))
        self._clear_runtime_sample_bindings()
        handles = [row.handle for row in self._runtime_cache.values()]
        handles.extend(self._calibration_runtime_cache.values())
        if self._prewarmed_runtime is not None:
            handles.append(self._prewarmed_runtime)
        for handle in handles:
            handle.backend.trace_sample_input = self.sample_inputs
            handle.backend.trace_sample_kwargs = self.sample_kwargs
            handle.plan.metadata["_example_inputs"] = self.sample_inputs
            handle.plan.metadata["_example_kwargs"] = self.sample_kwargs

    def retire_execution(self) -> None:
        """Finish a task while retaining only reusable native graph capacity."""
        if len(self._context_store):
            raise RuntimeError("A completed execution retains an unconsumed prefix context")
        self.server_model_proxy = None
        self.online_calibration_receipt = None
        self._round_config = {}
        self.train_data = self.evaluate_data = ()
        self.sample_inputs, self.sample_kwargs = (), {}
        self._sample_num_examples = None
        self._calibration_inputs = self._calibration_targets = None
        self._quality_rpc_spans = []
        self._quality_native_execution_intervals = []
        self._call_tail = type(self)._call_tail.__get__(self, type(self))
        self._clear_runtime_sample_bindings()
        self._clear_model_gradients()

    def close_execution(self) -> None:
        """Release all graph/runtime resources owned by this executor."""
        self.retire_execution()
        self._runtime_cache.clear()
        self._calibration_runtime_cache.clear()
        self._calibration_runtime_source = None
        self._prewarmed_runtime = None
        self.autosplit_session.discard_runtime_handles()

    # Short lifecycle aliases for integrations that treat the executor as a
    # reusable resource rather than a Flower NumPyClient.
    reset_runtime = retire_execution
    close = close_execution

    def set_server_model_proxy(self, proxy) -> None:
        self.server_model_proxy = proxy

    def discard_round_context(self, round_id: int | None = None) -> None:
        """Discard uncommitted prefix activations after cancellation."""
        if round_id is None:
            self._context_store.clear()
        else:
            self._context_store.discard_round(int(round_id))

    @property
    def pending_context_count(self) -> int:
        return len(self._context_store)

    @property
    def current_round_id(self) -> int:
        return int(self._round_config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0))

    def quality_measurements(self) -> tuple[list[float], list[tuple[int, int]]]:
        return list(getattr(self, "_quality_rpc_spans", ())), list(
            getattr(self, "_quality_native_execution_intervals", ())
        )

    def enable_quality_measurement(self) -> None:
        """Record real suffix RPC and native execution intervals for evidence."""
        original_tail_call = type(self)._call_tail.__get__(self, type(self))

        def measured_tail_call(**kwargs):
            began = time.perf_counter()
            try:
                return original_tail_call(**kwargs)
            finally:
                self._quality_rpc_spans.append(time.perf_counter() - began)

        self._quality_rpc_spans = []
        self._quality_native_execution_intervals = []
        self._call_tail = measured_tail_call

    def set_previous_plan_hint(self, plan_id: str) -> None:
        """Tell round preparation which prior deployment should be reused."""
        self._round_config["autosplit_plan_id"] = str(plan_id)

    def _clear_runtime_sample_bindings(self) -> None:
        handles = [row.handle for row in self._runtime_cache.values()]
        handles.extend(self._calibration_runtime_cache.values())
        if self._prewarmed_runtime is not None:
            handles.append(self._prewarmed_runtime)
        for handle in handles:
            handle.backend.trace_sample_input = ()
            handle.backend.trace_sample_kwargs = {}
            handle.plan.metadata["_example_inputs"] = ()
            handle.plan.metadata["_example_kwargs"] = {}

    def _clear_model_gradients(self) -> None:
        if self.backend_adapter.backend_name == "torch":
            for parameter in self.model.parameters():
                parameter.grad = None

    def prewarm_runtime(
        self,
        *,
        boundary: str = "50%",
        mode: str = "generated_eager",
        dynamic_batch: tuple[int, int] | None = None,
        trace_batch_mode: str | None = None,
    ) -> None:
        """Trace one reusable split while the server is still starting.

        This is deliberately opt-in: callers that can start this work before
        connecting to Flower remove TorchLens graph preparation from the first
        training round.  If the server selects another boundary, the captured
        graph is repartitioned when the round config arrives.
        """

        self.backend_adapter.set_training(self.model, True)
        self._calibration_runtime_cache.clear()
        self._calibration_runtime_source = None
        handle = self.autosplit_session.prepare_runtime(
            self.model,
            self.sample_inputs,
            sample_kwargs=self.sample_kwargs,
            batch_axes=self.batch_axes,
            boundary=boundary,
            mode=mode,
            trainable=True,
            dynamic_batch=dynamic_batch,
            trace_batch_mode=trace_batch_mode,
        )
        # Round plans use canonical operation cuts. Percentages and module
        # aliases can resolve to that cut while retaining a different request
        # hash; use the native recut API to align the complete graph contract.
        if str(handle.runtime.request.boundary) != str(handle.plan.boundary):
            handle = self.autosplit_session.repartition_runtime(handle, handle.plan.boundary)
        self._prewarmed_runtime = handle

    def calibrate_runtime(self, targets, *, boundaries) -> dict:
        """Measure the captured model, restoring state after temporary calibration."""
        from splitfleet.server.placement.greedy.calibration import calibrate_split

        if self._prewarmed_runtime is None:
            raise ValueError("Prepare the runtime before measuring calibration anchors")
        if self.task is None:
            raise ValueError("Greedy calibration requires a task loss")
        if self.optimizer_fn is None:
            raise ValueError("training cost calibration requires an optimizer factory")
        targets = move_value(targets, self.backend_adapter, self.device)
        boundaries = tuple(boundaries)
        cache_native = self.backend_adapter.backend_name == 'tf' and bool(
            getattr(self.model, '_splitfleet_tensorflow_function_forward', False))
        if self._calibration_runtime_source is not self._prewarmed_runtime.runtime:
            self._calibration_runtime_cache.clear()
            self._calibration_runtime_source = self._prewarmed_runtime.runtime
        self._calibration_runtime_cache = {
            cut: handle for cut, handle in self._calibration_runtime_cache.items() if cut in boundaries}
        cache_hits = 0

        def calibration_handle(cut):
            nonlocal cache_hits
            if cache_native and cut in self._calibration_runtime_cache:
                cache_hits += 1
                return self._calibration_runtime_cache[cut]
            handle = self.autosplit_session.repartition_runtime(self._prewarmed_runtime, cut)
            if cache_native:
                self._calibration_runtime_cache[cut] = handle
            return handle

        self.online_calibration_receipt = calibrate_split(
            self.model, self._calibration_inputs or ModelInputs(self.sample_inputs, self.sample_kwargs), targets,
            boundaries=boundaries,
            make_handle=calibration_handle,
            loss_fn=self.task.loss, device=self.device,
            source="client_private_sample_current_deployment",
            optimizer_fn=self.optimizer_fn,
        )
        if cache_native:
            self.online_calibration_receipt['native_runtime_cache_hits'] = cache_hits
        return self.online_calibration_receipt

    def get_properties(self, config):
        from splitfleet.common.network_probe import PROBE_CONFIG_KEY, PROBE_RESULT_KEY, measure_network
        if PROBE_CONFIG_KEY in config:
            receipt = measure_network(self._require_server_model_proxy(), round_id=config[PROBE_CONFIG_KEY])
            return {PROBE_RESULT_KEY: json.dumps(receipt, allow_nan=False)}
        if "splitfleet_probe_payload" in config:
            payload = config["splitfleet_probe_payload"]
            if not isinstance(payload, bytes) or not 1 <= len(payload) <= 2097152:
                raise ValueError("Invalid Greedy transport probe")
            reply_size = config.get("splitfleet_probe_reply_bytes", len(payload))
            if type(reply_size) is not int or not 1 <= reply_size <= 2097152:
                raise ValueError("Invalid Greedy transport reply size")
            # This echo changes no model, optimizer or random generator state.
            return {"splitfleet_probe_payload": (payload if reply_size == len(payload) else bytes(reply_size))}
        cuts = config.get("splitfleet_calibration_boundaries")
        if cuts:
            from flwr.common import parameters_to_ndarrays, serde
            from flwr.proto.transport_pb2 import Parameters

            from splitfleet.common.model_state import model_state_hash

            payload = config.get(PLACEMENT_CALIBRATION_PARAMETERS_CONFIG_KEY)
            expected_hash = config.get(PLACEMENT_CALIBRATION_MODEL_HASH_CONFIG_KEY)
            if expected_hash is not None and payload is None:
                if model_state_hash(self.model, self.backend_adapter) != expected_hash:
                    # No trace, temporary SGD or state mutation before the
                    # coordinator supplies the actual global model.
                    return {PLACEMENT_CALIBRATION_PARAMETERS_REQUIRED_KEY: True}
            if payload is not None:
                parameters = serde.parameters_from_proto(Parameters.FromString(payload))
                self.backend_adapter.load_ndarrays(self.model, parameters_to_ndarrays(parameters))
                if expected_hash is not None and model_state_hash(self.model, self.backend_adapter) != expected_hash:
                    raise ValueError("Calibration parameter payload differs from the requested global model")
            dynamic_batch = normalize_batch_window(config.get(AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY))
            trace_batch_mode = config.get(AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY)
            reusable = self._prewarmed_runtime
            if (reusable is None or normalize_batch_window(reusable.plan.dynamic_batch) != dynamic_batch
                    or (trace_batch_mode and reusable.plan.trace_batch_mode != trace_batch_mode)):
                self._prewarmed_runtime = None
                self._runtime_cache.clear()
                self.autosplit_session.discard_runtime_handles()
                self.prewarm_runtime(
                    mode=config.get(AUTOSPLIT_MODE_CONFIG_KEY, "generated_eager"),
                    dynamic_batch=dynamic_batch, trace_batch_mode=trace_batch_mode,
                )
                self.online_calibration_receipt = None
            boundaries = json.loads(cuts)
            receipt = self.online_calibration_receipt
            current_hash = model_state_hash(self.model, self.backend_adapter)
            if (receipt is None or receipt["model_hash_after"] != current_hash
                    or receipt.get("schema") != "splitfleet.training-calibration"
                    or {row["boundary"] for row in receipt["records"]} != set(boundaries)
                    or any(row["graph_signature"] != self._prewarmed_runtime.plan.graph_signature
                           for row in receipt["records"])):
                if self._calibration_targets is None:
                    raise ValueError("Provide a representative TaskBatch for Greedy calibration")
                self.calibrate_runtime(self._calibration_targets, boundaries=boundaries)
        properties = {
            PLACEMENT_CALIBRATION_HASH_HANDSHAKE_KEY: self.backend_adapter.backend_name in {'torch', 'tf', 'paddle'},
            "logical_client_id": f"{platform.node()}:{self.device}:{id(self)}",
            "framework_backend": self.backend_adapter.backend_name,
            "runtime_backend": "torchlens_native",
            "device_type": str(self.device).split(":", 1)[0],
            "accelerator": _accelerator_name(self.device),
            "precision": _model_precision(self.model),
        }
        if self._prewarmed_runtime is not None:
            batch_size = runtime_input_batch_size(
                self._prewarmed_runtime.runtime, self.sample_inputs, self.sample_kwargs,
            )
        else:
            batch_size = model_input_batch_size(
                self.model, self.sample_inputs, self.sample_kwargs, batch_axes=self.batch_axes,
            )
        properties["batch_size"] = batch_size or self._sample_num_examples or infer_num_examples(
            ModelInputs(self.sample_inputs, self.sample_kwargs))
        if hasattr(self.train_data, "__len__"):
            properties["num_batches"] = len(self.train_data)
        if self.online_calibration_receipt is not None:
            properties["online_calibration_receipt"] = json.dumps(self.online_calibration_receipt)
        return properties

    def get_parameters(self, config):
        _ = config
        return self.backend_adapter.export_ndarrays(self.model)

    def fit(self, parameters, config):
        fit_start = time.perf_counter()
        previous_plan_id = str(self._round_config.get(AUTOSPLIT_PLAN_ID_CONFIG_KEY, ""))
        prepare_start = time.perf_counter()
        round_runtime = self._prepare_round(parameters, config, training=True)
        runtime_prepare_sec = time.perf_counter() - prepare_start
        runtime_handle = round_runtime.handle
        contract = round_runtime.contract
        prefix_optimizer = self._build_optimizer()
        from splitfleet.runtime.auxiliary_state import supports_auxiliary_state
        auxiliary_ownership = None
        if supports_auxiliary_state(self.model):
            from splitfleet.autosplit.state_ownership import (
                read_ownership,
                state_ownership,
            )
            schema = self.backend_adapter.state_manifest(self.model).schema_hash
            auxiliary_ownership = read_ownership(config, runtime_handle, schema) or state_ownership(runtime_handle, schema)
        telemetry = _RoundTelemetry()
        round_id = int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0))
        # Physical Flower dispatch consumes the hidden ID while binding the
        # request proxy. Keep the boundary identity tied to that same binding.
        client_id = str(config.get(
            CLIENT_ID_CONFIG_KEY, getattr(self._require_server_model_proxy(), "cid", "")))
        plan_id = str(config[AUTOSPLIT_PLAN_ID_CONFIG_KEY])

        num_examples = 0
        weighted_loss = 0.0
        for batch in self.train_data:
            task_batch = prepare_task_batch(batch, task=self.task, batch_adapter=self.batch_adapter, training=True)
            call = task_batch.inputs.map_values(lambda value: move_value(value, self.backend_adapter, self.device))
            torch_inputs = bind_model_inputs(call.args, self.backend_adapter, model=self.model)
            torch_targets = move_value(task_batch.targets, self.backend_adapter, self.device)
            batch_size = task_batch.num_examples
            runtime_batch_size = runtime_input_batch_size(runtime_handle.runtime, torch_inputs, call.kwargs)
            if not self._accept_batch(runtime_batch_size, round_runtime, telemetry, num_examples=batch_size):
                continue

            zero_grad(self.model, prefix_optimizer)

            _synchronize_prefix_device(self.backend_adapter.backend_name, self.device, torch_inputs)
            prefix_started = time.perf_counter()
            native_started_ns = time.perf_counter_ns()
            boundary = runtime_handle.backend.run_prefix(
                *torch_inputs,
                training=True,
                input_kwargs=call.kwargs,
            )
            _synchronize_prefix_device(self.backend_adapter.backend_name, self.device, boundary.tensors)
            if hasattr(self, "_quality_native_execution_intervals"):
                self._quality_native_execution_intervals.append((native_started_ns, time.perf_counter_ns()))
            forward_sec = time.perf_counter() - prefix_started
            telemetry.prefix_compute_sec += forward_sec
            telemetry.record_component("client_forward", forward_sec * 1000.0)
            step_id = uuid.uuid4().hex
            self._context_store.put(round_id, client_id, step_id, boundary)
            serialize_started = time.perf_counter()
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
            telemetry.boundary_serialize_sec += time.perf_counter() - serialize_started
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
            _synchronize_prefix_device(self.backend_adapter.backend_name, self.device, local_boundary.tensors)
            backward_started = time.perf_counter()
            deserialize_started = time.perf_counter()
            boundary_grads = envelope_to_gradients(gradient_envelope, self.device)
            telemetry.gradient_deserialize_sec += time.perf_counter() - deserialize_started
            native_started_ns = time.perf_counter_ns()
            prefix_result = runtime_handle.backend.backward_prefix(
                local_boundary,
                boundary_grads=boundary_grads,
                optimizer=prefix_optimizer,
            )
            _synchronize_prefix_device(self.backend_adapter.backend_name, self.device, prefix_result)
            if hasattr(self, "_quality_native_execution_intervals"):
                self._quality_native_execution_intervals.append((native_started_ns, time.perf_counter_ns()))
            backward_sec = time.perf_counter() - backward_started
            telemetry.prefix_compute_sec += backward_sec
            telemetry.record_component("client_backward", backward_sec * 1000.0)
            if "auxiliary_state" in response:
                from splitfleet.runtime.auxiliary_state import commit_auxiliary_state
                from splitfleet.runtime.owned_auxiliary_state import (
                    expected_statistics_names,
                    statistics_for_stage,
                )
                if auxiliary_ownership is None:
                    raise RuntimeError("Received unnegotiated auxiliary model state")
                statistics = decode_bundle_wire(response["auxiliary_state"], self.device)
                selected = statistics_for_stage(self.model, statistics, auxiliary_ownership, "prefix")
                if set(selected) != set(statistics) or set(selected) != expected_statistics_names(self.model, auxiliary_ownership, "prefix"):
                    raise RuntimeError("Suffix returned incomplete or incorrectly owned auxiliary state")
                from splitfleet.backends.utils import native_device_scope
                with native_device_scope(self.backend_adapter.backend_name, self.device):
                    commit_auxiliary_state(self.model, selected)

            batch_examples = int(response["num_examples"])
            batch_loss = float(response["loss"])
            num_examples += batch_examples
            weighted_loss += batch_loss * batch_examples
            telemetry.record_batch(batch_size)

        average_loss = weighted_loss / max(num_examples, 1)
        fit_duration_sec = time.perf_counter() - fit_start
        current_plan_id = str(config[AUTOSPLIT_PLAN_ID_CONFIG_KEY])
        metrics = {
            "loss": average_loss,
            "fit_duration_sec": fit_duration_sec,
            "fit_duration_ms": fit_duration_sec * 1000.0,
            "runtime_prepare_sec": runtime_prepare_sec,
            "switch_ms": (
                self._last_split_prepare_sec * 1000.0
                if previous_plan_id and previous_plan_id != current_plan_id
                else 0.0
            ),
            "num_examples": num_examples,
            "boundary": str(runtime_handle.plan.boundary),
            "plan_id": str(runtime_handle.plan.plan_id),
            "device": str(self.device),
            "framework_backend": str(self.backend_adapter.backend_name),
            "runtime_backend": str(config[AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY]),
            "device_type": str(self.device).split(":", 1)[0].lower(),
            "accelerator": _accelerator_name(self.device),
            "precision": _model_precision(self.model),
            **telemetry.as_metrics(),
        }
        from splitfleet.autosplit.state_ownership import (
            DIGEST_KEY,
            export_owned,
            read_ownership,
        )
        ownership = read_ownership(config, runtime_handle, self.backend_adapter.state_manifest(self.model).schema_hash)
        export_started = time.perf_counter()
        values = export_owned(self.model, ownership, "prefix") if ownership else self.backend_adapter.export_ndarrays(self.model)
        metrics.update(state_export_sec=time.perf_counter() - export_started,
                       state_upload_tensor_count=len(values),
                       state_upload_bytes=sum(value.nbytes for value in values))
        if ownership:
            metrics[DIGEST_KEY] = ownership[DIGEST_KEY]
        metrics["client_fit_handler_ms"] = (time.perf_counter() - fit_start) * 1000
        return values, num_examples, metrics

    def evaluate(self, parameters, config):
        round_runtime = self._prepare_round(parameters, config, training=False)
        runtime_handle = round_runtime.handle
        contract = round_runtime.contract
        telemetry = _RoundTelemetry()
        round_id = int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0))
        client_id = str(config.get(
            CLIENT_ID_CONFIG_KEY, getattr(self._require_server_model_proxy(), "cid", "")))
        plan_id = str(config[AUTOSPLIT_PLAN_ID_CONFIG_KEY])

        num_examples = 0
        weighted_loss = 0.0
        with inference_context(self.backend_adapter.backend_name):
            for batch in self.evaluate_data:
                task_batch = prepare_task_batch(batch, task=self.task, batch_adapter=self.batch_adapter, training=False)
                call = task_batch.inputs.map_values(lambda value: move_value(value, self.backend_adapter, self.device))
                torch_inputs = bind_model_inputs(call.args, self.backend_adapter, model=self.model)
                torch_targets = move_value(task_batch.targets, self.backend_adapter, self.device)
                batch_size = task_batch.num_examples
                runtime_batch_size = runtime_input_batch_size(runtime_handle.runtime, torch_inputs, call.kwargs)
                if not self._accept_batch(runtime_batch_size, round_runtime, telemetry, num_examples=batch_size):
                    continue
                boundary = runtime_handle.backend.run_prefix(*torch_inputs, input_kwargs=call.kwargs)
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
        metrics = {
            "loss": average_loss,
            "boundary": str(runtime_handle.plan.boundary),
            **telemetry.as_metrics(),
        }
        return float(average_loss), num_examples, metrics

    def _prepare_round(self, parameters, config, *, training: bool) -> _RoundRuntime:
        if config.get(AUTOSPLIT_BACKEND_CONFIG_KEY) != AUTOSPLIT_BACKEND_VALUE_TORCHLENS:
            raise ValueError("Split config must explicitly declare backend='torchlens'.")
        client_stage_count = int(config.get(AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY, 1))
        if client_stage_count != 1:
            raise ValueError(
                "TorchLens autosplit backend supports exactly one client-local prefix stage."
            )
        self.backend_adapter.load_ndarrays(self.model, parameters)
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
        split_prepare_started = time.perf_counter()
        runtime = self._ensure_round_runtime(config, state_schema)
        self._last_split_prepare_sec = time.perf_counter() - split_prepare_started
        return runtime

    def _ensure_round_runtime(self, config, state_schema: str) -> _RoundRuntime:
        plan_id = str(config[AUTOSPLIT_PLAN_ID_CONFIG_KEY])
        split_id = str(config.get(AUTOSPLIT_SPLIT_ID_CONFIG_KEY, ""))
        graph_signature = str(config.get(AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY, ""))
        feature_abi_id = str(config.get(AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY, ""))
        module_mode = "train" if model_training(self.model) else "eval"
        # The server negotiates the dynamic batch window for the whole fleet.
        # A device that re-derives it from its own sample batch would prepare a
        # runtime with a different feature ABI and reject every other batch size.
        dynamic_batch = normalize_batch_window(config.get(AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY))
        trace_batch_mode = str(config.get(AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY, "")) or None
        trainable = bool(config.get(AUTOSPLIT_TRAINABLE_CONFIG_KEY, True))
        cache_key = "|".join([
            self.backend_adapter.backend_name, torchlens_runtime_version(), self.model.__class__.__qualname__,
            state_schema, plan_id, split_id, graph_signature, feature_abi_id,
            str(self.device), module_mode,
            describe_batch_window(dynamic_batch), str(trace_batch_mode or ""), str(trainable),
        ])
        cached = self._runtime_cache.get(cache_key)
        if cached is not None:
            raw_contract = config.get(AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY)
            if raw_contract:
                expected = GraphContract.from_json(raw_contract)
                if config.get(AUTOSPLIT_GRAPH_CONTRACT_DIGEST_CONFIG_KEY) != expected.digest:
                    raise RuntimeError("Configured graph contract digest is invalid")
                validate_contract(expected, graph_contract_for_runtime_handle(cached.handle))
            self._validate_feature_abi(config, cached.handle)
            return cached

        boundary = str(config.get(AUTOSPLIT_BOUNDARY_CONFIG_KEY, "50%"))
        reusable = self._prewarmed_runtime
        if reusable is None:
            reusable = (next(iter(self._runtime_cache.values())).handle
                        if self.max_cached_runtimes == 1 and self._runtime_cache else None)
        can_repartition = (
            reusable is not None
            and reusable.plan.graph_signature == graph_signature
            and reusable.plan.trainable == trainable
            and normalize_batch_window(reusable.plan.dynamic_batch) == dynamic_batch
            and str(reusable.plan.trace_batch_mode or "") == str(trace_batch_mode or "")
        )
        if reusable is not None and not can_repartition:
            print(json.dumps({
                "event": "client_runtime_reuse_miss",
                "expected_graph_signature": graph_signature,
                "prewarmed_graph_signature": reusable.plan.graph_signature,
                "expected_dynamic_batch": describe_batch_window(dynamic_batch),
                "prewarmed_dynamic_batch": describe_batch_window(reusable.plan.dynamic_batch),
                "expected_trace_batch_mode": trace_batch_mode or "",
                "prewarmed_trace_batch_mode": reusable.plan.trace_batch_mode or "",
                "expected_boundary": boundary,
                "prewarmed_boundary": reusable.plan.boundary,
                "expected_split_id": split_id,
                "prewarmed_split_id": reusable.plan.split_id,
                "expected_feature_abi_id": feature_abi_id,
                "prewarmed_feature_abi_id": reusable.plan.feature_abi_id,
            }, sort_keys=True), flush=True)
        if can_repartition:
            previous = reusable
            if str(previous.plan.boundary) == boundary:
                handle = previous
                self._prewarmed_runtime = None
                if self.max_cached_runtimes == 1 and self._runtime_cache:
                    # Keep the bounded cache invariant when a later round
                    # reuses the same boundary without repartitioning.
                    self._runtime_cache.clear()
            else:
                handle = self.autosplit_session.repartition_runtime(previous, boundary)
                self._prewarmed_runtime = None
                self._runtime_cache.clear()
                del previous
                gc.collect()
                if self.backend_adapter.backend_name == "torch" and str(self.device).startswith("cuda"):
                    import torch

                    torch.cuda.empty_cache()
        else:
            if self._prewarmed_runtime is not None:
                self._prewarmed_runtime = None
                self.autosplit_session.discard_runtime_handles()
            if (self.max_cached_runtimes is not None
                    and len(self._runtime_cache) >= self.max_cached_runtimes):
                self._runtime_cache.clear()
                self.autosplit_session.discard_runtime_handles()
                gc.collect()
            handle = self.autosplit_session.prepare_runtime(
                self.model,
                self.sample_inputs,
                sample_kwargs=self.sample_kwargs,
                batch_axes=self.batch_axes,
                boundary=boundary,
                mode=str(config.get(AUTOSPLIT_MODE_CONFIG_KEY, "generated_eager")),
                trainable=trainable,
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

    def _validate_feature_abi(self, config, handle: TorchLensRuntimeHandle) -> None:
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
        batch_size: int | None,
        round_runtime: _RoundRuntime,
        telemetry: _RoundTelemetry,
        *,
        num_examples: int | None = None,
    ) -> bool:
        """Apply the partial-batch policy to a batch the runtime cannot replay."""

        # A static image list has an example count but no symbolic replay batch
        # axis. Validate only dimensions declared by the native runtime.
        if batch_size is None or batch_window_contains(round_runtime.batch_window, batch_size):
            return True
        if self.partial_batch_policy == "skip":
            telemetry.record_skip(batch_size if num_examples is None else num_examples)
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
        telemetry: _RoundTelemetry | None = None,
    ):
        serialize_started = time.perf_counter()
        boundary_payload = encode_boundary(boundary)
        if telemetry is not None:
            telemetry.boundary_serialize_sec += time.perf_counter() - serialize_started
        target_payload = encode_bundle_wire(targets, backend=self.backend_adapter.backend_name)
        client_send_ns = time.time_ns()
        request = BatchData(
            data={
                "boundary": boundary_payload,
                "targets": target_payload,
                "metadata": json.dumps({"num_examples": num_examples}).encode("utf-8"),
            },
            control_code=ControlCode.OK,
            metadata={
                TRANSPORT_CLIENT_SEND_NS_METADATA_KEY: str(client_send_ns)
            },
        )
        started = time.perf_counter()
        response = getattr(self._require_server_model_proxy(), method_name)(
            request,
            _streams_=False,
        )
        client_receive_ns = time.time_ns()
        if telemetry is not None:
            rpc_elapsed_sec = time.perf_counter() - started
            telemetry.suffix_server_queue_sec += float(response.metadata.get("suffix_server_queue_sec", 0.))
            try:
                server_span_ms = int(response.metadata[TRANSPORT_SERVER_ELAPSED_NS_METADATA_KEY]) / 1e6
                resource_ms = server_span_ms - float(response.metadata.get("suffix_server_queue_sec", 0.)) * 1000.
            except (KeyError, TypeError, ValueError):
                resource_ms = None
            if resource_ms is not None and math.isfinite(resource_ms) and resource_ms >= 0:
                telemetry.record_component("server_resource_service", resource_ms)
            from splitfleet.server.telemetry import SERVER_LOAD_FIELDS
            load = json.loads(response.metadata.get("splitfleet_server_load", "{}"))
            for name in SERVER_LOAD_FIELDS:
                if name in load:
                    value = float(load[name])
                    if math.isfinite(value) and value >= 0:
                        telemetry._server_load_samples.setdefault(name, []).append(value)
            telemetry.tail_wait_sec += rpc_elapsed_sec
            telemetry.record_component("network_roundtrip", _transport_roundtrip_ms(
                response.metadata, rpc_elapsed_ms=rpc_elapsed_sec * 1000.0))
            telemetry.upload_bytes += len(boundary_payload) + len(target_payload)
            telemetry.download_bytes += sum(
                len(value) for value in response.data.values() if isinstance(value, bytes)
            )
            # Count the actual serialized application buffers. Keep targets and
            # response metadata separate; these are not NIC/TLS/gRPC byte counts.
            telemetry.activation_upload_bytes += len(boundary_payload)
            telemetry.target_upload_bytes += len(target_payload)
            telemetry.request_metadata_upload_bytes += len(request.data["metadata"])
            telemetry.gradient_download_bytes += len(response.data.get("gradients", b""))
            telemetry.response_metadata_download_bytes += sum(
                len(value) for key, value in response.data.items()
                if key != "gradients" and isinstance(value, bytes)
            )
        metadata = json.loads(response.data["metadata"].decode("utf-8"))
        if telemetry is not None:
            telemetry.record_component("server_forward", metadata.get("server_forward_ms"))
            telemetry.record_component("server_loss", metadata.get("server_loss_ms"))
            telemetry.record_component("server_backward", metadata.get("server_backward_ms"))
            telemetry.record_component("server_service", metadata.get("server_service_ms"))
            upload_ms, download_ms = _transport_phase_ms(
                response.metadata,
                client_send_ns=client_send_ns,
                client_receive_ns=client_receive_ns,
            )
            telemetry.record_component("network_upload", upload_ms)
            telemetry.record_component("network_download", download_ms)
        if "gradients" in response.data:
            deserialize_started = time.perf_counter()
            metadata["gradients"] = decode_gradients(response.data["gradients"])
            if telemetry is not None:
                telemetry.gradient_deserialize_sec += time.perf_counter() - deserialize_started
        if "auxiliary_state" in response.data:
            metadata["auxiliary_state"] = response.data["auxiliary_state"]
        return metadata

    def _build_optimizer(self):
        if self.optimizer_fn is not None:
            return self.optimizer_fn(self.model)
        return self.backend_adapter.build_optimizer(self.model, {"name": "sgd", "lr": 0.01})

    def _require_server_model_proxy(self):
        proxy = getattr(self, "server_model_proxy", None)
        if proxy is None:
            raise RuntimeError("AutoSplitSplitLearningClient requires a server_model_proxy.")
        return proxy
