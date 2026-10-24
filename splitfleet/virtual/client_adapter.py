"""Stateful orchestration around the existing AutoSplitSplitLearningClient."""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flwr.client import NumPyClient
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.common import BatchData, ControlCode
from splitfleet.common.network_probe import PROBE_CONFIG_KEY, PROBE_RESULT_KEY, measure_network
from splitfleet.common.constants import (
    AUTOSPLIT_MODEL_VERSION_CONFIG_KEY,
    CLIENT_ID_CONFIG_KEY,
    PLACEMENT_CALIBRATION_HASH_HANDSHAKE_KEY,
    PLACEMENT_CALIBRATION_PARAMETERS_REQUIRED_KEY,
)
from splitfleet.runtime.client_execution import server_model_binding

from .backend_ops import FrameworkOps
from .network import TaskNetwork
from .partitions import ClientBatches, partition_hash
from .resources import peak_rss_mb
from .state import client_seed, logical_state
from .suffix_service import decode_parameters, encode_parameters, suffix_connection


@dataclass(frozen=True)
class OptimizerFactory:
    name: str
    learning_rate: float
    backend: str | None = None

    def __call__(self, model):
        ops = FrameworkOps(self.backend) if self.backend else FrameworkOps.for_model(model)
        return ops.optimizer(model, self.name, self.learning_rate)


@dataclass(frozen=True)
class ClientFactory:
    config: object
    spec: object
    partitions: dict

    def __call__(self, context):
        return VirtualClientAdapter(context=context, factory=self)


class VirtualClientAdapter(NumPyClient):
    """Bind one Flower Context to the shared SFL executor for one task."""

    def __init__(self, *, context, factory):
        self.entered_ns = time.perf_counter_ns()
        self.context, self.factory = context, factory
        self.ops = FrameworkOps.for_spec(factory.spec)
        partition_id = context.node_config.get("partition-id")
        if type(partition_id) is not int or str(partition_id) not in factory.partitions:
            raise ValueError("Flower Context has an invalid partition-id")
        self.partition_id = partition_id
        self.indices = factory.partitions[str(partition_id)]
        self.state = logical_state(context, partition_id, partition_hash(self.indices))
        self.profile = factory.config.profile(partition_id)
        if self.profile.device.startswith("cuda"):
            # Resource telemetry runs before constructing the first CUDA model.
            self.ops.admit_device(self.profile.device)

    def _build_client(self, round_id, *, deadline=None, fail_after=None):
        spec, config = self.factory.spec, self.factory.config
        dataset = spec.train_dataset_factory()
        source = (spec.batch_factory or ClientBatches)(
            dataset,
            self.indices,
            batch_size=config.batch_size,
            seed=config.seed,
            partition_id=self.partition_id,
            round_id=round_id,
            local_epochs=config.local_epochs,
            collate_fn=spec.collate_fn,
            deadline=deadline,
            fail_after=fail_after,
        )
        from torch.utils.data import default_collate

        collate = spec.collate_fn or default_collate
        # Trace/calibrate with this client's private examples. Repeating one
        # example to match the negotiated trace shape never adds training steps.
        raw_sample = collate(
            [
                dataset[self.indices[i % len(self.indices)]]
                for i in range(config.batch_size)
            ]
        )
        sample = spec.task.make_adapter().prepare_batch(raw_sample)
        calibration = spec.task.make_adapter().prepare_batch(
            collate([dataset[i] for i in self.indices[: config.batch_size]])
        )
        with self.ops.rng_scope(
            seed=client_seed(config.seed, self.partition_id, "model"),
            device=self.profile.device,
        ):
            optimizer_factory = OptimizerFactory(
                config.optimizer,
                float(spec.learning_rate_fn(round_id)) if spec.learning_rate_fn else config.learning_rate,
                self.ops.backend,
            )

            def construct():
                model = self.ops.build_model(spec.model_factory, self.profile.device)
                return AutoSplitSplitLearningClient(
                    model=model, train_data=source,
                    sample_inputs=self.ops.prepare_batch(model, sample, self.profile.device),
                    task=spec.task.make_adapter(), batch_axes=spec.batch_axes,
                    device=self.profile.device, optimizer_fn=optimizer_factory,
                    max_cached_runtimes=1,
                    calibration_batch=self.ops.prepare_batch(model, calibration, self.profile.device),
                )

            # Flower owns Worker process reuse.  The executor is task-local so
            # model, optimizer, graph and RNG state cannot leak between Contexts.
            client = construct()
            # The shared executor owns its runtime/cache lifecycle.  The
            # virtual adapter only supplies this task's data and public state.
            call = self.ops.prepare_batch(client.model, sample, self.profile.device)
            private_calibration = self.ops.prepare_batch(client.model, calibration, self.profile.device)
            client.bind_execution_context(
                train_data=source,
                evaluate_data=source,
                sample_inputs=call.inputs.args,
                sample_kwargs=call.inputs.kwargs,
                sample_num_examples=call.num_examples,
                calibration_inputs=private_calibration.inputs,
                calibration_targets=private_calibration.targets,
                optimizer_fn=optimizer_factory,
            )
            if self.ops.backend != "torch":
                self.ops.reset_peak_memory(self.profile.device)
            if spec.trainer is not None:
                client.enable_quality_measurement()
        return client, source

    @contextmanager
    def _client_scope(self, round_id, **options):
        client = None
        try:
            client, source = self._build_client(round_id, **options)
            yield client, source
            if client.pending_context_count:
                raise RuntimeError("A completed client execution retains an unconsumed prefix context")
        finally:
            if client is not None:
                client.close_execution()

    def get_properties(self, config):
        if PROBE_CONFIG_KEY in config:
            # Emulation applies the plant's constraints here. Only observed
            # RPC durations leave this handler; no configured rate is returned.
            round_id = config[PROBE_CONFIG_KEY]
            runtime = self.factory.config
            network = runtime.network(self.partition_id, round_id)
            credentials = runtime.transport.credentials("client")
            address = config["virtual_suffix_address"]
            with TaskNetwork(address, network, round_id=round_id) as path:
                tls_name = address.rsplit(":", 1)[0].strip("[]") if credentials and network.constrained else None
                with suffix_connection(path.address, self.context.node_id,
                        timeout=runtime.transport.rpc_timeout_sec, certificates=credentials,
                        tls_server_name=tls_name,
                        metadata=(("authorization", f"Bearer {config['virtual_network_probe_token']}"),)) as proxy:
                    receipt = measure_network(proxy, round_id=round_id,
                                              timeout=runtime.transport.rpc_timeout_sec)
                    receipt["tcp_counters"] = path.metrics()
            return {PROBE_RESULT_KEY: json.dumps(receipt, allow_nan=False)}
        if "splitfleet_probe_payload" in config:
            # This is a real Flower control RPC. Its stateless echo must not
            # include model construction, data access or RNG initialization.
            payload = config["splitfleet_probe_payload"]
            if not isinstance(payload, bytes) or not 1 <= len(payload) <= 2_097_152:
                raise ValueError("Invalid Greedy transport probe")
            reply_size = config.get("splitfleet_probe_reply_bytes", len(payload))
            if type(reply_size) is not int or not 1 <= reply_size <= 2_097_152:
                raise ValueError("Invalid Greedy transport reply size")
            return {"splitfleet_probe_payload": payload if reply_size == len(payload) else bytes(reply_size)}
        runtime = self.factory.config
        properties = {
            PLACEMENT_CALIBRATION_HASH_HANDSHAKE_KEY: self.ops.backend in {'torch', 'tf', 'paddle'},
            "logical_client_id": self.state["logical_client_id"],
            "virtual_partition_id": self.partition_id,
            "partition_hash": self.state["partition_hash"],
            "framework_backend": self.ops.backend,
            "runtime_backend": "torchlens_native",
            "device_type": self.profile.device.split(":", 1)[0],
            "batch_size": min(runtime.batch_size, len(self.indices)),
            "num_batches": math.ceil(len(self.indices) / runtime.batch_size)
            * runtime.local_epochs,
        }
        if config.get("virtual_identity_only"):
            return properties
        with (
            self.ops.worker_resources(
                self.profile, mode=runtime.resources.mode
            ) as resource_receipt,
            self.ops.rng_scope(
                seed=client_seed(runtime.seed, self.partition_id),
                saved=self.state.get("rng"),
                device=self.profile.device,
            ),
            self._client_scope(int(self.state["last_round"]) + 1) as (client, _),
        ):
            result = client.get_properties(config)
            if ("splitfleet_probe_payload" in result
                    or result.get(PLACEMENT_CALIBRATION_PARAMETERS_REQUIRED_KEY) is True):
                # A hash mismatch asks the coordinator for weights before
                # calibration. That control reply has no hardware properties.
                return result
            result.update(properties)
            result["hardware_accelerator"] = result["accelerator"]
            result["accelerator"] = execution_group(result["accelerator"], resource_receipt)
            if self.factory.spec.trainer is not None and "online_calibration_receipt" in result:
                receipt = json.loads(result["online_calibration_receipt"])
                print(json.dumps({
                    "event": "quality_calibration_finished", "partition_id": self.partition_id,
                    "elapsed_sec": receipt["elapsed_sec"],
                    "temporary_optimizer_steps": receipt["optimizer_steps"],
                    "state_and_rng_preserved": receipt["state_and_torch_rng_preserved"],
                    "rng_scope": self.ops.rng_scope_description,
                }, sort_keys=True), flush=True)
            return result

    def get_parameters(self, config):
        # The server supplies initial parameters; no stale local global state.
        with self.ops.worker_resources(self.profile, mode=self.factory.config.resources.mode), self._client_scope(
            int(self.state["last_round"]) + 1
        ) as (client, _):
            return client.get_parameters(config)

    def fit(self, parameters, config):
        runtime = self.factory.config
        round_id = int(config[AUTOSPLIT_MODEL_VERSION_CONFIG_KEY])
        if round_id <= int(self.state["last_round"]):
            raise RuntimeError("Logical client round replay is forbidden")
        if str(config[CLIENT_ID_CONFIG_KEY]) != str(self.context.node_id):
            raise RuntimeError("Flower node and split RPC client identities differ")
        if config["virtual_partition_id"] != self.partition_id:
            raise RuntimeError("Dispatch was assigned to the wrong data partition")
        started_ns = time.perf_counter_ns()
        started = started_ns / 1e9
        deadline = started + runtime.virtual_clients.training_timeout_sec
        population = runtime.virtual_clients
        inject = (
            round_id in population.failure_rounds.get(str(self.partition_id), [])
            or np.random.default_rng(
                client_seed(runtime.seed, self.partition_id, f"fault:{round_id}")
            ).random()
            < population.failure_probability
        )
        network = runtime.network(self.partition_id, round_id)
        entry = self.entered_ns
        queue_sec = None
        boot_path = Path("/proc/sys/kernel/random/boot_id")
        boot_id = boot_path.read_text().strip() if boot_path.is_file() else ""
        shared_clock = not runtime.resources.ray_address or (
            boot_id and boot_id == config.get("virtual_dispatch_boot_id")
        )
        if config["virtual_dispatch_host"] == socket.gethostname() and shared_clock:
            queue_sec = max((entry - int(config["virtual_dispatch_ns"])) / 1e9, 0.0)
        metrics = {}
        with (
            self.ops.worker_resources(
                self.profile, mode=runtime.resources.mode
            ) as resource_receipt,
            self.ops.rng_scope(
                seed=client_seed(runtime.seed, self.partition_id),
                saved=self.state.get("rng"),
                device=self.profile.device,
            ),
            self._client_scope(round_id, deadline=deadline,
                               fail_after=population.fail_after_batches if inject else None) as (client, source),
        ):
            build_sec = time.perf_counter() - started
            with TaskNetwork(
                config["virtual_suffix_address"], network, round_id=round_id
            ) as path:
                credentials = runtime.transport.credentials("client")
                # TLS goes end to end through the local TCP proxy.
                tls_name = (
                    config["virtual_suffix_address"].rsplit(":", 1)[0].strip("[]")
                    if credentials is not None and network.constrained
                    else None
                )
                with suffix_connection(
                    path.address,
                    self.context.node_id,
                    timeout=min(
                        runtime.transport.rpc_timeout_sec,
                        population.training_timeout_sec,
                    ),
                    certificates=credentials,
                    tls_server_name=tls_name,
                    metadata=(
                        ("authorization", f"Bearer {config['virtual_task_token']}"),
                    ),
                ) as proxy:
                    leased, committed = False, False
                    try:
                        lease_began = time.perf_counter()
                        receipt = proxy.virtual_acquire(
                            BatchData(data={}, control_code=ControlCode.OK),
                            _streams_=False,
                        )
                        leased = True
                        metrics.update(json.loads(receipt.data["metadata"]))
                        metrics["quality_lease_rpc_sec"] = time.perf_counter() - lease_began
                        sync_started = time.perf_counter()
                        if network.constrained:
                            payload = proxy.virtual_download_state(
                                BatchData(data={}, control_code=ControlCode.OK),
                                _streams_=False,
                            )
                            state_payload = payload.data["parameters"]
                            parameters = parameters_to_ndarrays(
                                decode_parameters(state_payload)
                            )
                            metrics["model_state_download_bytes"] = len(state_payload)
                        else:
                            metrics["model_state_download_bytes"] = int(
                                config["virtual_state_download_bytes"]
                            )
                        client.backend_adapter.load_ndarrays(client.model, parameters)
                        received_hash = self.ops.state_hash(client.model)
                        if received_hash != config["virtual_global_model_hash"]:
                            raise RuntimeError(
                                "Client did not receive the current global model state"
                            )
                        metrics["global_model_hash_received"] = received_hash
                        sync_sec = time.perf_counter() - sync_started
                        ownership = json.loads(config["autosplit_state_ownership"])
                        initial_prefix = [
                            parameters[i]
                            for i, owner in enumerate(ownership["owners"])
                            if owner == "prefix"
                        ]
                        # Restore the same previous action without retaining its
                        # graph. Every new runtime still validates its current ABI.
                        changed = bool(
                            self.state["last_boundary"]
                            and self.state["last_boundary"]
                            != config["autosplit_boundary"]
                        )
                        if self.state["last_boundary"]:
                            client.set_previous_plan_hint(
                                "previous-boundary:" + self.state["last_boundary"]
                                if changed
                                else config["autosplit_plan_id"]
                            )
                        with server_model_binding(client, proxy):
                            values, count, fit_metrics = client.fit(parameters, config)
                        if time.perf_counter() >= deadline:
                            raise TimeoutError(
                                "Virtual client exceeded its training deadline"
                            )
                        if fit_metrics["num_batches"] != len(source):
                            raise RuntimeError(
                                "Virtual client failed to execute every declared minibatch"
                            )
                        metrics.update(fit_metrics)
                        metrics["virtual_boundary_changed"] = changed
                        metrics["switch_cost_scope"] = (
                            "cold split preparation when the boundary changes; overlaps runtime_prepare_sec"
                        )
                        if not shared_clock:
                            for direction in ("network_upload", "network_download"):
                                for suffix in ("_ms", "_mean_ms", "_sec", "_samples"):
                                    metrics.pop(direction + suffix, None)
                        if any(not np.isfinite(value).all() for value in values):
                            raise FloatingPointError(
                                "Prefix update contains nonfinite model state"
                            )
                        self.ops.assert_model_device(client.model, self.profile.device)
                        metrics["worker_model_device_verified"] = True
                        metrics["worker_cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                        memory = self.ops.native_memory_receipt(self.profile.device)
                        if memory is not None:
                            metrics["worker_native_memory"] = json.dumps(memory, sort_keys=True)
                        metrics["model_sync_sec"] = sync_sec
                        metrics["prefix_changed_tensors"] = sum(
                            not np.array_equal(before, after)
                            for before, after in zip(
                                initial_prefix, values, strict=True
                            )
                        )
                        metrics["prefix_update_hash"] = hashlib.sha256(
                            b"".join(v.tobytes() for v in values)
                        ).hexdigest()
                        upload_started = time.perf_counter()
                        if network.constrained:
                            payload = encode_parameters(ndarrays_to_parameters(values))
                            proxy.virtual_upload_state(
                                BatchData(
                                    data={"parameters": payload},
                                    control_code=ControlCode.OK,
                                ),
                                _streams_=False,
                            )
                            metrics["model_state_upload_bytes"] = len(payload)
                            values = []
                        else:
                            metrics["model_state_upload_bytes"] = len(
                                encode_parameters(ndarrays_to_parameters(values))
                            )
                        metrics["model_state_rpc_upload_sec"] = (
                            time.perf_counter() - upload_started
                        )
                        commit_began = time.perf_counter()
                        proxy.virtual_commit(
                            BatchData(data={}, control_code=ControlCode.OK),
                            _streams_=False,
                        )
                        committed = True
                        metrics["quality_commit_rpc_sec"] = time.perf_counter() - commit_began
                        self.state.update(
                            {
                                "last_round": round_id,
                                "completed_batches": int(
                                    self.state["completed_batches"]
                                )
                                + int(metrics["num_batches"]),
                                "last_boundary": str(metrics["boundary"]),
                                "last_plan_id": str(config["autosplit_plan_id"]),
                                "rng": self.ops.capture_rng(self.profile.device),
                            }
                        )
                    finally:
                        if leased and not committed:
                            try:
                                proxy.virtual_abort(
                                    BatchData(data={}, control_code=ControlCode.OK),
                                    _streams_=False,
                                    _timeout_=runtime.transport.rpc_timeout_sec,
                                )
                            except grpc_error_types():
                                pass
                metrics.update(path.metrics())
            metrics.update(
                {
                    "logical_client_id": self.state["logical_client_id"],
                    "partition_id": self.partition_id,
                    "partition_hash": self.state["partition_hash"],
                    "model_version": round_id,
                    "completed_batches": self.state["completed_batches"],
                    "rng_state_hash": hashlib.sha256(self.state["rng"]).hexdigest(),
                    "rng_state_hash_scope": self.ops.rng_scope_description,
                    "virtual_client_build_sec": build_sec,
                    "worker_capacity": runtime.resources.effective_worker_capacity or 1,
                    "virtual_handler_ms": (time.perf_counter() - started) * 1000,
                    "virtual_state_via_rpc": network.constrained,
                    "client_peak_memory_mb": peak_rss_mb(),
                    "worker_pid": os.getpid(),
                    "worker_host": socket.gethostname(),
                    "worker_boot_id": boot_id,
                    "worker_task_start_ns": started_ns,
                    "worker_task_end_ns": time.perf_counter_ns(),
                    "resource_source": resource_receipt["resource_source"],
                    "worker_affinity": json.dumps(resource_receipt["affinity"]),
                    "worker_torch_threads": resource_receipt["torch_threads"],
                    "hardware_accelerator": metrics["accelerator"],
                    "accelerator": execution_group(
                        metrics["accelerator"], resource_receipt
                    ),
                    "gpu_peak_memory_mb": self.ops.peak_gpu_memory_mb(self.profile.device),
                }
            )
            if metrics["gpu_peak_memory_mb"] is None:
                metrics.pop("gpu_peak_memory_mb")
            if self.ops.backend != "torch":
                metrics.pop("worker_torch_threads")
                metrics["worker_framework_backend"] = self.ops.backend
                metrics["backend_thread_environment"] = json.dumps(resource_receipt["backend_thread_environment"], sort_keys=True)
            if hasattr(source, "batch_hash"):
                metrics["data_batch_hash"] = source.batch_hash
                metrics["data_examples_seen"] = source.samples_seen
                rpc_spans, native_intervals = client.quality_measurements()
                metrics["quality_rpc_calls"] = len(rpc_spans)
                metrics["quality_rpc_max_sec"] = max(rpc_spans, default=0.0)
                metrics["native_training_intervals_ns"] = json.dumps(
                    native_intervals)
            if queue_sec is not None:
                metrics["virtual_worker_queue_sec"] = queue_sec
            return values, count, metrics


def grpc_error_types():
    import grpc

    return (grpc.RpcError, TimeoutError)


def execution_group(hardware, receipt):
    """Qualify the actual hardware by applied threads/affinity, never speed priors."""
    if "framework_backend" in receipt:
        return f"{hardware}@backend={receipt['framework_backend']},cpus={len(receipt['affinity'])},thread_env={json.dumps(receipt['backend_thread_environment'], sort_keys=True)}"
    return (
        f"{hardware}@threads={receipt['torch_threads']},cpus={len(receipt['affinity'])}"
    )
