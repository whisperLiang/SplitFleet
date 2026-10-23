"""Native Flower FedAvg scheduling with the shared owned-state aggregator."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from flwr.client import NumPyClient
from flwr.common import (
    FitIns,
    GetPropertiesIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import Server, ServerAppComponents, ServerConfig, SimpleClientManager
from flwr.server.strategy import FedAvg
from flwr.serverapp import ServerApp

from splitfleet.autosplit.state_ownership import aggregate_owned
from splitfleet.server.stage_runtime.storage import RoundStateStore

from .backend_ops import FrameworkOps
from .client_adapter import OptimizerFactory
from .client_app import make_client_app
from .partitions import ClientBatches, partition_hash
from .resources import peak_rss_mb
from .runtime import (
    check_simulation_dependencies,
    initialize_ray,
    round_learning_rate,
    training_partitions,
)
from .server_adapter import VirtualDispatchProxy, round_concurrency, task_concurrency
from .state import client_seed, logical_state
from .telemetry import Evidence


class _FedAvgClient(NumPyClient):
    def __init__(self, *, context, config, spec, partitions):
        self.context, self.config, self.spec = context, config, spec
        self.ops = FrameworkOps.for_spec(spec)
        self.partition_id = context.node_config["partition-id"]
        self.indices = partitions[str(self.partition_id)]
        self.profile = config.profile(self.partition_id)
        if self.profile.device.startswith("cuda"):
            self.ops.admit_device(self.profile.device)
        self.state = logical_state(context, self.partition_id, partition_hash(self.indices))

    def get_properties(self, config):
        return {"virtual_partition_id": self.partition_id,
                "logical_client_id": self.state["logical_client_id"],
                "partition_hash": self.state["partition_hash"]}

    def get_parameters(self, config):
        raise RuntimeError("The FedAvg coordinator supplies the frozen initial state")

    @contextmanager
    def _model_scope(self):
        def construct():
            # Model initialization cannot consume this logical task's training
            # RNG.
            with self.ops.rng_scope(seed=client_seed(self.config.seed, self.partition_id, "model"),
                                    device=self.profile.device):
                return self.ops.build_model(self.spec.model_factory, self.profile.device)

        # Flower/Ray reuses the Worker process; each task gets a fresh model so
        # optimizer, graph, buffers and RNG state remain owned by its Context.
        model = construct()
        try:
            yield model
        finally:
            _release_model(model)

    def fit(self, parameters, config):
        round_id = int(config["quality_round"])
        if round_id <= self.state["last_round"] or config["virtual_partition_id"] != self.partition_id:
            raise RuntimeError("FedAvg logical identity/round contract mismatch")
        started_ns = time.perf_counter_ns()
        deadline = time.perf_counter() + self.config.virtual_clients.training_timeout_sec
        with self.ops.worker_resources(self.profile, mode=self.config.resources.mode) as resource_receipt, self.ops.rng_scope(
            seed=client_seed(self.config.seed, self.partition_id),
            saved=self.state.get("rng"), device=self.profile.device,
        ), self._model_scope() as model:
            self.ops.set_training(model, True)
            adapter = self.ops.adapter
            adapter.load_ndarrays(model, parameters)
            received = self.ops.state_hash(model)
            if received != config["virtual_global_model_hash"]:
                raise RuntimeError("FedAvg client received stale global parameters")
            learning_rate = round_learning_rate(self.config, self.spec, round_id)
            optimizer = OptimizerFactory(self.config.optimizer, learning_rate, self.ops.backend)(model)
            if self.ops.backend != "torch":
                self.ops.reset_peak_memory(self.profile.device)
            source = (self.spec.batch_factory or ClientBatches)(
                self.spec.train_dataset_factory(), self.indices, batch_size=self.config.batch_size,
                seed=self.config.seed, partition_id=self.partition_id, round_id=round_id,
                local_epochs=self.config.local_epochs, collate_fn=self.spec.collate_fn,
                deadline=deadline, fail_after=None,
            )
            if self.spec.trainer is None:
                raise ValueError("Virtual FedAvg requires an explicit native full-model trainer")
            examples, weighted_loss = 0, 0.0
            native_intervals = []
            from splitfleet.client.autosplit_split_client import _synchronize_prefix_device

            for raw in source:
                _synchronize_prefix_device(self.ops.backend, self.profile.device)
                native_started_ns = time.perf_counter_ns()
                count, loss = self.spec.trainer(model, raw, optimizer, self.profile.device, learning_rate)
                _synchronize_prefix_device(self.ops.backend, self.profile.device)
                native_intervals.append((native_started_ns, time.perf_counter_ns()))
                examples += count
                weighted_loss += count * loss
            values = adapter.export_ndarrays(model)
            if source.executed != len(source) or any(not np.isfinite(v).all() for v in values):
                raise RuntimeError("FedAvg client did not finish a finite complete local epoch")
            self.ops.assert_model_device(model, self.profile.device)
            self.state.update(last_round=round_id,
                              completed_batches=self.state["completed_batches"] + source.executed,
                              rng=self.ops.capture_rng(self.profile.device))
            metrics = {
                "loss": weighted_loss / examples, "num_batches": source.executed,
                "num_examples": examples, "data_examples_seen": examples,
                "data_batch_hash": getattr(source, "batch_hash", ""),
                "completed_batches": self.state["completed_batches"],
                "logical_client_id": self.state["logical_client_id"],
                "partition_id": self.partition_id, "partition_hash": self.state["partition_hash"],
                "global_model_hash_received": received, "model_version": round_id,
                "worker_pid": os.getpid(), "worker_host": socket.gethostname(),
                "worker_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "worker_task_start_ns": started_ns, "worker_task_end_ns": time.perf_counter_ns(),
                "native_training_intervals_ns": json.dumps(native_intervals),
                "worker_capacity": self.config.resources.effective_worker_capacity or 1,
                "virtual_handler_ms": (time.perf_counter_ns() - started_ns) / 1e6,
                "rng_state_hash": hashlib.sha256(self.state["rng"]).hexdigest(),
                "rng_state_hash_scope": self.ops.rng_scope_description,
                "learning_rate": learning_rate, "client_peak_memory_mb": peak_rss_mb(),
                "updated_tensors": sum(not np.array_equal(a, b) for a, b in zip(parameters, values, strict=True)),
                "gpu_peak_memory_mb": self.ops.peak_gpu_memory_mb(self.profile.device),
                "device": self.profile.device, "worker_model_device_verified": True,
                "worker_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "resource_source": resource_receipt["resource_source"],
                "worker_affinity": json.dumps(resource_receipt["affinity"]),
                "worker_torch_threads": resource_receipt["torch_threads"],
            }
            if self.ops.backend != "torch":
                metrics.pop("worker_torch_threads")
                metrics["worker_framework_backend"] = self.ops.backend
                metrics["backend_thread_environment"] = json.dumps(resource_receipt["backend_thread_environment"], sort_keys=True)
            if metrics["gpu_peak_memory_mb"] is None:
                metrics.pop("gpu_peak_memory_mb")
            memory = self.ops.native_memory_receipt(self.profile.device)
            if memory is not None:
                metrics["worker_native_memory"] = json.dumps(memory, sort_keys=True)
            return values, examples, metrics


class _FedAvgFactory:
    def __init__(self, config, spec, partitions):
        self.config, self.spec, self.partitions = config, spec, partitions

    def __call__(self, context):
        return _FedAvgClient(context=context, config=self.config, spec=self.spec,
                             partitions=self.partitions)


def _release_model(model):
    import torch

    if isinstance(model, torch.nn.Module):
        for parameter in model.parameters():
            parameter.grad = None


class _FedAvgStrategy(FedAvg):
    """Use Flower selection/dispatch and the one shared aggregation implementation."""

    def __init__(self, *, config, spec, evidence, model, store):
        self.config, self.spec, self.evidence, self.model, self.store = config, spec, evidence, model, store
        self.ops = FrameworkOps.for_spec(spec)
        self.properties, self.round_evaluations = {}, []
        self.initial_round = None
        self.round_started = {}
        super().__init__(fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=config.virtual_clients.total,
                         min_evaluate_clients=0, min_available_clients=config.virtual_clients.total,
                         accept_failures=False, initial_parameters=ndarrays_to_parameters(
                             self.ops.export_arrays(model)))

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_started[server_round] = time.perf_counter()
        self.initial_round = parameters_to_ndarrays(parameters)
        global_hash = self.ops.hash_arrays(self.model, self.initial_round)
        instructions = super().configure_fit(server_round, parameters, client_manager)
        personalized = []
        identity_sec = 0.0
        for proxy, ins in instructions:
            if proxy.cid not in self.properties:
                identity_started = time.perf_counter()
                self.properties[proxy.cid] = dict(proxy.get_properties(GetPropertiesIns({}), timeout=300,
                                                                      group_id=server_round).properties)
                identity_sec += time.perf_counter() - identity_started
            props = self.properties[proxy.cid]
            personalized.append((proxy, FitIns(ins.parameters, {
                **ins.config, "quality_round": server_round,
                "virtual_partition_id": props["virtual_partition_id"],
                "virtual_global_model_hash": global_hash,
            })))
        personalized.sort(key=lambda item: self.properties[item[0].cid]["virtual_partition_id"])
        if {v["virtual_partition_id"] for v in self.properties.values()} != set(range(self.config.virtual_clients.total)):
            raise RuntimeError("FedAvg Flower partition identity is not bijective")
        if server_round == 1:
            self.evidence.write("first_configuration_timing.json", {
                "source": "measured", "round_id": server_round,
                "configuration_sec": time.perf_counter() - self.round_started[server_round],
                "identity_lookup_sec": identity_sec, "calibration": None,
            })
        return [(VirtualDispatchProxy(proxy, properties=self.properties[proxy.cid],
                                      state_store=self.store, service=None), ins) for proxy, ins in personalized]

    def aggregate_fit(self, server_round, results, failures):
        if failures or len(results) != self.config.virtual_clients.total:
            self.evidence.append("round_metrics.jsonl", {"round_id": server_round, "status": "failed",
                                                         "failures": [str(v) for v in failures]})
            raise RuntimeError("FedAvg round is incomplete; partial results cannot enter aggregation")
        results.sort(key=lambda item: self.properties[item[0].cid]["virtual_partition_id"])
        schema = {"names": self.ops.names(self.model), "owners": ["prefix"] * len(self.initial_round)}
        updates = ((schema, parameters_to_ndarrays(result.parameters), [], result.num_examples)
                   for _, result in results)
        arrays = aggregate_owned(self.initial_round, updates,
                                 integer_policy=self.spec.integer_aggregation_policy)
        model_hash = self.ops.hash_arrays(self.model, arrays)
        if self.config.workload.get("checkpoint_each_round") is True:
            from .checkpoint import save_round_checkpoint

            save_round_checkpoint(self.ops, self.model, arrays, self.evidence.directory,
                                  round_id=server_round, model_hash=model_hash)
        samples = sum(result.num_examples for _, result in results)
        train_loss = sum(result.num_examples * result.metrics["loss"] for _, result in results) / samples
        for proxy, result in results:
            self.evidence.append("client_metrics.jsonl", {
                "round_id": server_round, "node_id": proxy.cid,
                "partition_id": self.properties[proxy.cid]["virtual_partition_id"],
                "logical_client_id": self.properties[proxy.cid]["logical_client_id"],
                "status": "success", "num_examples": result.num_examples, "metrics": dict(result.metrics),
            })
        metrics = [r.metrics for _, r in results]
        self.evidence.append("round_metrics.jsonl", {
            "round_id": server_round, "source": "measured", "successful_clients": len(results),
            "round_wall_sec": time.perf_counter() - self.round_started[server_round],
            "failed_clients": 0, "train_loss": train_loss, "examples": samples,
            "model_hash": model_hash,
            "prefix_optimization_steps": sum(m["num_batches"] for m in metrics),
            "worker_processes_observed": len({(m["worker_host"], m["worker_pid"]) for m in metrics}),
            **task_concurrency(metrics),
        })
        if self.spec.round_callback is not None:
            self.spec.round_callback(server_round, arrays, {"train_loss": train_loss, "num_examples": samples,
                "successful_clients": len(results), "optimization_steps": sum(m["num_batches"] for m in metrics)},
                                     self.evidence.directory, phase="training")
        print(json.dumps({"event": "virtual_round_finished", "method": "fedavg", "round": server_round,
                          "successful_clients": len(results), "examples": samples,
                          "steps": sum(m["num_batches"] for m in metrics),
                          "configured_workers": self.config.resources.max_concurrent_clients,
                          "actual_workers_observed": len({(m["worker_host"], m["worker_pid"]) for m in metrics}),
                          **task_concurrency(metrics)}, sort_keys=True), flush=True)
        self.store.clear()
        self.initial_round = None
        return ndarrays_to_parameters(arrays), {"loss": train_loss}

    def evaluate(self, server_round, parameters):
        if self.spec.evaluate_fn is None or server_round == 0:
            return None
        started = time.perf_counter()
        arrays = parameters_to_ndarrays(parameters)
        self.ops.load_arrays(self.model, arrays)
        metrics = self.spec.evaluate_fn(self.model, self.config.server_device)
        observation = {"round_id": server_round, "metrics": metrics, "source": "measured",
                       "evaluation_sec": time.perf_counter() - started,
                       "elapsed_wall_sec": time.perf_counter() - self.evidence.started}
        self.round_evaluations.append(observation)
        self.evidence.append("evaluation_metrics.jsonl", observation)
        if self.spec.round_callback is not None:
            self.spec.round_callback(server_round, arrays, metrics, self.evidence.directory, phase="evaluation")
        return float(metrics["loss"]), metrics


def run_fedavg(config, spec, *, output):
    ops = FrameworkOps.for_spec(spec)
    ops.prepare_process(isolate_unused_frameworks=config.workload.get("isolate_unused_frameworks", False))
    ray, run_simulation = check_simulation_dependencies()
    evidence = Evidence(output, config)
    started = time.perf_counter()
    scratch = tempfile.TemporaryDirectory(prefix="splitfleet-fedavg-state-")
    store = RoundStateStore(scratch.name)
    try:
        with ops.rng_scope(seed=config.seed, device=config.server_device):
            model = ops.build_model(spec.model_factory, config.server_device)
        initial_hash = ops.state_hash(model)
        evidence.write("native_backend.json", ops.environment_receipt(model, config.server_device))
        partitions = training_partitions(config, spec, spec.train_dataset_factory())
        evidence.write("partition_manifest.json", {"data": spec.data_manifest, "clients": {
            cid: {"indices": indices, "hash": partition_hash(indices), "num_samples": len(indices)}
            for cid, indices in partitions.items()}})
        init_args, pool_size = initialize_ray(config, ray, ops)
        evidence.record_capacity_observation(
            config.resources, native_pool_size=pool_size
        )
        strategy = _FedAvgStrategy(config=config, spec=spec, evidence=evidence, model=model, store=store)
        server = Server(client_manager=SimpleClientManager(), strategy=strategy)
        # Match SplitFleet's coordinator fan-out; Ray still bounds actual
        # client workers using the shared resource configuration.
        server.set_max_workers(config.virtual_clients.total)
        app = ServerApp(server_fn=lambda context: ServerAppComponents(
            server=server, config=ServerConfig(num_rounds=config.rounds, round_timeout=spec.round_timeout_sec)))
        run_simulation(server_app=app, client_app=make_client_app(_FedAvgFactory(
                           config, spec, partitions)),
                       num_supernodes=config.virtual_clients.total, backend_config={
                           "init_args": init_args, "client_resources": {
                               "num_cpus": config.resources.client_cpus_per_worker or 1.0,
                               "num_gpus": config.resources.client_gpus_per_worker}})
        arrays = parameters_to_ndarrays(server.parameters)
        if any(not np.isfinite(v).all() for v in arrays):
            raise FloatingPointError("FedAvg final state is nonfinite")
        checkpoint = ops.save_checkpoint(model, arrays, evidence.directory / "final_model")
        rows = evidence.read_rows("client_metrics.jsonl")
        expected = config.rounds * config.virtual_clients.total
        if len(rows) != expected:
            raise RuntimeError("FedAvg did not complete the declared client/round budget")
        result = {"status": "passed", "native_flower_ray": True, "real_split_rpc": False,
                  "logical_clients": config.virtual_clients.total, "worker_limit": pool_size,
                  "worker_target_count": config.resources.worker_target_count,
                  "capacity_observation": config.resources.capacity_observation,
                  "native_worker_pool_size": pool_size,
                  "worker_count_policy": config.resources.worker_count_policy,
                  "requested_worker_count": config.resources.worker_target_count,
                  **round_concurrency(evidence.read_rows("round_metrics.jsonl")),
                  "rounds": config.rounds, "training_tasks": len(rows),
                  "optimization_steps": sum(row["metrics"]["num_batches"] for row in rows),
                  "examples_seen": sum(row["num_examples"] for row in rows),
                  "initial_model_hash": initial_hash, "final_model_hash": ops.hash_arrays(model, arrays),
                  "framework_backend": ops.backend, "checkpoint": checkpoint.name,
                  "model_hash_layout": "native_adapter_schema",
                  "wall_sec": time.perf_counter() - started}
        evidence.write("model_metrics.json", {"evaluations": strategy.round_evaluations})
        evidence.write("validation.json", result)
        evidence.write("summary.json", result)
        return result
    except BaseException as error:
        for name in ("validation.json", "summary.json"):
            if not (evidence.directory / name).exists():
                evidence.write(
                    name,
                    {
                        "status": "failed",
                        "reason": repr(error),
                        "capacity_observation": config.resources.capacity_observation,
                    },
                )
        raise
    finally:
        store.clear()
        scratch.cleanup()
        ray.shutdown()
