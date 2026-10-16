"""Run actual SFL through Flower's ClientApp/ServerApp/RayBackend."""

from __future__ import annotations

import importlib.metadata
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from flwr.common import parameters_to_ndarrays
from flwr.server import SimpleClientManager
from torch.utils.data import default_collate

from splitfleet.autosplit import ExecutionSchedulePolicy
from splitfleet.server.placement.cosplit_ucb import (
    CoSplitUCBConfig,
    CoSplitUCBPlacementPolicy,
)
from splitfleet.server.placement.cosplit_ucb.candidate_provider import (
    TorchLensCandidateProvider,
)
from splitfleet.server.stage_runtime.manager import StageRuntimeManager
from splitfleet.server.stage_runtime.storage import RoundStateStore
from splitfleet.tasks import TaskSpec

from .backend_ops import FrameworkOps
from .client_adapter import ClientFactory, OptimizerFactory
from .client_app import make_client_app
from .config import VirtualConfig
from .partitions import partition_dataset, partition_hash
from .resources import peak_rss_mb
from .server_adapter import RuntimeConditions, VirtualServer, VirtualStrategy, round_concurrency
from .server_app import make_server_app
from .state import client_seed
from .suffix_service import LeasedSuffixServicer, SuffixService
from .telemetry import Evidence


@dataclass(frozen=True)
class TrainingSpec:
    """Existing task/model/data interfaces; factories may lazily read shared data."""

    task: TaskSpec
    model_factory: Callable
    train_dataset_factory: Callable
    collate_fn: Callable | None = None
    batch_axes: dict | None = None
    partition_labels: tuple[int, ...] | None = None
    evaluate_fn: Callable | None = None
    candidate_provider_factory: Callable | None = None
    source: str = "real"
    data_manifest: dict | None = None
    frozen_partitions: dict[str, list[int]] | None = None
    batch_factory: Callable | None = None
    learning_rate_fn: Callable | None = None
    trainer: Callable | None = None
    placement_policy_factory: Callable | None = None
    round_callback: Callable | None = None
    integer_aggregation_policy: str = "weighted_round"
    round_timeout_sec: float | None = None
    framework_backend: str = "torch"


def training_partitions(config, spec, dataset):
    """Admit explicit shared partitions, retaining the historical default."""
    if spec.frozen_partitions is None:
        return partition_dataset(
            len(dataset), config.virtual_clients.total, config=config.partition,
            seed=config.seed, labels=spec.partition_labels, family=spec.task.family.value,
        )
    partitions = {str(cid): list(indices) for cid, indices in spec.frozen_partitions.items()}
    if set(partitions) != {str(cid) for cid in range(config.virtual_clients.total)}:
        raise ValueError("Frozen partition identities differ from the declared population")
    flat = [index for indices in partitions.values() for index in indices]
    if (len(flat) != len(set(flat)) or not all(indices for indices in partitions.values())
            or any(type(index) is not int or not 0 <= index < len(dataset) for index in flat)):
        raise ValueError("Frozen partitions are empty, overlapping or outside the dataset")
    return partitions


def round_learning_rate(config, spec, round_id):
    return float(spec.learning_rate_fn(round_id)) if spec.learning_rate_fn else config.learning_rate


def initialize_ray(config, ray, ops=None):
    """Use the native Ray resource budget and admit the exact worker pool."""
    resources = config.resources
    ops = ops or FrameworkOps()
    init_args = {"include_dashboard": False, "ignore_reinit_error": False}
    if resources.ray_address:
        init_args["address"] = resources.ray_address
    else:
        cpu_budget = resources.client_cpus_per_worker * resources.max_concurrent_clients
        integer_budget = math.isclose(cpu_budget, round(cpu_budget), rel_tol=0, abs_tol=1e-9)
        if not integer_budget and cpu_budget > 1:
            raise ValueError("Ray 2.51.1 requires an integer local CPU budget")
        init_args.update(num_cpus=round(cpu_budget) if integer_budget else cpu_budget,
                         object_store_memory=resources.object_store_memory_mb * 2**20)
        if resources.client_gpus_per_worker:
            available = ops.gpu_count()
            gpu_pool = resources.client_gpu_pool_size
            if gpu_pool is not None and gpu_pool > available:
                raise RuntimeError("The local client GPU pool exceeds native visible GPU devices")
            init_args["num_gpus"] = available if gpu_pool is None else gpu_pool
        else:
            # Ray otherwise autodetects GPU resources, including a GPU reserved
            # for this process's suffix/coordinator.
            init_args["num_gpus"] = 0
    if not ray.is_initialized():
        environment = {"env_vars": {
            "PYTHONPATH": os.pathsep.join(sys.path),
            **{name: os.environ[name] for name in (
                "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
                "TF_ENABLE_ONEDNN_OPTS", "JAX_PLATFORMS", "JAX_DEFAULT_MATMUL_PRECISION",
                "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_ALLOCATOR",
                "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION", "CACHELEVEL",
                "XLA_FLAGS", "PYTHONFAULTHANDLER",
            ) if name in os.environ},
        }}
        if ops.backend == "paddle" or (
                config.workload.get("isolate_unused_frameworks") and ops.backend != "tf"):
            environment["worker_process_setup_hook"] = "splitfleet.virtual.backend_ops.isolate_unused_tensorflow"
        if resources.ray_address:
            environment["working_dir"] = str(Path(__file__).resolve().parents[2])
        ray.init(runtime_env=environment, **init_args)
    from flwr.simulation.ray_transport.ray_actor import pool_size_from_resources
    actual = pool_size_from_resources({"num_cpus": resources.client_cpus_per_worker,
                                       "num_gpus": resources.client_gpus_per_worker})
    if actual != resources.max_concurrent_clients:
        raise ValueError(
            f"Native Flower pool has {actual} workers; expected {resources.max_concurrent_clients} "
            f"for worker_count_policy={resources.worker_count_policy}. "
            "Adjust the CPU/GPU reservations or supply a larger Ray cluster; "
            "the runtime will not fall back to fewer workers."
        )
    return init_args, actual


def check_simulation_dependencies():
    try:
        import ray
        from flwr.simulation import run_simulation
    except ImportError as exc:
        raise RuntimeError(
            "Virtual SFL needs the native simulation dependencies. Install with: uv sync --extra simulation"
        ) from exc
    if importlib.metadata.version("flwr") != "1.29.0" or ray.__version__ != "2.51.1":
        raise RuntimeError(
            "This adapter is verified against Flower 1.29.0 / Ray 2.51.1; use uv sync --extra simulation"
        )
    return ray, run_simulation


def _admit_resources(config, model, ops=None):
    ops = ops or FrameworkOps.for_model(model)
    ops.admit_device(config.server_device)
    if (
        not config.resources.ray_address
        and any(p.device.startswith("cuda") for p in config.profiles.values())
        and not ops.gpu_count()
    ):
        raise RuntimeError(
            "No local GPU is available for the declared virtual client profiles"
        )
    capacities = [
        config.resources.server_concurrency,
        *config.resources.server_concurrency_by_round.values(),
    ]
    if (config.resources.resident_suffix_capacity > config.resources.suffix_capacity
            or max(capacities) > config.resources.resident_suffix_capacity):
        raise ValueError(
            "Server execution lanes cannot exceed the admitted suffix replica capacity"
        )
    if ops.backend == "torch" and max(capacities) > 1 and any(
        isinstance(m, (torch.nn.modules.dropout._DropoutNd, torch.nn.RNNBase))
        for m in model.modules()
    ):
        raise ValueError(
            "Stochastic suffix models require server_concurrency=1 for isolated Torch RNG state"
        )
    state_bytes = ops.state_bytes(model)
    # Explicit state/optimizer floor; graph/activation memory is additionally
    # measured in runtime receipts, rather than passed off as this estimate.
    if (config.resources.per_client_suffix_models
            and config.resources.suffix_capacity != config.virtual_clients.total):
        raise ValueError("Per-client suffix model count must equal the logical client population")
    required = state_bytes * (4 * config.resources.resident_suffix_capacity + 3) / 2**20
    if (
        config.resources.server_memory_budget_mb is not None
        and required > config.resources.server_memory_budget_mb
    ):
        raise MemoryError(
            f"Declared suffix memory budget is below state/optimizer admission floor: {required:.1f} MiB"
        )
    if config.server_device.startswith("cuda"):
        free = ops.free_gpu_memory(config.server_device)
        if free is not None and required * 2**20 > free:
            raise MemoryError(
                "Available server GPU memory is below the suffix state/optimizer admission floor"
            )
    return {
        "state_bytes": state_bytes,
        "state_optimizer_floor_mb": required,
        "floor_source": "estimated",
        "framework_backend": ops.backend,
        "suffix_model_count": config.resources.suffix_capacity,
        "resident_suffix_limit": config.resources.resident_suffix_capacity,
    }


def run_virtual(config: VirtualConfig, spec: TrainingSpec, *, output: str | Path):
    """Synchronous entry point. Existing results are never overwritten."""
    ops = FrameworkOps.for_spec(spec)
    ops.prepare_process(isolate_unused_frameworks=config.workload.get("isolate_unused_frameworks", False))
    ray, run_simulation = check_simulation_dependencies()
    evidence = Evidence(output, config)
    started = time.perf_counter()
    store = None
    service = None
    try:
        with ops.rng_scope(seed=config.seed, device=config.server_device):
            torch.set_num_threads(1)
            model = ops.build_model(spec.model_factory, config.server_device)
            initial_hash = ops.state_hash(model)
            admission = _admit_resources(config, model, ops)
            evidence.write("native_backend.json", ops.environment_receipt(model, config.server_device))
            dataset = spec.train_dataset_factory()
            partitions = training_partitions(config, spec, dataset)
            evidence.write(
                "partition_manifest.json",
                {
                    "source": spec.source,
                    "data": spec.data_manifest,
                    "seed": config.seed,
                    "method": config.partition.method,
                    "disjoint": True,
                    "dataset_size": len(dataset),
                    "clients": {
                        cid: {
                            "indices": indices,
                            "num_samples": len(indices),
                            "hash": partition_hash(indices),
                        }
                        for cid, indices in partitions.items()
                    },
                },
            )
            collate = spec.collate_fn or default_collate
            sample = spec.task.make_adapter().prepare_batch(
                collate(
                    [
                        dataset[partitions["0"][i % len(partitions["0"])]]
                        for i in range(config.batch_size)
                    ]
                )
            )
            sample = ops.prepare_batch(model, sample, config.server_device)
            scratch = tempfile.TemporaryDirectory(prefix="splitfleet-round-state-")
            store = RoundStateStore(scratch.name)
            policy = None
            if config.placement == "cosplit_ucb" or spec.placement_policy_factory is not None:
                provider_model = ops.adapter.clone_model(model)
                provider_kwargs = {
                    "model": provider_model,
                    "sample_inputs": ops.prepare_batch(provider_model, sample, config.server_device).inputs,
                    "batch_axes": spec.batch_axes,
                    "dynamic_batch": (1, config.batch_size),
                }
                if spec.candidate_provider_factory is None:
                    provider_kwargs["require_trainable_prefix"] = True
                    provider_kwargs["resident_parameter_transport"] = ops.backend == "jax" and config.optimizer == "sgd"
                provider = (
                    spec.candidate_provider_factory or TorchLensCandidateProvider
                )(**provider_kwargs)
                policy = (spec.placement_policy_factory or CoSplitUCBPlacementPolicy)(
                    candidate_provider=provider,
                    config=CoSplitUCBConfig(
                        server_concurrency=config.resources.server_concurrency,
                        seed=config.seed,
                        state_path=str(evidence.directory / "cosplit_state.json"),
                    ),
                )
            strategy = VirtualStrategy(
                virtual_config=config,
                evidence=evidence,
                spec=spec,
                partitions=partitions,
                model=model,
                sample_inputs=sample,
                batch_axes=spec.batch_axes,
                task=spec.task.make_adapter(),
                boundary=config.boundary,
                placement_policy=policy,
                aggregation_policy="splitfed",
                owned_state_exchange=True,
                round_state_store=store,
                dynamic_batch=(1, config.batch_size),
                runtime_device=config.server_device,
                defer_server_models=config.resources.per_client_suffix_models,
                optimizer_fn=OptimizerFactory(config.optimizer, config.learning_rate, ops.backend),
                integer_aggregation_policy=spec.integer_aggregation_policy,
                execution_schedule_policy=ExecutionSchedulePolicy(
                    process_clients_as_batch=False
                ),
                fraction_fit=config.virtual_clients.participation_fraction,
                fraction_evaluate=0.0,
                min_evaluate_clients=0,
                min_available_clients=config.virtual_clients.total,
            )
            manager = StageRuntimeManager(
                init_server_model_fn=strategy.init_server_model_fn,
                autosplit_session=strategy.autosplit_session,
                replica_capacity=config.resources.resident_suffix_capacity,
                state_store=store,
                reuse_completed_replicas=not config.resources.per_client_suffix_models,
                per_client_models=config.resources.per_client_suffix_models,
                model_capacity=config.resources.suffix_capacity,
            )
            strategy.bind_stage_runtime_manager(manager)
            server = VirtualServer(
                evidence=evidence,
                client_manager=SimpleClientManager(),
                strategy=strategy,
                server_model_manager=manager,
            )
            server.set_max_workers(config.virtual_clients.total)
            max_concurrency = max(
                [
                    config.resources.server_concurrency,
                    *config.resources.server_concurrency_by_round.values(),
                ]
            )
            class NativeLeasedSuffixServicer(LeasedSuffixServicer):
                def _compute_group(self, group, method_name):
                    if ops.backend == "torch":
                        return super()._compute_group(group, method_name)
                    from splitfleet.server.grpc.servicer import ServerModelServicer
                    return ServerModelServicer._compute_group(self, group, method_name)

            servicer = NativeLeasedSuffixServicer(
                manager,
                capacity=config.resources.resident_suffix_capacity,
                concurrency=max_concurrency,
                state_store=store,
                seed=config.seed,
            )
            transport = config.transport
            service = SuffixService(
                manager,
                bind_host=transport.bind_host,
                advertise_host=transport.advertise_host,
                port=transport.port,
                certificates=transport.credentials("server"),
                servicer=servicer,
            )
            strategy.service = service
            if policy is not None:
                policy.runtime_context_provider = RuntimeConditions(strategy)
            import asyncio

            manager.before_collect = lambda: asyncio.run_coroutine_threadsafe(
                servicer.reconcile(), service.loop
            ).result(timeout=300)
            app = make_server_app(server, rounds=config.rounds, service=service)

            resources = config.resources
            init_args, actual = initialize_ray(config, ray, ops)
            admission["native_worker_pool_size"] = actual
            client_factory = ClientFactory(
                config=config, spec=spec, partitions=partitions, run_scope=str(evidence.directory)
            )
            run_simulation(
                server_app=app,
                client_app=make_client_app(client_factory),
                num_supernodes=config.virtual_clients.total,
                backend_config={
                    "init_args": init_args,
                    "client_resources": {
                        "num_cpus": resources.client_cpus_per_worker,
                        "num_gpus": resources.client_gpus_per_worker,
                    },
                },
            )
            arrays = parameters_to_ndarrays(server.client_parameters)
            if any(not np.isfinite(value).all() for value in arrays):
                raise FloatingPointError(
                    "Aggregated global state is nonfinite; the run cannot pass validation"
                )
            final_hash = ops.hash_arrays(model, arrays)
            checkpoint = ops.save_checkpoint(model, arrays, evidence.directory / "final_model")
            rows = evidence.read_rows("client_metrics.jsonl")
            prefixes = [r for r in rows if r["status"] == "success"]
            suffixes = {
                (r["round_id"], r["node_id"]): r
                for r in rows
                if r["status"] == "suffix_committed"
            }
            valid = server.rounds_finished == config.rounds and all(
                (r["round_id"], r["node_id"]) in suffixes
                and r["metrics"]["num_batches"] > 0
                and suffixes[(r["round_id"], r["node_id"])]["metrics"][
                    "suffix_optimization_steps"
                ]
                == r["metrics"]["num_batches"]
                for r in prefixes
            )
            unexpected_failures = []
            for row in rows:
                if row["status"] not in {"failure", "timeout"}:
                    continue
                pid, round_id = row["partition_id"], row["round_id"]
                population = config.virtual_clients
                planned = pid is not None and (
                    round_id in population.failure_rounds.get(str(pid), [])
                    or np.random.default_rng(
                        client_seed(config.seed, pid, f"fault:{round_id}")
                    ).random()
                    < population.failure_probability
                )
                if (
                    not planned
                    or "Injected virtual client dropout" not in row["reason"]
                ):
                    unexpected_failures.append(
                        {
                            "partition_id": pid,
                            "round_id": round_id,
                            "reason": row["reason"],
                        }
                    )
            valid = valid and not unexpected_failures
            validation = {
                "status": "passed"
                if valid and prefixes and final_hash != initial_hash
                else "failed",
                "native_flower_ray": True,
                "real_split_rpc": True,
                "framework_backend": ops.backend,
                "checkpoint": checkpoint.name,
                "model_hash_layout": "native_adapter_schema",
                "rounds_completed": server.rounds_finished,
                "successful_training_tasks": len(prefixes),
                "failed_training_tasks": sum(
                    r["status"] in {"failure", "timeout"} for r in rows
                ),
                "unexpected_failures": unexpected_failures,
                "prefix_updates": sum(
                    r["metrics"]["prefix_changed_tensors"] > 0 for r in prefixes
                ),
                "suffix_updates": sum(
                    suffixes[(r["round_id"], r["node_id"])]["metrics"].get(
                        "suffix_changed_tensors", 0
                    )
                    > 0
                    for r in prefixes
                ),
                "optimization_steps": sum(
                    r["metrics"]["num_batches"] for r in prefixes
                ),
                "initial_model_hash": initial_hash,
                "final_model_hash": final_hash,
                "model_changed": final_hash != initial_hash,
                "data_source": spec.source,
                "multi_node": "passed"
                if len({r["metrics"]["worker_host"] for r in prefixes}) > 1
                else "not_executed",
            }
            evidence.write("validation.json", validation)
            evidence.write(
                "model_metrics.json",
                {
                    "evaluations": strategy.round_evaluations,
                    "initial_hash": initial_hash,
                    "final_hash": final_hash,
                },
            )
            evidence.write(
                "resource_metrics.json",
                {
                    "source": "measured",
                    "server_peak_rss_mb": peak_rss_mb(),
                    "server_gpu_peak_memory_mb": ops.peak_gpu_memory_mb(config.server_device),
                    "worker_peak_rss_mb": max(
                        (r["metrics"]["client_peak_memory_mb"] for r in prefixes),
                        default=None,
                    ),
                    "worker_limit": resources.max_concurrent_clients,
                    "worker_count_policy": resources.worker_count_policy,
                    "requested_worker_count": config.workload.get("worker_target_count", config.virtual_clients.total),
                    "native_worker_pool_size": actual,
                    "worker_execution_slots_per_process": resources.client_runtime_slots or 1,
                    "worker_execution_resident_slot_bound": actual * (resources.client_runtime_slots or 1),
                    "peak_suffix_replicas": manager._delegate.peak_live,
                    "per_client_suffix_models": resources.per_client_suffix_models,
                    "suffix_model_count": resources.suffix_capacity,
                    "observed_suffix_model_count": manager._delegate.peak_client_models if resources.per_client_suffix_models else None,
                    "resident_suffix_limit": resources.resident_suffix_capacity,
                    "suffix_execution_limit": max_concurrency,
                    **round_concurrency(evidence.read_rows("round_metrics.jsonl")),
                    "suffix_replicas_created": manager._delegate.total_created,
                    "suffix_replicas_reused": manager._delegate.total_reused,
                    "peak_server_executions": servicer.peak_executions,
                    "admission": admission,
                    "per_task_state": "Context stores identity, progress and host RNG only; optimizer rebuilt each round",
                    "rng_scope": ops.rng_scope_description,
                    "aggregation_storage": "prefix/suffix updates spill to disk; one logical accumulator in RAM",
                },
            )
            summary = {
                "status": validation["status"],
                "source": "measured",
                "output": str(evidence.directory),
                "logical_clients": config.virtual_clients.total,
                "worker_limit": resources.max_concurrent_clients,
                "worker_count_policy": resources.worker_count_policy,
                "requested_worker_count": config.workload.get("worker_target_count", config.virtual_clients.total),
                "suffix_model_count": manager._delegate.peak_client_models if resources.per_client_suffix_models else resources.suffix_capacity,
                "resident_suffix_limit": resources.resident_suffix_capacity,
                "suffix_execution_limit": max_concurrency,
                **round_concurrency(evidence.read_rows("round_metrics.jsonl")),
                "rounds": server.rounds_finished,
                "training_tasks": len(prefixes),
                "optimization_steps": validation["optimization_steps"],
                "wall_sec": time.perf_counter() - started,
                "model_changed": final_hash != initial_hash,
            }
            evidence.write("summary.json", summary)
            if validation["status"] != "passed":
                raise RuntimeError(
                    "Virtual SFL execution failed validation; inspect validation.json"
                )
            return summary
    except BaseException as exc:
        if not (evidence.directory / "validation.json").exists():
            evidence.write(
                "validation.json",
                {"status": "failed", "reason": str(exc), "multi_node": "not_executed"},
            )
        if not (evidence.directory / "summary.json").exists():
            evidence.write(
                "summary.json",
                {
                    "status": "failed",
                    "reason": str(exc),
                    "wall_sec": time.perf_counter() - started,
                },
            )
        raise
    finally:
        if service is not None and service.loop is not None:
            service.stop()
        if store is not None:
            store.clear()
            scratch.cleanup()
        ray.shutdown()
