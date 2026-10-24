"""Run actual SFL through Flower's ClientApp/ServerApp/RayBackend."""

from __future__ import annotations

import inspect
import json
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
from splitfleet.server.placement.greedy import (
    GreedyConfig,
    GreedyPlacementPolicy,
)
from splitfleet.server.placement.greedy.candidate_provider import (
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
    initialized = ray.is_initialized()
    init_args = {"include_dashboard": False, "ignore_reinit_error": False}
    if resources.ray_address:
        init_args["address"] = resources.ray_address
    else:
        # A single-host simulation must not depend on a physical interface or
        # public DNS being reachable when Ray selects its node address.
        init_args["_node_ip_address"] = "127.0.0.1"
        worker_capacity = (
            resources.effective_worker_capacity
            or resources.max_concurrent_clients
            or config.virtual_clients.total
        )
        cpu_budget = (resources.client_cpus_per_worker or 1.0) * worker_capacity
        # Actor reservations may be fractional; a Ray node's total CPUs must
        # be integral. Memory admission can reduce the worker count after its
        # reservation was derived. Spread the rounded budget over precisely
        # those workers so rounding up does not create an oversized pool.
        native_cpu_budget = (round(cpu_budget)
                             if math.isclose(cpu_budget, round(cpu_budget), rel_tol=0, abs_tol=1e-9)
                             else math.ceil(cpu_budget))
        if not initialized and native_cpu_budget != cpu_budget:
            reservation = native_cpu_budget / worker_capacity
            if int(native_cpu_budget / reservation) < worker_capacity:
                reservation = math.nextafter(reservation, 0.)
            object.__setattr__(resources, "client_cpus_per_worker", reservation)
        init_args.update(num_cpus=native_cpu_budget,
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
    if not initialized:
        worker_threads = str(max(profile.torch_threads for profile in config.profiles.values()))
        environment = {"env_vars": {
            "PYTHONPATH": os.pathsep.join(sys.path),
            # Set native pools before the reused actor imports its frameworks.
            # Per-task Torch scopes should not rebuild a host-sized pool.
            "OMP_NUM_THREADS": worker_threads,
            "MKL_NUM_THREADS": worker_threads,
            "OPENBLAS_NUM_THREADS": worker_threads,
            "TF_NUM_INTRAOP_THREADS": worker_threads,
            "TF_NUM_INTEROP_THREADS": "1",
            **{name: os.environ[name] for name in (
                "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
                "TF_ENABLE_ONEDNN_OPTS", "TF_FORCE_GPU_ALLOW_GROWTH",
                "CACHELEVEL", "PYTHONFAULTHANDLER",
            ) if name in os.environ},
        }}
        if ops.backend == "paddle" or (
                config.workload.get("isolate_unused_frameworks") and ops.backend != "tf"):
            environment["worker_process_setup_hook"] = "splitfleet.virtual.backend_ops.isolate_unused_tensorflow"
        if resources.ray_address:
            environment["working_dir"] = str(Path(__file__).resolve().parents[2])
        ray.init(runtime_env=environment, **init_args)
    from flwr.simulation.ray_transport.ray_actor import pool_size_from_resources
    actual = pool_size_from_resources({"num_cpus": resources.client_cpus_per_worker or 1.0,
                                       "num_gpus": resources.client_gpus_per_worker})
    predicted = resources.effective_worker_capacity or resources.max_concurrent_clients
    if predicted is None:
        raise RuntimeError("Worker capacity was not admitted before Ray initialization")
    if actual != predicted:
        message = {
            "event": "worker_capacity_mismatch",
            "predicted": int(predicted),
            "actual": int(actual),
            "policy": resources.worker_count_policy,
            "configured_policy": resources.worker_count_policy,
            "source": "native_flower_ray_pool",
        }
        # Fixed experiments are reproducibility contracts.  The maximizing
        # policy can adopt a smaller native pool and keep logical clients
        # queued in Flower; it never fabricates capacity.
        if resources.strict_worker_capacity or actual > predicted:
            resources.record_capacity_observation(
                int(actual), predicted=int(predicted)
            )
            raise ValueError(
                f"Native Flower pool has {actual} workers; expected {predicted} "
                f"for admitted worker_count_policy={resources.worker_count_policy}. "
                f"Use a Ray pool within the admitted limit or increase the per-worker "
                f"resource reservation: {message}"
            )
        print(json.dumps(message, sort_keys=True), flush=True)
        resources.adopt_capacity(int(actual), predicted=int(predicted))
    else:
        resources.adopt_capacity(int(actual), predicted=int(predicted))
    _validate_explicit_suffix_concurrency(resources)
    capacities = [
        resources.effective_server_concurrency,
        *resources.server_concurrency_by_round.values(),
    ]
    if max(capacities) > resources.resident_suffix_capacity:
        raise ValueError(
            "Native worker admission left fewer resident suffix slots than the "
            "configured server execution concurrency"
        )
    return init_args, actual


def _validate_explicit_suffix_concurrency(resources):
    """Reject explicit lane requests that no longer fit resident slots."""
    resident = resources.resident_suffix_capacity
    if resources.server_concurrency is not None and resources.server_concurrency > resident:
        raise ValueError(
            "server_concurrency exceeds the admitted resident suffix capacity"
        )
    if any(capacity > resident for capacity in resources.server_concurrency_by_round.values()):
        raise ValueError(
            "server_concurrency_by_round exceeds the admitted resident suffix capacity"
        )


def check_simulation_dependencies():
    """Load the current Flower Simulation Runtime entry point.

    SplitFleet deliberately targets the modern ``ClientApp``/``ServerApp``
    entry point.  There is no legacy ``start_simulation`` or old-version
    compatibility branch here.  The version itself is left to the lockfile;
    the callable signature is checked so an incompatible installation fails
    with an actionable message before Ray is started.
    """
    try:
        import ray
        from flwr.simulation import run_simulation
    except ImportError as exc:
        raise RuntimeError(
            "Virtual SFL needs the native simulation dependencies. Install with: uv sync --extra simulation"
        ) from exc
    required = {"server_app", "client_app", "num_supernodes"}
    if not callable(run_simulation) or not required.issubset(inspect.signature(run_simulation).parameters):
        raise RuntimeError(
            "The installed Flower package does not expose the current "
            "ClientApp/ServerApp run_simulation API; install flwr[simulation] "
            "from the locked project environment."
        )
    return ray, run_simulation


def _stochastic_torch_modules(model):
    """Return native Torch modules that consume random values.

    The list is retained as an admission receipt and diagnostics field.  It is
    not a concurrency gate: Torch suffix replicas always receive a private
    thread-local RNG by default, including when several lanes are active.
    """
    if not isinstance(model, torch.nn.Module):
        return ()
    stochastic = []
    for module in model.modules():
        if isinstance(module, (torch.nn.modules.dropout._DropoutNd, torch.nn.RNNBase)):
            stochastic.append(module.__class__.__name__)
    return tuple(stochastic)


def _admit_resources(config, model, ops=None):
    ops = ops or FrameworkOps.for_model(model)
    ops.admit_device(config.server_device)
    _validate_explicit_suffix_concurrency(config.resources)
    if (
        not config.resources.ray_address
        and any(p.device.startswith("cuda") for p in config.profiles.values())
        and not ops.gpu_count()
    ):
        raise RuntimeError(
            "No local GPU is available for the declared virtual client profiles"
        )
    capacities = [
        config.resources.effective_server_concurrency,
        *config.resources.server_concurrency_by_round.values(),
    ]
    if max(capacities) > config.resources.resident_suffix_capacity:
        raise ValueError(
            "Server execution lanes cannot exceed resident suffix capacity"
        )
    stochastic = _stochastic_torch_modules(model) if ops.backend == "torch" else ()
    state_bytes = ops.state_bytes(model)
    # Explicit state/optimizer floor; graph/activation memory is additionally
    # measured in runtime receipts, rather than passed off as this estimate.
    if (config.resources.per_client_suffix_models
            and config.resources.suffix_capacity != config.virtual_clients.total):
        raise ValueError("Per-client suffix model count must equal the logical client population")
    required = state_bytes * (4 * config.resources.resident_suffix_capacity + 3) / 2**20
    # Native suffix state/optimizer memory is a physical-slot constraint.  A
    # maximizing run may lower its admitted worker count before the manager is
    # built; a fixed run fails closed instead of silently queueing a smaller
    # experiment than requested.
    def constrain_by_memory(available_mb, source):
        if available_mb is None:
            return
        capacity = math.floor((available_mb * 2**20 / max(state_bytes, 1) - 3) / 4)
        capacity = max(0, capacity)
        # Server memory limits materialized suffix replicas.  A run may have
        # many client workers while keeping only a smaller resident suffix
        # population (``max_suffix_replicas``/``max_resident_suffix_replicas``).
        # Compare against that physical suffix bound instead of shrinking the
        # unrelated client worker pool.
        current = config.resources.resident_suffix_capacity
        if capacity >= current:
            return
        explicit_suffix_bound = (
            config.resources.max_suffix_replicas is not None
            or config.resources.max_resident_suffix_replicas is not None
        )
        if not explicit_suffix_bound and config.resources.strict_worker_capacity:
            raise MemoryError(
                f"{source} admits {capacity} suffix slots, below strict worker capacity {current}"
            )
        if capacity < 1:
            raise MemoryError(f"{source} cannot admit one suffix execution slot")
        print(
            json.dumps(
                {
                    "event": (
                        "suffix_capacity_memory_bound"
                        if explicit_suffix_bound
                        else "worker_capacity_memory_bound"
                    ),
                    "predicted": current,
                    "actual": capacity,
                    "source": source,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if explicit_suffix_bound:
            # Keep logical suffix-model count and client-worker capacity intact;
            # only the number of native suffixes resident at once is reduced.
            object.__setattr__(
                config.resources,
                "max_resident_suffix_replicas",
                int(capacity),
            )
        else:
            config.resources.adopt_capacity(
                int(capacity), predicted=int(current), source=source
            )

    constrain_by_memory(config.resources.server_memory_budget_mb, "declared_server_memory_budget")
    _validate_explicit_suffix_concurrency(config.resources)
    required = state_bytes * (4 * config.resources.resident_suffix_capacity + 3) / 2**20
    if max(
        [
            config.resources.effective_server_concurrency,
            *config.resources.server_concurrency_by_round.values(),
        ]
    ) > config.resources.resident_suffix_capacity:
        raise ValueError(
            "Server execution lanes exceed the memory-admitted resident suffix capacity"
        )
    if (
        config.resources.server_memory_budget_mb is not None
        and required > config.resources.server_memory_budget_mb
    ):
        raise MemoryError(
            f"Declared suffix memory budget is below state/optimizer admission floor: {required:.1f} MiB"
        )
    if config.server_device.startswith("cuda"):
        free = ops.free_gpu_memory(config.server_device)
        constrain_by_memory(
            free / 2**20 if free is not None else None,
            "server_gpu_free_memory",
        )
        _validate_explicit_suffix_concurrency(config.resources)
        if free is not None and required * 2**20 > free:
            required = state_bytes * (4 * config.resources.resident_suffix_capacity + 3) / 2**20
            if required * 2**20 > free:
                raise MemoryError(
                    "Available server GPU memory is below the suffix state/optimizer admission floor"
                )
    required = state_bytes * (4 * config.resources.resident_suffix_capacity + 3) / 2**20
    return {
        "state_bytes": state_bytes,
        "state_optimizer_floor_mb": required,
        "floor_source": "estimated",
        "framework_backend": ops.backend,
        "suffix_model_count": config.resources.suffix_capacity,
        "logical_suffix_model_count": config.resources.logical_suffix_models,
        "resident_suffix_limit": config.resources.resident_suffix_capacity,
        "effective_worker_capacity": config.resources.effective_worker_capacity,
        "effective_suffix_slots": config.resources.effective_suffix_slots,
        "effective_server_concurrency": config.resources.effective_server_concurrency,
        "stochastic_suffix_modules": list(sorted(set(stochastic))),
        "replica_rng_isolation": ops.backend == "torch",
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
            # Admit the native Flower/Ray pool before constructing server
            # executors.  Any non-strict capacity mismatch is therefore
            # reflected in both worker and suffix-slot limits.
            init_args, actual = initialize_ray(config, ray, ops)
            admission["native_worker_pool_size"] = actual
            evidence.record_capacity_observation(
                config.resources, native_pool_size=actual
            )
            scratch = tempfile.TemporaryDirectory(prefix="splitfleet-round-state-")
            store = RoundStateStore(scratch.name)
            policy = None
            if config.placement == "greedy" or spec.placement_policy_factory is not None:
                provider_model = ops.adapter.clone_model(model)
                provider_kwargs = {
                    "model": provider_model,
                    "sample_inputs": ops.prepare_batch(provider_model, sample, config.server_device).inputs,
                    "batch_axes": spec.batch_axes,
                    "dynamic_batch": (1, config.batch_size),
                }
                if spec.candidate_provider_factory is None:
                    provider_kwargs["require_trainable_prefix"] = True
                provider = (
                    spec.candidate_provider_factory or TorchLensCandidateProvider
                )(**provider_kwargs)
                policy = (spec.placement_policy_factory or GreedyPlacementPolicy)(
                    candidate_provider=provider,
                    config=GreedyConfig(
                        server_concurrency=config.resources.effective_server_concurrency,
                        seed=config.seed,
                        state_path=str(evidence.directory / "greedy_state.json"),
                    ),
                )
            # Placement policies may create one logical suffix configuration
            # per client, but those configurations still use resettable native
            # slots unless persistent per-client suffix state was requested.
            # This keeps logical model identity separate from physical model
            # residency while retaining Worker/slot reuse by default.
            independent_suffix_models = bool(config.resources.per_client_suffix_models)
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
                defer_server_models=independent_suffix_models,
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
                reuse_completed_replicas=not independent_suffix_models,
                per_client_models=independent_suffix_models,
                model_capacity=(
                    config.virtual_clients.total
                    if independent_suffix_models
                    else config.resources.suffix_capacity
                ),
                # Every Torch suffix replica owns a private, thread-local RNG.
                # This remains enabled for deterministic models as well, so a
                # later cut change cannot silently alter concurrency semantics.
                replica_rng_isolation=bool(admission.get("replica_rng_isolation", ops.backend == "torch")),
            )
            strategy.bind_stage_runtime_manager(manager)
            server = VirtualServer(
                evidence=evidence,
                client_manager=SimpleClientManager(),
                strategy=strategy,
                server_model_manager=manager,
            )
            # This is only the coordinator's proxy-call thread pool.  It may
            # fan out logical Flower requests while Ray's native actor pool
            # remains bounded by the admitted Worker capacity below.
            server.set_max_workers(config.virtual_clients.total)
            max_concurrency = max(
                [
                    config.resources.effective_server_concurrency,
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
            from splitfleet.server.telemetry import LiveServerTelemetry
            servicer.telemetry = LiveServerTelemetry(device=config.server_device,
                                                     concurrency=max_concurrency)
            if policy is not None:
                policy.runtime_context_provider = RuntimeConditions(strategy)
                bind_telemetry = getattr(policy, "bind_server_telemetry", None)
                if bind_telemetry is not None:
                    bind_telemetry(servicer.telemetry)
            import asyncio

            manager.before_collect = lambda: asyncio.run_coroutine_threadsafe(
                servicer.reconcile(), service.loop
            ).result(timeout=300)
            app = make_server_app(server, rounds=config.rounds, service=service)

            resources = config.resources
            client_factory = ClientFactory(config=config, spec=spec, partitions=partitions)
            run_simulation(
                server_app=app,
                client_app=make_client_app(client_factory),
                num_supernodes=config.virtual_clients.total,
                backend_config={
                    "init_args": init_args,
                    "client_resources": {
                        "num_cpus": resources.client_cpus_per_worker or 1.0,
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
                    "worker_limit": resources.effective_worker_capacity,
                    "worker_target_count": resources.worker_target_count,
                    "capacity_observation": resources.capacity_observation,
                    "worker_count_policy": resources.worker_count_policy,
                    "requested_worker_count": resources.worker_target_count,
                    "native_worker_pool_size": actual,
                    "worker_capacity": resources.effective_worker_capacity,
                    "peak_suffix_replicas": manager._delegate.peak_live,
                    "per_client_suffix_models": independent_suffix_models,
                    "suffix_model_count": manager._delegate.peak_logical_model_count,
                    "logical_suffix_model_count": resources.logical_suffix_models,
                    "observed_suffix_model_count": (
                        manager._delegate.peak_logical_model_count
                        if independent_suffix_models else None
                    ),
                    "resident_suffix_model_peak": manager._delegate.peak_client_models,
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
                "worker_limit": resources.effective_worker_capacity,
                "worker_target_count": resources.worker_target_count,
                "capacity_observation": resources.capacity_observation,
                "worker_count_policy": resources.worker_count_policy,
                "requested_worker_count": resources.worker_target_count,
                "suffix_model_count": manager._delegate.peak_logical_model_count,
                "resident_suffix_model_peak": manager._delegate.peak_client_models,
                "logical_suffix_model_count": resources.logical_suffix_models,
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
                {
                    "status": "failed",
                    "reason": str(exc),
                    "capacity_observation": config.resources.capacity_observation,
                    "multi_node": "not_executed",
                },
            )
        if not (evidence.directory / "summary.json").exists():
            evidence.write(
                "summary.json",
                {
                    "status": "failed",
                    "reason": str(exc),
                    "capacity_observation": config.resources.capacity_observation,
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
