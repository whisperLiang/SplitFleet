"""Federated experiment runner for RA-SplitFed and all required baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .config_utils import load_config, set_reproducible_seed, stable_hash, tensor_state_hash
from .logical_state import LogicalClientModelState, aggregate_named_states
from .metrics import classification_metrics, descriptive_stats, jain_fairness_index
from .model_data import (
    build_model,
    cifar10_datasets,
    dataset_targets,
    dirichlet_partition,
    make_loader,
    partition_manifest,
    stratified_client_holdout,
)
from .resource_emulator import ResourceEmulator
from .resource_monitor import ClientResourceState, ResourceMonitor, ServerJobPool
from .result_schema import ClientMetricRecord, ResultWriter, SplitDecisionRecord
from .split_candidates import ExperimentSplitCandidate, discover_split_candidates
from .split_cost_model import SplitCostModel, SplitCostPrediction
from .split_scheduler import ResourceAdaptiveSplitScheduler, select_edge_local
from .training_runtime import BatchMeasurement, LogicalClientRuntime, SwitchMeasurement


METHODS = {
    "fedavg_full_local",
    "fixed_early",
    "fixed_middle",
    "fixed_late",
    "best_global_fixed",
    "static_heterogeneous",
    "compute_only_adaptive",
    "edge_local_adaptive",
    "resource_adaptive_splitfed",
    "oracle",
    "fedavg_full_local_with_timeout",
    "fedavg_strong_clients_only",
    "no_network",
    "no_server_state",
    "no_memory_constraint",
    "no_hysteresis",
    "no_switch_cost",
    "offline_profile_only",
    "full_resource_adaptive",
}


@dataclass
class ClientRoundResult:
    client_id: str
    split_key: str
    state: dict[str, torch.Tensor] | None
    num_examples: int
    metrics: ClientMetricRecord
    switch: SwitchMeasurement
    measured_uplink_mbps: float | None
    measured_downlink_mbps: float | None


def _load_profile_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Profile records not found at {source}. Run resource_profile first; "
            "the runner never fabricates a cost table."
        )
    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid profile JSONL at line {line_number}: {exc}") from exc
    if not any(record.get("success") for record in records):
        raise ValueError("The selected profile contains no successful measured records.")
    return records


def _profile_assignment(config: Mapping[str, Any], num_clients: int) -> dict[str, str]:
    configured = config.get("client_profiles") or {"weak": num_clients}
    total = sum(int(count) for count in configured.values())
    if total != num_clients:
        raise ValueError(
            f"client_profiles counts must sum to num_clients: got {total} for "
            f"num_clients={num_clients} from {dict(configured)!r}."
        )
    result: dict[str, str] = {}
    cursor = 0
    for profile, count in configured.items():
        for _ in range(int(count)):
            result[str(cursor)] = str(profile)
            cursor += 1
    return result


def _batches_per_round(
    num_examples: int,
    *,
    batch_size: int,
    local_epochs: int,
    max_batches_per_epoch: int | None,
) -> int:
    """Count the batches one client actually executes in a round."""

    per_epoch = math.ceil(num_examples / max(1, int(batch_size)))
    if max_batches_per_epoch is not None:
        per_epoch = min(per_epoch, int(max_batches_per_epoch))
    return max(1, per_epoch * max(1, int(local_epochs)))


def _sampling_plan(config: Mapping[str, Any], num_clients: int, seed: int) -> list[list[str]]:
    rounds = int(config.get("rounds", 100))
    fraction = float(config.get("client_fraction", 1.0))
    count = max(1, min(num_clients, math.ceil(num_clients * fraction)))
    rng = np.random.default_rng(seed + 71_923)
    clients = np.arange(num_clients)
    return [
        [str(value) for value in sorted(rng.choice(clients, size=count, replace=False).tolist())]
        for _ in range(rounds)
    ]


def _resource_phase(config: Mapping[str, Any], round_id: int) -> dict[str, Any]:
    for phase in config.get("resource_phases", []) or []:
        first, last = phase.get("rounds", (1, int(config.get("rounds", 1))))
        if int(first) <= round_id <= int(last):
            return dict(phase)
    return {"name": "static"}


def _affected_clients(phase: Mapping[str, Any], clients: Sequence[str], seed: int) -> set[str]:
    ratio = float(phase.get("affected_client_ratio", 0.0) or 0.0)
    if ratio <= 0:
        return set()
    count = max(1, min(len(clients), round(len(clients) * ratio)))
    rng = np.random.default_rng(seed + int(stable_hash(phase.get("name", ""))[:8], 16))
    return set(str(value) for value in rng.choice(list(clients), size=count, replace=False))


def _controls_for_client(
    config: Mapping[str, Any], profile: str, phase: Mapping[str, Any], affected: bool
) -> dict[str, Any]:
    controls = dict((config.get("device_profile_controls") or {}).get(profile, {}) or {})
    network_name = str((config.get("client_network_profiles") or {}).get(profile, "default"))
    controls.update(dict((config.get("network_profiles") or {}).get(network_name, {}) or {}))
    if affected:
        for key in ("cpu_background_load", "uplink_mbps", "downlink_mbps", "rtt_ms"):
            if phase.get(key) is not None:
                controls[key] = phase[key]
    return controls


def _prediction_inputs(
    cost_model: SplitCostModel,
    clients: Sequence[str],
    resources: Mapping[str, ClientResourceState],
    server_state: Any,
    candidates: Mapping[str, ExperimentSplitCandidate],
    batch_size: int,
    current_splits: Mapping[str, str],
    method: str,
    batches_per_round: Mapping[str, int],
) -> dict[str, dict[str, SplitCostPrediction]]:
    result = {}
    for client_id in clients:
        resource = asdict(resources[client_id])
        resource["batch_size"] = batch_size
        resource["batches_per_round"] = int(batches_per_round.get(client_id, 1))
        resource["current_split_key"] = current_splits.get(client_id, "full_local")
        if method == "no_network":
            resource["uplink_mbps"] = None
            resource["downlink_mbps"] = None
            resource["rtt_ms"] = 0.0
        per_client = {}
        for key, candidate in candidates.items():
            candidate_data = candidate.to_dict()
            if method == "no_switch_cost":
                candidate_data["predicted_switch_ms"] = 0.0
            prediction = cost_model.predict(resource, server_state, candidate_data)
            if method == "no_network":
                prediction.predicted_round_ms -= prediction.predicted_network_ms
                prediction.predicted_network_ms = 0.0
            if method == "compute_only_adaptive":
                prediction.predicted_round_ms = prediction.predicted_client_compute_ms
                prediction.predicted_network_ms = 0.0
                prediction.predicted_server_compute_ms = 0.0
                prediction.predicted_server_queue_ms = 0.0
            per_client[key] = prediction
        result[client_id] = per_client
    return result


def _fixed_key(name: str) -> str:
    return {"fixed_early": "stem", "fixed_middle": "layer2", "fixed_late": "layer4"}[name]


def _select_splits(
    method: str,
    clients: Sequence[str],
    profiles: Mapping[str, str],
    predictions: Mapping[str, Mapping[str, SplitCostPrediction]],
    scheduler: ResourceAdaptiveSplitScheduler,
    resources: Mapping[str, ClientResourceState],
    server_state: Any,
    round_id: int,
    best_global_fixed: str,
    oracle_choices: Mapping[str, str],
) -> dict[str, str]:
    if method in {"fedavg_full_local", "fedavg_full_local_with_timeout"}:
        return {client_id: "full_local" for client_id in clients}
    if method in {"fixed_early", "fixed_middle", "fixed_late"}:
        return {client_id: _fixed_key(method) for client_id in clients}
    if method == "best_global_fixed":
        return {client_id: best_global_fixed for client_id in clients}
    if method == "static_heterogeneous":
        mapping = {"weak": "stem", "medium": "layer2", "strong": "layer4"}
        return {client_id: mapping.get(profiles[client_id], "full_local") for client_id in clients}
    if method == "fedavg_strong_clients_only":
        return {client_id: "full_local" for client_id in clients if profiles[client_id] == "strong"}
    if method == "oracle":
        return {client_id: oracle_choices[client_id] for client_id in clients}
    if method in {"edge_local_adaptive", "compute_only_adaptive"}:
        return select_edge_local(predictions)
    return scheduler.select_splits(
        [{"client_id": client_id} for client_id in clients],
        resources,
        server_state,
        predictions,
        round_id=round_id,
    )


def _best_profile_split(records: Sequence[Mapping[str, Any]]) -> str:
    grouped: dict[str, list[float]] = {}
    for record in records:
        if record.get("success") and record.get("end_to_end_batch_ms") is not None:
            grouped.setdefault(str(record["split_key"]), []).append(float(record["end_to_end_batch_ms"]))
    if not grouped:
        raise ValueError("best_global_fixed calibration has no successful measured splits.")
    return min(grouped, key=lambda key: statistics.fmean(grouped[key]))


def _average_batches(values: list[BatchMeasurement]) -> dict[str, float | int | None]:
    if not values:
        raise ValueError("A client must execute at least one real training batch.")
    total_examples = sum(value.num_examples for value in values)

    def weighted(name: str):
        present = [(getattr(value, name), value.num_examples) for value in values if getattr(value, name) is not None]
        if not present:
            return None
        return sum(float(value) * count for value, count in present) / sum(count for _, count in present)

    result = {name: weighted(name) for name in BatchMeasurement.__dataclass_fields__ if name not in {"num_examples"}}
    result["num_examples"] = total_examples
    return result


def _train_client(
    *,
    run_id: str,
    method: str,
    seed: int,
    round_id: int,
    phase_name: str,
    client_id: str,
    split_key: str,
    runtime: LogicalClientRuntime,
    global_state: Mapping[str, torch.Tensor],
    train_dataset,
    indices: Sequence[int],
    batch_size: int,
    local_epochs: int,
    max_batches: int | None,
    server_pool: ServerJobPool,
    controls: Mapping[str, Any],
    deadline_ns: int | None = None,
) -> ClientRoundResult:
    _require_before_deadline(deadline_ns)
    started = time.perf_counter_ns()
    state = LogicalClientModelState(
        client_id=client_id,
        full_state_dict={name: tensor.detach().cpu().clone() for name, tensor in global_state.items()},
        split_key=runtime.current_split_key,
    )
    with ResourceEmulator(dict(controls)) as emulator:
        monitor = ResourceMonitor(runtime.device)
        handle, switch = runtime.activate(state, split_key)
        _require_before_deadline(deadline_ns)
        batches: list[BatchMeasurement] = []
        for epoch in range(local_epochs):
            loader = make_loader(
                train_dataset,
                indices,
                batch_size=batch_size,
                shuffle=True,
                seed=seed * 1_000_003 + round_id * 10_007 + int(client_id) * 101 + epoch,
            )
            for batch_index, (inputs, targets) in enumerate(loader):
                _require_before_deadline(deadline_ns)
                if max_batches is not None and batch_index >= max_batches:
                    break
                batches.append(
                    runtime.train_batch(
                        handle,
                        inputs,
                        targets,
                        round_id=round_id,
                        loss_fn=nn.CrossEntropyLoss(),
                        server_pool=server_pool,
                        network=emulator.link,
                        energy=monitor.energy,
                        deadline_ns=deadline_ns,
                    )
                )
        _require_before_deadline(deadline_ns)
        memory_limit = controls.get("memory_limit_mb")
        measured_peaks = [
            float(value.client_peak_memory_mb)
            for value in batches
            if value.client_peak_memory_mb is not None
        ]
        if (
            memory_limit is not None
            and measured_peaks
            and max(measured_peaks) > float(memory_limit)
        ):
            raise MemoryError(
                "client out of memory: measured peak "
                f"{max(measured_peaks):.2f} MiB exceeds limit {float(memory_limit):.2f} MiB"
            )
        averaged = _average_batches(batches)
        state.capture_model(runtime.model)
        completion_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        metric = ClientMetricRecord(
            run_id=run_id,
            method=method,
            seed=seed,
            round_id=round_id,
            client_id=client_id,
            split_key=split_key,
            resource_phase=phase_name,
            num_examples=int(averaged["num_examples"]),
            completion_ms=completion_ms,
            client_forward_ms=float(averaged["client_forward_ms"] or 0.0),
            client_backward_ms=float(averaged["client_backward_ms"] or 0.0),
            server_forward_ms=float(averaged["server_forward_ms"] or 0.0),
            server_backward_ms=float(averaged["server_backward_ms"] or 0.0),
            network_upload_ms=float(averaged["network_upload_ms"] or 0.0),
            network_download_ms=float(averaged["network_download_ms"] or 0.0),
            server_queue_ms=float(averaged["server_queue_ms"] or 0.0),
            boundary_forward_bytes=round(float(averaged["boundary_forward_bytes"] or 0.0)),
            boundary_gradient_bytes=round(float(averaged["boundary_gradient_bytes"] or 0.0)),
            client_peak_memory_mb=averaged["client_peak_memory_mb"],
            server_peak_memory_mb=averaged["server_peak_memory_mb"],
            client_energy_j=averaged["client_energy_j"],
            server_energy_j=averaged["server_energy_j"],
            loss=float(averaged["loss"]),
            success=True,
            overlapping_host_load_clients=int(emulator.overlapping_host_load_clients),
        )
        return ClientRoundResult(
            client_id=client_id,
            split_key=split_key,
            state=state.full_state_dict,
            num_examples=metric.num_examples,
            metrics=metric,
            switch=switch,
            measured_uplink_mbps=averaged["measured_uplink_mbps"],
            measured_downlink_mbps=averaged["measured_downlink_mbps"],
        )


def _evaluate(model: torch.nn.Module, dataset, *, batch_size: int, device: str) -> dict[str, Any]:
    model.eval()
    targets_all: list[int] = []
    predictions: list[int] = []
    loader = make_loader(dataset, None, batch_size=batch_size, shuffle=False, seed=0)
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(device))
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            targets_all.extend(targets.tolist())
    return classification_metrics(targets_all, predictions, 10)


def _oracle_probe(
    *,
    client_id: str,
    model_factory,
    candidates: Mapping[str, ExperimentSplitCandidate],
    global_state: Mapping[str, torch.Tensor],
    sample_inputs: torch.Tensor,
    batch,
    device: str,
    learning_rate: float,
    client_controls: Mapping[str, Any],
    server_controls: Mapping[str, Any],
    server_concurrency: int,
) -> tuple[str, dict[str, float]]:
    measured: dict[str, float] = {}
    inputs, targets = batch
    for split_key in candidates:
        probe = LogicalClientRuntime(
            f"oracle-{client_id}-{split_key}", model_factory, sample_inputs, candidates,
            device=device, learning_rate=learning_rate, max_batch_size=int(inputs.shape[0]),
        )
        state = LogicalClientModelState(
            probe.client_id,
            {name: tensor.detach().cpu().clone() for name, tensor in global_state.items()},
        )
        with ResourceEmulator(dict(server_controls)), ResourceEmulator(
            dict(client_controls)
        ) as emulator:
            started = time.perf_counter_ns()
            handle, _ = probe.activate(state, split_key)
            measurement = probe.train_batch(
                handle,
                inputs,
                targets,
                round_id=0,
                loss_fn=nn.CrossEntropyLoss(),
                server_pool=ServerJobPool(server_concurrency),
                network=emulator.link,
                energy=ResourceMonitor(device).energy,
            )
            memory_limit = client_controls.get("memory_limit_mb")
            if (
                memory_limit is not None
                and measurement.client_peak_memory_mb is not None
                and measurement.client_peak_memory_mb > float(memory_limit)
            ):
                continue
            measured[split_key] = (time.perf_counter_ns() - started) / 1_000_000.0
    if not measured:
        raise RuntimeError(f"Oracle found no executable split for client {client_id!r}.")
    return min(measured, key=measured.get), measured


def _require_before_deadline(deadline_ns: int | None) -> None:
    if deadline_ns is not None and time.perf_counter_ns() >= int(deadline_ns):
        raise TimeoutError("client round deadline exceeded")


def _predicted_memory_failure(
    prediction: SplitCostPrediction,
    controls: Mapping[str, Any],
) -> str | None:
    memory_limit = controls.get("memory_limit_mb")
    predicted_peak = float(prediction.predicted_client_peak_memory_mb)
    if (
        memory_limit is None
        or not math.isfinite(predicted_peak)
        or predicted_peak <= float(memory_limit)
    ):
        return None
    return (
        "MemoryError: client out of memory: predicted peak "
        f"{predicted_peak:.2f} MiB exceeds limit {float(memory_limit):.2f} MiB"
    )


def run_experiment(config: dict[str, Any], method: str, seed: int, run_id: str) -> Path:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(METHODS)}")
    set_reproducible_seed(seed)
    if config.get("optimizer", {}).get("name", "sgd").lower() != "sgd" or float(config.get("optimizer", {}).get("momentum", 0.0)) != 0.0:
        raise ValueError("RA-SplitFed currently requires SGD with momentum=0; optimizer state is never silently discarded.")
    torch.set_num_threads(int(config.get("torch_num_threads", max(1, torch.get_num_threads()))))
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model_factory = lambda: build_model(
        str(config.get("model", "resnet18")),
        normalization=str(config.get("normalization", "groupnorm")),
    )
    global_model = model_factory().to(device)
    global_state = {name: tensor.detach().cpu().clone() for name, tensor in global_model.state_dict().items()}
    initial_model_hash = tensor_state_hash(global_state)
    sample_inputs = torch.zeros((2, 3, 32, 32), device=device)
    candidates_list = discover_split_candidates(global_model, sample_inputs)
    candidates = {item.split_key: item for item in candidates_list}
    train_dataset, test_dataset = cifar10_datasets(
        config.get("data_root", "data"),
        download=bool(config.get("download", True)),
        max_train_samples=config.get("max_train_samples"),
        max_test_samples=config.get("max_test_samples"),
    )
    num_clients = int(config.get("num_clients", 20))
    profiles = _profile_assignment(config, num_clients)
    partition_config = config.get("partition", {}) or {}
    fairness = config.get("fairness_partition", {}) or {}
    targets = dataset_targets(train_dataset)
    complete_assignments = dirichlet_partition(
        targets,
        num_clients,
        alpha=float(partition_config.get("alpha", 0.5)),
        seed=seed,
        client_profiles=profiles,
        rare_classes=fairness.get("rare_classes", []),
        weak_rare_multiplier=float(fairness.get("weak_rare_multiplier", 1.0)),
    )
    assignments, client_holdouts = stratified_client_holdout(
        complete_assignments,
        targets,
        fraction=float(config.get("client_holdout_fraction", 0.10)),
        seed=seed,
    )
    manifest = partition_manifest(assignments, targets, client_holdouts)
    sampling = _sampling_plan(config, num_clients, seed)
    profile_records = _load_profile_records(config["profile_records"])
    cost_model = SplitCostModel(ema_alpha=float(config.get("cost_model_ema_alpha", 0.25))).fit(profile_records)
    best_fixed = _best_profile_split(profile_records)
    scheduler_config = dict(config.get("scheduler", {}) or {})
    scheduler = ResourceAdaptiveSplitScheduler(
        min_relative_improvement_to_switch=float(scheduler_config.get("min_relative_improvement_to_switch", 0.10)),
        min_rounds_between_switches=int(scheduler_config.get("min_rounds_between_switches", 3)),
        max_switches_per_round=scheduler_config.get("max_switches_per_round"),
        use_memory_constraint=method != "no_memory_constraint",
        use_hysteresis=method != "no_hysteresis",
        use_server_state=method != "no_server_state",
    )
    emulated = bool(
        config.get("resource_phases")
        or any((config.get("device_profile_controls") or {}).values())
        or any((config.get("network_profiles") or {}).values())
    )
    writer = ResultWriter(
        config.get("results_root", "results/resource_adaptive_splitfed"),
        run_id,
        config=config,
        method=method,
        seed=seed,
        emulation_mode="controlled_in_process" if emulated else "none",
    )
    writer.write_json("split_candidates.json", [item.to_dict() for item in candidates_list])
    assignment_payload = {
        **manifest,
        "client_profiles": profiles,
        "initial_model_hash": initial_model_hash,
        "client_sampling_order": sampling,
        "client_sampling_hash": stable_hash(sampling),
    }
    writer.write_json("client_assignments.json", assignment_payload)
    runtimes = {
        str(index): LogicalClientRuntime(
            str(index), model_factory, sample_inputs, candidates,
            device=device,
            learning_rate=float(config.get("optimizer", {}).get("lr", 0.01)),
            max_batch_size=int(config.get("batch_size", 32)),
        )
        for index in range(num_clients)
    }
    previous_completion: dict[str, float] = {}
    oracle_by_phase: dict[str, dict[str, str]] = {}
    oracle_measurements: dict[str, dict[str, dict[str, float]]] = {}
    round_rows: list[dict[str, Any]] = []
    total_wall_s = 0.0
    target_accuracy = config.get("target_accuracy")
    wall_to_target = None
    try:
        for round_id, sampled in enumerate(sampling, 1):
            round_started = time.perf_counter_ns()
            phase = _resource_phase(config, round_id)
            phase_name = str(phase.get("name", "static"))
            affected = _affected_clients(phase, sampled, seed)
            selected_for_method = sampled
            if method == "fedavg_strong_clients_only":
                selected_for_method = [client_id for client_id in sampled if profiles[client_id] == "strong"]
            server_concurrency = int(phase.get("server_concurrency", config.get("server_concurrency", 4)))
            pool = ServerJobPool(server_concurrency)
            monitor = ResourceMonitor(device)
            server_state = monitor.server_state(pool)
            resources: dict[str, ClientResourceState] = {}
            controls_by_client = {}
            probes = {}
            for client_id in selected_for_method:
                controls = _controls_for_client(config, profiles[client_id], phase, client_id in affected)
                controls_by_client[client_id] = controls
                with ResourceEmulator(controls) as emulator:
                    up, down, rtt = emulator.link.probe(
                        int(config.get("network_probe_bytes", 64 * 1024))
                    )
                    probes[client_id] = (up, down, rtt)
                    state = monitor.client_state(
                        client_id,
                        profiles[client_id],
                        uplink_mbps=up,
                        downlink_mbps=down,
                        rtt_ms=rtt,
                    )
                limit = controls.get("memory_limit_mb")
                if limit is not None:
                    state.client_available_memory_mb = min(state.client_available_memory_mb, float(limit))
                resources[client_id] = state
                writer.append(
                    "resource_metrics.jsonl",
                    {
                        "run_id": run_id, "method": method, "seed": seed, "round_id": round_id,
                        "client_id": client_id, "split_key": runtimes[client_id].current_split_key,
                        "resource_phase": phase_name, **asdict(state), **asdict(server_state),
                    },
                )
            current = {client_id: runtimes[client_id].current_split_key for client_id in selected_for_method}
            batches_per_round = {
                client_id: _batches_per_round(
                    len(assignments[client_id]),
                    batch_size=int(config.get("batch_size", 32)),
                    local_epochs=int(config.get("local_epochs", 1)),
                    max_batches_per_epoch=config.get("max_batches_per_epoch"),
                )
                for client_id in selected_for_method
            }
            predictions = _prediction_inputs(
                cost_model, selected_for_method, resources, server_state, candidates,
                int(config.get("batch_size", 32)), current, method, batches_per_round,
            )
            server_load = {}
            capacity_ratio = phase.get("server_compute_capacity_ratio")
            if capacity_ratio is not None and float(capacity_ratio) < 1.0:
                server_load = {
                    "cpu_background_load": max(0.01, 1.0 - float(capacity_ratio)),
                    "cpu_load_workers": 1,
                }
            if method == "oracle":
                oracle_by_phase.setdefault(phase_name, {})
                oracle_measurements.setdefault(phase_name, {})
                for client_id in selected_for_method:
                    if client_id in oracle_by_phase[phase_name]:
                        continue
                    loader = make_loader(
                        train_dataset, assignments[client_id], batch_size=int(config.get("batch_size", 32)),
                        shuffle=False, seed=seed,
                    )
                    choice, measurements = _oracle_probe(
                        client_id=client_id, model_factory=model_factory, candidates=candidates,
                        global_state=global_state, sample_inputs=sample_inputs,
                        batch=next(iter(loader)), device=device,
                        learning_rate=float(config.get("optimizer", {}).get("lr", 0.01)),
                        client_controls=controls_by_client[client_id],
                        server_controls=server_load,
                        server_concurrency=server_concurrency,
                    )
                    oracle_by_phase[phase_name][client_id] = choice
                    oracle_measurements[phase_name][client_id] = measurements
            oracle_choices = oracle_by_phase.get(phase_name, {})
            splits = _select_splits(
                method, selected_for_method, profiles, predictions, scheduler, resources,
                server_state, round_id, best_fixed, oracle_choices,
            )
            worker_count = min(len(splits), int(config.get("client_worker_threads", len(splits) or 1)))
            futures = {}
            failed_clients: list[tuple[str, str]] = []
            timed_out_clients: set[str] = set()

            def record_failure(client_id: str, split_key: str, reason: str) -> None:
                failed_clients.append((client_id, reason))
                writer.append(
                    "failures.jsonl",
                    {
                        "run_id": run_id,
                        "method": method,
                        "seed": seed,
                        "round_id": round_id,
                        "client_id": client_id,
                        "split_key": split_key,
                        "resource_phase": phase_name,
                        "failure_reason": reason,
                    },
                )

            timeout_ms = (
                config.get("client_timeout_ms")
                if method == "fedavg_full_local_with_timeout"
                else None
            )
            deadline_ns = (
                time.perf_counter_ns() + int(float(timeout_ms) * 1_000_000.0)
                if timeout_ms is not None
                else None
            )
            with ResourceEmulator(server_load):
                with ThreadPoolExecutor(max_workers=max(1, worker_count), thread_name_prefix="ra-splitfed") as executor:
                    for client_id, split_key in splits.items():
                        reason = _predicted_memory_failure(
                            predictions[client_id][split_key],
                            controls_by_client[client_id],
                        )
                        if reason is not None:
                            record_failure(client_id, split_key, reason)
                            continue
                        future = executor.submit(
                            _train_client,
                            run_id=run_id, method=method, seed=seed, round_id=round_id,
                            phase_name=phase_name, client_id=client_id, split_key=split_key,
                            runtime=runtimes[client_id], global_state=global_state,
                            train_dataset=train_dataset, indices=assignments[client_id],
                            batch_size=int(config.get("batch_size", 32)),
                            local_epochs=int(config.get("local_epochs", 1)),
                            max_batches=config.get("max_batches_per_epoch"), server_pool=pool,
                            controls=controls_by_client[client_id],
                            deadline_ns=deadline_ns,
                        )
                        futures[future] = (client_id, split_key)
                    results: list[ClientRoundResult] = []
                    for future in as_completed(futures):
                        client_id, split_key = futures[future]
                        try:
                            results.append(future.result())
                        except Exception as exc:
                            if isinstance(exc, TimeoutError):
                                reason = "timeout"
                                timed_out_clients.add(client_id)
                            else:
                                reason = f"{type(exc).__name__}: {exc}"
                            record_failure(client_id, split_key, reason)
                            warnings.warn(f"Client {client_id} failed in round {round_id}: {exc}", RuntimeWarning)
            accepted = list(results)
            for result in results:
                writer.append("client_metrics.jsonl", result.metrics)
                prediction = predictions[result.client_id][result.split_key]
                old_completion = previous_completion.get(result.client_id)
                actual_improvement = None
                if old_completion is not None and result.switch.old_split_key != result.switch.new_split_key:
                    actual_improvement = (old_completion - result.metrics.completion_ms) / max(old_completion, 1e-9)
                scheduler_decision = next(
                    (item for item in scheduler.last_decisions if item.client_id == result.client_id), None
                )
                decision = SplitDecisionRecord(
                    run_id, method, seed, round_id, result.client_id, result.split_key, phase_name,
                    result.switch.old_split_key, result.switch.new_split_key,
                    scheduler_decision.predicted_improvement_ratio if scheduler_decision else 0.0,
                    actual_improvement,
                    scheduler_decision.switch_reason if scheduler_decision else "baseline_policy",
                    result.switch.runtime_prepare_ms, result.switch.model_transfer_ms,
                    result.switch.optimizer_state_transfer_ms, result.switch.total_switch_ms,
                )
                writer.append("split_decisions.jsonl", decision)
                previous_completion[result.client_id] = result.metrics.completion_ms
                if method != "offline_profile_only":
                    cost_model.update_online(
                        {
                            **asdict(result.metrics),
                            "predicted_completion_ms": prediction.predicted_round_ms,
                        }
                    )
            if not accepted:
                raise RuntimeError(f"Round {round_id} has no successful client updates.")
            global_state = aggregate_named_states(
                [result.state for result in accepted if result.state is not None],
                [result.num_examples for result in accepted if result.state is not None],
            )
            global_model.load_state_dict(global_state, strict=True)
            evaluation = _evaluate(
                global_model, test_dataset,
                batch_size=int(config.get("evaluation_batch_size", 256)), device=device,
            )
            elapsed_ms = (time.perf_counter_ns() - round_started) / 1_000_000.0
            total_wall_s += elapsed_ms / 1000.0
            if target_accuracy is not None and wall_to_target is None and evaluation["test_accuracy"] >= float(target_accuracy):
                wall_to_target = total_wall_s
            completions = [result.metrics.completion_ms for result in results]
            successful_ids = {result.client_id for result in accepted}
            weak_ids = [cid for cid in selected_for_method if profiles[cid] == "weak"]
            split_switch_count = sum(
                result.switch.old_split_key != result.switch.new_split_key for result in results
            )
            prediction_errors = [
                abs(predictions[result.client_id][result.split_key].predicted_round_ms - result.metrics.completion_ms)
                / max(result.metrics.completion_ms, 1e-9)
                for result in results
                if math.isfinite(predictions[result.client_id][result.split_key].predicted_round_ms)
            ]
            oracle_regrets = []
            phase_oracle = oracle_measurements.get(phase_name, {})
            for result in results:
                if result.client_id in phase_oracle:
                    best = min(phase_oracle[result.client_id].values())
                    oracle_regrets.append((result.metrics.completion_ms - best) / max(best, 1e-9))
            final_client_accuracies = None
            if round_id == int(config.get("rounds", 100)):
                final_client_accuracies = []
                for client_id in sorted(client_holdouts, key=int):
                    if not client_holdouts[client_id]:
                        continue
                    client_eval = _evaluate(
                        global_model,
                        torch.utils.data.Subset(train_dataset, client_holdouts[client_id]),
                        batch_size=int(config.get("evaluation_batch_size", 256)),
                        device=device,
                    )
                    final_client_accuracies.append(float(client_eval["test_accuracy"]))
            oom_clients = [
                client_id for client_id, reason in failed_clients
                if "out of memory" in reason.lower() or "outofmemory" in reason.lower()
            ]
            row = {
                "run_id": run_id,
                "method": method,
                "seed": seed,
                "round_id": round_id,
                "resource_phase": phase_name,
                "round_time_ms": elapsed_ms,
                "round_p95_time_ms": descriptive_stats(completions)["p95"],
                "straggler_completion_ms": max(completions),
                "wall_clock_to_target_accuracy_s": wall_to_target,
                **evaluation,
                "global_accuracy": evaluation["test_accuracy"],
                "client_peak_memory_mb": max(
                    (result.metrics.client_peak_memory_mb for result in results if result.metrics.client_peak_memory_mb is not None),
                    default=None,
                ),
                "server_peak_memory_mb": max(
                    (result.metrics.server_peak_memory_mb for result in results if result.metrics.server_peak_memory_mb is not None),
                    default=None,
                ),
                "server_queue_ms": sum(result.metrics.server_queue_ms for result in results),
                "client_idle_wait_ms": sum(max(completions) - value for value in completions),
                "oom_count": len(oom_clients),
                "timeout_count": len(timed_out_clients),
                "timeout_ratio": len(timed_out_clients) / max(len(selected_for_method), 1),
                "successful_client_ratio": len(accepted) / max(len(selected_for_method), 1),
                "successful_participation_ratio": len(accepted) / max(len(sampled), 1),
                "weak_client_completion_ratio": sum(cid in successful_ids for cid in weak_ids) / max(len(weak_ids), 1),
                "max_overlapping_host_load_clients": max(
                    (result.metrics.overlapping_host_load_clients for result in results),
                    default=0,
                ),
                "split_switch_count": split_switch_count,
                "early_cut_ratio": sum(
                    result.split_key in {"stem", "maxpool"} for result in results
                ) / max(len(results), 1),
                "split_switch_cost_ms": sum(result.switch.total_switch_ms for result in results),
                "cost_prediction_error": statistics.fmean(prediction_errors) if prediction_errors else None,
                "oracle_regret": statistics.fmean(oracle_regrets) if oracle_regrets else None,
                "samples_lost_due_to_timeout": sum(
                    len(assignments[client_id]) for client_id in timed_out_clients
                ),
                "samples_lost_due_to_oom": sum(len(assignments[client_id]) for client_id in oom_clients),
                "jain_fairness_index": jain_fairness_index([1.0 if cid in successful_ids else 0.0 for cid in sampled]),
                "worst_client_accuracy": min(final_client_accuracies) if final_client_accuracies else None,
                "worst_client_accuracy_source": "cifar10_train_client_holdout",
                "server_gpu_utilization": server_state.server_gpu_utilization,
                "server_gpu_memory_mb": (
                    max(
                        (
                            result.metrics.server_peak_memory_mb
                            for result in results
                            if result.metrics.server_peak_memory_mb is not None
                        ),
                        default=None,
                    )
                    if torch.cuda.is_available()
                    else None
                ),
                "initial_model_hash": initial_model_hash,
                "partition_hash": manifest["partition_hash"],
                "client_sampling_hash": stable_hash(sampling),
                "final_accuracy_source": "cifar10_test_set",
            }
            writer.append("round_metrics.jsonl", row)
            round_rows.append(row)
        _write_round_summary(round_rows, writer.path / "summary.csv")
        from .validate_results import validate_run

        report = validate_run(writer.path, write_report=False)
        writer.write_json("validation_report.json", report)
    finally:
        writer.close()
    return writer.path


def _write_round_summary(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    metrics = [
        "round_time_ms", "round_p95_time_ms", "straggler_completion_ms", "test_accuracy",
        "macro_f1", "client_peak_memory_mb", "server_peak_memory_mb", "server_queue_ms",
        "client_idle_wait_ms", "oom_count", "timeout_count", "successful_client_ratio",
        "split_switch_count", "split_switch_cost_ms", "cost_prediction_error", "oracle_regret",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "mean", "std", "median", "p95", "confidence_interval_95", "number_of_valid_runs"],
        )
        writer.writeheader()
        for metric in metrics:
            stats = descriptive_stats(row[metric] for row in rows if row.get(metric) is not None)
            stats["confidence_interval_95"] = json.dumps(stats["confidence_interval_95"])
            writer.writerow({"metric": metric, **stats})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    output = run_experiment(load_config(args.config), args.method, args.seed, args.run_id)
    print(output)


if __name__ == "__main__":
    main()
