"""Offline real-batch resource profiling for every split candidate."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .config_utils import load_config, set_reproducible_seed, tensor_state_hash
from .logical_state import LogicalClientModelState
from .model_data import build_model, cifar10_datasets, make_loader
from .resource_emulator import ResourceEmulator
from .resource_monitor import ResourceMonitor, ServerJobPool
from .result_schema import ProfileRecord, ResultWriter
from .split_candidates import ExperimentSplitCandidate, discover_split_candidates
from .training_runtime import BatchMeasurement, LogicalClientRuntime


def _cycle(loader: Iterable[Any]):
    while True:
        yield from loader


def _mean(measurements: list[BatchMeasurement], name: str) -> float | None:
    values = [getattr(item, name) for item in measurements if getattr(item, name) is not None]
    return statistics.fmean(values) if values else None


def _profile_one(
    *,
    run_id: str,
    model_factory,
    initial_state: dict[str, torch.Tensor],
    sample_inputs: torch.Tensor,
    candidate: ExperimentSplitCandidate,
    batch_size: int,
    repetition: int,
    loader,
    warmup_steps: int,
    measured_steps: int,
    device_profile: str,
    network_profile: str,
    emulation_config: dict[str, Any],
    device: str,
    learning_rate: float,
    server_concurrency: int,
) -> ProfileRecord:
    with ResourceEmulator(emulation_config) as emulator:
        monitor = ResourceMonitor(device)
        runtime = LogicalClientRuntime(
            client_id=f"profile-{device_profile}",
            model_factory=model_factory,
            sample_inputs=sample_inputs,
            candidates={candidate.split_key: candidate},
            device=device,
            learning_rate=learning_rate,
            max_batch_size=batch_size,
        )
        state = LogicalClientModelState(
            client_id=runtime.client_id,
            full_state_dict={name: tensor.clone() for name, tensor in initial_state.items()},
        )
        handle, switch = runtime.activate(state, candidate.split_key)
        pool = ServerJobPool(server_concurrency)
        batches = _cycle(loader)
        loss_fn = nn.CrossEntropyLoss()
        for step in range(warmup_steps):
            inputs, targets = next(batches)
            runtime.train_batch(
                handle,
                inputs,
                targets,
                round_id=-(step + 1),
                loss_fn=loss_fn,
                server_pool=pool,
                network=emulator.link,
                energy=monitor.energy,
            )
        values = []
        for step in range(measured_steps):
            inputs, targets = next(batches)
            values.append(
                runtime.train_batch(
                    handle,
                    inputs,
                    targets,
                    round_id=step + 1,
                    loss_fn=loss_fn,
                    server_pool=pool,
                    network=emulator.link,
                    energy=monitor.energy,
                )
            )
        return ProfileRecord(
            run_id=run_id,
            device_profile=device_profile,
            network_profile=network_profile,
            split_key=candidate.split_key,
            batch_size=batch_size,
            repetition=repetition,
            client_forward_ms=_mean(values, "client_forward_ms"),
            client_backward_ms=_mean(values, "client_backward_ms"),
            server_forward_ms=_mean(values, "server_forward_ms"),
            server_backward_ms=_mean(values, "server_backward_ms"),
            boundary_forward_bytes=round(_mean(values, "boundary_forward_bytes") or 0),
            boundary_gradient_bytes=round(_mean(values, "boundary_gradient_bytes") or 0),
            client_peak_memory_mb=_mean(values, "client_peak_memory_mb"),
            server_peak_memory_mb=_mean(values, "server_peak_memory_mb"),
            client_energy_j=_mean(values, "client_energy_j"),
            server_energy_j=_mean(values, "server_energy_j"),
            network_upload_ms=_mean(values, "network_upload_ms"),
            network_download_ms=_mean(values, "network_download_ms"),
            server_queue_ms=_mean(values, "server_queue_ms"),
            end_to_end_batch_ms=_mean(values, "end_to_end_batch_ms"),
            success=True,
            failure_reason=None,
            emulation_mode=emulator.mode,
            runtime_prepare_ms=switch.runtime_prepare_ms,
            client_compute_score=monitor.compute_score(),
            measured_uplink_mbps=_mean(values, "measured_uplink_mbps"),
            measured_downlink_mbps=_mean(values, "measured_downlink_mbps"),
        )


def run_profile(config: dict[str, Any], run_id: str) -> Path:
    seed = int(config.get("seed", 1))
    set_reproducible_seed(seed)
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model_factory = lambda: build_model(
        str(config.get("model", "resnet18")),
        normalization=str(config.get("normalization", "groupnorm")),
    )
    base_model = model_factory().to(device)
    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
    batch_sizes = [int(value) for value in config.get("batch_sizes", [1, 2, 4, 8])]
    sample_batch = max(2, min(batch_sizes))
    sample_inputs = torch.zeros((sample_batch, 3, 32, 32), device=device)
    candidates = discover_split_candidates(base_model, sample_inputs)
    candidate_map = {item.split_key: item for item in candidates}
    selected_keys = config.get("split_keys") or list(candidate_map)
    train, _ = cifar10_datasets(
        config.get("data_root", "data"),
        download=bool(config.get("download", True)),
        max_train_samples=config.get("max_train_samples"),
    )
    output_root = Path(config.get("results_root", "results/resource_adaptive_splitfed"))
    device_profiles = config.get("device_profiles") or {"measured_host": {}}
    network_profiles = config.get("network_profiles") or {"measured_link": {}}
    emulation_modes = []
    for value in itertools.chain(device_profiles.values(), network_profiles.values()):
        if value:
            emulation_modes.append("controlled_in_process")
    writer = ResultWriter(
        output_root,
        run_id,
        config=config,
        method="resource_profile",
        seed=seed,
        emulation_mode="controlled_in_process" if emulation_modes else "none",
    )
    profile_dir = writer.path / "profiles"
    profile_dir.mkdir()
    record_path = profile_dir / "profile_records.jsonl"
    records: list[ProfileRecord] = []
    writer.write_json("split_candidates.json", [item.to_dict() for item in candidates])
    writer.write_json(
        "client_assignments.json",
        {"profile_dataset": "cifar10", "initial_model_hash": tensor_state_hash(initial_state)},
    )
    try:
        with record_path.open("w", encoding="utf-8") as stream:
            for device_name, device_control in device_profiles.items():
                for network_name, network_control in network_profiles.items():
                    controls = {**dict(device_control or {}), **dict(network_control or {})}
                    for batch_size in batch_sizes:
                        loader = make_loader(
                            train,
                            None,
                            batch_size=batch_size,
                            shuffle=False,
                            seed=seed,
                        )
                        for split_key in selected_keys:
                            candidate = candidate_map[str(split_key)]
                            for repetition in range(int(config.get("repetitions", 5))):
                                try:
                                    record = _profile_one(
                                        run_id=run_id,
                                        model_factory=model_factory,
                                        initial_state=initial_state,
                                        sample_inputs=sample_inputs,
                                        candidate=candidate,
                                        batch_size=batch_size,
                                        repetition=repetition,
                                        loader=loader,
                                        warmup_steps=int(config.get("warmup_steps", 5)),
                                        measured_steps=int(config.get("local_steps_per_profile", 20)),
                                        device_profile=str(device_name),
                                        network_profile=str(network_name),
                                        emulation_config=controls,
                                        device=device,
                                        learning_rate=float(config.get("learning_rate", 0.01)),
                                        server_concurrency=int(config.get("server_concurrency", 1)),
                                    )
                                except Exception as exc:
                                    warnings.warn(
                                        f"Profile failed for {device_name}/{network_name}/{split_key}/"
                                        f"batch={batch_size}: {exc}",
                                        RuntimeWarning,
                                    )
                                    record = ProfileRecord(
                                        run_id, str(device_name), str(network_name), str(split_key),
                                        batch_size, repetition,
                                        None, None, None, None, None, None, None, None,
                                        None, None, None, None, None, None,
                                        False, f"{type(exc).__name__}: {exc}",
                                        "controlled_in_process" if controls else "none",
                                    )
                                    writer.append(
                                        "failures.jsonl",
                                        {
                                            "run_id": run_id,
                                            "stage": "profile",
                                            "device_profile": device_name,
                                            "network_profile": network_name,
                                            "split_key": split_key,
                                            "batch_size": batch_size,
                                            "repetition": repetition,
                                            "failure_reason": record.failure_reason,
                                        },
                                    )
                                records.append(record)
                                stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
                                stream.flush()
        summary_path = profile_dir / "profile_summary.csv"
        _write_summary(records, summary_path)
        (writer.path / "summary.csv").write_bytes(summary_path.read_bytes())
        from .plot_results import plot_profile_figures

        plot_profile_figures(records, writer.path / "figures")
        from .validate_results import validate_run

        writer.write_json(
            "validation_report.json",
            validate_run(writer.path, write_report=False),
        )
    finally:
        writer.close()
    return writer.path


def _write_summary(records: list[ProfileRecord], path: Path) -> None:
    successful = [record for record in records if record.success]
    groups: dict[tuple[str, str, str, int], list[ProfileRecord]] = {}
    for record in successful:
        key = (record.device_profile, record.network_profile, record.split_key, record.batch_size)
        groups.setdefault(key, []).append(record)
    metric_names = [
        "client_forward_ms", "client_backward_ms", "server_forward_ms", "server_backward_ms",
        "boundary_forward_bytes", "boundary_gradient_bytes", "client_peak_memory_mb",
        "server_peak_memory_mb", "network_upload_ms", "network_download_ms", "server_queue_ms",
        "end_to_end_batch_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["device_profile", "network_profile", "split_key", "batch_size", "valid_repetitions"] + metric_names,
        )
        writer.writeheader()
        for key, values in sorted(groups.items()):
            row = dict(zip(("device_profile", "network_profile", "split_key", "batch_size"), key))
            row["valid_repetitions"] = len(values)
            for metric in metric_names:
                data = [getattr(item, metric) for item in values if getattr(item, metric) is not None]
                row[metric] = statistics.fmean(data) if data else None
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    output = run_profile(load_config(args.config), args.run_id)
    print(output)


if __name__ == "__main__":
    main()
