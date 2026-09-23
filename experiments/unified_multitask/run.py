"""Sequential local federated benchmark for four real or fixture task datasets.

This is a controlled in-process correctness/convergence runner.  Its timing
includes TorchLens capture and local wire encode/decode and is not a physical
network or heterogeneous-device measurement.  Physical system conclusions
must use the separate multi-host RA-SplitFed harness.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from itertools import chain
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.resource_adaptive_splitfed.config_utils import git_commit, stable_hash, tensor_state_hash
from experiments.resource_adaptive_splitfed.logical_state import aggregate_named_states
from experiments.resource_adaptive_splitfed.model_data import dirichlet_partition
from experiments.resource_adaptive_splitfed.training_runtime import fedprox_penalty
from splitfleet.autosplit.torchlens_backend import prepare_torchlens_runtime
from splitfleet.server.placement import CapabilityAwarePlacementPolicy
from splitfleet.split_engine import graph_contract_for_runtime_handle
from splitfleet.tasks import ModelInputs
from splitfleet.transport import decode_boundary, decode_gradients, encode_boundary, encode_gradients
from splitfleet.transport.split_wire import (
    boundary_to_envelope,
    envelope_to_boundary,
    envelope_to_gradients,
    gradients_to_envelope,
)

from .data import Workload, load_workload
from .models import decode_detections


METHODS = ("fedavg", "fedprox", "splitfed_fixed", "splitfleet")


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


def _batch(workload: Workload, raw: Any, device: torch.device):
    adapter = workload.task.make_adapter()
    batch = adapter.prepare_batch(raw)
    return batch.inputs.map_values(lambda item: _move(item, device)), _move(batch.targets, device), batch.num_examples


def _forward(model: torch.nn.Module, call: ModelInputs) -> Any:
    return model(*call.args, **call.kwargs)


def _train_client(
    workload: Workload,
    global_state: Mapping[str, torch.Tensor],
    indices: list[int],
    *,
    method: str,
    boundary: str | None,
    seed: int,
    round_id: int,
    client_id: str,
    batch_size: int,
    max_batches: int | None,
    learning_rate: float,
    proximal_mu: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model = workload.model_factory().to(device)
    model.load_state_dict(global_state, strict=True)
    model.train()
    adapter = workload.task.make_adapter()
    subset = torch.utils.data.Subset(workload.train_dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=workload.collate_fn,
        generator=torch.Generator().manual_seed(seed * 1_000_003 + round_id * 10_007 + int(client_id)),
    )
    if len(loader) == 0:
        raise ValueError(f"Client {client_id} has fewer than one full batch; reduce batch_size.")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    reference = {
        name: tensor.detach().to(device).clone()
        for name, tensor in global_state.items()
        if name in dict(model.named_parameters())
    }
    started = time.perf_counter()
    runtime_prepare_started = time.perf_counter()
    runtime = None
    iterator = iter(loader)
    first_raw = next(iterator)
    first_call, first_targets, first_count = _batch(workload, first_raw, device)
    if boundary is not None:
        runtime = prepare_torchlens_runtime(
            model,
            first_call.args,
            sample_kwargs=dict(first_call.kwargs),
            boundary=boundary,
            trainable=True,
            batch_axes={},
            model_name=model.__class__.__name__,
        )
    runtime_prepare_sec = time.perf_counter() - runtime_prepare_started
    counts = 0
    losses = 0.0
    upload_bytes = 0
    download_bytes = 0
    batches = 0
    for raw_index, raw in enumerate(chain((first_raw,), iterator)):
        if max_batches is not None and raw_index >= max_batches:
            break
        call, targets, count = (
            (first_call, first_targets, first_count)
            if raw_index == 0 else _batch(workload, raw, device)
        )
        optimizer.zero_grad(set_to_none=True)
        if runtime is None:
            loss = adapter.loss(_forward(model, call), targets)
            objective = loss
            if method == "fedprox":
                objective = loss + proximal_mu * fedprox_penalty(model, reference)
            objective.backward()
            optimizer.step()
        else:
            local = runtime.backend.run_prefix(
                *call.args, training=True, input_kwargs=dict(call.kwargs)
            )
            contract = graph_contract_for_runtime_handle(runtime)
            envelope = boundary_to_envelope(
                local,
                round_id=round_id,
                client_id=client_id,
                step_id=str(raw_index),
                plan_id=runtime.plan.plan_id,
                split_id=contract.split_id,
                canonical_graph_hash=contract.canonical_graph_hash,
                boundary_schema_hash=contract.boundary_schema_hash,
                model_version=round_id,
            )
            wire = encode_boundary(envelope)
            upload_bytes += len(wire)
            remote = envelope_to_boundary(decode_boundary(wire), runtime.runtime, device)
            loss, gradients = runtime.backend.train_suffix(
                remote, targets, loss_fn=adapter.loss, optimizer=optimizer
            )
            gradient_wire = encode_gradients(gradients_to_envelope(envelope, gradients))
            download_bytes += len(gradient_wire)
            received = envelope_to_gradients(decode_gradients(gradient_wire), device)
            runtime.backend.backward_prefix(local, boundary_grads=received, optimizer=optimizer)
        counts += count
        losses += float(loss.detach().cpu()) * count
        batches += 1
    if counts == 0:
        raise RuntimeError("Client executed no training examples.")
    elapsed = time.perf_counter() - started
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    return state, {
        "client_id": client_id,
        "round_id": round_id,
        "method": method,
        "boundary": boundary or "full_local",
        "num_examples": counts,
        "num_batches": batches,
        "task_loss": losses / counts,
        "fit_duration_sec": elapsed,
        "runtime_prepare_sec": runtime_prepare_sec,
        "boundary_upload_bytes": upload_bytes,
        "boundary_download_bytes": download_bytes,
    }


def _evaluate(workload: Workload, model: torch.nn.Module, *, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    loader = DataLoader(
        workload.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=workload.collate_fn,
    )
    all_targets: list[Any] = []
    all_predictions: list[Any] = []
    with torch.inference_mode():
        for raw in loader:
            call, targets, _count = _batch(workload, raw, device)
            outputs = _forward(model, call)
            if workload.task.name == "object_detection":
                all_targets.extend(_move(targets, torch.device("cpu")))
                all_predictions.extend(decode_detections(outputs))
            else:
                if isinstance(outputs, dict):
                    output_key = "out" if workload.task.name == "semantic_segmentation" else "logits"
                    outputs = outputs[output_key]
                all_targets.extend(targets.detach().cpu().numpy())
                all_predictions.extend(outputs.argmax(dim=1).detach().cpu().numpy())
    if workload.task.name == "object_detection":
        return workload.task.evaluate(all_targets, all_predictions)
    return workload.task.evaluate(np.asarray(all_targets), np.asarray(all_predictions))


def run_benchmark(
    workload: Workload,
    *,
    method: str,
    output: str | Path,
    seed: int = 2026,
    rounds: int = 1,
    num_clients: int = 2,
    batch_size: int = 2,
    max_batches_per_client: int | None = None,
    learning_rate: float = 0.01,
    proximal_mu: float = 0.01,
    fixed_boundary: str = "50%",
    dirichlet_alpha: float = 0.5,
    device: str = "cpu",
) -> Path:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}.")
    if rounds < 1 or num_clients < 2 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("rounds, num_clients, batch_size and learning_rate must be positive.")
    if method == "fedprox" and proximal_mu <= 0:
        raise ValueError("fedprox requires proximal_mu > 0.")
    if len(workload.train_dataset) < num_clients * batch_size:
        raise ValueError("Training dataset is too small for one full batch per client.")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device_obj = torch.device(device)
    model = workload.model_factory().to(device_obj)
    global_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    initial_hash = tensor_state_hash(global_state)
    assignments = dirichlet_partition(
        workload.partition_labels,
        num_clients,
        alpha=dirichlet_alpha,
        seed=seed,
        min_partition_size=batch_size,
    )
    assignment_hash = stable_hash(assignments)
    candidate_cuts = workload.task.resolve_candidate_cuts(model)
    if fixed_boundary not in candidate_cuts:
        raise ValueError(f"fixed_boundary must be one of {candidate_cuts}.")
    policy = CapabilityAwarePlacementPolicy(boundary_ladder=candidate_cuts) if method == "splitfleet" else None
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema": "splitfleet.unified-benchmark.v1",
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "UNVERIFIED",
            "version_label": "exp_result_v1",
        },
        "scope": "controlled_in_process; physical network and device gains unmeasured",
        "task": workload.task.name,
        "source": workload.source,
        "sample_selection": "evenly_spaced_v1" if workload.source == "real" else "fixture_v1",
        "method": method,
        "seed": seed,
        "rounds": rounds,
        "num_clients": num_clients,
        "batch_size": batch_size,
        "max_batches_per_client": max_batches_per_client,
        "learning_rate": learning_rate,
        "dirichlet_alpha": dirichlet_alpha,
        "fixed_boundary": fixed_boundary if method == "splitfed_fixed" else None,
        "proximal_mu": proximal_mu if method == "fedprox" else None,
        "candidate_cuts": candidate_cuts,
        "initial_model_hash": initial_hash,
        "partition_hash": assignment_hash,
        "data_content_hash": workload.data_content_hash,
        "git_commit": git_commit(),
        "completed": False,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "hostname": platform.node(),
        "device": str(device_obj),
        "device_name": torch.cuda.get_device_name(device_obj) if device_obj.type == "cuda" else None,
        "train_examples": len(workload.train_dataset),
        "test_examples": len(workload.test_dataset),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    (destination / "assignments.json").write_text(json.dumps(assignments, sort_keys=True) + "\n", encoding="utf-8")
    with (destination / "client_metrics.jsonl").open("w", encoding="utf-8") as client_stream, (
        destination / "round_metrics.jsonl"
    ).open("w", encoding="utf-8") as round_stream:
        for round_id in range(1, rounds + 1):
            started = time.perf_counter()
            updates = []
            weights = []
            records = []
            for client_id in sorted(assignments, key=int):
                boundary = None
                if method == "splitfed_fixed":
                    boundary = fixed_boundary
                elif policy is not None:
                    boundary = policy(round_id, client_id, True)
                state, record = _train_client(
                    workload,
                    global_state,
                    assignments[client_id],
                    method=method,
                    boundary=boundary,
                    seed=seed,
                    round_id=round_id,
                    client_id=client_id,
                    batch_size=batch_size,
                    max_batches=max_batches_per_client,
                    learning_rate=learning_rate,
                    proximal_mu=proximal_mu,
                    device=device_obj,
                )
                updates.append(state)
                weights.append(record["num_examples"])
                records.append(record)
                client_stream.write(json.dumps(record, sort_keys=True) + "\n")
                client_stream.flush()
                if policy is not None:
                    policy.observe_fit_metrics(
                        round_id=round_id,
                        cid=client_id,
                        num_examples=record["num_examples"],
                        metrics=record,
                    )
            global_state = aggregate_named_states(updates, weights)
            model.load_state_dict(global_state, strict=True)
            metrics = _evaluate(workload, model, batch_size=batch_size, device=device_obj)
            row = {
                "round_id": round_id,
                "method": method,
                "task": workload.task.name,
                "source": workload.source,
                "primary_metric": workload.task.primary_metric,
                "metrics": metrics,
                "round_time_sec": time.perf_counter() - started,
                "successful_clients": len(records),
                "num_examples": sum(weights),
                "boundary_upload_bytes": sum(item["boundary_upload_bytes"] for item in records),
                "boundary_download_bytes": sum(item["boundary_download_bytes"] for item in records),
                "placements": {item["client_id"]: item["boundary"] for item in records},
                "initial_model_hash": initial_hash,
                "partition_hash": assignment_hash,
                "data_content_hash": workload.data_content_hash,
            }
            round_stream.write(json.dumps(row, sort_keys=True) + "\n")
            round_stream.flush()
    metadata["completed"] = True
    metadata["finished_unix"] = time.time()
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=(
        "image_classification", "text_classification", "object_detection", "semantic_segmentation"
    ))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--source", choices=("real", "fixture"), default="real")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches-per-client", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--proximal-mu", type=float, default=0.01)
    parser.add_argument("--fixed-boundary", default="50%")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    workload = load_workload(
        args.task,
        data_root=args.data_root,
        source=args.source,
        download=args.download,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )
    result = run_benchmark(
        workload,
        method=args.method,
        output=args.output,
        seed=args.seed,
        rounds=args.rounds,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        max_batches_per_client=args.max_batches_per_client,
        learning_rate=args.learning_rate,
        proximal_mu=args.proximal_mu,
        fixed_boundary=args.fixed_boundary,
        dirichlet_alpha=args.dirichlet_alpha,
        device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
