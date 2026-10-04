"""Batch, training and evaluation helpers for the physical four-task study."""

from __future__ import annotations

from statistics import median
import time
from itertools import chain
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.common.training import fedprox_penalty
from splitfleet.autosplit.torchlens_backend import prepare_torchlens_runtime
from splitfleet.split_engine import graph_contract_for_runtime_handle
from splitfleet.tasks import ModelInputs
from splitfleet.transport import decode_boundary, decode_gradients, encode_boundary, encode_gradients
from splitfleet.transport.split_wire import (
    boundary_to_envelope,
    envelope_to_boundary,
    envelope_to_gradients,
    gradients_to_envelope,
)

from experiments.unified_multitask.data import Workload


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
    drop_last: bool = True,
    optimizer_name: str = "sgd",
    model: torch.nn.Module | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model = (workload.model_factory() if model is None else model).to(device)
    model.load_state_dict(global_state, strict=True)
    model.train()
    adapter = workload.task.make_adapter()
    subset = torch.utils.data.Subset(workload.train_dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        collate_fn=workload.collate_fn,
        generator=torch.Generator().manual_seed(seed * 1_000_003 + round_id * 10_007 + int(client_id)),
    )
    if len(loader) == 0:
        raise ValueError(f"Client {client_id} has fewer than one full batch; reduce batch_size.")
    optimizer_class = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer {optimizer_name!r}")
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    reference = {
        name: tensor.detach().to(device).clone()
        for name, tensor in global_state.items()
        if name in dict(model.named_parameters())
    } if method == "fedprox" else {}
    def synchronize_device() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize_device()
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
    synchronize_device()
    runtime_prepare_sec = time.perf_counter() - runtime_prepare_started
    counts = 0
    losses = 0.0
    upload_bytes = 0
    download_bytes = 0
    batches = 0
    client_forward_samples = []
    client_backward_samples = []
    server_service_samples = []
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
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Native training loss is nonfinite; refusing the optimizer update")
            objective = loss
            if method == "fedprox":
                objective = loss + proximal_mu * fedprox_penalty(model, reference)
            objective.backward()
            optimizer.step()
        else:
            synchronize_device()
            phase_started = time.perf_counter()
            local = runtime.backend.run_prefix(
                *call.args, training=True, input_kwargs=dict(call.kwargs)
            )
            synchronize_device()
            client_forward_samples.append((time.perf_counter() - phase_started) * 1000.0)
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
            measurements: dict[str, float] = {}
            loss, gradients = runtime.backend.train_suffix(
                remote,
                targets,
                loss_fn=adapter.loss,
                optimizer=optimizer,
                measurements=measurements,
            )
            server_service_samples.append(float(measurements["server_total_ms"]))
            gradient_wire = encode_gradients(gradients_to_envelope(envelope, gradients))
            download_bytes += len(gradient_wire)
            received = envelope_to_gradients(decode_gradients(gradient_wire), device)
            synchronize_device()
            phase_started = time.perf_counter()
            runtime.backend.backward_prefix(local, boundary_grads=received, optimizer=optimizer)
            synchronize_device()
            client_backward_samples.append((time.perf_counter() - phase_started) * 1000.0)
        counts += count
        losses += float(loss.detach().cpu()) * count
        batches += 1
    if counts == 0:
        raise RuntimeError("Client executed no training examples.")
    synchronize_device()
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
        # Runtime preparation is not an incremental cut-switch observation.
        "switch_ms": None,
        "client_forward_ms": float(median(client_forward_samples)) if client_forward_samples else None,
        "client_backward_ms": float(median(client_backward_samples)) if client_backward_samples else None,
        "server_service_ms": float(median(server_service_samples)) if server_service_samples else None,
        "boundary_upload_bytes": upload_bytes,
        "boundary_download_bytes": download_bytes,
    }


def _evaluate(workload: Workload, model: torch.nn.Module, *, batch_size: int, device: torch.device) -> dict[str, float]:
    if workload.task.name == "object_detection":
        from experiments.rfdetr_nano_physical import evaluate

        return evaluate(workload, model, device=device)
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
            if isinstance(outputs, dict):
                output_key = "out" if workload.task.name == "semantic_segmentation" else "logits"
                outputs = outputs[output_key]
            all_targets.extend(targets.detach().cpu().numpy())
            all_predictions.extend(outputs.argmax(dim=1).detach().cpu().numpy())
    return workload.task.evaluate(np.asarray(all_targets), np.asarray(all_predictions))
