"""Phase-A ResNet50 numerical gate through the production Torch split RPC path.

Run in its own process, for example:
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m \
experiments.resnet50_quality.gate_torch --output results/resnet50_quality/gates/torch_A
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .model_contract import (
    NAMES,
    TRAINABLE_NAMES,
    TRAINABLE_PARAMETERS,
    canonical_state,
    state_hash,
)

ATOL = 1e-5
RTOL = 1e-4
LEARNING_RATE = .05
SEED = 233


def compare_tensor(actual, expected, *, name):
    """Report strict schema and elementwise accuracy without relaxing tolerances."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    report = {
        "name": name, "shape": list(expected.shape), "dtype": str(expected.dtype),
        "actual_shape": list(actual.shape), "actual_dtype": str(actual.dtype),
        "passed": False,
    }
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        report["reason"] = "schema_mismatch"
        return report
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        report["reason"] = "nonfinite"
        return report
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    exact = expected.dtype.kind not in "fc"
    allowance = 0.0 if exact else ATOL + RTOL * np.abs(expected.astype(np.float64))
    mismatch = difference > allowance
    report.update(
        passed=not bool(np.any(mismatch)), exact=exact,
        mismatched_elements=int(np.count_nonzero(mismatch)),
        max_absolute_error=float(difference.max(initial=0)),
        max_relative_error=float((difference / np.maximum(np.abs(expected), 1e-12)).max(initial=0)),
    )
    return report


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class _Tee:
    def __init__(self, console, durable):
        self.console, self.durable = console, durable

    def write(self, text):
        self.console.write(text)
        self.durable.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.durable.flush()


class _RecordingLoss:
    def __init__(self):
        self.calls = []

    def __call__(self, logits, labels):
        from splitfleet.tasks.losses import sparse_cross_entropy

        from .native import numpy_value

        loss = sparse_cross_entropy(logits, labels)
        self.calls.append({"logits": numpy_value(logits), "loss": float(numpy_value(loss))})
        return loss


def _stage_boundaries(handle):
    """Resolve each complete residual stage to its final canonical operation."""
    result = {}
    for stage, channels, spatial in ((1, 256, 16), (2, 512, 8), (3, 1024, 4)):
        layer = getattr(handle.model, f"layer{stage}")
        module = f"layer{stage}.{len(layer) - 1}.relu"
        matches = [node for node in handle.runtime.trace_graph.nodes if node.module_path == module]
        if not matches:
            raise RuntimeError(f"Native TorchLens capture has no terminal ReLU for {module}")
        node = matches[-1]
        if node.output_shape is None or tuple(node.output_shape[1:]) != (channels, spatial, spatial):
            raise RuntimeError(f"Unexpected native stage{stage} endpoint shape: {node.output_shape}")
        result[f"stage{stage}"] = {
            "boundary": "after:" + node.canonical_id,
            "canonical_node": node.canonical_id, "label": node.label,
            "module": module, "captured_shape": list(node.output_shape),
        }
    return result


def _owned_snapshot(client, tail, ownership, initial):
    from .native import numpy_value

    owner_by_name = dict(zip(ownership["names"], ownership["owners"], strict=True))
    if tuple(owner_by_name) != NAMES:
        raise RuntimeError("Native split ownership does not match every canonical state name")
    replicas = {"prefix": client.model, "suffix": tail.model}
    native_states = {owner: model.state_dict() for owner, model in replicas.items()}
    native_parameters = {owner: dict(model.named_parameters()) for owner, model in replicas.items()}
    state, gradients = {}, {}
    updates = {owner: {"trainable_tensors": 0, "updated_tensors": 0, "updated_names": []}
               for owner in replicas}
    for name in NAMES:
        owner = owner_by_name[name]
        state[name] = initial[name].copy() if owner == "initial" else numpy_value(native_states[owner][name])
        if name in TRAINABLE_NAMES:
            if owner not in replicas:
                raise RuntimeError(f"Trainable parameter {name} has no executed owner")
            parameter = native_parameters[owner][name]
            if parameter.grad is None:
                raise RuntimeError(f"Missing {owner} gradient for {name}")
            gradients[name] = numpy_value(parameter.grad)
            updates[owner]["trainable_tensors"] += 1
            if np.any(state[name] != initial[name]):
                updates[owner]["updated_tensors"] += 1
                updates[owner]["updated_names"].append(name)
    return state, gradients, updates


def _run_cut(*, stage, endpoint, base, session, images, labels, initial, reference, device, event,
             client_device=None, client_images=None, client_labels=None):
    import torch

    from splitfleet.autosplit.planner import _build_placement
    from splitfleet.autosplit.state_ownership import state_ownership, validate_owned
    from splitfleet.autosplit.types import WorkerSpec
    from splitfleet.backends.utils import adapter_for
    from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
    from splitfleet.common.constants import CLIENT_ID_CONFIG_KEY
    from splitfleet.runtime.client_execution import server_model_binding
    from splitfleet.server.grpc.servicer import ServerModelServicer
    from splitfleet.server.stage_runtime.manager import StageRuntimeManager
    from splitfleet.server.strategy import AutoSplitStrategy
    from splitfleet.virtual.suffix_service import SuffixService, suffix_connection

    from .native import build_model, export_state

    client_device = client_device or device
    client_images = images if client_images is None else client_images
    client_labels = labels if client_labels is None else client_labels

    event("recut_start", stage=stage, **endpoint)
    handle = session.repartition_runtime(base, endpoint["boundary"])
    if state_hash(export_state("pytorch", base.model)) != state_hash(initial):
        raise RuntimeError("Capture/repartition changed canonical initial state")
    recorder = _RecordingLoss()

    def make_optimizer(model):
        return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0, weight_decay=0)

    strategy = AutoSplitStrategy(
        model=base.model, sample_inputs=images, boundary=handle.plan.boundary,
        autosplit_session=session, dynamic_batch=(1, 16), runtime_device=device,
        loss_fn=recorder, optimizer_fn=make_optimizer,
        aggregation_policy="splitfed", owned_state_exchange=True,
    )
    manager = StageRuntimeManager(
        autosplit_session=session, init_server_model_fn=strategy._make_server_model,
    )
    manager.set_strategy(strategy)
    strategy.bind_stage_runtime_manager(manager)
    placement = _build_placement(
        handle, candidate=handle.backend.current_candidate,
        validation={"success": True, "verification": "native_capability_only", "replay_performed": False},
        worker_specs=[WorkerSpec(worker_id="coordinator", device=device)],
        constraints=strategy.constraints, objective=strategy.objective,
        model_name="ResNet50-V1.5-CIFAR10",
    )
    placement.metadata["_training"] = True
    strategy._placement_plans[handle.plan.boundary] = placement
    strategy._placement_plan = placement
    manager.set_placement_plan(placement)
    cid = "phaseA_" + stage
    initial_arrays = [initial[name] for name in NAMES]
    strategy._round_initial_client_states[1] = initial_arrays
    strategy._round_active_clients = [cid]
    server_configs = strategy.configure_server_fit(1, initial_arrays, [cid])
    manager.initialize_server_models(server_configs)
    tail = manager.get_server_model(cid)
    client_model = base.model if client_device == device else build_model("pytorch", initial, client_device).train()
    client = AutoSplitSplitLearningClient(
        model=client_model, train_data=[(client_images, client_labels)], sample_inputs=client_images,
        optimizer_fn=make_optimizer, device=client_device,
        autosplit_session=type(session)(device=client_device), max_cached_runtimes=1,
    )
    # The production manager verifies the full graph contract while binding
    # independent live state; this reuses capture without bypassing admission.
    if client_device == device:
        client._prewarmed_runtime = manager.clone_runtime_for_model(client.model, plan_id=placement.plan_id)
    else:
        # Capture on the actual prefix device; server CUDA literal arguments
        # must never redirect CPU or other-GPU prefix operations.
        client._prewarmed_runtime = client.autosplit_session.prepare_runtime(
            client.model, client_images, boundary=handle.plan.boundary,
            trainable=True, dynamic_batch=(1, 16))
        from splitfleet.split_engine import graph_contract_for_runtime_handle
        from splitfleet.split_engine.contracts import validate_contract

        validate_contract(graph_contract_for_runtime_handle(handle),
                          graph_contract_for_runtime_handle(client._prewarmed_runtime))
    from splitfleet.virtual.backend_ops import FrameworkOps

    FrameworkOps("torch").assert_model_device(client.model, client_device)
    FrameworkOps("torch").assert_model_device(tail.model, device)
    schema = adapter_for(client.model, client_images).state_manifest(client.model).schema_hash
    ownership = state_ownership(client._prewarmed_runtime, schema)
    reference_evidence = {}
    if client_device != device:
        from .staged_reference import placement_reference
        reference = placement_reference("pytorch", initial, client_images, client_labels,
            stage=int(stage.removeprefix("stage")), client_device=client_device, server_device=device,
            learning_rate=LEARNING_RATE, full_reference=reference, ownership=ownership)
        reference_evidence = {name: reference[name] for name in ("reference_kind", "native_devices",
            "native_autograd_device_copy", "native_vjp_composition", "same_device_native_control", "full_client_device_reference_differences")}
    config = dict(server_configs[0].config)
    config[CLIENT_ID_CONFIG_KEY] = cid
    if state_hash(export_state("pytorch", client.model)) != state_hash(initial):
        raise RuntimeError("Prefix preparation changed initial state")
    if state_hash(export_state("pytorch", tail.model)) != state_hash(initial):
        raise RuntimeError("Suffix preparation changed initial state")
    event("rpc_step_start", stage=stage, boundary=handle.plan.boundary,
          dynamic_batch=list(handle.plan.dynamic_batch),
          batch_validation=handle.runtime.batch_validation,
          prefix_nodes=handle.plan.prefix_node_count, suffix_nodes=handle.plan.suffix_node_count)
    servicer = ServerModelServicer(manager, execution_concurrency=1)
    service = SuffixService(manager, servicer=servicer).start()
    try:
        with suffix_connection(service.address, cid, timeout=120) as proxy, server_model_binding(client, proxy):
            prefix_arrays, examples, metrics = client.fit(initial_arrays, config)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        server_result = tail.get_fit_result()
        actual_state, gradients, updates = _owned_snapshot(client, tail, ownership, initial)
        uploaded = {
            **validate_owned(prefix_arrays, ownership, "prefix", initial_arrays),
            **validate_owned(server_result.parameters, ownership, "suffix", initial_arrays),
        }
        for index, value in uploaded.items():
            if not np.array_equal(value, actual_state[NAMES[index]]):
                raise RuntimeError(f"Owned state upload differs from executed state at {NAMES[index]}")
        if len(recorder.calls) != 1:
            raise RuntimeError(f"Suffix objective ran {len(recorder.calls)} times for one actual batch")
        loss_call = recorder.calls[0]
        comparisons = {
            "logits": compare_tensor(loss_call["logits"], reference["logits"], name="logits"),
            "ce": compare_tensor(np.asarray(loss_call["loss"], dtype=np.float32),
                                 np.asarray(reference["loss"], dtype=np.float32), name="ce"),
            "gradients": {name: compare_tensor(gradients[name], reference["gradients"][name], name=name)
                          for name in TRAINABLE_NAMES},
            "sgd_updates": {
                name: compare_tensor(actual_state[name] - initial[name],
                                     reference["state"][name] - initial[name], name=name)
                for name in TRAINABLE_NAMES
            },
            "state": {name: compare_tensor(actual_state[name], reference["state"][name], name=name)
                      for name in NAMES},
        }
        counters = {name: int(actual_state[name]) for name in NAMES if name.endswith("num_batches_tracked")}
        budget_ok = (
            examples == 2 and server_result.config["num_examples"] == 2
            and metrics["num_batches"] == 1 and metrics["skipped_batches"] == 0
            and metrics["activation_upload_bytes"] > 0 and metrics["gradient_download_bytes"] > 0
            and len(tail._processed_steps) == 1
        )
        counters_ok = len(counters) == 53 and set(counters.values()) == {1}
        updates_ok = all(row["updated_tensors"] > 0 for row in updates.values())
        flat = [comparisons["logits"], comparisons["ce"], *comparisons["gradients"].values(),
                *comparisons["sgd_updates"].values(), *comparisons["state"].values()]
        passed = budget_ok and counters_ok and updates_ok and all(row["passed"] for row in flat)
        failed_tensors = [row for row in flat if not row["passed"]]
        result = {
            "stage": stage, "status": "passed" if passed else "failed", **endpoint,
            **reference_evidence,
            "actual_boundary": handle.plan.boundary, "graph_signature": handle.plan.graph_signature,
            "graph_contract": client._runtime_cache[next(iter(client._runtime_cache))].contract.to_dict(),
            "dynamic_batch": list(handle.plan.dynamic_batch), "batch_validation": handle.runtime.batch_validation,
            "ownership": ownership, "updates": updates, "comparisons": comparisons,
            "bn_counters": counters, "counters_ok": counters_ok, "budget_ok": budget_ok,
            "updates_ok": updates_ok, "gradient_tensor_count": len(gradients),
            "state_tensor_count": len(actual_state), "failed_tensors": failed_tensors,
            "rpc_address": service.address, "rpc_processed_steps": len(tail._processed_steps),
            "client_metrics": metrics, "server_metrics": server_result.config,
            "prefix_upload_tensors": len(prefix_arrays), "suffix_upload_tensors": len(server_result.parameters),
            "initial_hash": state_hash(initial), "updated_hash": state_hash(actual_state),
            "captured_training_logits": loss_call["logits"].tolist(), "captured_ce": loss_call["loss"],
        }
        event("rpc_step_finished", stage=stage, status=result["status"],
              failed_tensor_count=len(failed_tensors), counters_ok=counters_ok,
              prefix_updated_tensors=updates["prefix"]["updated_tensors"],
              suffix_updated_tensors=updates["suffix"]["updated_tensors"])
        return result
    finally:
        service.stop()


def run_gate(output: Path, *, device="cuda:0", client_device=None, data_root="data/cifar-10-batches-py"):
    """Write a durable gate receipt, retaining every failed cut and traceback."""
    client_device = client_device or device
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": "splitfleet.resnet50.phaseA.torch.v1", "status": "running", "device": device,
        "client_device": client_device, "server_device": device,
        "started_utc": datetime.now(UTC).isoformat(), "seed": SEED,
        "atol": ATOL, "rtol": RTOL, "learning_rate": LEARNING_RATE,
        "momentum": 0, "weight_decay": 0, "batch_size": 2,
        "trainable_parameters": TRAINABLE_PARAMETERS, "cuts": [],
    }
    summary["source_hashes"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("gate_torch.py", "native.py", "data.py", "model_contract.py", "torch_model.py", "staged_reference.py")
    }
    started = time.monotonic()

    def event(phase, **details):
        record = {"event": phase, "elapsed_seconds": time.monotonic() - started, **details}
        summary["last_event"] = record
        _write_json(output / "summary.json", summary)
        print(json.dumps(record, sort_keys=True), flush=True)

    with (output / "stdout.log").open("x", encoding="utf-8") as log:  # noqa: SIM117 - explicit durable stream lifetime
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            try:
                event("gate_started")
                for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                    os.environ[variable] = "1"
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                import torch

                from splitfleet.autosplit import AutoSplitSession

                from .data import load_cifar
                from .native import (
                    build_model,
                    configure_backend,
                    export_state,
                    native_batch,
                    train_step,
                )

                configure_backend("pytorch")
                if device.startswith("cuda") and not torch.cuda.is_available():
                    raise RuntimeError("Requested CUDA device is unavailable; the gate will not fall back")
                from splitfleet.virtual.backend_ops import FrameworkOps
                FrameworkOps("torch").admit_device(device)
                FrameworkOps("torch").admit_device(client_device)
                summary["versions"] = {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "torchvision", "torchlens", "flwr", "numpy", "Pillow")
                }
                summary["environment"] = {
                    name: os.environ.get(name)
                    for name in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
                                 "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
                }
                summary["precision"] = {
                    "dtype": "float32", "amp": False,
                    "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                }
                if device.startswith("cuda"):
                    summary["accelerator"] = torch.cuda.get_device_name(device)
                event("data_loading")
                data = load_cifar(data_root)
                sample_ids = data.train_indices[:2]
                image_arrays = data.images(sample_ids)
                label_arrays = data.train_labels[sample_ids]
                images, labels = native_batch("pytorch", image_arrays, label_arrays, device)
                client_images, client_labels = native_batch("pytorch", image_arrays, label_arrays, client_device)
                summary["data"] = {
                    "dataset": "CIFAR-10", "sample_ids": sample_ids.tolist(),
                    "labels": label_arrays.tolist(), "input_shape": list(image_arrays.shape),
                    "input_hash": hashlib.sha256(image_arrays.tobytes()).hexdigest(),
                    "mean": data.mean.tolist(), "std": data.std.tolist(),
                    "preprocessing": "Pillow.BILINEAR 32-to-64; frozen training normalization; NCHW FP32",
                }
                del data
                initial = canonical_state(SEED)
                summary["initial_hash"] = state_hash(initial)
                oracle = build_model("pytorch", initial, client_device)
                if sum(p.numel() for p in oracle.parameters() if p.requires_grad) != TRAINABLE_PARAMETERS:
                    raise RuntimeError("The numerical oracle is not the full required ResNet50")
                event("native_oracle_start")
                reference = train_step("pytorch", oracle, client_images, client_labels, learning_rate=LEARNING_RATE, capture=True)
                reference["state"] = export_state("pytorch", oracle)
                if set(reference["gradients"]) != set(TRAINABLE_NAMES):
                    raise RuntimeError("Native oracle does not cover every canonical gradient")
                oracle_counters = [int(v) for k, v in reference["state"].items() if k.endswith("num_batches_tracked")]
                if len(oracle_counters) != 53 or set(oracle_counters) != {1}:
                    raise RuntimeError("Native oracle BN counters did not advance exactly once")
                event("native_oracle_finished", loss=reference["loss"], gradients=len(reference["gradients"]))
                summary["oracle"] = {"loss": reference["loss"], "logits": reference["logits"].tolist(),
                                     "updated_hash": state_hash(reference["state"])}
                del oracle
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                template = build_model("pytorch", initial, device).train()
                session = AutoSplitSession(device=device)
                event("native_capture_start", dynamic_batch=[1, 16])
                base = session.prepare_runtime(template, images, boundary="50%", trainable=True, dynamic_batch=(1, 16))
                if state_hash(export_state("pytorch", template)) != state_hash(initial):
                    raise RuntimeError("Native TorchLens graph capture changed initialization or BN state")
                batch_validation = base.runtime.batch_validation
                if (batch_validation.get("status") != "passed"
                        or batch_validation.get("traced_batch_size") != 1
                        or batch_validation.get("probe_batch_size") != 2):
                    raise RuntimeError(f"Native dynamic batch admission did not pass: {batch_validation}")
                endpoints = _stage_boundaries(base)
                summary["endpoints"] = endpoints
                event("native_capture_finished", graph_nodes=len(base.runtime.trace_graph.nodes),
                      batch_validation=base.runtime.batch_validation)
                for stage, endpoint in endpoints.items():
                    try:
                        result = _run_cut(
                            stage=stage, endpoint=endpoint, base=base, session=session,
                            images=images, labels=labels, initial=initial, reference=reference,
                            device=device, event=event, client_device=client_device,
                            client_images=client_images, client_labels=client_labels,
                        )
                    except Exception as error:  # noqa: BLE001 - retain each native cut failure
                        result = {"stage": stage, **endpoint, "status": "failed", "error": repr(error),
                                  "traceback": traceback.format_exc()}
                        traceback.print_exc()
                        event("cut_exception", stage=stage, error=repr(error))
                    _write_json(output / f"{stage}.json", result)
                    summary["cuts"].append(result)
                    _write_json(output / "summary.json", summary)
                    gc.collect()
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                summary["status"] = "passed" if all(row["status"] == "passed" for row in summary["cuts"]) else "failed"
            except Exception as error:  # noqa: BLE001 - persist native gate failure
                summary.update(status="failed", error=repr(error), traceback=traceback.format_exc())
                traceback.print_exc()
            finally:
                summary["elapsed_seconds"] = time.monotonic() - started
                summary["finished_utc"] = datetime.now(UTC).isoformat()
                event("gate_finished", status=summary["status"])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="Actual suffix/template device")
    parser.add_argument("--client-device", help="Actual FedAvg oracle and prefix device; defaults to --device")
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar-10-batches-py"))
    arguments = parser.parse_args(argv)
    result = run_gate(arguments.output, device=arguments.device, client_device=arguments.client_device,
                      data_root=arguments.data_root)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
