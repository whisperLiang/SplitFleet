"""Capture complete cut catalogs and pair real split training with native updates.

Each model runs in its own process. Captures are reused, while client, server
and native models have independent parameter storage. Rejected, untested and
numerically failed boundaries are distinct evidence states.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import torch


CHECKS = ("outputs", "loss", "gradients", "updates", "buffers", "optimizer",
          "activation_wire", "gradient_wire")


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(v) for v in value)
    return value


def restore_model_state(model, state, buffers):
    """Reset each trajectory without replacing buffers referenced by capture."""
    model.load_state_dict(state)
    # state_dict omits non-persistent buffers. Restore them in place as well,
    # so captures keep their live tensor references across cuts and RPC checks.
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            buffer.copy_(buffers[name])


def compare(expected, actual, *, rtol, atol, path="value"):
    """Report errors even on failure; retain exact structure and integer checks."""
    groups, failures = {}, []
    def walk(left, right, name):
        if isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                failures.append(f"{name}: keys differ")
                return
            for key, value in left.items():
                walk(value, right[key], f"{name}.{key}")
        elif isinstance(left, (tuple, list)):
            if type(left) is not type(right) or len(left) != len(right):
                failures.append(f"{name}: container differs")
                return
            for index, (a, b) in enumerate(zip(left, right)):
                walk(a, b, f"{name}[{index}]")
        elif isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
                failures.append(f"{name}: tensor schema differs")
                return
            if not left.numel():
                return
            right = right.to(left.device)
            floating = left.is_floating_point() or left.is_complex()
            if floating:
                passed = (torch.isfinite(left) & torch.isfinite(right) & torch.isclose(right, left, rtol=rtol, atol=atol)).all()
                error = (left - right).abs().max()
            else:
                passed = (left == right).all()
                error = (left.to(torch.int64) - right.to(torch.int64)).abs().max().float()
            groups.setdefault(str(left.device), []).append((name, passed, torch.nan_to_num(error, nan=0., posinf=0., neginf=0.)))
        elif left != right:
            failures.append(f"{name}: value differs")
    walk(expected, actual, path)
    maximum = 0.
    for rows in groups.values():
        # Transfer scalar reductions together, rather than synchronizing CUDA
        # for every parameter in a full Transformer.
        passed = torch.stack([row[1] for row in rows]).cpu().tolist()
        maximum = max(maximum, float(torch.stack([row[2] for row in rows]).max()))
        failures.extend(row[0] for row, ok in zip(rows, passed) if not ok)
    return {"passed": not failures, "max_abs": maximum,
            **({"reason": failures[0]} if failures else {})}


def graph_record(runtime):
    graph = runtime.trace_graph
    compute = tuple(graph.compute_nodes)
    positions = {node.canonical_id: i for i, node in enumerate(compute)}
    edges = sorted({(positions[parent], positions[node.canonical_id])
                    for node in compute for parent in node.parents if parent in positions})
    return {"backend": graph.backend, "graph_hash": runtime.graph_ir.graph_hash,
            "nodes": [{"id": n.canonical_id, "operation": n.op_type,
                       "module": n.module_path, "shape": n.output_shape,
                       "dtype": n.dtype} for n in compute],
            "edges": edges, "long_edges": sum(b - a > 1 for a, b in edges)}


def catalog(handle):
    from splitfleet.autosplit.state_ownership import state_ownership
    from splitfleet.autosplit.torchlens_backend import TorchLensRuntimeHandle
    from splitfleet.backends.utils import adapter_for
    from torchlens.split.errors import SplitRequestError, SplitUnsupportedError

    runtime = handle.runtime
    view = copy.copy(runtime)
    view.request = replace(view.request, validation="permissive")
    nodes = list(runtime.trace_graph.compute_nodes)
    positions = {node.canonical_id: i for i, node in enumerate(nodes)}
    schema = adapter_for(handle.model, ()).state_manifest(handle.model).schema_hash
    rows = []
    analyses = {}
    parameters = dict(handle.model.named_parameters())
    for site in view.split_points(diagnose=False).candidates:
        row = {"boundary": f"{site.kind}:{site.node_id}", "kind": site.kind,
               "position": positions[site.node_id], "status": "rejected",
               "frontier_count": None, "reason": None}
        rows.append(row)
        try:
            analysis = view.analyze(site.point)
            row["frontier_count"] = len(analysis.plan.boundary_node_ids)
            report = analysis.capability_report
            if report is None or not (report.preflight.supported and report.replay.supported and report.training.supported):
                row["reason"] = "; ".join(report.unsupported_reasons) if report else "missing_capability_report"
                continue
            # State ownership uses graph references only; no segment execution.
            candidate = TorchLensRuntimeHandle(handle.model, copy.copy(runtime), handle.plan, handle.backend)
            candidate.runtime.plan = analysis.plan
            ownership = state_ownership(candidate, schema)
            # Non-persistent buffers (e.g. BERT position_ids) are absent from
            # state_dict, but belong in numerical state checks as well.
            known = set(ownership["names"])
            for name, value in handle.model.named_buffers():
                if name in known:
                    continue
                stages = set()
                for node in runtime.trace_graph.nodes:
                    refs = list(node.buffer_refs) + list(node.param_refs)
                    for ref in node.param_refs:
                        module = getattr(ref, "module", None)
                        if module is not None:
                            refs.extend(module.buffers.values())
                    if any(getattr(ref, "handle", None) is value for ref in refs):
                        if node.canonical_id in analysis.plan.prefix_node_ids:
                            stages.add("prefix")
                        if node.canonical_id in analysis.plan.suffix_node_ids:
                            stages.add("suffix")
                if len(stages) > 1:
                    raise ValueError(f"Non-persistent buffer {name} is shared across stages")
                ownership["names"].append(name)
                ownership["owners"].append(next(iter(stages), "initial"))
            owners = dict(zip(ownership["names"], ownership["owners"]))
            if not all(any(owners.get(name) == side and p.requires_grad for name, p in parameters.items())
                       for side in ("prefix", "suffix")):
                row["reason"] = "both_stages_must_have_trainable_parameters"
                continue
            row.update(status="admitted_unchecked", split_id=analysis.plan.split_id)
            analyses[row["boundary"]] = (site.point, ownership)
        except (SplitRequestError, SplitUnsupportedError, ValueError) as exc:
            row["reason"] = f"{type(exc).__name__}: {exc}"
    return rows, analyses


def rng_state():
    return torch.get_rng_state(), torch.cuda.get_rng_state() if torch.cuda.is_available() else None


def restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1])


def optimizer_state(model, optimizer):
    return {name: snapshot(optimizer.state.get(p, {})) for name, p in model.named_parameters()}


def native_step(model, optimizer, call, targets, loss_fn):
    optimizer.zero_grad(set_to_none=True)
    before = snapshot(dict(model.named_parameters()))
    outputs = model(*call.args, **call.kwargs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    gradients = {n: snapshot(p.grad) for n, p in model.named_parameters()}
    optimizer.step()
    return {"outputs": snapshot(outputs), "loss": snapshot(loss), "gradients": gradients,
            "updates": {n: snapshot(p) - before[n] for n, p in model.named_parameters()},
            "buffers": snapshot(dict(model.named_buffers())), "optimizer": optimizer_state(model, optimizer)}


def rpc_suffix(suffix, optimizer, wire, targets, loss_fn):
    """One loopback call through the production protobuf service and tail model."""
    import asyncio
    from types import SimpleNamespace
    import grpc
    from splitfleet.autosplit import AutoSplitSession
    from splitfleet.common.serde import to_grpc_format, from_grpc_format
    from splitfleet.proto import server_model_pb2 as pb, server_model_pb2_grpc as rpc
    from splitfleet.server.grpc.servicer import ServerModelServicer
    from splitfleet.server.server_model.autosplit_tail_server_model import AutoSplitTailServerModel
    from splitfleet.server.server_model.utils import ClientRequestGroup
    from splitfleet.split_engine import graph_contract_for_runtime_handle
    from splitfleet.transport import encode_boundary, encode_bundle_wire

    device = str(next(suffix.model.parameters()).device)
    manager = SimpleNamespace(autosplit_session=AutoSplitSession())
    tail = AutoSplitTailServerModel(runtime_manager=manager, model=suffix.model,
                                    loss_fn=loss_fn, device=device)
    tail.model, tail.runtime_handle, tail.optimizer = suffix.model, suffix, optimizer
    tail.graph_contract = graph_contract_for_runtime_handle(suffix)
    tail.plan_id, tail.model_version, tail.sid = wire.plan_id, wire.model_version, "coverage"
    tail.batch_window = (wire.batch_size, wire.batch_size)
    group = ClientRequestGroup(tail.sid)
    strategy = SimpleNamespace(route_client_request=lambda **_: (group, None),
        mark_ready_requests=group.mark_as_ready)
    serving_manager = SimpleNamespace(strategy=strategy, get_server_model=lambda _: tail)
    payload = {"boundary": encode_boundary(wire), "targets": encode_bundle_wire(targets, backend="torch"),
               "metadata": json.dumps({"num_examples": wire.batch_size}).encode()}
    async def exchange():
        options = [("grpc.max_send_message_length", 256 * 1024**2),
                   ("grpc.max_receive_message_length", 256 * 1024**2)]
        service = grpc.aio.server(options=options)
        rpc.add_ServerModelServicer_to_server(ServerModelServicer(serving_manager), service)
        port = service.add_insecure_port("127.0.0.1:0")
        await service.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}", options=options) as channel:
                response = await rpc.ServerModelStub(channel).UnaryRequest(
                    pb.BatchData(method="train_tail", cid="coverage", data=to_grpc_format(payload)), timeout=120)
                return from_grpc_format(response.data)
        finally:
            await service.stop(0)
    return asyncio.run(exchange())


def split_step(prefix, suffix, optimizers, ownership, call, targets, loss_fn, *, rpc=False):
    from splitfleet.split_engine import graph_contract_for_runtime_handle
    from splitfleet.transport import encode_boundary, decode_boundary, encode_gradients, decode_gradients
    from splitfleet.transport.split_wire import (boundary_to_envelope, envelope_to_boundary,
        gradients_to_envelope, envelope_to_gradients)

    owners = dict(zip(ownership["names"], ownership["owners"]))
    models = {"prefix": prefix.model, "suffix": suffix.model}
    for model in models.values():
        model.zero_grad(set_to_none=True)
    combined = {n: p for side, model in models.items() for n, p in model.named_parameters() if owners[n] == side}
    before = snapshot(combined)
    payload = prefix.backend.run_prefix(*call.args, training=True, input_kwargs=dict(call.kwargs))
    contract = graph_contract_for_runtime_handle(prefix)
    wire = boundary_to_envelope(payload, round_id=1, client_id="coverage", step_id="1",
        plan_id=prefix.plan.plan_id, split_id=contract.split_id,
        canonical_graph_hash=contract.canonical_graph_hash, boundary_schema_hash=contract.boundary_schema_hash,
        model_version=1)
    remote = envelope_to_boundary(decode_boundary(encode_boundary(wire)), suffix.runtime, next(suffix.model.parameters()).device)
    # A boundary may alias a parameter. Compare the transmitted values before
    # either optimizer mutates that storage, rather than after the train step.
    activation_pair = (snapshot(payload.tensors), snapshot(remote.tensors))
    captured = {}
    def measured_loss(outputs, labels):
        captured["outputs"] = snapshot(outputs)
        return loss_fn(outputs, labels)
    if rpc:
        response = rpc_suffix(suffix, optimizers[1], wire, targets, measured_loss)
        loss = torch.tensor(json.loads(response["metadata"])["loss"], device=next(prefix.model.parameters()).device)
        received = envelope_to_gradients(decode_gradients(response["gradients"]), next(prefix.model.parameters()).device)
        gradients = received
    else:
        loss, gradients = suffix.backend.train_suffix(remote, targets, loss_fn=measured_loss, optimizer=optimizers[1])
        gradient_wire = gradients_to_envelope(wire, gradients)
        received = envelope_to_gradients(decode_gradients(encode_gradients(gradient_wire)),
                                        next(prefix.model.parameters()).device)
    prefix.backend.backward_prefix(payload, boundary_grads=received, optimizer=optimizers[0])
    state = {}
    for side, model, optimizer in zip(("prefix", "suffix"), models.values(), optimizers):
        state.update({n: snapshot(optimizer.state.get(p, {})) for n, p in model.named_parameters() if owners[n] == side})
    return {"outputs": captured["outputs"], "loss": snapshot(loss),
            "gradients": {n: snapshot(p.grad) for n, p in combined.items()},
            "updates": {n: snapshot(p) - before[n] for n, p in combined.items()},
            "buffers": {n: snapshot(v) for side, model in models.items() for n, v in model.named_buffers() if owners[n] == side},
            "optimizer": state, "activation_wire": activation_pair,
            "gradient_wire": (gradients, received)}


def pressure_boundaries(rows):
    """Preselect the widest admitted frontier in each third, before any test."""
    admitted = [r for r in rows if r["status"] == "admitted_unchecked"]
    length = max((r["position"] for r in rows), default=0) + 1
    chosen = []
    for third in range(3):
        group = [r for r in admitted if min(2, 3 * r["position"] // length) == third]
        if group:
            chosen.append(max(group, key=lambda r: (r["frontier_count"], -r["position"], r["kind"] == "after"))["boundary"])
    return chosen


def validate(model, batches, loss_fn, *, device, steps, output, rtol=2e-4, atol=2e-6, numeric=True):
    from splitfleet.autosplit import prepare_torchlens_runtime
    from torchlens.split.errors import SplitRequestError, SplitUnsupportedError

    model = model.to(device).train()
    if numeric and len(batches) < steps:
        raise ValueError(f"Need {steps} distinct input batches, received {len(batches)}")
    initial = snapshot(model.state_dict())
    initial_buffers = snapshot(dict(model.named_buffers()))
    native, client, server = (copy.deepcopy(model) for _ in range(3))
    stores = [{p.untyped_storage().data_ptr() for p in replica.parameters()} for replica in (native, client, server)]
    if any(a & b for index, a in enumerate(stores) for b in stores[index + 1:]):
        raise ValueError("Native/client/server replicas share parameter storage")
    call, targets = batches[0]
    sample = call.args[0] if call.args else next(iter(call.kwargs.values()))
    batch_size = int(sample.shape[0])
    # Capture only once per independent replica, never per cut.
    handles = [prepare_torchlens_runtime(replica, call.args, sample_kwargs=dict(call.kwargs),
        boundary="50%", trainable=True, batch_axes={}, dynamic_batch=(batch_size, batch_size))
        for replica in (client, server)]
    for replica in (native, client, server):
        restore_model_state(replica, initial, initial_buffers)
    graph = graph_record(handles[0].runtime)
    rows, analyses = catalog(handles[0])
    pressure = pressure_boundaries(rows)
    record = {"schema": "splitfleet.execution-coverage.v1", "graph": graph, "cuts": rows,
              "pressure_boundaries": pressure, "trajectories": [], "checks": CHECKS,
              "rtol": rtol, "atol": atol, "steps": steps, "device": str(device),
              "batch_size": batch_size, "transport": "typed activation and gradient serialization",
              "independent_model_storage": True, "numeric_selection": "all_admitted" if numeric else "none"}
    save(output, record)
    if not numeric:
        return record
    initial_rng = rng_state()
    references = []
    reference_rngs = []
    optimizer = torch.optim.Adam(native.parameters(), lr=1e-4)
    for index in range(steps):
        reference_rngs.append(rng_state())
        references.append(native_step(native, optimizer, *batches[index], loss_fn))
    restore_model_state(model, initial, initial_buffers)
    restore_rng(reference_rngs[0])
    control = native_step(model, torch.optim.Adam(model.parameters(), lr=1e-4), call, targets, loss_fn)
    record["native_repeat"] = {name: compare(references[0][name], control[name], rtol=rtol, atol=atol) for name in CHECKS[:6]}
    del control
    native_initial_parameters = {n for n, _ in native.named_parameters()}
    for index, row in enumerate(rows):
        if row["boundary"] not in analyses:
            continue
        started = time.monotonic()
        try:
            for replica in (client, server):
                restore_model_state(replica, initial, initial_buffers)
            prefix = handles[0].backend.repartition(row["boundary"])
            suffix = handles[1].backend.repartition(row["boundary"])
            ownership = analyses[row["boundary"]][1]
            optimizers = [torch.optim.Adam(replica.parameters(), lr=1e-4) for replica in (client, server)]
            trajectory = []
            count = steps if row["boundary"] in pressure else 1
            for step in range(count):
                restore_rng(reference_rngs[step])
                actual = split_step(prefix, suffix, optimizers, ownership,
                                    *batches[step], loss_fn)
                # Unused parameters/buffers retain the native None/initial state.
                for n, p in native.named_parameters():
                    if n not in actual["gradients"]:
                        actual["gradients"][n] = None
                        actual["updates"][n] = torch.zeros_like(p)
                        actual["optimizer"][n] = {}
                for n, v in native.named_buffers():
                    if n not in actual["buffers"]:
                        actual["buffers"][n] = initial_buffers[n]
                checks = {name: compare(references[step][name], actual[name], rtol=rtol, atol=atol)
                          for name in CHECKS[:6]}
                for name in CHECKS[6:]:
                    checks[name] = compare(*actual[name], rtol=0, atol=0)
                trajectory.append({"step": step + 1, "checks": checks,
                                   "native_loss": float(references[step]["loss"]),
                                   "split_loss": float(actual["loss"])})
                if not all(v["passed"] for v in checks.values()):
                    break
            row.update(status="passed" if len(trajectory) == count and all(v["passed"] for v in trajectory[-1]["checks"].values()) else "failed",
                       checks=trajectory[0]["checks"], completed_steps=len(trajectory), planned_steps=count,
                       elapsed_sec=time.monotonic() - started)
            if count > 1:
                record["trajectories"].append({"boundary": row["boundary"], "planned_steps": count,
                                               "status": row["status"], "steps": trajectory})
        except (SplitRequestError, SplitUnsupportedError) as exc:
            row.update(status="execution_refused", reason=f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        record["counts"] = dict(Counter(r["status"] for r in rows))
        save(output, record)
        if index % 20 == 0 or row["status"] != "passed":
            print(f"CUT {index+1}/{len(rows)} {row['status']} {row.get('reason') or ''}", flush=True)
    restore_rng(initial_rng)
    if pressure:
        boundary = max((r for r in rows if r["boundary"] in pressure), key=lambda r: r["frontier_count"])["boundary"]
        rpc_record = {"boundary": boundary, "transport": "production ServerModelServicer.UnaryRequest + AutoSplitTailServerModel.train_tail", "deployment": "same-host loopback"}
        try:
            for replica in (client, server):
                restore_model_state(replica, initial, initial_buffers)
            prefix = handles[0].backend.repartition(boundary)
            suffix = handles[1].backend.repartition(boundary)
            optimizers = [torch.optim.Adam(replica.parameters(), lr=1e-4) for replica in (client, server)]
            restore_rng(reference_rngs[0])
            actual = split_step(prefix, suffix, optimizers, analyses[boundary][1], call, targets, loss_fn, rpc=True)
            for n, p in native.named_parameters():
                if n not in actual["gradients"]:
                    actual["gradients"][n], actual["updates"][n], actual["optimizer"][n] = None, torch.zeros_like(p), {}
            for n, v in native.named_buffers():
                actual["buffers"].setdefault(n, initial_buffers[n])
            rpc_record["checks"] = {name: compare(references[0][name], actual[name], rtol=rtol, atol=atol) for name in CHECKS[:6]}
            rpc_record["status"] = "passed" if all(c["passed"] for c in rpc_record["checks"].values()) else "failed"
        except Exception as exc:
            rpc_record.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        record["rpc"] = rpc_record
    record["counts"] = dict(Counter(r["status"] for r in rows))
    record["parameter_count"] = sum(p.numel() for p in model.parameters())
    record["unique_parameters_checked"] = len(native_initial_parameters)
    save(output, record)
    return record


def tail_batch_check(model, batches, loss_fn, boundary, device):
    """Reuse one B=2 capture for a B=2 then B=1 Adam trajectory."""
    from splitfleet.autosplit import prepare_torchlens_runtime
    from splitfleet.tasks import ModelInputs

    def join(a, b):
        if isinstance(a, torch.Tensor):
            return torch.cat((a, b), dim=0)
        if isinstance(a, dict):
            return {k: join(v, b[k]) for k, v in a.items()}
        return type(a)(join(x, y) for x, y in zip(a, b))
    initial = snapshot(model.state_dict())
    initial_buffers = snapshot(dict(model.named_buffers()))
    native, client, server = (copy.deepcopy(model) for _ in range(3))
    first, second = batches[0], batches[1]
    call = ModelInputs(join(first[0].args, second[0].args), join(first[0].kwargs, second[0].kwargs))
    labels = join(first[1], second[1])
    axes = {f"/args/{i}": 0 for i in range(len(call.args))} or {f"/kwargs/{key}": 0 for key in call.kwargs}
    handles = [prepare_torchlens_runtime(replica, call.args, sample_kwargs=dict(call.kwargs), boundary=boundary,
                                        trainable=True, batch_axes=axes, dynamic_batch=(1, 2)) for replica in (client, server)]
    for replica in (native, client, server):
        restore_model_state(replica, initial, initial_buffers)
    rows, analyses = catalog(handles[0])
    if boundary not in analyses:
        raise ValueError("Tail-batch boundary rejected by the B=2 training catalog")
    optimizer = torch.optim.Adam(native.parameters(), lr=1e-4)
    split_optimizers = [torch.optim.Adam(replica.parameters(), lr=1e-4) for replica in (client, server)]
    trajectory = []
    for index, (inputs, targets) in enumerate(((call, labels), batches[2])):
        state = rng_state()
        expected = native_step(native, optimizer, inputs, targets, loss_fn)
        restore_rng(state)
        actual = split_step(*handles, split_optimizers, analyses[boundary][1], inputs, targets, loss_fn)
        for n, p in native.named_parameters():
            if n not in actual["gradients"]:
                actual["gradients"][n], actual["updates"][n], actual["optimizer"][n] = None, torch.zeros_like(p), {}
        for n, value in native.named_buffers():
            actual["buffers"].setdefault(n, value.detach())
        checks = {name: compare(expected[name], actual[name], rtol=2e-4, atol=2e-6) for name in CHECKS[:6]}
        trajectory.append({"batch_size": 2 if index == 0 else 1, "checks": checks})
    return {"boundary": boundary, "capture_batch_size": 2, "recaptured": False,
            "steps": trajectory, "status": "passed" if all(c["passed"] for step in trajectory for c in step["checks"].values()) else "failed"}


def build_case(name, bundle_path, device, *, sample_count=10):
    from torch.utils.data import DataLoader
    from experiments.common.workload_training import _batch
    from experiments.physical_multitask import load_bundle, _configure_primary_math

    workload, bundle = load_bundle(bundle_path)
    _configure_primary_math(bundle)
    torch.manual_seed(20261009)
    if name == "mobilenet_v3":
        from torchvision.models import mobilenet_v3_large
        model = mobilenet_v3_large(weights=None, num_classes=10)
    elif name == "distilbert":
        from transformers import DistilBertConfig, DistilBertForSequenceClassification
        class TextModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                config = DistilBertConfig(num_labels=4)
                config._attn_implementation = "eager"
                self.model = DistilBertForSequenceClassification(config)
            def forward(self, input_ids, attention_mask):
                return {"logits": self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]}
        model = TextModel()
    elif name == "yolo26":
        from ultralytics import YOLO
        from ultralytics.utils import DEFAULT_CFG
        model = YOLO("yolo26n.yaml").model
        model.args = copy.copy(DEFAULT_CFG)
    else:
        model = workload.model_factory()
        model.load_state_dict(bundle["initial_model_state"])
    model = model.to(device).train()
    loader = DataLoader(workload.train_dataset, batch_size=1, shuffle=False, collate_fn=workload.collate_fn)
    batches = []
    for raw in loader:
        call, labels, _ = _batch(workload, raw, torch.device(device))
        if name == "yolo26":
            # Use real VOC images/boxes; convert COCO category IDs to YOLO's
            # contiguous 80-class order without dropping annotated instances.
            coco = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
            # The workload's RF-DETR adapter has already mapped VOC to COCO.
            inverse = {category: index for index, category in enumerate(coco)}
            classes = torch.tensor([inverse[int(v)] for v in labels[0]["labels"]], device=device).float().view(-1, 1)
            labels = {"batch_idx": torch.zeros(len(classes), device=device), "cls": classes,
                      "bboxes": labels[0]["boxes"], "img": call.args[0]}
        batches.append((call, labels))
        if len(batches) == sample_count:
            break
    if len(batches) < sample_count:
        raise ValueError(f"The training bundle has {len(batches)} batches; need {sample_count}")
    loss_fn = workload.task.make_adapter().loss
    if name == "yolo26":
        criterion = model.init_criterion()
        loss_fn = lambda outputs, labels: criterion(outputs, labels)[0].sum()
    return model, batches, loss_fn, {"model": name, "bundle": str(Path(bundle_path).resolve()),
        "bundle_sha256": hashlib.sha256(Path(bundle_path).read_bytes()).hexdigest(),
        "data": workload.task.name, "input_samples": len(batches),
        "input_bundle_role": bundle["role"],
        "input_source_indices": bundle.get("source_indices", [])[:len(batches)],
        "local_data_hash": bundle["local_data_hash"],
        "initial_model_hash": bundle["initial_model_hash"],
        "sample_selection": "first distinct training examples in recorded order",
        "initialization": "seeded_random" if name in ("mobilenet_v3", "distilbert", "yolo26") else "frozen_pretrained_bundle",
        "added_architecture": name in ("mobilenet_v3", "distilbert", "yolo26"), "manual_split_code": False,
        "executor_modified": False, "model_modes": dict(Counter(type(m).__name__ for m in model.modules() if not m.training))}


def run_suite(output, bundle_catalog, *, steps=10, timeout=3600):
    from concurrent.futures import ThreadPoolExecutor
    import importlib.metadata
    import os
    import subprocess
    import sys
    from experiments.common.evidence import begin_run, finish_run

    bundles = {job["model"]: job["bundle"] for job in json.loads(Path(bundle_catalog).read_text())["jobs"]}
    jobs = [(name, bundles[model], lane) for lane, assignments in enumerate((
        (("resnet50", "resnet50_pretrained"), ("mobilenet_v3", "resnet50_pretrained"),
         ("deeplab", "deeplabv3_resnet50"), ("yolo26", "rfdetr_nano")),
        (("bert", "bert_base"), ("distilbert", "bert_base"), ("rfdetr", "rfdetr_nano"))))
        for name, model in assignments]
    output = Path(output).resolve()
    config = {"jobs": jobs, "steps": steps, "timeout_sec": timeout, "rtol": 2e-4, "atol": 2e-6,
              "selection": "all admitted cuts; widest frontier in each graph third for multi-step checks",
              "ultralytics_version": importlib.metadata.version("ultralytics"),
              "scope": "same-device native-versus-split Adam, real-data representative batches, same-host production RPC"}
    manifest = begin_run(output, config)
    env = {**os.environ, "PYTHONPATH": str(output / "runtime_snapshot"), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1"}
    def lane(number):
        for name, bundle, assigned in jobs:
            if assigned != number:
                continue
            command = [sys.executable, "-u", "-m", "experiments.analysis.execution_coverage",
                       "--model", name, "--bundle", bundle, "--output", str(output / f"{name}.json"),
                       "--device", f"cuda:{number}", "--steps", str(steps)]
            print(f"START {name} cuda:{number}", flush=True)
            try:
                with (output / f"{name}.log").open("x") as stream:
                    child = subprocess.run(command, env=env, cwd=output / "runtime_snapshot",
                                           stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
                execution = {"model": name, "exit_code": child.returncode, "command": command}
            except subprocess.TimeoutExpired:
                execution = {"model": name, "status": "timeout", "timeout_sec": timeout, "command": command}
            save(output / f"{name}.execution.json", execution)
            print(f"END {name} {execution}", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lane, (0, 1)))
    rows = [json.loads((output / f"{name}.json").read_text()) if (output / f"{name}.json").exists()
            else {"model": name, "status": "failed", "reason": "no output"} for name, _, _ in jobs]
    result = {"schema": "splitfleet.execution-coverage-suite.v1", "source_identity": manifest["source_identity"],
              "config": config, "models": rows,
              "status": "completed" if all(row.get("status") == "completed" for row in rows) else "partial"}
    finish_run(output, result, validation={"status": result["status"], "failures_retained": True})
    print(json.dumps({"status": result["status"], "counts": {r["model"]: r.get("counts") for r in rows}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("resnet50", "bert", "rfdetr", "deeplab", "mobilenet_v3", "distilbert", "yolo26"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--suite-bundles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()
    if args.suite_bundles:
        run_suite(args.output, args.suite_bundles, steps=args.steps)
        return
    if not args.model or not args.bundle:
        parser.error("Select --suite-bundles or both --model and --bundle")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Keep the previous attempt; select a fresh output")
    torch.set_num_threads(1)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    started = time.monotonic()
    try:
        model, batches, loss_fn, metadata = build_case(args.model, args.bundle, args.device,
                                                     sample_count=max(args.steps, 3))
        record = validate(model, batches, loss_fn, device=args.device, steps=args.steps,
                          output=args.output, numeric=not args.structure_only)
        if args.model in ("mobilenet_v3", "distilbert") and not args.structure_only and record["pressure_boundaries"]:
            try:
                record["tail_batch"] = tail_batch_check(model, batches, loss_fn, record["pressure_boundaries"][1], args.device)
            except Exception as exc:
                record["tail_batch"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        record.update(metadata, elapsed_sec=time.monotonic() - started, status="completed")
        save(args.output, record)
        print(json.dumps({"model": args.model, "counts": record.get("counts"), "elapsed_sec": record["elapsed_sec"]}), flush=True)
    except Exception as exc:
        record = json.loads(args.output.read_text()) if args.output.exists() else {}
        record.update(model=args.model, status="failed", reason=f"{type(exc).__name__}: {exc}", elapsed_sec=time.monotonic()-started)
        save(args.output, record)
        raise


if __name__ == "__main__":
    main()
