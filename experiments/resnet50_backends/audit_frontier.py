"""Audit actual native boundaries, including negotiated JAX resident weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def audit(args):
    if args.backend != "tensorflow":
        sys.modules["tensorflow"] = None
    from .common import Images, save_json
    from .worker import initialize, endpoints, execution, receipt
    from splitfleet.autosplit import AutoSplitSession
    from splitfleet.autosplit.state_ownership import state_ownership
    from splitfleet.backends.utils import adapter_for
    from splitfleet.runtime.native_frontier import minimal_frontier_node_ids
    from splitfleet.split_engine import graph_contract_for_runtime_handle
    from splitfleet.transport import encode_boundary
    from splitfleet.transport.split_wire import boundary_to_envelope
    from tests.integration.resnet50_reference.native import call_inputs, native_batch

    config, initial, model, sample, _ = initialize(SimpleNamespace(backend=args.backend,
        bundle=args.bundle, device=args.device, identity=None, index=0, command="server"))
    with execution(args.backend, args.device):
        inputs = call_inputs(args.backend, model, sample)
        handle = AutoSplitSession(device=args.device).prepare_runtime(model, inputs, trainable=True,
            batch_axes={"/args/1" if args.backend == "jax" else "/args/0": 0}, dynamic_batch=(1, 16))
        handle = handle.backend.repartition(endpoints(handle, args.backend)["stage2"]["boundary"])
        plan, graph = handle.runtime.plan, handle.runtime.trace_graph
        minimal = minimal_frontier_node_ids(graph, prefix_node_ids=plan.prefix_node_ids,
                                            suffix_node_ids=plan.suffix_node_ids)
        surplus = set(plan.boundary_node_ids) - set(minimal)
        contract = graph_contract_for_runtime_handle(handle)
        resident, residency_contract = {}, None
        if args.backend == "jax":
            from splitfleet.runtime.functional_sgd import resident_parameter_bindings, resident_contract
            ownership = state_ownership(handle, adapter_for(model, inputs).state_manifest(model).schema_hash)
            resident = resident_parameter_bindings(handle, ownership)
            residency_contract = resident_contract(handle, ownership, resident)
        data = Images(args.bundle / "validation.npz", config)
        report = {"status": "running", "backend": args.backend, "device": args.device,
            "environment": receipt(args.backend, model, args.device), "measurements": [],
            "required_fields": len(minimal), "native_fields": len(plan.boundary_node_ids),
            "surplus_node_ids": list(sorted(surplus)), "resident_parameter_fields": len(resident),
            "scope": "real CIFAR inputs, native forward and actual boundary codec; not an SGD or LAN timing check"}
        save_json(args.output, report)
        for size in (16, 12):
            images, _ = native_batch(args.backend, data.images(list(range(size))), data.labels[:size], args.device)
            boundary = handle.backend.run_prefix(*call_inputs(args.backend, model, images), training=True)
            envelope = boundary_to_envelope(boundary, round_id=1, client_id="frontier-audit", step_id="1",
                plan_id=handle.plan.plan_id, split_id=contract.split_id,
                canonical_graph_hash=contract.canonical_graph_hash,
                boundary_schema_hash=contract.boundary_schema_hash, model_version=1,
                resident_tensor_ids=set(resident), resident_state_contract=residency_contract)
            fields = [{"key": tensor.tensor_id, "shape": list(tensor.shape), "bytes": len(tensor.payload),
                       "surplus": plan.boundary_bindings[tensor.tensor_id] in surplus} for tensor in envelope.tensors]
            report["measurements"].append({"batch_size": size, "encoded_bytes": len(encode_boundary(envelope)),
                "tensor_bytes": sum(row["bytes"] for row in fields),
                "surplus_tensor_bytes": sum(row["bytes"] for row in fields if row["surplus"]), "fields": fields})
            save_json(args.output, report)
        report["status"] = "completed"
        save_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"environment", "measurements"}},
                     ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pytorch", "tensorflow", "paddle", "jax"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    audit(args)


if __name__ == "__main__":
    main()
