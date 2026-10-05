"""Characterize every captured cut without replay or materialization.

Native capability, trainable parameters, and admission into the ownership-aware
training catalog are separate facts. These counts describe structural support;
they are not numerical output/gradient/update equivalence results.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import copy


def _torchlens_source_identity():
    import hashlib
    import importlib.util
    import json
    from pathlib import Path

    package = Path(importlib.util.find_spec("torchlens").origin).parent
    hashes = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(package.rglob("*.py"))}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def characterize_runtime(runtime, *, eligible_boundaries=None, catalog_rejections=None,
                         require_trainable_prefix=False, kinds=("before", "after")):
    from splitfleet.autosplit.torchlens_candidate import ParameterCountIndex
    from torchlens.split.errors import SplitRequestError, SplitUnsupportedError

    if not kinds or any(kind not in ("before", "after") for kind in kinds):
        raise ValueError("Declare before and/or after boundaries")
    if not runtime.request.features.training:
        raise ValueError("Characterize training support using a training-enabled capture")
    # A shallow view shares the capture and live state. Permissive analysis
    # exposes separate replay/training statuses without constructing segments.
    view = copy.copy(runtime)
    view.request = replace(runtime.request, validation="permissive")
    parameters = ParameterCountIndex.from_runtime(runtime)
    eligible = None if eligible_boundaries is None else set(eligible_boundaries)
    rejections = catalog_rejections or {}
    rows = []
    for site in view.split_points(diagnose=False).candidates:
        if site.kind not in kinds:
            continue
        boundary = f"{site.kind}:{site.node_id}"
        row = {"boundary": boundary, "replay_supported": None, "native_training_supported": None,
               "native_training_with_parameters_supported": None,
               "training_supported": None, "trainable_prefix_parameters": None,
               "trainable_suffix_parameters": None, "catalog_eligible": None if eligible is None else boundary in eligible,
               "refusal_reasons": [], "numeric_replay_performed": False}
        try:
            analysis = view.analyze(site.point)
        except (SplitRequestError, SplitUnsupportedError) as error:
            row.update(replay_supported=False, native_training_supported=False,
                       native_training_with_parameters_supported=False)
            row["refusal_reasons"].append(str(error))
        else:
            report = analysis.capability_report
            prefix = parameters.count(analysis.plan.prefix_node_ids, trainable_only=True)
            suffix = parameters.count(analysis.plan.suffix_node_ids, trainable_only=True)
            row.update(trainable_prefix_parameters=prefix, trainable_suffix_parameters=suffix,
                       graph_signature=analysis.graph_ir.graph_hash,
                       native_split_id=analysis.plan.split_id)
            if report is not None:
                replay = bool(report.replay.supported and report.preflight.supported)
                native_training = bool(replay and report.training.supported)
                row.update(replay_supported=replay, native_training_supported=native_training,
                           native_training_with_parameters_supported=bool(native_training and suffix > 0 and
                                                   (not require_trainable_prefix or prefix > 0)))
                row["refusal_reasons"].extend(report.unsupported_reasons)
            if not suffix:
                row["refusal_reasons"].append("suffix_not_trainable")
                row["native_training_with_parameters_supported"] = False
            if require_trainable_prefix and not prefix:
                row["refusal_reasons"].append("prefix_not_trainable")
                row["native_training_with_parameters_supported"] = False
        if boundary in rejections:
            reasons = rejections[boundary]
            row["refusal_reasons"].extend([reasons] if isinstance(reasons, str) else reasons)
        elif eligible is not None and boundary not in eligible:
            row["refusal_reasons"].append("not_in_ownership_aware_training_catalog")
        row["refusal_reasons"] = sorted(set(row["refusal_reasons"]))
        parameter_support = row["native_training_with_parameters_supported"]
        if row["catalog_eligible"] and parameter_support is not True:
            raise ValueError(f"Catalog contains a boundary without independently confirmed training support: {boundary}")
        row["training_supported"] = False if parameter_support is False or row["catalog_eligible"] is False else (
            True if parameter_support is True and row["catalog_eligible"] is True else None)
        rows.append(row)
    if not rows:
        raise ValueError("Capture has no requested boundaries")
    if eligible is not None and eligible - {row["boundary"] for row in rows}:
        raise ValueError("Catalog includes boundaries outside the characterized domain")
    total = len(rows)
    count = lambda key: sum(row[key] is True for row in rows)
    unknown = lambda key: sum(row[key] is None for row in rows)
    summary = {"enumerated_boundaries": total,
               "captured_graph_nodes": len(runtime.trace_graph.nodes),
               "captured_operation_nodes": sum(not any(getattr(node, flag, False) for flag in
                   ("is_input", "is_output", "is_buffer", "is_buffer_only_source", "is_param_source"))
                   for node in runtime.trace_graph.nodes),
               "replay_supported_boundaries": count("replay_supported"),
               "native_training_supported_boundaries": count("native_training_supported"),
               "native_training_with_parameters_boundaries": count("native_training_with_parameters_supported"),
               "training_supported_boundaries": count("training_supported"),
               "training_catalog_size": None if eligible is None else count("catalog_eligible"),
               "rejected_training_boundaries": None if eligible is None else total - count("catalog_eligible"),
               "unknown_replay_support": unknown("replay_supported"),
               "unknown_training_support": unknown("training_supported"),
               "ReplayCoverage": None if unknown("replay_supported") else count("replay_supported") / total,
               "TrainCoverage": None if eligible is None else count("catalog_eligible") / total,
               "numeric_replay_validations": 0,
               "verification": "structural native capability; no numerical equivalence claim"}
    reasons = Counter(reason for row in rows for reason in row["refusal_reasons"])
    return {"summary": summary, "candidates": rows,
            "rejection_summary": [{"reason": reason, "count": amount} for reason, amount in sorted(reasons.items())]}


def _collect(bundle_path, output, device):
    import csv
    import json
    import torch
    from torch.utils.data import DataLoader
    from experiments.physical_multitask import load_bundle, _make_candidate_provider, _configure_primary_math
    from experiments.common.workload_training import _batch
    from experiments.www2027_study import save_json
    from experiments.ablations.layer_level_candidates import LayerCandidateProvider

    torch.set_num_threads(1)
    workload, bundle = load_bundle(bundle_path)
    _configure_primary_math(bundle)
    torch.manual_seed(bundle["seed"])
    model = workload.model_factory().to(device).train()
    model.load_state_dict(bundle["initial_model_state"])
    raw = next(iter(DataLoader(workload.train_dataset, batch_size=bundle["batch_size"],
                               collate_fn=workload.collate_fn)))
    call, _, _ = _batch(workload, raw, torch.device(device))
    provider = _make_candidate_provider(model=model, sample_inputs=call, bundle=bundle)
    catalog = provider.get_candidates(training=True)
    runtime = provider._backends[True].runtime
    result = characterize_runtime(runtime, eligible_boundaries=[value.boundary for value in catalog],
        catalog_rejections=provider.catalog_diagnostics[True]["rejected_candidates"],
        require_trainable_prefix=provider.require_trainable_prefix)
    try:
        layers = LayerCandidateProvider.from_torchlens(provider)
        retained = layers.get_candidates(training=True)
        layer_domain = {"status": "supported", **layers.catalog_diagnostics[True],
                        "boundaries": [value.boundary for value in retained]}
    except ValueError as error:
        layer_domain = {"status": "unsupported", "reason": str(error), "catalog_size": None}
    save_json(output / "layer_candidate_domain.json", layer_domain)
    save_json(output / "partition_catalog.json", result["candidates"])
    save_json(output / "candidate_catalog.json", [{key: getattr(value, key) for key in (
        "boundary", "split_id", "graph_signature", "feature_abi_id", "graph_position_ratio",
        "prefix_node_count", "suffix_node_count", "trainable")} for value in catalog])
    for name, rows in (("partition_summary.csv", [{"model_id": bundle["model_id"], **result["summary"]}]),
                       ("rejection_summary.csv", result["rejection_summary"])):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["reason", "count"])
            writer.writeheader()
            writer.writerows(rows)
    with (output / "metrics.jsonl").open("w") as stream:
        for row in result["candidates"]:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    record = {"schema": "splitfleet.partition-characterization.v1", "status": "completed",
        "model_id": bundle["model_id"], "backend": provider.framework_backend, "device": device,
        "bundle_seed": bundle["seed"], "checkpoint_sha256": bundle.get("pretrain_checkpoint_sha256"),
        "summary": result["summary"], "layer_candidate_domain": layer_domain,
        "performance_measured": False, "best_layer_latency_ms": None, "best_operation_latency_ms": None}
    save_json(output / "result.json", record)
    save_json(output / "validation_report.json", {
        "complete_structural_characterization": True, "catalog_identities_confirmed": True,
        "training_catalog_size": len(catalog), "numeric_equivalence_tested": False,
        "layer_candidate_status": layer_domain["status"], "latency_advantage_tested": False})
    print(json.dumps(record, sort_keys=True))


def main():
    import argparse
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    from experiments.www2027_study import freeze_runtime, save_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output, bundle = args.output.resolve(), args.bundle.resolve()
    if args.frozen:
        config = json.loads((output / "config.json").read_text())
        if Path(__file__).resolve().parents[2] != output / "runtime_snapshot" or args.device != config["device"]:
            raise ValueError("Run the recorded characterization from its frozen source and device configuration")
        if (output / "result.json").exists():
            raise FileExistsError("Characterization already attempted; preserve this result and use a new directory")
        if _torchlens_source_identity() != config["torchlens_source_sha256"]:
            raise ValueError("TorchLens source differs from the frozen characterization dependency")
        if hashlib.sha256(bundle.read_bytes()).hexdigest() != config["bundle_sha256"]:
            raise ValueError("Bundle changed after the characterization source/config freeze")
        _collect(bundle, output, args.device)
        return
    output.mkdir(parents=True, exist_ok=False)
    runtime, manifest = freeze_runtime(Path(__file__).resolve().parents[2], output)
    save_json(output / "config.json", {"bundle": str(bundle),
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(), "device": args.device,
        "torchlens_source_sha256": _torchlens_source_identity(),
        "source_identity": manifest["source_identity"], "numeric_replay": False,
        "scope": "complete structural capability characterization; no measured latency advantage"})
    with (output / "stdout.log").open("w") as stream:
        try:
            completed = subprocess.run([sys.executable, "-m", "experiments.analysis.partition_coverage",
                "--bundle", str(bundle), "--output", str(output), "--device", args.device, "--frozen"],
                cwd=runtime, env={**os.environ, "PYTHONPATH": str(runtime)}, stdout=stream, stderr=subprocess.STDOUT,
                timeout=3600)
        except Exception as error:
            save_json(output / "result.json", {"status": "failed", "reason": str(error)})
            save_json(output / "validation_report.json", {"complete_structural_characterization": False})
            raise
    if completed.returncode:
        save_json(output / "result.json", {"status": "failed", "exit_code": completed.returncode})
        save_json(output / "validation_report.json", {"complete_structural_characterization": False})
        raise RuntimeError("Characterization failed; retain the attempt and inspect stdout.log")


if __name__ == "__main__":
    main()
