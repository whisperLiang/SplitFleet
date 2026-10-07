"""Measured split landscapes and explicitly labeled RQ2/RQ3 simulation.

Profiles execute real local split forward/backward, with state/RNG restoration.
They do not measure a distributed round. Simulation reuses production policies,
contexts and the lane simulator; it never reports simulated costs as physical.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

from experiments.common.evidence import begin_run, claim_frozen_run, finish_run, write_csv


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tensor_tree_hash(value):
    """Content identity independent of tensor storage, serialization and device."""
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(json.dumps(["tensor", str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item):
                digest.update(json.dumps(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(json.dumps([type(item).__name__, len(item)]).encode())
            for child in item: visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def collect_profile(output, config):
    import torch
    from splitfleet.tasks import ModelInputs
    from splitfleet.server.placement.cosplit_ucb import TorchLensCandidateProvider
    from splitfleet.server.placement.cosplit_ucb.calibration import calibrate_split, provider_handle
    from splitfleet.transport import encode_boundary, encode_gradients
    from splitfleet.transport.split_wire import gradients_to_envelope
    from splitfleet.validation.matrix import _wire_boundary
    from experiments.analysis.partition_coverage import characterize_runtime
    from experiments.ablations.layer_level_candidates import LayerCandidateProvider
    from experiments.www2027_study import save_json

    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    if config.get("bundle"):
        from torch.utils.data import DataLoader
        from experiments.physical_multitask import load_bundle, _make_candidate_provider, _configure_primary_math
        from experiments.common.workload_training import _batch
        if _hash(config["bundle"]) != config["bundle_sha256"]:
            raise ValueError("Bundle changed after source/config freeze")
        workload, bundle = load_bundle(config["bundle"])
        _configure_primary_math(bundle)
        model = workload.model_factory().to(device).train()
        model.load_state_dict(bundle["initial_model_state"])
        raw = next(iter(DataLoader(workload.train_dataset, batch_size=bundle["batch_size"], collate_fn=workload.collate_fn)))
        call, targets, _ = _batch(workload, raw, device)
        loss_fn = workload.task.make_adapter().loss
        provider = _make_candidate_provider(model=model, sample_inputs=call, bundle=bundle)
        identity = {key: bundle.get(key) for key in ("model_id", "seed", "pretrain_checkpoint_sha256", "partition_hash", "data_content_hash", "batch_size")}
        scope = "real model representative local split timing; no optimizer steps or distributed latency"
    else:
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
            torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)).to(device).train()
        # Use the same CPU random stream before transfer so paired CPU/GPU
        # profiles contain identical numerical inputs as well as graph shapes.
        call = ModelInputs(args=(torch.randn(2, 4).to(device),))
        targets = torch.randn(2, 3).to(device)
        loss_fn = torch.nn.functional.mse_loss
        provider = TorchLensCandidateProvider(model=model, sample_inputs=call, batch_axes={},
            dynamic_batch=(2, 2), require_trainable_prefix=True)
        identity = {"model_id": "download_free_mlp_fixture", "seed": config["seed"], "batch_size": 2}
        scope = "measured fixture split timing; synthetic inputs; no real workload performance claim"
    identity.update(initial_state_sha256=_tensor_tree_hash(dict(model.state_dict())),
                    sample_inputs_sha256=_tensor_tree_hash([call.args, dict(call.kwargs), targets]))
    catalog = provider.get_candidates(training=True)
    capture = provider._backends[True].runtime
    character = characterize_runtime(capture, eligible_boundaries=[value.boundary for value in catalog],
        catalog_rejections=provider.catalog_diagnostics[True]["rejected_candidates"],
        require_trainable_prefix=provider.require_trainable_prefix)
    try:
        layers = LayerCandidateProvider.from_torchlens(provider)
        retained = layers.get_candidates(training=True)
        layer_domain = {"status": "supported", **layers.catalog_diagnostics[True],
                        "boundaries": [value.boundary for value in retained]}
    except ValueError as exc:
        layer_domain = {"status": "unsupported", "reason": str(exc), "boundaries": []}
    samples, receipts, builds = {}, [], {}
    # Persist each completed repeat. A later refusal retains measured partial
    # receipts, and never fills in the unexecuted candidate costs.
    for repeat in range(config["repeats"]):
        def materialize(boundary):
            started = time.perf_counter()
            handle = provider_handle(provider, boundary)
            builds.setdefault(boundary, []).append((time.perf_counter() - started) * 1000)
            return handle
        receipt = calibrate_split(model, call, targets, boundaries=[value.boundary for value in catalog],
            make_handle=materialize, loss_fn=loss_fn, device=device,
            source="measured_offline_candidate_profile", optimizer_fn=None)
        receipts.append(receipt)
        save_json(output / "calibration_receipts.json", receipts)
        for row in receipt["records"]:
            samples.setdefault(row["boundary"], []).append(row)
    wire_rows = {}
    # Wire sizes are actually encoded, including multi-tensor frontiers; do not
    # infer a gradient payload size from the activation size.
    for candidate in catalog:
        sizes = {}
        def loss_and_sizes(boundary):
            handle = provider_handle(provider, boundary)
            local = handle.backend.run_prefix(*call.args, input_kwargs=dict(call.kwargs), training=True)
            remote, envelope = _wire_boundary(handle, local)
            loss, gradients = handle.backend.train_suffix(remote, targets, loss_fn=loss_fn)
            sizes.update(activation_wire_bytes=len(encode_boundary(envelope)),
                         gradient_wire_bytes=len(encode_gradients(gradients_to_envelope(envelope, gradients))),
                         gradient_tensor_bytes=sum(value.numel() * value.element_size() for value in gradients.values()))
            return handle
        # The calibration guard restores state/RNG after the auxiliary wire
        # probe as well. Its extra timings are not mixed into the timing sample.
        calibrate_split(model, call, targets, boundaries=[candidate.boundary], make_handle=loss_and_sizes,
            loss_fn=loss_fn, device=device, source="wire_size_probe", optimizer_fn=None)
        wire_rows[candidate.boundary] = sizes
    measured = []
    for candidate in catalog:
        rows = samples[candidate.boundary]
        measured.append({"boundary": candidate.boundary,
            "client_forward_ms": median(row["client_forward_ms"] for row in rows),
            "client_backward_ms": median(row["client_backward_ms"] for row in rows),
            "server_service_ms": median(row["local_tail_service_ms"] for row in rows),
            "materialization_ms": median(builds[candidate.boundary]), "repeats": len(rows),
            **wire_rows[candidate.boundary]})
    by_cut = {row["boundary"]: row for row in measured}
    cost = lambda cut: sum(by_cut[cut][key] for key in ("client_forward_ms", "client_backward_ms", "server_service_ms"))
    best = min(by_cut, key=lambda cut: (cost(cut), cut))
    layer_cuts = layer_domain["boundaries"]
    best_layer = min(layer_cuts, key=lambda cut: (cost(cut), cut)) if layer_cuts else None
    landscape = []
    for candidate in catalog:
        row = by_cut[candidate.boundary]
        landscape.append({**row, "graph_position_ratio": candidate.graph_position_ratio,
                          "local_forward_backward_ms": cost(candidate.boundary), "layer_boundary": candidate.boundary in layer_cuts})
    result = {"schema": "splitfleet.measured-candidate-profile.v1", "status": "completed", "provenance": "measured_local",
        "scope": scope, "device": config["device"], "identity": identity,
        "candidates": [asdict(value) for value in catalog], "costs": measured, "layer_domain": layer_domain,
        "partition_summary": character["summary"], "candidate_count": len(catalog), "layer_candidate_count": len(layer_cuts),
        "best_operation_boundary": best, "best_operation_local_ms": cost(best),
        "best_layer_boundary": best_layer, "best_layer_local_ms": cost(best_layer) if best_layer else None,
        "distributed_round_latency_measured": False, "optimizer_steps": 0,
        "fixed_boundaries": {label: min(catalog, key=lambda value: (abs(value.graph_position_ratio - ratio), value.boundary)).boundary
                             for label, ratio in (("Fixed-25", .25), ("Fixed-50", .5), ("Fixed-75", .75))}}
    save_json(output / "candidate_catalog.json", result["candidates"])
    save_json(output / "partition_catalog.json", character["candidates"])
    write_csv(output / "partition_summary.csv", [{"model_id": identity["model_id"], **character["summary"]}])
    write_csv(output / "rejection_summary.csv", character["rejection_summary"], ["reason", "count"])
    write_csv(output / "split_landscape.csv", landscape)
    finish_run(output, result, validation={"status": "passed", "complete_profile": True,
        "state_rng_restored": True, "optimizer_steps": 0, "distributed_latency_tested": False}, metrics=landscape)
    return result


def profile(output, *, device="cpu", bundle=None, seed=2027, repeats=3):
    if repeats < 1:
        raise ValueError("Profile repeats must be positive")
    output = output.resolve()
    config = {"device": device, "seed": seed, "repeats": repeats, "bundle": str(bundle.resolve()) if bundle else None,
              "bundle_sha256": _hash(bundle) if bundle else None}
    begin_run(output, config)
    command = [sys.executable, "-m", "experiments.placement_study", "profile", "--output", str(output), "--frozen"]
    with (output / "stdout.log").open("w") as log:
        try:
            completed = subprocess.run(command, cwd=output / "runtime_snapshot",
                env={**os.environ, "PYTHONPATH": str(output / "runtime_snapshot")}, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode: raise RuntimeError(f"Profile process exited {completed.returncode}; see stdout.log")
        except Exception as exc:
            finish_run(output, {"status": "failed", "reason": str(exc)}, validation={"status": "failed", "complete_profile": False})
            raise
    return json.loads((output / "result.json").read_text())


def validate_profiles(profiles):
    """Refuse incomplete costs, identity drift and synthetic measurements."""
    from splitfleet.server.placement.cosplit_ucb import SplitCandidateDescriptor
    if not profiles: raise ValueError("At least one client profile is required")
    reference = None
    for profile in profiles:
        if profile.get("provenance") != "measured_local" or profile.get("status") != "completed":
            raise ValueError("A completed measured-local profile is required; synthetic costs are not measurements")
        candidates = [SplitCandidateDescriptor(**value) for value in profile["candidates"]]
        identities = {value.boundary: (value.split_id, value.graph_signature, value.feature_abi_id) for value in candidates}
        if len(identities) != len(candidates) or not identities:
            raise ValueError("Profile candidate identities must be unique and nonempty")
        signature = (identities, profile.get("identity"))
        if reference is not None and signature != reference:
            raise ValueError("Paired profiles must share exact graph/ABI/model/checkpoint/input identities")
        reference = signature
        rows = {row["boundary"]: row for row in profile["costs"]}
        if len(rows) != len(profile["costs"]) or set(rows) != set(identities):
            raise ValueError("Every admitted candidate requires one complete recorded cost row")
        for row in rows.values():
            for key in ("client_forward_ms", "client_backward_ms", "server_service_ms", "materialization_ms",
                        "activation_wire_bytes", "gradient_wire_bytes"):
                value = row[key]
                if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid recorded cost: {key}")
                if key.endswith("bytes") and int(value) != value:
                    raise ValueError("Byte counts must be integer")
        if not set(profile["layer_domain"]["boundaries"]) <= set(identities):
            raise ValueError("Layer domain invented a candidate")
    return tuple(SplitCandidateDescriptor(**value) for value in profiles[0]["candidates"])


def simulate(config, profiles):
    from splitfleet.server.placement.cosplit_ucb import CoSplitUCBConfig, CoSplitUCBPlacementPolicy, PlacementFeedback, StaticCandidateProvider
    from splitfleet.server.placement.cosplit_ucb.types import CandidateEstimate
    from splitfleet.server.placement.cosplit_ucb.solver import GlobalPlacementSolver
    from experiments.baselines.independent_ucb import IndependentUCB
    from experiments.baselines.oracle_split import offline_oracle
    from experiments.ablations.no_cooperation import NoCooperation
    from experiments.ablations.no_joint_solver import NoJointSolver
    from experiments.ablations.no_nonstationary import no_nonstationary_updates
    from experiments.heterogeneity.scenarios import HeterogeneityScenario, Phase
    from experiments.heterogeneity.profiles import NetworkProfile, ResourceProfile
    from experiments.analysis.oracle_gap import oracle_gap, adaptation_rounds

    catalog = validate_profiles(profiles)
    clients = config["clients"]
    ids = [str(client["id"]) for client in clients]
    if not ids or len(set(ids)) != len(ids) or len(clients) != len(profiles):
        raise ValueError("One distinct client ID and profile is required per client")
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds): raise ValueError("Paired seeds must be unique")
    phases = [Phase(row["starts_round"], NetworkProfile(**row["network"]), ResourceProfile(**row.get("resources", {})))
              for row in config["scenario"]["phases"]]
    scenario = HeterogeneityScenario(config["scenario"]["name"], phases, provenance="simulation")
    rounds = config["rounds"]
    if rounds < 1 or phases[-1].starts_round > rounds:
        raise ValueError("Every scenario phase must occur within the round budget")
    if config.get("candidate_boundaries") is not None:
        domain = set(config["candidate_boundaries"])
        if not domain or not domain <= {value.boundary for value in catalog}:
            raise ValueError("Explicit oracle domain must be a nonempty catalog subset")
        catalog = tuple(value for value in catalog if value.boundary in domain)
    options = {value.boundary: value for value in catalog}
    costs = {cid: {row["boundary"]: row for row in profile["costs"]} for cid, profile in zip(ids, profiles)}
    server_index = config.get("server_profile_index", 0)
    if isinstance(server_index, bool) or not isinstance(server_index, int) or not 0 <= server_index < len(profiles):
        raise ValueError("Declare a valid common server profile index")
    server_costs = {row["boundary"]: row for row in profiles[server_index]["costs"]}
    counts = {cid: client["num_batches"] for cid, client in zip(ids, clients)}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts.values()):
        raise ValueError("Client batch counts must be positive integers")
    full = StaticCandidateProvider(catalog)
    layer_cuts = set(profiles[0]["layer_domain"]["boundaries"]) & set(options)
    methods = config.get("methods", ["Fixed-25", "Fixed-50", "Fixed-75", "Independent-UCB", "CoSplit-UCB", "Oracle",
                                     "Layer-only", "w/o Cooperation", "w/o Joint Solver", "w/o Non-stationary Update"])
    allowed = {"Fixed-25", "Fixed-50", "Fixed-75", "Independent-UCB", "CoSplit-UCB", "Oracle", "Layer-only",
               "w/o Cooperation", "w/o Joint Solver", "w/o Non-stationary Update"}
    if len(set(methods)) != len(methods) or not methods or not set(methods) <= allowed:
        raise ValueError("Unknown or duplicate comparison methods")
    if "Layer-only" in methods and not layer_cuts: raise ValueError("No layer boundary in the declared candidate domain")
    trace, summaries = [], []
    for seed in seeds:
        policy_config = CoSplitUCBConfig(**{**config.get("cosplit", {}), "seed": seed, "state_path": None})
        for method in methods:
            telemetry = {cid: {"num_batches": counts[cid], "batch_size": profiles[index]["identity"]["batch_size"],
                "device_type": clients[index].get("device_type", profiles[index]["device"].split(":")[0]),
                "accelerator": clients[index].get("accelerator", profiles[index]["device"]), "precision": "fp32"}
                for index, cid in enumerate(ids)}
            provider = StaticCandidateProvider(tuple(value for value in catalog if value.boundary in layer_cuts)) if method == "Layer-only" else full
            factories = {"Independent-UCB": IndependentUCB, "CoSplit-UCB": CoSplitUCBPlacementPolicy,
                "Layer-only": CoSplitUCBPlacementPolicy, "w/o Cooperation": NoCooperation,
                "w/o Joint Solver": NoJointSolver, "w/o Non-stationary Update": no_nonstationary_updates}
            policy = factories[method](candidate_provider=provider, config=policy_config, telemetry_provider=telemetry) if method in factories else None
            previous, round_rows, switches, bytes_total = {}, [], 0, 0
            for round_id in range(1, rounds + 1):
                phase = scenario.at(round_id)
                simulator = GlobalPlacementSolver(server_concurrency=phase.resources.server_concurrency)
                if policy:
                    targets = policy.clients.values() if isinstance(policy, IndependentUCB) else [policy]
                    for target in targets: target.solver.server_concurrency = phase.resources.server_concurrency
                links = {}
                for cid, client in zip(ids, clients):
                    multiplier = client.get("bandwidth_multiplier", 1)
                    links[cid] = NetworkProfile(phase.network.upload_mbps * multiplier,
                        phase.network.download_mbps * multiplier, phase.network.rtt_ms + client.get("rtt_offset_ms", 0))
                    telemetry[cid].update(links[cid].telemetry())
                estimates = {}
                for cid in ids:
                    estimates[cid] = []
                    for candidate in catalog:
                        row = costs[cid][candidate.boundary]
                        switch = row["materialization_ms"] if previous.get(cid) != candidate.boundary else 0
                        estimates[cid].append(CandidateEstimate(cid, candidate.boundary,
                            row["client_forward_ms"] * phase.resources.client_compute_multiplier,
                            row["client_backward_ms"] * phase.resources.client_compute_multiplier,
                            links[cid].transfer_ms(row["activation_wire_bytes"], direction="upload"),
                            links[cid].transfer_ms(row["gradient_wire_bytes"], direction="download"),
                            server_costs[candidate.boundary]["server_service_ms"] * phase.resources.server_service_multiplier, switch,
                            0, 0, 0, 0, 0, 0))
                oracle = offline_oracle(estimates, server_concurrency=phase.resources.server_concurrency,
                    batch_counts=counts, max_assignments=config.get("oracle_max_assignments", 1_000_000))
                if policy: selected = policy.plan_round(round_id=round_id, client_ids=ids, training=True)
                elif method == "Oracle": selected = {cid: value.boundary for cid, value in oracle.assignment.items()}
                else:
                    ratio = int(method.split("-")[1]) / 100
                    cut = min(catalog, key=lambda value: (abs(value.graph_position_ratio - ratio), value.boundary)).boundary
                    selected = {cid: cut for cid in ids}
                assignment = {cid: next(value for value in estimates[cid] if value.boundary == selected[cid]) for cid in ids}
                observed = simulator.simulate(assignment, batch_counts=counts)
                gap = oracle_gap(observed.max_client_completion_ms, oracle.simulation.max_client_completion_ms)
                round_rows.append({"round_id": round_id, "observed_ms": observed.max_client_completion_ms,
                                   "oracle_ms": oracle.simulation.max_client_completion_ms})
                feedback = []
                for cid in ids:
                    value, timeline = assignment[cid], observed.timelines[cid]
                    changed = cid in previous and previous[cid] != selected[cid]
                    switches += int(changed)
                    measurement = costs[cid][selected[cid]]
                    communication = counts[cid] * (measurement["activation_wire_bytes"] + measurement["gradient_wire_bytes"])
                    bytes_total += communication
                    diagnostic = (policy.clients[cid].round_diagnostics[round_id] if isinstance(policy, IndependentUCB)
                                  else policy.round_diagnostics[round_id]) if policy else {}
                    estimated = diagnostic.get("estimates", {}).get(cid, {}).get(selected[cid], {}).get("mean_total_without_queue_ms")
                    trace.append({"seed": seed, "method": method, "round_id": round_id, "client_id": cid,
                        "boundary": selected[cid], "split_id": options[selected[cid]].split_id,
                        "estimated_cost_ms": estimated, "observed_cost_ms": value.mean_total_without_queue_ms,
                        "round_makespan_ms": observed.max_client_completion_ms, "oracle_ms": oracle.simulation.max_client_completion_ms,
                        "oracle_gap": gap, "cut_switch": changed, "queue_ms": timeline.queue_ms,
                        "server_concurrency": phase.resources.server_concurrency, "network": asdict(links[cid]),
                        "resources": asdict(phase.resources), "boundary_application_bytes": communication,
                        "provenance": "simulation_from_measured_local_profile", "physical_shaping_applied": False})
                    feedback.append(PlacementFeedback(round_id=round_id, client_id=cid, boundary=selected[cid],
                        client_forward_ms=value.client_forward_mean_ms, client_backward_ms=value.client_backward_mean_ms,
                        network_upload_ms=value.network_upload_mean_ms, network_download_ms=value.network_download_mean_ms,
                        network_roundtrip_ms=value.network_upload_mean_ms + value.network_download_mean_ms,
                        server_service_ms=value.server_service_mean_ms, switch_ms=value.switch_mean_ms,
                        completion_ms=timeline.completion_ms, num_batches=counts[cid]))
                if policy: policy.observe_round(round_id=round_id, feedback=feedback)
                previous = dict(selected)
            recovery = [{"change_round": phase.starts_round, "adaptation_rounds": adaptation_rounds(round_rows,
                change_round=phase.starts_round, tolerance=config["adaptation_tolerance"], sustained_rounds=config["sustained_rounds"])}
                for phase in phases[1:]]
            summaries.append({"seed": seed, "method": method, "candidate_count": len(layer_cuts) if method == "Layer-only" else len(catalog),
                "mean_round_makespan_ms": sum(row["observed_ms"] for row in round_rows) / rounds,
                "mean_oracle_gap": sum(oracle_gap(row["observed_ms"], row["oracle_ms"]) for row in round_rows) / rounds,
                "cut_switches": switches, "boundary_application_bytes": bytes_total, "adaptation": recovery,
                "final_assignment": previous, "oracle_deployable": False})
    return {"schema": "splitfleet.placement-simulation.v1", "status": "completed", "provenance": "simulation_from_measured_local_profile",
        "physical_measurement": False, "oracle_scope": "exact declared candidate domain; lane simulation and each arm's previous placement",
        "oracle_is_deployable": False, "candidate_domain": list(options), "scenario": scenario.to_dict(),
        "server_profile_index": server_index, "server_cost_scope": "one common suffix device profile for all clients",
        "summaries": summaries, "placement_trace": trace, "missing_measurements": ["distributed round wall time", "model synchronization", "task quality"]}


def run_simulation(output, config):
    from experiments.www2027_study import save_json
    output = output.resolve()
    paths = [Path(client["profile"]).resolve() for client in config["clients"]]
    profiles = [json.loads(path.read_text()) for path in paths]
    frozen = {**config, "profile_sha256": {str(path): _hash(path) for path in paths},
              "profile_contents_sha256": hashlib.sha256(
                  json.dumps(profiles, sort_keys=True, allow_nan=False).encode()).hexdigest()}
    begin_run(output, frozen)
    save_json(output / "input_profiles.json", profiles)
    command = [sys.executable, "-m", "experiments.placement_study", "simulate",
               "--output", str(output), "--config", str(output / "config.json"), "--frozen"]
    try:
        with (output / "stdout.log").open("w") as log:
            completed = subprocess.run(command, cwd=output / "runtime_snapshot",
                env={**os.environ, "PYTHONPATH": str(output / "runtime_snapshot")},
                stdout=log, stderr=subprocess.STDOUT, timeout=config.get("timeout_sec", 1800))
        if completed.returncode:
            raise RuntimeError(f"Simulation process exited {completed.returncode}; see stdout.log")
    except Exception as exc:
        if json.loads((output / "result.json").read_text()).get("status") != "failed":
            finish_run(output, {"status": "failed", "reason": str(exc)}, validation={"status": "failed"})
        raise
    return json.loads((output / "result.json").read_text())


def complete_simulation(output, config, profiles):
    from experiments.www2027_study import save_json
    try:
        if hashlib.sha256(json.dumps(profiles, sort_keys=True, allow_nan=False).encode()).hexdigest() != config["profile_contents_sha256"]:
            raise ValueError("Frozen measured profile contents changed before simulation")
        result = simulate(config, profiles)
        save_json(output / "candidate_catalog.json", profiles[0]["candidates"])
        write_csv(output / "placement_summary.csv", [{**row, "adaptation": json.dumps(row["adaptation"]),
            "final_assignment": json.dumps(row["final_assignment"])} for row in result["summaries"]])
        with (output / "placement_trace.jsonl").open("w") as stream:
            for row in result["placement_trace"]: stream.write(json.dumps(row, sort_keys=True) + "\n")
        save_json(output / "communication.json", {"scope": "simulated boundary/gradient application buffers only",
            "model_state_bytes": None, "total_communication_bytes": None})
        finish_run(output, result, validation={"status": "passed", "profile_costs_complete": True,
            "simulation_labeled": True, "oracle_exact_in_declared_domain": True, "physical_performance_claim": False}, metrics=result["placement_trace"])
    except Exception as exc:
        finish_run(output, {"status": "failed", "reason": str(exc)}, validation={"status": "failed"})
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    collect = sub.add_parser("profile")
    collect.add_argument("--bundle", type=Path)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--device", default="cpu")
    collect.add_argument("--seed", type=int, default=2027)
    collect.add_argument("--repeats", type=int, default=3)
    collect.add_argument("--frozen", action="store_true", help=argparse.SUPPRESS)
    study = sub.add_parser("simulate")
    study.add_argument("--config", type=Path, required=True)
    study.add_argument("--output", type=Path, required=True)
    study.add_argument("--frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == "simulate":
        config = json.loads(args.config.read_text())
        if args.frozen:
            if args.config.resolve() != args.output.resolve() / "config.json":
                raise ValueError("Use the recorded frozen configuration for simulation execution")
            config = claim_frozen_run(args.output)
            complete_simulation(args.output, config, json.loads((args.output / "input_profiles.json").read_text()))
        else: run_simulation(args.output, config)
    elif args.frozen:
        collect_profile(args.output, claim_frozen_run(args.output))
    else: profile(args.output, device=args.device, bundle=args.bundle, seed=args.seed, repeats=args.repeats)


if __name__ == "__main__":
    main()
