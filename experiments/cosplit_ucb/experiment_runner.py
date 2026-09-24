"""Deterministic CoSplit-UCB dynamic-resource smoke experiment.

The online method is the production implementation from
``splitfleet.server.placement.cosplit_ucb``. The oracle below is used only for
offline regret measurement.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from splitfleet.server.placement.cosplit_ucb import (
    CandidateEstimate,
    CoSplitUCBConfig,
    CoSplitUCBPlacementPolicy,
    GlobalPlacementSolver,
    PlacementFeedback,
    SplitCandidateDescriptor,
    StaticCandidateProvider,
)

from .config_utils import load_config


def _candidates(count: int = 5) -> list[SplitCandidateDescriptor]:
    if count < 5:
        raise ValueError("the smoke experiment requires at least five candidates")
    values = []
    for index in range(count):
        position = (index + 1) / (count + 1)
        prefix = index + 1
        values.append(
            SplitCandidateDescriptor(
                boundary=f"after:operation_{index + 1}",
                split_id=f"split-{index + 1}",
                graph_position_ratio=position,
                prefix_node_count=prefix,
                suffix_node_count=count - index,
                total_node_count=count + 1,
                boundary_forward_bytes=int((1.1 - position) * 1_000_000),
                boundary_gradient_bytes=int((1.1 - position) * 1_000_000),
                boundary_tensor_count=1 + index % 2,
                prefix_parameter_bytes=int(position * 8_000_000),
                suffix_parameter_bytes=int((1.0 - position) * 8_000_000),
                client_memory_bytes=None,
                server_memory_bytes=None,
                trainable=True,
                feature_abi_id=f"abi-{index + 1}",
                graph_signature="synthetic-operation-graph-v1",
                framework_backend="synthetic",
                runtime_backend="synthetic",
            )
        )
    return values


def _phase(round_id: int) -> str:
    if round_id <= 6:
        return "normal"
    if round_id <= 12:
        return "client_compute_contention"
    if round_id <= 18:
        return "poor_uplink"
    if round_id <= 24:
        return "server_contention"
    return "recovered"


def _actual_estimate(
    client_id: str,
    candidate: SplitCandidateDescriptor,
    *,
    round_id: int,
    previous: str | None,
) -> CandidateEstimate:
    phase = _phase(round_id)
    position = candidate.graph_position_ratio
    client_factor = 3.5 if phase == "client_compute_contention" and client_id == "client-0" else 1.0
    network_factor = 4.0 if phase == "poor_uplink" else 1.0
    server_factor = 3.5 if phase == "server_contention" else 1.0
    edge_forward = client_factor * (8.0 + 55.0 * position)
    edge_backward = client_factor * (5.0 + 35.0 * position)
    upload = network_factor * (4.0 + 35.0 * (1.0 - position))
    download = network_factor * (2.0 + 18.0 * (1.0 - position))
    server = server_factor * (4.0 + 65.0 * (1.0 - position))
    switch = 3.0 if previous is not None and previous != candidate.boundary else 0.0
    return CandidateEstimate(
        client_id=client_id,
        boundary=candidate.boundary,
        client_forward_mean_ms=edge_forward,
        client_backward_mean_ms=edge_backward,
        network_upload_mean_ms=upload,
        network_download_mean_ms=download,
        server_service_mean_ms=server,
        switch_mean_ms=switch,
        client_forward_uncertainty_ms=0,
        client_backward_uncertainty_ms=0,
        network_upload_uncertainty_ms=0,
        network_download_uncertainty_ms=0,
        server_service_uncertainty_ms=0,
        switch_uncertainty_ms=0,
    )


def _oracle(
    candidates: Sequence[SplitCandidateDescriptor],
    *,
    client_ids: Sequence[str],
    round_id: int,
    previous: Mapping[str, str],
    solver: GlobalPlacementSolver,
) -> tuple[dict[str, CandidateEstimate], float]:
    best_assignment: dict[str, CandidateEstimate] | None = None
    best_objective: tuple[float, float] | None = None
    for values in itertools.product(candidates, repeat=len(client_ids)):
        assignment = {
            client_id: _actual_estimate(
                client_id,
                candidate,
                round_id=round_id,
                previous=previous.get(client_id),
            )
            for client_id, candidate in zip(client_ids, values)
        }
        objective = solver.simulate(assignment).objective
        if best_objective is None or objective < best_objective:
            best_assignment = assignment
            best_objective = objective
    assert best_assignment is not None and best_objective is not None
    return best_assignment, best_objective[0]


def run_experiment(
    config: Mapping[str, Any],
    method: str,
    seed: int,
    run_id: str,
) -> Path:
    """Run 20--40 synthetic rounds and write online-learning diagnostics."""

    rounds = int(config.get("rounds", 30))
    if not 20 <= rounds <= 40:
        raise ValueError("CoSplit-UCB smoke rounds must be in [20, 40]")
    candidates = _candidates(int(config.get("candidate_count", 5)))
    num_clients = int(config.get("num_clients", 3))
    if not 3 <= num_clients <= 5:
        raise ValueError("the exhaustive smoke oracle supports 3–5 clients")
    if len(candidates) ** num_clients > 5000:
        raise ValueError("the exhaustive smoke oracle is limited to 5000 assignments")
    if config.get("client_profiles"):
        raise ValueError("client_profiles are not implemented by the synthetic smoke")
    client_ids = tuple(f"client-{index}" for index in range(num_clients))
    concurrency = int(config.get("server_concurrency", 1))
    solver = GlobalPlacementSolver(server_concurrency=concurrency)
    allowed = {"cosplit_ucb", "fixed_early", "fixed_middle", "fixed_late", "oracle"}
    if method not in allowed:
        raise ValueError(f"unsupported smoke method {method!r}; expected one of {sorted(allowed)}")
    policy = CoSplitUCBPlacementPolicy(
        candidate_provider=StaticCandidateProvider(candidates),
        solver=solver,
        config=CoSplitUCBConfig(
            discount_gamma=float(config.get("discount_gamma", 0.80)),
            safe_exploration_epsilon=float(config.get("safe_exploration_epsilon", 0.25)),
            max_explorations_per_round=int(config.get("max_explorations_per_round", 1)),
            min_residence_rounds=int(config.get("min_residence_rounds", 1)),
            forced_probe_interval=int(config.get("forced_probe_interval", 4)),
            target_scale_ms=float(config.get("target_scale_ms", 100.0)),
            server_concurrency=concurrency,
            seed=int(seed),
        ),
    )
    root = Path(config.get("results_root", "results/cosplit_ucb")) / run_id
    root.mkdir(parents=True, exist_ok=False)
    output = root / "round_metrics.jsonl"
    previous: dict[str, str] = {}
    cumulative_regret = 0.0
    seen: dict[str, set[str]] = {client_id: set() for client_id in client_ids}
    with output.open("w", encoding="utf-8") as stream:
        for round_id in range(1, rounds + 1):
            if method == "cosplit_ucb":
                selected = dict(
                    policy.plan_round(round_id=round_id, client_ids=client_ids, training=True)
                )
            elif method == "oracle":
                oracle_assignment, _ = _oracle(
                    candidates, client_ids=client_ids, round_id=round_id, previous=previous, solver=solver
                )
                selected = {cid: value.boundary for cid, value in oracle_assignment.items()}
            else:
                index = {"fixed_early": 0, "fixed_middle": len(candidates) // 2, "fixed_late": -1}[method]
                selected = {client_id: candidates[index].boundary for client_id in client_ids}
            selected_estimates = {
                client_id: _actual_estimate(
                    client_id,
                    next(value for value in candidates if value.boundary == boundary),
                    round_id=round_id,
                    previous=previous.get(client_id),
                )
                for client_id, boundary in selected.items()
            }
            simulation = solver.simulate(selected_estimates)
            _, oracle_makespan = _oracle(
                candidates, client_ids=client_ids, round_id=round_id, previous=previous, solver=solver
            )
            regret = simulation.max_client_completion_ms - oracle_makespan
            cumulative_regret += regret
            diagnostics = policy.round_diagnostics.get(round_id, {}) if method == "cosplit_ucb" else {}
            exploration_by_client = {
                row["client_id"]: row for row in diagnostics.get("exploration", [])
            }
            client_rows = []
            feedback = []
            for client_id in client_ids:
                estimate = selected_estimates[client_id]
                boundary = estimate.boundary
                seen[client_id].add(boundary)
                learned = diagnostics.get("estimates", {}).get(client_id, {}).get(boundary, {})
                timeline = simulation.timelines[client_id]
                client_rows.append(
                    {
                        "client_id": client_id,
                        "boundary": boundary,
                        "mean_estimate_ms": learned.get("mean_total_without_queue_ms"),
                        "uncertainty_ms": learned.get("uncertainty_total_ms"),
                        "lcb_ms": learned.get("lcb_ms"),
                        "ucb_ms": learned.get("ucb_ms"),
                        "component_mean_ms": learned.get("component_mean_ms"),
                        "component_uncertainty_ms": learned.get("component_uncertainty_ms"),
                        "execution_profile": learned.get("execution_profile"),
                        "exploration": client_id in exploration_by_client,
                        "exploration_reason": exploration_by_client.get(client_id, {}).get("reason"),
                        "actual_completion_ms": timeline.completion_ms,
                    }
                )
                feedback.append(
                    PlacementFeedback(
                        round_id=round_id,
                        client_id=client_id,
                        boundary=boundary,
                        client_forward_ms=estimate.client_forward_mean_ms,
                        client_backward_ms=estimate.client_backward_mean_ms,
                        network_upload_ms=estimate.network_upload_mean_ms,
                        network_download_ms=estimate.network_download_mean_ms,
                        server_service_ms=estimate.server_service_mean_ms,
                        switch_ms=estimate.switch_mean_ms,
                        completion_ms=timeline.completion_ms,
                        num_examples=32,
                        num_batches=1,
                    )
                )
            if method == "cosplit_ucb":
                policy.observe_round(round_id=round_id, feedback=feedback)
                diagnostics = policy.round_diagnostics[round_id]
                counts = diagnostics.get("learner_update_counts", {})
                for client_row in client_rows:
                    client_row.update(counts.get(client_row["client_id"], {}))
                    client_row["prediction_residual_ms"] = diagnostics.get(
                        "prediction_residual_ms", {}
                    ).get(client_row["client_id"])
            record = {
                "round_id": round_id,
                "phase": _phase(round_id),
                "method": method,
                "clients": client_rows,
                "baseline_makespan_ms": diagnostics.get("baseline_makespan_ms"),
                "baseline_upper_makespan_ms": diagnostics.get("baseline_upper_makespan_ms"),
                "safe_budget_ms": diagnostics.get("safe_budget_ms"),
                "predicted_final_makespan_ms": diagnostics.get("predicted_final_makespan_ms"),
                "selected_makespan_ms": simulation.max_client_completion_ms,
                "oracle_makespan_ms": oracle_makespan,
                "instantaneous_regret_ms": regret,
                "cumulative_dynamic_regret_ms": cumulative_regret,
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            previous = selected
    metadata = {
        "run_id": run_id,
        "method": method,
        "placement_policy": "cosplit_ucb" if method == "cosplit_ucb" else method,
        "seed": int(seed),
        "rounds": rounds,
        "num_clients": num_clients,
        "candidate_count": len(candidates),
        "server_concurrency": concurrency,
        "distinct_boundaries_per_client": {
            client_id: len(boundaries) for client_id, boundaries in seen.items()
        },
        "oracle_usage": "offline_evaluation_only",
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if method == "cosplit_ucb" and max(metadata["distinct_boundaries_per_client"].values()) < 2:
        raise RuntimeError("smoke did not exercise a real boundary change")
    return root


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--method", default="cosplit_ucb")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--run-id", default="smoke")
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else {}
    print(run_experiment(config, args.method, args.seed, args.run_id))


if __name__ == "__main__":
    main()
