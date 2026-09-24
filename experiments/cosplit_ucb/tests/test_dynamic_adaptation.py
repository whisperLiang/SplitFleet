from __future__ import annotations

import json
from collections import Counter

import pytest

from experiments.cosplit_ucb.experiment_runner import run_experiment
from experiments.cosplit_ucb.validate_results import validate_run


def test_dynamic_feedback_changes_selection_distribution_and_emits_regret(tmp_path) -> None:
    run = run_experiment(
        {
            "rounds": 40,
            "results_root": str(tmp_path),
            "candidate_count": 5,
            "server_concurrency": 1,
        },
        "cosplit_ucb",
        233,
        "dynamic",
    )
    rows = [
        json.loads(line)
        for line in (run / "round_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    distributions = {
        phase: Counter(
            client["boundary"]
            for row in rows
            if row["phase"] == phase
            for client in row["clients"]
        )
        for phase in {row["phase"] for row in rows}
    }

    def clients_for(phase: str):
        return [
            client
            for row in rows
            if row["phase"] == phase
            for client in row["clients"]
        ]

    def mean_position(phase: str) -> float:
        values = clients_for(phase)
        return sum(
            int(client["boundary"].rsplit("_", 1)[1]) for client in values
        ) / len(values)

    def mean_component(selected_rows, component: str) -> float:
        values = [
            client["component_mean_ms"][component]
            for row in selected_rows
            for client in row["clients"]
        ]
        return sum(values) / len(values)

    # Phases alter only synthetic component observations. The production policy
    # sees no phase label or phase-to-cut rule, so distribution changes arise
    # from discounted feedback and global placement.
    assert len({tuple(sorted(value.items())) for value in distributions.values()}) > 1
    # A poorer uplink and a congested server both make later, lower-transfer /
    # lower-suffix-work candidates more common than during client contention.
    assert mean_position("poor_uplink") > mean_position("client_compute_contention")
    assert mean_position("server_contention") > mean_position("client_compute_contention")
    assert mean_component(
        [row for row in rows if row["phase"] == "poor_uplink"],
        "network_upload",
    ) > mean_component(
        [row for row in rows if row["phase"] == "client_compute_contention"],
        "network_upload",
    )
    assert mean_component(
        [row for row in rows if row["phase"] == "server_contention"],
        "server_service",
    ) > mean_component(
        [row for row in rows if row["phase"] == "poor_uplink"],
        "server_service",
    )

    recovered = [row for row in rows if row["phase"] == "recovered"]
    assert mean_component(recovered[-6:], "network_upload") < mean_component(
        recovered[:6], "network_upload"
    )
    assert mean_component(recovered[-6:], "server_service") < mean_component(
        recovered[:6], "server_service"
    )
    assert all(row["instantaneous_regret_ms"] >= -1e-7 for row in rows)
    assert rows[-1]["cumulative_dynamic_regret_ms"] >= 0
    assert validate_run(run)["valid"]


def test_smoke_client_count_is_effective_and_profile_config_is_rejected(tmp_path) -> None:
    run = run_experiment(
        {"rounds": 20, "num_clients": 4, "results_root": str(tmp_path)},
        "fixed_middle",
        233,
        "four-clients",
    )
    metadata = json.loads((run / "metadata.json").read_text())
    first = json.loads((run / "round_metrics.jsonl").read_text().splitlines()[0])
    assert metadata["num_clients"] == 4
    assert len(first["clients"]) == 4

    with pytest.raises(ValueError, match="client_profiles"):
        run_experiment(
            {"rounds": 20, "results_root": str(tmp_path), "client_profiles": {"weak": 3}},
            "cosplit_ucb",
            233,
            "invalid-profiles",
        )
