"""Physical runner command and result contracts without requiring SSH hosts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.physical_cosplit_ucb.orchestrate import (
    _validate_client_starts,
    build_commands,
)
from experiments.physical_cosplit_ucb.run import build_physical_parser, run_client, run_server
from experiments.physical_cosplit_ucb.validate_run import validate_result


def test_role_commands_cover_each_configured_host() -> None:
    deployment = {
        "server": {
            "python": "python", "workdir": "/srv/splitfleet",
            "bind": "0.0.0.0:18097", "connect": "10.0.0.1:18097",
        },
        "clients": [
            {"id": "win", "ssh": "win@example", "shell": "powershell",
             "python": ".\\venv\\python.exe", "workdir": "C:\\SplitFleet"},
            {"id": "linux", "ssh": "lin@example", "shell": "posix",
             "python": "./venv/bin/python", "workdir": "/srv/splitfleet"},
        ],
        "experiment": {"rounds": 2, "seed": 5},
    }
    server, clients = build_commands(deployment, run_id="check", output=Path("/tmp/check"))
    assert server[:4] == ["python", "-m", "experiments.physical_cosplit_ucb.run", "server"]
    assert server[server.index("--rounds") + 1] == "2"
    assert len(clients) == 2
    assert "--client-index' '0'" in clients[0][-1]
    assert "--client-index 1" in clients[1][-1]
    assert "--server 10.0.0.1:18097" in clients[1][-1]


def test_role_parser_exposes_runnable_server_and_client() -> None:
    parser = build_physical_parser()
    server = parser.parse_args(["server", "--run-id", "check", "--output", "/tmp/check"])
    client = parser.parse_args(["client", "--server", "localhost:18097",
                                "--client-id", "one", "--client-index", "0"])
    assert server.run is run_server
    assert client.run is run_client


def _valid_result() -> dict:
    fits = []
    suffix = []
    evaluations = []
    diagnostics = {}
    for round_id in (1, 2):
        diagnostics[str(round_id)] = {"assignment": {"cid-a": "cut-a", "cid-b": "cut-b"}}
        evaluations.append({"round_id": round_id, "num_examples": 10000})
        for index, cid in enumerate(("cid-a", "cid-b")):
            fits.append({
                "round_id": round_id, "flower_cid": cid,
                "logical_client_id": f"host-{index}", "num_examples": 6,
                "metrics": {"client_index": index, "boundary": f"cut-{'ab'[index]}"},
            })
            suffix.append({"round_id": round_id, "flower_cid": cid, "num_examples": 6})
    return {
        "schema": "splitfleet.physical-cosplit-ucb.v1",
        "placement_policy": "cosplit_ucb", "rounds": 2, "expected_clients": 2,
        "candidate_count": 2, "partition_hash": "hash", "history": "History()",
        "local_epochs": 2,
        "partition_manifest": {"partition_hash": "hash", "clients": {
            "0": {"num_examples": 3}, "1": {"num_examples": 3},
        }},
        "fit_records": fits, "server_fit_records": suffix,
        "evaluation_records": evaluations, "round_diagnostics": diagnostics,
        "fit_failures": [],
    }


def test_validator_requires_matching_boundaries_and_complete_local_epochs() -> None:
    result = _valid_result()
    assert validate_result(result)["valid"]
    result["fit_records"][0]["num_examples"] = 3
    result["fit_records"][1]["metrics"]["boundary"] = "wrong"
    report = validate_result(result)
    assert not report["valid"]
    assert any("complete local epochs" in value for value in report["errors"])
    assert any("different boundary" in value for value in report["errors"])


def test_orchestrator_rejects_client_partition_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(json.dumps({"partition_hash": "server-hash"}))
    (tmp_path / "run-1.node-1.log").write_text(
        json.dumps({"event": "client_start", "client_id": "node-1",
                    "partition_hash": "client-hash"}) + "\n"
    )
    with pytest.raises(RuntimeError, match="different data partition"):
        _validate_client_starts({"clients": [{"id": "node-1"}]}, run_id="run-1", output=run_dir)
