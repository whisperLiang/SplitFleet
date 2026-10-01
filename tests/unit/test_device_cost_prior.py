from __future__ import annotations

import json

from splitfleet.server.placement.cosplit_ucb import (
    CoSplitUCBConfig, CoSplitUCBPlacementPolicy, SplitCandidateDescriptor,
    StaticCandidateProvider,
)
from splitfleet.server.placement.cosplit_ucb.device_cost import DeviceCostPrior


def candidate(name, position):
    return SplitCandidateDescriptor(
        boundary=name, split_id=name, graph_position_ratio=position/3,
        prefix_node_count=position, suffix_node_count=3-position,
        total_node_count=3, boundary_forward_bytes=100,
        boundary_gradient_bytes=100, boundary_tensor_count=1,
        prefix_parameter_bytes=4, suffix_parameter_bytes=4,
        client_memory_bytes=None, server_memory_bytes=None,
        trainable=True, feature_abi_id="abi", graph_signature="graph",
        framework_backend="torch",
    )


def test_physical_cpu_and_gpu_choose_independently_from_catalog(tmp_path):
    def profile(values):
        return {"graph_signature": "graph", "nodes": [
            {"node": str(i), "forward_ms": value, "backward_ms": 0.0}
            for i, value in enumerate(values)]}
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"schema": "splitfleet.device-cost-profiles.v1",
                                "graph_signature": "graph", "profiles": {
                                    "server": profile([0, 10, 10]),
                                    "cpu": profile([0, 50, 50]),
                                    "gpu": profile([0, 2, 2]),
                                }}))
    prior = DeviceCostPrior(path)
    prior.clients = {cid: {"logical_client_id": cid, "num_batches": 1} for cid in ("cpu", "gpu")}
    policy = CoSplitUCBPlacementPolicy(
        candidate_provider=StaticCandidateProvider([candidate("early", 1), candidate("late", 2)]),
        device_cost_prior=prior,
        config=CoSplitUCBConfig(safe_exploration_epsilon=0,
                                min_residence_rounds=0, max_explorations_per_round=0),
    )
    selected = policy.plan_round(round_id=1, client_ids=["cpu", "gpu"], training=True)
    assert selected == {"cpu": "early", "gpu": "late"}
