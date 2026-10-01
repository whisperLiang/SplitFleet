"""Measured per-operation priors for every legal split on each physical device."""
from __future__ import annotations

import json
from pathlib import Path

from flwr.common import GetPropertiesIns

from .types import CandidateEstimate


class DeviceCostPrior:
    def __init__(self, path: str | Path, *, link_bytes_per_ms: float = 50_000.0,
                 link_base_ms: float = 1.0):
        data = json.loads(Path(path).read_text())
        if data.get("schema") != "splitfleet.device-cost-profiles.v1":
            raise ValueError("Invalid device cost profile schema")
        self.graph_signature = data["graph_signature"]
        self.profiles = data["profiles"]
        self.link_bytes_per_ms = float(link_bytes_per_ms)
        self.link_base_ms = float(link_base_ms)
        self.clients: dict[str, dict] = {}
        self._curves = {}
        for key, profile in self.profiles.items():
            if profile["graph_signature"] != self.graph_signature:
                raise ValueError(f"Graph mismatch in profile {key}")
            nodes = profile["nodes"]
            forward, backward = [0.0], [0.0]
            for row in nodes:
                forward.append(forward[-1] + max(float(row["forward_ms"]), 0.0))
                backward.append(backward[-1] + max(float(row["backward_ms"]), 0.0))
            self._curves[key] = forward, backward

    def bind_clients(self, clients, round_id: int) -> None:
        for client in clients:
            if client.cid in self.clients:
                continue
            result = client.get_properties(GetPropertiesIns(config={}), timeout=30,
                                           group_id=round_id)
            props = dict(result.properties)
            identity = str(props.get("logical_client_id", ""))
            if identity not in self.profiles:
                raise RuntimeError(f"Missing measured device profile for {identity!r}")
            self.clients[str(client.cid)] = props

    def client_context(self, cid):
        props = self.clients.get(str(cid), {})
        return {"num_batches": int(props.get("num_batches", 1)),
                "device_type": props.get("device_type", "cpu"),
                "batch_size": 1}

    def estimate(self, cid, candidate, original: CandidateEstimate) -> CandidateEstimate:
        props = self.clients.get(str(cid))
        if props is None:
            raise RuntimeError(f"Physical client {cid} has no bound device profile")
        key = str(props["logical_client_id"])
        forward, backward = self._curves[key]
        server_forward, server_backward = self._curves["server"]
        if candidate.graph_signature != self.graph_signature:
            raise RuntimeError("Cut catalog and device profiles describe different graphs")
        count = int(candidate.prefix_node_count)
        if not 0 <= count < len(forward) or len(forward) != len(server_forward):
            raise RuntimeError("Cut node count differs from device profile")
        transfer = self.link_base_ms + 2 * float(candidate.boundary_forward_bytes or 0) / self.link_bytes_per_ms
        # At present the tail runs synchronously on the coordinator's CUDA
        # stream; model one shared server lane in the placement solver.
        return CandidateEstimate(
            client_id=original.client_id, boundary=original.boundary,
            client_forward_mean_ms=forward[count],
            client_backward_mean_ms=backward[count],
            network_upload_mean_ms=transfer/2,
            network_download_mean_ms=transfer/2,
            server_service_mean_ms=(server_forward[-1]-server_forward[count]
                                    + server_backward[-1]-server_backward[count]),
            switch_mean_ms=original.switch_mean_ms,
            client_forward_uncertainty_ms=0, client_backward_uncertainty_ms=0,
            network_upload_uncertainty_ms=0, network_download_uncertainty_ms=0,
            server_service_uncertainty_ms=0, switch_uncertainty_ms=0,
            feasible=original.feasible,
            infeasible_reason=original.infeasible_reason,
        )
