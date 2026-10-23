"""Independent minimum predicted round cost, with measured online feedback.

Each client owns its calibration anchors, latest per-cut computation observations
and affine link costs. Current network conditions replace stale bandwidth
estimates when available. There are no confidence bounds, shared learners or
joint placement solver. Executor concurrency is a separate resource setting.
"""

from __future__ import annotations

from dataclasses import asdict
import copy
import json
import math
from pathlib import Path

import numpy as np

from .config import GreedyConfig
from .feasibility import FeasibilityFilter
from .feedback import aggregate_feedback
from .profiles import graph_workload_features


class GreedyPlacementPolicy:
    name = "greedy"
    algorithm = "Greedy"
    batch_cost_statistic = "batch_mean_preferred_v1"
    uses_active_network_probes = True

    def __init__(self, *, candidate_provider, config=None, telemetry_provider=None,
                 feasibility_filter=None, runtime_context_provider=None):
        self.config = config or GreedyConfig()
        self.candidate_provider = candidate_provider
        self.telemetry_provider = telemetry_provider
        self.runtime_context_provider = runtime_context_provider
        self.server_load_provider = None
        self.feasibility_filter = feasibility_filter or FeasibilityFilter(
            failure_cooldown_rounds=self.config.failure_cooldown_rounds)
        self.state_download_bytes = 0
        self.client_concurrency = 1
        self.dispatch_order = ()
        self.profiles = None
        self.cost_tables = {}
        self.round_diagnostics = {}
        self.evaluation_diagnostics = {}
        self.decision_receipts = []
        self._catalog_by_boundary = {}
        self._round_assignments = {}
        self._observed_actions = set()
        self._last_boundary = {}
        self._state_restored = False
        self._saved_tables = {}
        self._pending_state = None
        if self.config.state_path and self.config.warm_start:
            path = Path(self.config.state_path)
            if path.exists():
                self._pending_state = json.loads(path.read_text())

    def _catalog(self):
        if not self._catalog_by_boundary:
            catalog = tuple(self.candidate_provider.get_candidates(training=True))
            if not catalog or len({c.boundary for c in catalog}) != len(catalog):
                raise ValueError("Greedy requires distinct admitted training candidates")
            if len({(c.graph_signature, c.framework_backend) for c in catalog}) != 1:
                raise ValueError("Candidate catalog must describe one graph and backend")
            self._catalog_by_boundary = {c.boundary: c for c in catalog}
            if self._pending_state is not None:
                pending, self._pending_state = self._pending_state, None
                self.load_state_dict(pending)
        return tuple(sorted(self._catalog_by_boundary.values(),
                            key=lambda c: (c.graph_position_ratio, c.boundary)))

    def bind_server_telemetry(self, provider):
        self.server_load_provider = provider

    def _server_telemetry(self):
        values = {}
        for provider in (self.telemetry_provider, self.runtime_context_provider,
                         self.server_load_provider):
            method = getattr(provider, "server_context", None)
            if method is not None:
                values.update(method() or {})
        return values

    def _client_telemetry(self, cid):
        provider = self.telemetry_provider
        method = getattr(provider, "client_context", None)
        if method is not None:
            values = dict(method(str(cid)) or {})
        elif callable(provider):
            values = dict(provider(str(cid)) or {})
        elif isinstance(provider, dict):
            values = dict(provider.get(str(cid), {}) or {})
        else:
            values = {}
        values.update(getattr(provider, "clients", {}).get(str(cid), {}))
        runtime = getattr(self.runtime_context_provider, "client_context", None)
        if runtime is not None:
            values.update(runtime(str(cid)) or {})
        return values

    def bind_clients(self, clients, round_id):
        method = getattr(self.telemetry_provider, "bind_clients", None)
        if method is not None:
            method(tuple(clients), round_id)

    def bind_evaluation_clients(self, clients, round_id):
        from .calibration import ClientTelemetry
        if self.telemetry_provider is None:
            self.telemetry_provider = ClientTelemetry(policy=self)
        method = getattr(self.telemetry_provider, "bind_evaluation_clients", None)
        if method is not None:
            method(tuple(clients), round_id)

    def prepare_calibration(self, inputs, targets, *, loss_fn, device,
                            parameters=None, optimizer_fn=None):
        from flwr.common import ndarrays_to_parameters, serde
        from splitfleet.backends.utils import adapter_for
        from splitfleet.common.model_state import model_state_hash
        from .calibration import (ClientTelemetry, CalibratedTelemetry,
                                  calibrate_split, calibration_boundaries, provider_handle)
        self._catalog()
        previous = self.telemetry_provider
        if previous is not None and not isinstance(previous, ClientTelemetry):
            return
        if isinstance(previous, CalibratedTelemetry) and previous.initialized:
            # Keep the calibration handshake aligned with the current global
            # model.  Existing clients retain their measured anchors, while a
            # client that was offline during bootstrap can be calibrated when
            # it first joins a later round without comparing its receipt to a
            # stale initial-model hash.
            if parameters is not None:
                model = self.candidate_provider.model
                adapter = adapter_for(model, inputs.args if inputs.args else inputs.kwargs)
                adapter.load_ndarrays(model, parameters)
                previous.parameters = serde.parameters_to_proto(
                    ndarrays_to_parameters(parameters)
                ).SerializeToString()
                previous.calibration_model_hash = model_state_hash(model, adapter)
            return
        if self._state_restored:
            self.telemetry_provider = ClientTelemetry(policy=self)
            if previous is not None:
                self.telemetry_provider.clients.update(previous.clients)
            return
        if targets is None or optimizer_fn is None:
            raise ValueError("Greedy calibration requires representative targets and the training optimizer factory")
        model = self.candidate_provider.model
        adapter = adapter_for(model, inputs.args if inputs.args else inputs.kwargs)
        if parameters is not None:
            adapter.load_ndarrays(model, parameters)
        arrays = parameters if parameters is not None else adapter.export_ndarrays(model)
        payload = serde.parameters_to_proto(ndarrays_to_parameters(arrays)).SerializeToString()
        receipt = calibrate_split(model, inputs, targets,
            boundaries=calibration_boundaries(self._catalog()),
            make_handle=lambda cut: provider_handle(self.candidate_provider, cut),
            loss_fn=loss_fn, device=device, source="server_shape_matched_current_deployment",
            optimizer_fn=optimizer_fn)
        telemetry = CalibratedTelemetry(initial_model_hash=receipt["model_hash_before"],
            server_receipt=receipt, parameters=payload, policy=self)
        if previous is not None:
            telemetry.clients.update(previous.clients)
        self.telemetry_provider = telemetry

    def _initialize_reference(self):
        if self.profiles is None:
            self.profiles = graph_workload_features(self.candidate_provider)

    def _live_network_context(self, cid, round_id):
        """Use current measured samples; never read network-shaping settings."""
        if not self.uses_active_network_probes:
            return None
        method = getattr(self.telemetry_provider, "network_context", None)
        return method(str(cid), int(round_id)) if method is not None else None

    @staticmethod
    def _affine_samples(samples, *, state=False):
        if state:
            small = [r["roundtrip_ms"] / 1000 for r in samples
                     if r["request_bytes"] + r["reply_bytes"] == 2 * 65536]
            large = [r["roundtrip_ms"] / 1000 for r in samples
                     if r["request_bytes"] + r["reply_bytes"] == 2097152 + 65536]
            delta = 2097152 - 65536
        else:
            small = [r["roundtrip_ms"] / 1000 for r in samples if r["payload_bytes"] == 65536]
            large = [r["roundtrip_ms"] / 1000 for r in samples if r["payload_bytes"] == 2097152]
            delta = 2 * (2097152 - 65536)
        if not small or not large or not all(math.isfinite(v) and v >= 0 for v in small + large):
            raise ValueError("Greedy requires measured small and large bootstrap RPC samples")
        low, high = float(np.mean(small)), float(np.mean(large))
        slope = max((high - low) / delta, 0.)
        return slope, max(low - 2 * 65536 * slope, 0.)

    def _table(self, cid):
        if cid in self.cost_tables:
            return self.cost_tables[cid]
        props = self._client_telemetry(cid)
        logical = str(props.get("logical_client_id", cid))
        if logical in self._saved_tables:
            table = copy.deepcopy(self._saved_tables[logical])
        else:
            receipt = getattr(self.telemetry_provider, "receipt", None)
            if receipt is None:
                raise ValueError("Greedy needs measured calibration before training placement")
            server = {r["boundary"]: r["local_tail_service_ms"] / 1000
                      for r in receipt["server_calibration"]["records"]}
            samples = [r for r in receipt["client_samples"] if r["logical_client_id"] == logical]
            if not samples:
                raise ValueError(f"No calibration anchors for {logical!r}")
            anchors = sorted((self.profiles[r["boundary"]]["prefix_mac_fraction"],
                (r["client_forward_ms"] + r["client_backward_ms"]) / 1000,
                server[r["boundary"]], r["boundary"]) for r in samples)
            link = [r for r in receipt["transport_bootstrap"]["samples"]
                    if r["logical_client_id"] == logical]
            exchange = [r for r in receipt["state_exchange_bootstrap"]["samples"]
                        if r["logical_client_id"] == logical]
            network_slope, network_intercept = self._affine_samples(link)
            state_slope, state_intercept = self._affine_samples(exchange, state=True)
            table = {"anchors": anchors, "observed_front": {}, "observed_tail": {},
                "network_seconds_per_byte": network_slope, "network_intercept_seconds": network_intercept,
                "state_seconds_per_byte": state_slope, "state_intercept_seconds": state_intercept}
        self.cost_tables[cid] = table
        return table

    def _front(self, cid, candidate):
        table = self._table(cid)
        if candidate.boundary in table["observed_front"]:
            return table["observed_front"][candidate.boundary]
        return self._interpolate(candidate, table, 1)

    def _interpolate(self, candidate, table, column):
        return float(np.interp(self.profiles[candidate.boundary]["prefix_mac_fraction"],
                              [r[0] for r in table["anchors"]], [r[column] for r in table["anchors"]]))

    def _payload_volumes(self, candidate, batch):
        sizes = candidate.metadata.get("boundary_forward_bytes_by_batch_size", {})
        activation = sizes.get(batch, sizes.get(str(batch)))
        if activation is None and candidate.boundary_forward_bytes is not None:
            activation = candidate.boundary_forward_bytes * batch
        upload = candidate.metadata.get("optimizer_prefix_parameter_bytes", candidate.prefix_parameter_bytes)
        if activation is None or upload is None:
            raise ValueError("Greedy needs boundary and owned-parameter byte estimates")
        gradient_sizes = candidate.metadata.get("boundary_gradient_bytes_by_batch_size", {})
        gradient = gradient_sizes.get(batch, gradient_sizes.get(str(batch)))
        if gradient is None:
            gradient = self.profiles[candidate.boundary]["gradient_volume_proxy_bytes_per_sample"] * batch
        return float(activation), float(gradient), float(self.state_download_bytes or 0), float(upload)

    def _volumes(self, candidate, batch):
        activation, gradient, state_download, state_upload = self._payload_volumes(candidate, batch)
        return activation + gradient, state_download + state_upload

    def _live_network_cost(self, candidate, *, batch, context):
        activation, gradient, state_download, state_upload = self._payload_volumes(candidate, batch)
        upload_bps = context["upload_mbps"] * 1_000_000
        download_bps = context["download_mbps"] * 1_000_000
        fit_transport = (activation * 8 / upload_bps + gradient * 8 / download_bps
                         + context["rtt_ms"] / 1000.)
        state_transport = (state_upload * 8 / upload_bps + state_download * 8 / download_bps
                           + context["rtt_ms"] / 1000.)
        # The measured small-RPC RTT already includes fixed RPC overhead.
        # Adding the bootstrap intercept would count that overhead twice.
        return fit_transport, state_transport

    def _score(self, cid, candidate, *, batch, count, network_context=None):
        table = self._table(cid)
        tail = table["observed_tail"].get(candidate.boundary)
        if tail is None:
            tail = self._interpolate(candidate, table, 2)
        if network_context is None:
            volume, exchange = self._volumes(candidate, batch)
            network = table["network_intercept_seconds"] + volume * table["network_seconds_per_byte"]
            state = table["state_intercept_seconds"] + exchange * table["state_seconds_per_byte"]
        else:
            network, state = self._live_network_cost(
                candidate, batch=batch, context=network_context)
        return count * (self._front(cid, candidate) + tail + network) + state

    def plan_round(self, *, round_id, client_ids, training):
        client_ids = sorted({str(cid) for cid in client_ids})
        if not client_ids:
            return {}
        key = (int(round_id), bool(training))
        if key in self._round_assignments:
            cached = self._round_assignments[key]
            if set(cached) != set(client_ids):
                raise ValueError("A round's selected client set must remain stable")
            return dict(cached)
        if not training:
            catalog = tuple(c for c in self.candidate_provider.get_candidates(training=False) if c.valid)
            if not catalog:
                raise ValueError("No valid evaluation candidates")
            by_boundary = {c.boundary: c for c in catalog}
            return {cid: (self._last_boundary[cid] if self._last_boundary.get(cid) in by_boundary
                          else min(catalog, key=lambda c: (abs(c.graph_position_ratio - .5), c.boundary)).boundary)
                    for cid in client_ids}
        self._catalog()
        self._initialize_reference()
        begin = getattr(self.server_load_provider, "begin_round", None)
        if begin is not None:
            begin(round_id)
        server = self._server_telemetry()
        assignment, estimates = {}, {}
        network_contexts = {}
        for cid in client_ids:
            props = self._client_telemetry(cid)
            batch, count = int(props.get("batch_size", 0)), int(props.get("num_batches", 0))
            if batch < 1 or count < 1:
                raise ValueError("Greedy needs positive batch_size and num_batches")
            feasible = [c for c in self._catalog() if self.feasibility_filter.check(c,
                client_id=cid, round_id=round_id, capabilities={**props, **server}).feasible]
            if not feasible:
                raise ValueError(f"Client {cid!r} has no feasible Greedy candidates")
            network_context = self._live_network_context(cid, round_id)
            if network_context is not None:
                network_contexts[cid] = network_context
            scores = [self._score(cid, c, batch=batch, count=count,
                                  network_context=network_context) for c in feasible]
            position = int(np.argmin(scores))
            chosen = feasible[position]
            assignment[cid] = chosen.boundary
            estimates[cid] = {c.boundary: {"round_seconds": float(score)} for c, score in zip(feasible, scores)}
            self.decision_receipts.append({"round_id": int(round_id),
                "logical_client_id": props.get("logical_client_id", cid), "algorithm": self.algorithm,
                "boundary": chosen.boundary, "candidate_count": len(feasible),
                "score_seconds": float(scores[position]),
                "predicted_frontend_seconds": self._front(cid, chosen) * count,
                "prefix_mac_fraction": self.profiles[chosen.boundary]["prefix_mac_fraction"],
                "fully_local_escape_available": False, "decision_granularity": "federated_round"})
        self._round_assignments[key] = assignment
        self.round_diagnostics[int(round_id)] = {"algorithm": self.name,
            "placement_objective": "predicted_solo_round_cost", "assignment": assignment,
            "estimates": estimates, "server_load": server,
            "network_contexts": network_contexts,
            "network_probes": {cid: receipt for cid, receipt in
                getattr(self.telemetry_provider, "round_probes", {}).get(int(round_id), {}).items()
                if cid in assignment},
            "latency_model_uses_live_network_context": bool(network_contexts),
            "latency_model_uses_live_load_features": False,
            "queue_wait_observed": server.get("queue_wait_mean_ms") is not None,
            "feedback_clients": [],
            "server_concurrency": self.config.server_concurrency}
        self._round_assignments = {k: v for k, v in self._round_assignments.items() if k[0] >= int(round_id) - 2}
        return dict(assignment)

    def observe_round(self, *, round_id, feedback):
        for observation in aggregate_feedback(feedback):
            if observation.round_id != int(round_id):
                raise ValueError("Feedback round does not match observe_round")
            key = (int(round_id), observation.client_id, observation.boundary)
            if key in self._observed_actions:
                continue
            planned = self._round_assignments.get((int(round_id), True), {})
            if planned.get(observation.client_id) != observation.boundary:
                raise ValueError("Feedback must match the client's executed round plan")
            if not observation.success:
                self.observe_failure(round_id=round_id, client_id=observation.client_id,
                                     boundary=observation.boundary)
                continue
            if observation.num_batches < 1:
                raise ValueError("Greedy feedback requires actual executed batches")
            cid = observation.client_id
            candidate = self._catalog_by_boundary[observation.boundary]
            table = self._table(cid)
            for field in ("client_forward_ms", "client_backward_ms", "server_service_ms"):
                value = getattr(observation, field)
                if value is None or not math.isfinite(value) or value < 0:
                    raise ValueError("Greedy requires finite measured computation costs")
            table["observed_front"][candidate.boundary] = (
                observation.client_forward_ms + observation.client_backward_ms) / 1000
            table["observed_tail"][candidate.boundary] = observation.server_service_ms / 1000
            volume, exchange = self._volumes(candidate, int(self._client_telemetry(cid)["batch_size"]))
            if volume > 0 and observation.network_roundtrip_ms is not None:
                table["network_seconds_per_byte"] = max((observation.network_roundtrip_ms / 1000
                    - table["network_intercept_seconds"]) / volume, 0.)
            if exchange > 0 and observation.state_exchange_ms is not None:
                table["state_seconds_per_byte"] = max((observation.state_exchange_ms / 1000
                    - table["state_intercept_seconds"]) / exchange, 0.)
            self.feasibility_filter.observe_memory(client_id=cid, boundary=candidate.boundary,
                client_peak_memory_mb=observation.client_peak_memory_mb,
                server_peak_memory_mb=observation.server_peak_memory_mb)
            self._last_boundary[cid] = candidate.boundary
            self._observed_actions.add(key)
            self.round_diagnostics[int(round_id)]["feedback_clients"].append(cid)
        self._save_state()

    def observe_failure(self, *, round_id, client_id, boundary=None, kind="runtime", reason=None):
        self.feasibility_filter.observe_failure(client_id=client_id,
            boundary=boundary or self._last_boundary.get(str(client_id)),
            round_id=round_id, kind=kind, reason=reason)

    def observe_evaluation(self, *, round_id, feedback):
        self.evaluation_diagnostics[int(round_id)] = {"observations": len(feedback)}

    def observe_round_wall_time(self, *, round_id, duration_ms):
        self.round_diagnostics.setdefault(int(round_id), {})["observed_round_wall_ms"] = float(duration_ms)

    def state_dict(self):
        tables = copy.deepcopy(self._saved_tables)
        for cid, table in self.cost_tables.items():
            logical = str(self._client_telemetry(cid).get("logical_client_id", cid))
            tables[logical] = copy.deepcopy(table)
        catalog = self._catalog()
        return {"schema": "splitfleet.greedy-state.v1", "algorithm": self.name,
            "graph_signature": catalog[0].graph_signature, "framework_backend": catalog[0].framework_backend,
            "server_concurrency": self.config.server_concurrency, "cost_tables": tables,
            "feasibility": self.feasibility_filter.state_dict()}

    def load_state_dict(self, state):
        catalog = self._catalog()
        expected = {"schema": "splitfleet.greedy-state.v1", "algorithm": self.name,
            "graph_signature": catalog[0].graph_signature, "framework_backend": catalog[0].framework_backend,
            "server_concurrency": self.config.server_concurrency}
        if any(state.get(k) != v for k, v in expected.items()):
            raise ValueError("Incompatible Greedy state identity")
        tables = copy.deepcopy(state["cost_tables"])
        for table in tables.values():
            for name in ("network_seconds_per_byte", "network_intercept_seconds",
                         "state_seconds_per_byte", "state_intercept_seconds"):
                if not math.isfinite(float(table[name])) or float(table[name]) < 0:
                    raise ValueError("Invalid persisted Greedy communication cost")
            if not table["anchors"]:
                raise ValueError("Persisted Greedy state requires calibration anchors")
        self._saved_tables = tables
        self.feasibility_filter.load_state_dict(state["feasibility"])
        self._state_restored = True

    def _save_state(self):
        if self.config.state_path:
            path = Path(self.config.state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(self.state_dict(), indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
