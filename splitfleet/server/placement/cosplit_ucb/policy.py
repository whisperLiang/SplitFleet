"""Production CoSplit-UCB round placement policy."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from .candidate_provider import CandidateProvider
from .config import CoSplitUCBConfig
from .context import ContextEncoder
from .exploration import ExplorationDecision, SafeExplorationController
from .feasibility import FeasibilityFilter
from .feedback import aggregate_feedback
from .learners import CandidateContexts, CooperativeLearners
from .solver import GlobalPlacementSolver
from .state import BanditStateStore
from .types import (
    CandidateEstimate,
    ExecutionProfileKey,
    PlacementFeedback,
    RuntimeTelemetryProvider,
    SplitCandidateDescriptor,
)


LOGGER = logging.getLogger(__name__)


class CoSplitUCBPlacementPolicy:
    """Joint online split placement with cooperative discounted cost learning."""

    def __init__(
        self,
        *,
        candidate_provider: CandidateProvider,
        context_encoder: ContextEncoder | None = None,
        feasibility_filter: FeasibilityFilter | None = None,
        learners: CooperativeLearners | None = None,
        solver: GlobalPlacementSolver | None = None,
        exploration_controller: SafeExplorationController | None = None,
        config: CoSplitUCBConfig | None = None,
        telemetry_provider: RuntimeTelemetryProvider | Mapping[str, Any] | None = None,
        state_store: BanditStateStore | None = None,
    ) -> None:
        self.config = config or CoSplitUCBConfig()
        self.candidate_provider = candidate_provider
        self.context_encoder = context_encoder or ContextEncoder()
        self.feasibility_filter = feasibility_filter or FeasibilityFilter(
            failure_cooldown_rounds=self.config.failure_cooldown_rounds
        )
        self.learners = learners or CooperativeLearners(self.config, self.context_encoder)
        self.solver = solver or GlobalPlacementSolver(
            server_concurrency=self.config.server_concurrency,
            max_coordinate_passes=self.config.max_coordinate_passes,
        )
        self.exploration_controller = exploration_controller or SafeExplorationController(
            epsilon=self.config.safe_exploration_epsilon,
            max_explorations_per_round=self.config.max_explorations_per_round,
            forced_probe_interval=self.config.forced_probe_interval,
            seed=self.config.seed,
        )
        self.telemetry_provider = telemetry_provider
        self.state_store = state_store or (
            BanditStateStore(self.config.state_path) if self.config.state_path else None
        )
        self._catalog_by_boundary: dict[str, SplitCandidateDescriptor] = {}
        self._round_assignments: dict[tuple[int, bool], dict[str, str]] = {}
        self._round_contexts: dict[
            tuple[int, bool, str, str], tuple[ExecutionProfileKey, CandidateContexts]
        ] = {}
        self._last_boundary: dict[str, str] = {}
        self._last_executed_boundary: dict[str, str] = {}
        self._last_switch_round: dict[str, int] = {}
        self._last_component_observation: dict[tuple[str, str], int] = {}
        self._observed_profiles: dict[str, ExecutionProfileKey] = {}
        self._last_num_batches: dict[str, int] = {}
        self._round_batch_counts: dict[tuple[int, bool], dict[str, int]] = {}
        self._observed_actions: set[tuple[int, str, str]] = set()
        self._pending_store_state = (
            self.state_store.load()
            if self.state_store is not None and self.config.warm_start
            else None
        )
        self.round_diagnostics: dict[int, dict[str, Any]] = {}
        self.evaluation_diagnostics: dict[int, dict[str, Any]] = {}

    def _catalog(self) -> tuple[SplitCandidateDescriptor, ...]:
        if not self._catalog_by_boundary:
            catalog = tuple(self.candidate_provider.get_candidates(training=True))
            if len({candidate.boundary for candidate in catalog}) != len(catalog):
                raise ValueError("candidate provider returned duplicate boundaries")
            if not catalog:
                raise ValueError("candidate provider returned no split candidates")
            if len({(candidate.graph_signature, candidate.framework_backend) for candidate in catalog}) != 1:
                raise ValueError("candidate catalog must describe one graph and framework backend")
            self._catalog_by_boundary = {candidate.boundary: candidate for candidate in catalog}
            LOGGER.info(
                "[CoSplitUCB] candidate_catalog_ready candidates=%d graph=%s",
                len(catalog),
                catalog[0].graph_signature[:12] if catalog else "",
            )
            if self._pending_store_state is not None:
                self.load_state_dict(self._pending_store_state)
                self._pending_store_state = None
        return tuple(
            sorted(
                self._catalog_by_boundary.values(),
                key=lambda value: (value.graph_position_ratio, value.boundary),
            )
        )

    def _client_telemetry(self, client_id: str) -> Mapping[str, Any]:
        provider = self.telemetry_provider
        if provider is None:
            return {}
        method = getattr(provider, "client_context", None)
        if method is not None:
            return dict(method(str(client_id)) or {})
        if callable(provider):
            return dict(provider(str(client_id)) or {})
        if isinstance(provider, Mapping):
            return dict(provider.get(str(client_id), {}) or {})
        return {}

    def _server_telemetry(self) -> Mapping[str, Any]:
        method = getattr(self.telemetry_provider, "server_context", None)
        return dict(method() or {}) if method is not None else {}

    def _execution_profile(
        self,
        client_id: str,
        telemetry: Mapping[str, Any],
        candidate: SplitCandidateDescriptor,
    ) -> ExecutionProfileKey:
        method = getattr(self.telemetry_provider, "execution_profile", None)
        if method is not None:
            provided = method(str(client_id))
            if provided is not None:
                return ExecutionProfileKey.from_value(provided)
        raw = telemetry.get("execution_profile")
        if raw is not None:
            return ExecutionProfileKey.from_value(raw)
        observed = self._observed_profiles.get(str(client_id))
        if observed is not None:
            return observed
        device_type = str(telemetry.get("device_type", "cpu"))
        accelerator = str(
            telemetry.get(
                "accelerator",
                "generic_cuda" if "cuda" in device_type.lower() else "generic_cpu",
            )
        )
        return ExecutionProfileKey(
            framework_backend=candidate.framework_backend or "unknown",
            runtime_backend=candidate.runtime_backend or "unknown",
            device_type=device_type,
            accelerator=accelerator,
            precision=str(telemetry.get("precision", "fp32")),
        )

    def _contexts(
        self,
        *,
        client_id: str,
        candidate: SplitCandidateDescriptor,
        client_telemetry: Mapping[str, Any],
        server_telemetry: Mapping[str, Any],
    ) -> CandidateContexts:
        previous = self._catalog_by_boundary.get(
            self._last_executed_boundary.get(client_id, "")
        )
        return CandidateContexts(
            edge=self.context_encoder.edge_context(
                candidate,
                client_telemetry,
                batch_size=client_telemetry.get("batch_size"),
            ),
            upload=self.context_encoder.network_context(
                candidate, client_telemetry, direction="upload"
            ),
            download=self.context_encoder.network_context(
                candidate, client_telemetry, direction="download"
            ),
            server=self.context_encoder.server_context(
                candidate,
                server_telemetry,
                max_concurrency=self.solver.server_concurrency,
            ),
            switch=self.context_encoder.switch_context(
                candidate,
                previous=previous,
                telemetry=client_telemetry,
            ),
        )

    def _estimate_round(
        self,
        *,
        round_id: int,
        client_ids: Sequence[str],
        training: bool,
        allowed_boundaries: set[str] | None = None,
    ) -> tuple[dict[str, list[CandidateEstimate]], set[str]]:
        catalog = self._catalog()
        server_telemetry = self._server_telemetry()
        estimates: dict[str, list[CandidateEstimate]] = {}
        residence_locked: set[str] = set()
        for client_id in sorted({str(value) for value in client_ids}):
            telemetry = self._client_telemetry(client_id)
            client_capabilities = telemetry.get("capabilities", telemetry)
            capabilities = {
                **(
                    client_capabilities
                    if isinstance(client_capabilities, Mapping)
                    else {}
                ),
                **server_telemetry,
            }
            client_estimates: list[CandidateEstimate] = []
            previous_boundary = self._last_boundary.get(client_id)
            residence_window_active = (
                previous_boundary is not None
                and int(round_id) - self._last_switch_round.get(client_id, int(round_id))
                < self.config.min_residence_rounds
            )
            hard_feasibility = {
                candidate.boundary: self.feasibility_filter.check(
                    candidate,
                    client_id=client_id,
                    round_id=round_id,
                    capabilities=capabilities,
                )
                for candidate in catalog
            }
            # Residence is a stability guard, never a reason to remain on an
            # OOM/ABI-invalid/otherwise hard-infeasible placement.
            lock_previous = bool(
                residence_window_active
                and previous_boundary is not None
                and (allowed_boundaries is None or previous_boundary in allowed_boundaries)
                and hard_feasibility.get(previous_boundary) is not None
                and hard_feasibility[previous_boundary].feasible
            )
            for candidate in catalog:
                if allowed_boundaries is not None and candidate.boundary not in allowed_boundaries:
                    continue
                feasibility = hard_feasibility[candidate.boundary]
                if lock_previous and candidate.boundary != previous_boundary and feasibility.feasible:
                    feasibility = type(feasibility)(False, "minimum_residence")
                contexts = self._contexts(
                    client_id=client_id,
                    candidate=candidate,
                    client_telemetry=telemetry,
                    server_telemetry=server_telemetry,
                )
                profile = self._execution_profile(client_id, telemetry, candidate)
                self._round_contexts[(int(round_id), bool(training), client_id, candidate.boundary)] = (
                    profile,
                    contexts,
                )
                client_estimates.append(
                    self.learners.predict(
                        client_id=client_id,
                        boundary=candidate.boundary,
                        profile=profile,
                        contexts=contexts,
                        feasible=feasibility.feasible,
                        infeasible_reason=feasibility.reason,
                    )
                )
            if lock_previous and any(
                value.feasible and value.boundary == previous_boundary for value in client_estimates
            ):
                residence_locked.add(client_id)
            estimates[client_id] = client_estimates
        return estimates, residence_locked

    def plan_round(
        self,
        *,
        round_id: int,
        client_ids: Sequence[str],
        training: bool,
    ) -> Mapping[str, str]:
        """Compute exactly one joint assignment for the selected clients."""

        key = (int(round_id), bool(training))
        normalized_ids = tuple(sorted({str(value) for value in client_ids}))
        cached = self._round_assignments.get(key)
        if cached is not None:
            missing = set(normalized_ids) - set(cached)
            if missing:
                raise ValueError(f"round {round_id} was already planned without clients {sorted(missing)}")
            return {client_id: cached[client_id] for client_id in normalized_ids}
        if not normalized_ids:
            self._round_assignments[key] = {}
            return {}
        batch_counts: dict[str, int] = {}
        batch_counts_known = True
        for client_id in normalized_ids:
            raw_count = self._client_telemetry(client_id).get("num_batches")
            if raw_count is None:
                raw_count = self._last_num_batches.get(client_id)
            if raw_count is None:
                batch_counts_known = False
                raw_count = 1
            batch_counts[client_id] = int(raw_count)
        if any(count < 1 for count in batch_counts.values()):
            raise ValueError("predicted batch counts must be positive")
        self._round_batch_counts[key] = batch_counts
        self._catalog()
        if training:
            advance_round = getattr(self.learners, "advance_round", None)
            if advance_round is not None:
                advance_round(int(round_id))
        allowed_boundaries = None
        if not training:
            allowed_boundaries = {
                candidate.boundary
                for candidate in self.candidate_provider.get_candidates(training=False)
            }
            if not allowed_boundaries.intersection(self._catalog_by_boundary):
                raise ValueError("training and evaluation graphs share no valid split boundary")
        estimates, residence_locked = self._estimate_round(
            round_id=int(round_id), client_ids=normalized_ids, training=bool(training),
            allowed_boundaries=allowed_boundaries,
        )
        LOGGER.info("[CoSplitUCB] round_context_ready round=%d clients=%d", round_id, len(normalized_ids))
        baseline = self.solver.solve(estimates, batch_counts=batch_counts)
        baseline_simulation = self.solver.simulate(baseline, batch_counts=batch_counts)
        LOGGER.info(
            "[CoSplitUCB] exploitation_assignment round=%d makespan_ms=%.3f",
            round_id,
            baseline_simulation.max_client_completion_ms,
        )
        decisions: list[ExplorationDecision] = []
        final = baseline
        if training and batch_counts_known:
            final, decisions = self.exploration_controller.apply(
                round_id=int(round_id),
                baseline=baseline,
                estimates=estimates,
                solver=self.solver,
                batch_counts=batch_counts,
                residence_locked=residence_locked,
                component_last_observation=self._last_component_observation,
            )
        assignment = {client_id: final[client_id].boundary for client_id in normalized_ids}
        if training:
            for client_id, boundary in assignment.items():
                previous = self._last_boundary.get(client_id)
                if previous != boundary:
                    self._last_switch_round[client_id] = int(round_id)
                self._last_boundary[client_id] = boundary
        predicted = self.solver.simulate(final, batch_counts=batch_counts)
        baseline_upper = self.solver.simulate(
            baseline, use_upper=True, batch_counts=batch_counts
        )
        self._round_assignments[key] = assignment
        diagnostic_store = self.round_diagnostics if training else self.evaluation_diagnostics
        diagnostic_store[int(round_id)] = {
            "training": bool(training),
            "predicted_batch_counts": dict(batch_counts),
            "exploration_suppressed_reason": (
                "unknown_batch_count" if training and not batch_counts_known else None
            ),
            "baseline_assignment": {cid: value.boundary for cid, value in baseline.items()},
            "assignment": dict(assignment),
            "baseline_makespan_ms": baseline_simulation.max_client_completion_ms,
            "baseline_upper_makespan_ms": baseline_upper.max_client_completion_ms,
            "safe_budget_ms": (
                (1.0 + self.config.safe_exploration_epsilon)
                * baseline_upper.max_client_completion_ms
            ),
            "predicted_final_makespan_ms": predicted.max_client_completion_ms,
            "predicted_client_completion_ms": {
                cid: timeline.completion_ms for cid, timeline in predicted.timelines.items()
            },
            "exploration": [decision.__dict__.copy() for decision in decisions],
            "estimates": {
                cid: {
                    value.boundary: {
                        "mean_total_without_queue_ms": value.mean_total_without_queue_ms,
                        "uncertainty_total_ms": value.uncertainty_total_ms,
                        "component_mean_ms": {
                            "client_forward": value.client_forward_mean_ms,
                            "client_backward": value.client_backward_mean_ms,
                            "network_upload": value.network_upload_mean_ms,
                            "network_download": value.network_download_mean_ms,
                            "server_service": value.server_service_mean_ms,
                            "switch": value.switch_mean_ms,
                        },
                        "component_uncertainty_ms": {
                            "client_forward": value.client_forward_uncertainty_ms,
                            "client_backward": value.client_backward_uncertainty_ms,
                            "network_upload": value.network_upload_uncertainty_ms,
                            "network_download": value.network_download_uncertainty_ms,
                            "server_service": value.server_service_uncertainty_ms,
                            "switch": value.switch_uncertainty_ms,
                        },
                        "lcb_ms": value.lcb_total_without_queue_ms,
                        "ucb_ms": value.ucb_total_without_queue_ms,
                        "execution_profile": self._round_contexts[
                            (int(round_id), bool(training), cid, value.boundary)
                        ][0].stable_id,
                        "feasible": value.feasible,
                        "infeasible_reason": value.infeasible_reason,
                    }
                    for value in values
                }
                for cid, values in estimates.items()
            },
        }
        if decisions:
            LOGGER.info("[CoSplitUCB] safe_exploration round=%d count=%d", round_id, len(decisions))
        LOGGER.info("[CoSplitUCB] assignment_final round=%d assignment=%s", round_id, assignment)
        return dict(assignment)

    def observe_evaluation(
        self, *, round_id: int, executions: Mapping[str, str]
    ) -> None:
        """Record only evaluation placements confirmed by successful clients."""

        planned = self._round_assignments.get((int(round_id), False), {})
        for client_id, boundary in executions.items():
            cid = str(client_id)
            if planned.get(cid) != str(boundary):
                raise ValueError("evaluation boundary does not match the round assignment")
        self._last_executed_boundary.update(
            {str(client_id): str(boundary) for client_id, boundary in executions.items()}
        )

    def observe_round(
        self,
        *,
        round_id: int,
        feedback: Sequence[PlacementFeedback],
    ) -> None:
        """Update component learners once per aggregate client/cut observation."""

        observations = aggregate_feedback(feedback)
        planned = self._round_assignments.get((int(round_id), True))
        if planned is None and observations:
            raise ValueError("cannot observe a training round before planning it")
        for value in observations:
            if value.round_id != int(round_id):
                raise ValueError("feedback round_id does not match observe_round")
            if planned is not None and planned.get(value.client_id) != value.boundary:
                raise ValueError("feedback boundary does not match the round assignment")
        for value in observations:
            action = (int(round_id), value.client_id, value.boundary)
            if action in self._observed_actions:
                continue
            if not value.success:
                self.observe_failure(
                    round_id=round_id,
                    client_id=value.client_id,
                    boundary=value.boundary,
                    kind="runtime",
                )
                self._observed_actions.add(action)
                continue
            retained = self._round_contexts.get((int(round_id), True, value.client_id, value.boundary))
            if retained is None:
                candidate = self._catalog_by_boundary.get(value.boundary)
                if candidate is None:
                    raise ValueError(f"feedback references unknown boundary {value.boundary!r}")
                telemetry = self._client_telemetry(value.client_id)
                contexts = self._contexts(
                    client_id=value.client_id,
                    candidate=candidate,
                    client_telemetry=telemetry,
                    server_telemetry=self._server_telemetry(),
                )
                profile = self._execution_profile(value.client_id, telemetry, candidate)
            else:
                profile, contexts = retained
            if value.execution_profile is not None:
                profile = ExecutionProfileKey.from_value(value.execution_profile)
                self._observed_profiles[value.client_id] = profile
            self.learners.edge.update(
                profile,
                contexts.edge,
                forward_ms=value.client_forward_ms,
                backward_ms=value.client_backward_ms,
                round_id=round_id,
            )
            self.learners.network.update(
                value.client_id,
                contexts.upload,
                contexts.download,
                upload_ms=value.network_upload_ms,
                download_ms=value.network_download_ms,
                round_id=round_id,
            )
            if value.server_service_ms is not None:
                self.learners.server.update(contexts.server, value.server_service_ms, round_id=round_id)
            if value.switch_ms is not None:
                self.learners.switch.update(contexts.switch, value.switch_ms, round_id=round_id)
            self.feasibility_filter.observe_memory(
                client_id=value.client_id,
                boundary=value.boundary,
                client_peak_memory_mb=value.client_peak_memory_mb,
                server_peak_memory_mb=value.server_peak_memory_mb,
            )
            for component, measurement in (
                ("network", value.network_upload_ms),
                ("network", value.network_download_ms),
            ):
                if measurement is not None:
                    self._last_component_observation[(value.client_id, component)] = int(round_id)
            if value.server_service_ms is not None:
                self._last_component_observation[("__global__", "server")] = int(round_id)
            explored = any(
                row.get("client_id") == value.client_id
                and row.get("boundary") == value.boundary
                for row in self.round_diagnostics.get(int(round_id), {}).get(
                    "exploration", []
                )
            )
            self.exploration_controller.record_observation(
                round_id=round_id,
                client_id=value.client_id,
                boundary=value.boundary,
                was_probe=explored,
            )
            self._last_executed_boundary[value.client_id] = value.boundary
            self._observed_actions.add(action)
            if value.num_batches > 0:
                self._last_num_batches[value.client_id] = int(value.num_batches)
            diagnostics = self.round_diagnostics.setdefault(int(round_id), {})
            actual = diagnostics.setdefault("actual_client_completion_ms", {})
            residuals = diagnostics.setdefault("prediction_residual_ms", {})
            if value.completion_ms is not None:
                actual[value.client_id] = value.completion_ms
                diagnostics["client_fit_duration_max_ms"] = max(actual.values())
                # Direct policy users have no server dispatch clock. The
                # production strategy replaces this proxy with wall time.
                diagnostics["actual_round_makespan_ms"] = max(actual.values())
                predicted = diagnostics.get("predicted_client_completion_ms", {}).get(
                    value.client_id
                )
                predicted_count = self._round_batch_counts.get((int(round_id), True), {}).get(value.client_id)
                if predicted is not None and predicted_count == value.num_batches:
                    residuals[value.client_id] = value.completion_ms - float(predicted)
            counts = diagnostics.setdefault("learner_update_counts", {})
            counts[value.client_id] = {
                "execution_profile": profile.stable_id,
                "group_update_count": self.learners.edge.update_count(profile),
                "network_update_count": self.learners.network.update_count(value.client_id),
                "server_update_count": self.learners.server.model.num_updates,
            }
            LOGGER.info(
                "[CoSplitUCB] feedback round=%d client=%s boundary=%s batches=%d",
                round_id,
                value.client_id,
                value.boundary,
                value.num_batches,
            )
        LOGGER.info("[CoSplitUCB] learner_update round=%d", round_id)
        self._prune_round_state(int(round_id))
        if self.state_store is not None:
            self.state_store.save(self.state_dict())

    def observe_round_wall_time(self, *, round_id: int, duration_ms: float) -> None:
        """Record the server-observed interval from fit dispatch to collection."""

        if duration_ms < 0:
            raise ValueError("round wall duration must be non-negative")
        diagnostics = self.round_diagnostics.setdefault(int(round_id), {})
        diagnostics["actual_round_wall_ms"] = float(duration_ms)
        diagnostics["actual_round_makespan_ms"] = float(duration_ms)

    def observe_failure(
        self,
        *,
        round_id: int,
        client_id: str,
        boundary: str | None = None,
        kind: str = "runtime",
        reason: Any = None,
    ) -> None:
        """Update feasibility only; failures never become synthetic latency."""

        self.feasibility_filter.observe_failure(
            round_id=round_id,
            client_id=client_id,
            boundary=boundary or self._last_executed_boundary.get(str(client_id)),
            kind=kind,
            reason=reason,
        )
        LOGGER.warning(
            "[CoSplitUCB] failure round=%d client=%s boundary=%s kind=%s",
            round_id,
            client_id,
            boundary,
            kind,
        )
        if self.state_store is not None:
            self.state_store.save(self.state_dict())

    def _prune_round_state(self, round_id: int) -> None:
        for key in [key for key in self._round_contexts if key[0] <= round_id]:
            self._round_contexts.pop(key, None)
        for key in [key for key in self._round_assignments if key[0] < round_id]:
            self._round_assignments.pop(key, None)
        for key in [key for key in self._round_batch_counts if key[0] < round_id]:
            self._round_batch_counts.pop(key, None)
        self._observed_actions = {
            action for action in self._observed_actions if action[0] >= round_id
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a versioned, JSON-compatible warm-start snapshot."""

        catalog = self._catalog()
        graph_signature = catalog[0].graph_signature
        backend = catalog[0].framework_backend
        payload = {
            "learners": self.learners.state_dict(),
            "feasibility": self.feasibility_filter.state_dict(),
            "exploration": self.exploration_controller.state_dict(),
            "last_boundary": dict(sorted(self._last_boundary.items())),
            "last_executed_boundary": dict(sorted(self._last_executed_boundary.items())),
            "last_switch_round": dict(sorted(self._last_switch_round.items())),
            "last_num_batches": dict(sorted(self._last_num_batches.items())),
            "observed_profiles": {
                cid: profile.stable_id
                for cid, profile in sorted(self._observed_profiles.items())
            },
            "last_component_observation": [
                {"client_id": cid, "component": component, "round_id": value}
                for (cid, component), value in sorted(self._last_component_observation.items())
            ],
            "observed_actions": [
                {"round_id": round_id, "client_id": cid, "boundary": boundary}
                for round_id, cid, boundary in sorted(self._observed_actions)
            ],
        }
        store = self.state_store or BanditStateStore()
        return store.envelope(
            payload,
            graph_signature=graph_signature,
            backend=backend,
            feature_schema_version=self.context_encoder.feature_schema_version,
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore learned statistics after validating graph/backend/schema identity."""

        catalog = self._catalog()
        store = self.state_store or BanditStateStore()
        payload = store.validate(
            state,
            graph_signature=catalog[0].graph_signature,
            backend=catalog[0].framework_backend,
            feature_schema_version=self.context_encoder.feature_schema_version,
        )
        self.learners.load_state_dict(payload["learners"])
        self.feasibility_filter.load_state_dict(payload.get("feasibility", {}))
        self.exploration_controller.load_state_dict(payload.get("exploration", {}))
        self._last_boundary = {
            str(key): str(value) for key, value in payload.get("last_boundary", {}).items()
        }
        self._last_executed_boundary = {
            str(key): str(value)
            for key, value in payload.get(
                "last_executed_boundary", payload.get("last_boundary", {})
            ).items()
        }
        self._last_switch_round = {
            str(key): int(value) for key, value in payload.get("last_switch_round", {}).items()
        }
        self._last_num_batches = {
            str(key): int(value) for key, value in payload.get("last_num_batches", {}).items()
        }
        self._observed_profiles = {
            str(cid): ExecutionProfileKey.from_value(value)
            for cid, value in payload.get("observed_profiles", {}).items()
        }
        self._last_component_observation = {
            (str(row["client_id"]), str(row["component"])): int(row["round_id"])
            for row in payload.get("last_component_observation", [])
        }
        self._observed_actions = {
            (int(row["round_id"]), str(row["client_id"]), str(row["boundary"]))
            for row in payload.get("observed_actions", [])
        }


__all__ = ["CoSplitUCBPlacementPolicy"]
