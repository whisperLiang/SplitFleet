"""TorchLens autosplit-aware strategy facade built on top of PlainSlStrategy."""

from __future__ import annotations

from logging import ERROR, WARNING
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from flwr.common import log, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate

from splitfleet.autosplit import (
    AggregationPolicy,
    AutoSplitPlanner,
    AutoSplitSession,
    ExecutionSchedulePolicy,
    PartitionSelectionPolicy,
    PlacementConstraint,
    PlacementObjective,
    ReplicaScope,
    ReplicaScopePolicy,
    WorkerSpec,
    normalize_batch_window,
)
from splitfleet.server.client_selection import (
    ClientSelector,
    OortSelector,
    OortSelectorConfig,
    RandomSelector,
)
from splitfleet.autosplit.planner import validate_stage_counts
from splitfleet.autosplit.torchlens_contract import runtime_contract_digest, stable_json
from splitfleet.split_engine import graph_contract_for_runtime_handle
from splitfleet.split_engine.contracts import ModelVersionContract
from splitfleet.backends.utils import adapter_for
from splitfleet.common.constants import (
    AUTOSPLIT_BACKEND_CONFIG_KEY,
    AUTOSPLIT_BACKEND_VALUE_TORCHLENS,
    AUTOSPLIT_FRAMEWORK_BACKEND_CONFIG_KEY,
    AUTOSPLIT_BOUNDARY_CONFIG_KEY,
    AUTOSPLIT_BOUNDARY_TENSOR_LABELS_CONFIG_KEY,
    AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY,
    AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY,
    AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY,
    AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_GRAPH_CONTRACT_DIGEST_CONFIG_KEY,
    AUTOSPLIT_MODEL_VERSION_CONFIG_KEY,
    AUTOSPLIT_MODEL_VERSION_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_MODE_CONFIG_KEY,
    AUTOSPLIT_PLAN_ID_CONFIG_KEY,
    AUTOSPLIT_SPLIT_ID_CONFIG_KEY,
    AUTOSPLIT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_BACKEND_VALUE_TORCHLENS_NATIVE,
    AUTOSPLIT_RUNTIME_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_CONTRACT_DIGEST_CONFIG_KEY,
    AUTOSPLIT_TORCHLENS_VERSION_CONFIG_KEY,
    AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY,
)
from splitfleet.server.server_model.autosplit_tail_server_model import AutoSplitTailServerModel
from splitfleet.server.server_model.autosplit_server_model import AutoSplitServerModel
from splitfleet.server.strategy.plain_strategy import PlainSlStrategy


TIED_WEIGHT_UPDATE_MODES = ("reject", "additive_sgd")


def _coerce_replica_scope_policy(
    replica_scope_policy: Optional[ReplicaScopePolicy | ReplicaScope | str],
) -> ReplicaScopePolicy:
    if replica_scope_policy is None:
        return ReplicaScopePolicy(replica_scope=ReplicaScope.SHARED)
    if isinstance(replica_scope_policy, ReplicaScopePolicy):
        return replica_scope_policy
    if isinstance(replica_scope_policy, ReplicaScope):
        return ReplicaScopePolicy(replica_scope=replica_scope_policy)
    normalized = replica_scope_policy.strip().lower().replace("-", "_")
    return ReplicaScopePolicy(replica_scope=ReplicaScope(normalized))


def _coerce_aggregation_policy(
    aggregation_policy: Optional[AggregationPolicy | str],
) -> AggregationPolicy:
    if aggregation_policy is None:
        return AggregationPolicy()
    if isinstance(aggregation_policy, AggregationPolicy):
        return aggregation_policy
    return AggregationPolicy(name=aggregation_policy)


class AutoSplitStrategy(PlainSlStrategy):
    """Strategy that computes and propagates TorchLens split metadata."""

    uses_stage_runtime = True

    def __init__(
        self,
        *,
        model,
        sample_inputs,
        sample_kwargs: Optional[Dict[str, Any]] = None,
        worker_specs: Optional[Sequence[WorkerSpec]] = None,
        autosplit_session: Optional[AutoSplitSession] = None,
        planner: Optional[AutoSplitPlanner] = None,
        constraints: Optional[PlacementConstraint] = None,
        objective: Optional[PlacementObjective] = None,
        partition_selection_policy: Optional[PartitionSelectionPolicy] = None,
        replica_scope_policy: Optional[ReplicaScopePolicy | ReplicaScope | str] = None,
        execution_schedule_policy: Optional[ExecutionSchedulePolicy] = None,
        aggregation_policy: Optional[AggregationPolicy | str] = None,
        preferred_stage_count: Optional[int] = 2,
        client_stage_count: int = 1,
        boundary: str = "50%",
        dynamic_batch: tuple[int, int] | list[int] | None = None,
        trace_batch_mode: Optional[str] = None,
        mode: str = "generated_eager",
        loss_fn=None,
        optimizer_fn=None,
        runtime_device: str = "cpu",
        client_placement_fn: Optional[Callable[[int, str, bool], Any]] = None,
        tied_weight_update_mode: str = "reject",
        client_selection: str | None = "flower_default",
        client_selector: Optional[ClientSelector] = None,
        oort_config: Optional[OortSelectorConfig] = None,
        **kwargs,
    ) -> None:
        if sample_kwargs:
            raise ValueError("TorchLens autosplit backend accepts positional model inputs only.")
        validate_stage_counts(
            preferred_stage_count=preferred_stage_count,
            client_stage_count=client_stage_count,
        )
        self.model = model
        self.sample_inputs = sample_inputs
        self.backend_adapter = adapter_for(model, sample_inputs)
        self.sample_kwargs = {}
        self.worker_specs = list(worker_specs or [WorkerSpec(worker_id="coordinator", device="cpu")])
        self.autosplit_session = autosplit_session or AutoSplitSession(
            planner=planner or AutoSplitPlanner()
        )
        self.constraints = constraints or PlacementConstraint()
        self.objective = objective or PlacementObjective()
        self.partition_selection_policy = partition_selection_policy or PartitionSelectionPolicy(
            preferred_stage_count=preferred_stage_count
        )
        self.aggregation_policy = _coerce_aggregation_policy(aggregation_policy)
        self.replica_scope_policy = _coerce_replica_scope_policy(replica_scope_policy)
        if (
            replica_scope_policy is None
            and (
                self.aggregation_policy.requires_per_client_replica_scope()
                or client_placement_fn is not None
            )
        ):
            self.replica_scope_policy = ReplicaScopePolicy(replica_scope=ReplicaScope.PER_CLIENT)
        if (
            self.aggregation_policy.requires_per_client_replica_scope()
            and self.replica_scope_policy.resolve() != ReplicaScope.PER_CLIENT
        ):
            raise ValueError(
                "SplitFed aggregation requires `ReplicaScope.PER_CLIENT` server replicas."
            )
        if (
            client_placement_fn is not None
            and self.replica_scope_policy.resolve() != ReplicaScope.PER_CLIENT
        ):
            raise ValueError(
                "Per-client placement requires `ReplicaScope.PER_CLIENT` server replicas."
            )
        self.execution_schedule_policy = execution_schedule_policy or ExecutionSchedulePolicy()
        self.client_stage_count = int(client_stage_count)
        self.boundary = boundary
        # The dynamic batch window is part of the feature ABI, so it is planned
        # once here and broadcast to every device instead of being re-derived
        # from each client's own sample batch.
        self.dynamic_batch = normalize_batch_window(dynamic_batch)
        self.trace_batch_mode = str(trace_batch_mode) if trace_batch_mode else None
        self.mode = mode
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.runtime_device = runtime_device
        self.client_placement_fn = client_placement_fn
        normalized_tied_mode = str(tied_weight_update_mode).strip().lower()
        if normalized_tied_mode not in TIED_WEIGHT_UPDATE_MODES:
            raise ValueError(
                f"Unsupported tied_weight_update_mode {tied_weight_update_mode!r}; "
                f"expected one of {TIED_WEIGHT_UPDATE_MODES}."
            )
        self.tied_weight_update_mode = normalized_tied_mode
        self._placement_plan = None
        self._placement_plans: dict[str, Any] = {}
        self._client_placement_cache: dict[tuple[int, str, bool], Any] = {}
        self._autosplit_config_cache: dict[tuple[str, bool], Dict[str, Any]] = {}
        self._round_initial_client_states: dict[int, list[np.ndarray]] = {}
        self._round_initial_server_states: dict[int, list[np.ndarray]] = {}
        self._pending_client_updates: dict[
            int, dict[str, tuple[list[np.ndarray], float]]
        ] = {}
        self._reassembled_client_states: dict[int, list[np.ndarray]] = {}
        self._logical_state_names: list[str] | None = None
        self._logical_tied_groups: tuple[tuple[int, ...], ...] | None = None
        self._runtime_manager = None
        self.client_selector = self._make_client_selector(
            client_selection=client_selection,
            client_selector=client_selector,
            oort_config=oort_config,
        )
        self._last_selection_result = None

        init_server_model_fn = kwargs.pop("init_server_model_fn", None) or self._make_server_model
        super().__init__(
            init_server_model_fn=init_server_model_fn,
            common_server_model=not self.replica_scope_policy.requires_independent_replicas(),
            process_clients_as_batch=self.execution_schedule_policy.should_batch(),
            **kwargs,
        )

    @staticmethod
    def _make_client_selector(
        *,
        client_selection: str | None,
        client_selector: Optional[ClientSelector],
        oort_config: Optional[OortSelectorConfig],
    ) -> Optional[ClientSelector]:
        if client_selector is not None:
            return client_selector
        if client_selection is None:
            return None
        normalized = str(client_selection).strip().lower()
        if normalized in ("flower_default", "default"):
            return None
        if normalized == "oort":
            return OortSelector(oort_config or OortSelectorConfig())
        if normalized == "random":
            seed = oort_config.seed if oort_config is not None else 233
            return RandomSelector(seed=seed)
        raise ValueError(f"Unsupported client_selection: {client_selection!r}")

    def bind_stage_runtime_manager(self, runtime_manager) -> None:
        self._runtime_manager = runtime_manager
        if self._placement_plan is not None:
            runtime_manager.set_placement_plan(self._placement_plan)
        for placement in self._placement_plans.values():
            if placement is not self._placement_plan:
                runtime_manager.register_placement_plan(placement)

    def initialize_parameters(self, client_manager):
        _ = client_manager
        return ndarrays_to_parameters(
            self.backend_adapter.export_ndarrays(self.model)
        )

    def initialize_server_parameters(self):
        return self.backend_adapter.export_ndarrays(self.model)

    def get_or_create_placement_plan(self, boundary: str | None = None):
        requested_boundary = str(boundary or self.boundary)
        placement = self._placement_plans.get(requested_boundary)
        if placement is None:
            placement = self.autosplit_session.plan(
                self.model,
                self.sample_inputs,
                worker_specs=self.worker_specs,
                constraints=self.constraints,
                objective=self.objective,
                preferred_stage_count=self.partition_selection_policy.preferred_stage_count,
                client_stage_count=self.client_stage_count,
                model_name=self.model.__class__.__name__,
                boundary=requested_boundary,
                mode=self.mode,
                trainable=True,
                dynamic_batch=self.dynamic_batch,
                trace_batch_mode=self.trace_batch_mode,
            )
            self._placement_plans[requested_boundary] = placement
            if self._runtime_manager is not None:
                self._runtime_manager.register_placement_plan(placement)
        if boundary is None and self._placement_plan is None:
            self._placement_plan = placement
            if self._runtime_manager is not None:
                self._runtime_manager.set_placement_plan(self._placement_plan)
        return placement

    def _placement_for_client(self, server_round: int, cid: str, *, training: bool):
        if self.client_placement_fn is None:
            return self.get_or_create_placement_plan()
        cache_key = (int(server_round), str(cid), bool(training))
        cached = self._client_placement_cache.get(cache_key)
        if cached is not None:
            return cached
        self._prune_placement_cache(int(server_round))
        selected = self.client_placement_fn(int(server_round), str(cid), bool(training))
        if selected is None:
            placement = self.get_or_create_placement_plan()
        elif hasattr(selected, "plan_id") and hasattr(selected, "boundary"):
            placement = selected
            self._placement_plans.setdefault(str(placement.boundary), placement)
            if self._runtime_manager is not None:
                self._runtime_manager.register_placement_plan(placement)
        else:
            placement = self.get_or_create_placement_plan(str(selected))
        self._client_placement_cache[cache_key] = placement
        return placement

    def _prune_placement_cache(self, server_round: int) -> None:
        """Keep only the current round so long runs do not accumulate entries."""

        stale = [key for key in self._client_placement_cache if key[0] < server_round]
        for key in stale:
            self._client_placement_cache.pop(key, None)

    def _discard_round_state(self, through: int) -> None:
        """Drop per-round bookkeeping for every round up to and including ``through``.

        A round that never reaches ``aggregate_server_fit`` (no client was
        selected, or the round raised) would otherwise keep a full copy of the
        client and server model for the lifetime of the run.
        """

        stores = (
            self._round_initial_client_states,
            self._round_initial_server_states,
            self._pending_client_updates,
            self._reassembled_client_states,
        )
        for store in stores:
            for stale in [key for key in store if key <= int(through)]:
                store.pop(stale, None)

    def _make_server_model(self):
        if self._runtime_manager is None:
            raise RuntimeError(
                "AutoSplitStrategy has not been bound to a StageRuntimeManager yet."
            )
        if self.client_stage_count == 1:
            return AutoSplitTailServerModel(
                runtime_manager=self._runtime_manager,
                model=self.model,
                optimizer_fn=self.optimizer_fn,
                loss_fn=self.loss_fn,
                boundary=self.boundary,
                mode=self.mode,
                device=self.runtime_device,
            )
        return AutoSplitServerModel(
            runtime_manager=self._runtime_manager,
            model=self.model,
            optimizer_fn=self.optimizer_fn,
            loss_fn=self.loss_fn,
            device=self.runtime_device,
        )

    def _autosplit_config(
        self,
        model_version: int = 0,
        *,
        training: bool = True,
        placement=None,
    ) -> Dict[str, Any]:
        placement = placement or self.get_or_create_placement_plan()
        cache_key = (placement.plan_id, bool(training))
        base_config = self._autosplit_config_cache.get(cache_key)
        if base_config is None:
            reference_model = self.backend_adapter.clone_model(self.model)
            self.backend_adapter.set_training(reference_model, training)
            reference_handle = self.autosplit_session.prepare_runtime(
                reference_model,
                self.sample_inputs,
                boundary=placement.boundary,
                mode=placement.mode,
                trainable=True,
                dynamic_batch=placement.dynamic_batch,
                trace_batch_mode=placement.trace_batch_mode,
            )
            contract = graph_contract_for_runtime_handle(reference_handle)
            state_schema_hash = self.backend_adapter.state_manifest(reference_model).schema_hash
            base_config = {
                AUTOSPLIT_BACKEND_CONFIG_KEY: AUTOSPLIT_BACKEND_VALUE_TORCHLENS,
                AUTOSPLIT_FRAMEWORK_BACKEND_CONFIG_KEY: self.backend_adapter.backend_name,
                AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY: AUTOSPLIT_RUNTIME_BACKEND_VALUE_TORCHLENS_NATIVE,
                AUTOSPLIT_TORCHLENS_VERSION_CONFIG_KEY: placement.torchlens_version,
                AUTOSPLIT_PLAN_ID_CONFIG_KEY: placement.plan_id,
                AUTOSPLIT_SPLIT_ID_CONFIG_KEY: reference_handle.plan.split_id,
                AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY: reference_handle.plan.graph_signature,
                AUTOSPLIT_BOUNDARY_CONFIG_KEY: placement.boundary,
                AUTOSPLIT_BOUNDARY_TENSOR_LABELS_CONFIG_KEY: stable_json(placement.boundary_tensor_labels),
                AUTOSPLIT_MODE_CONFIG_KEY: placement.mode,
                AUTOSPLIT_STAGE_COUNT_CONFIG_KEY: 2,
                AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY: 1,
                AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY: placement.feature_abi_id,
                AUTOSPLIT_RUNTIME_CONTRACT_CONFIG_KEY: stable_json(placement.runtime_contract),
                AUTOSPLIT_RUNTIME_CONTRACT_DIGEST_CONFIG_KEY: runtime_contract_digest(placement.runtime_contract),
                AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY: placement.trace_batch_mode,
                AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY: stable_json(placement.dynamic_batch),
                AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY: contract.to_json().decode("utf-8"),
                AUTOSPLIT_GRAPH_CONTRACT_DIGEST_CONFIG_KEY: contract.digest,
                "_state_schema_hash": state_schema_hash,
            }
            self._autosplit_config_cache[cache_key] = base_config
        state_schema_hash = str(base_config["_state_schema_hash"])
        version_contract = ModelVersionContract(
            round_model_version=int(model_version),
            prefix_state_version=int(model_version),
            suffix_state_version=int(model_version),
            state_schema_hash=state_schema_hash,
        )
        config = {
            key: value for key, value in base_config.items() if not key.startswith("_")
        }
        config.update({
            AUTOSPLIT_MODEL_VERSION_CONFIG_KEY: int(model_version),
            AUTOSPLIT_MODEL_VERSION_CONTRACT_CONFIG_KEY: version_contract.to_json().decode("utf-8"),
        })
        return config

    def configure_fit(self, server_round, parameters, client_manager):
        self._discard_round_state(int(server_round) - 1)
        self._round_initial_client_states[int(server_round)] = [
            np.array(value, copy=True) for value in parameters_to_ndarrays(parameters)
        ]
        instructions = super().configure_fit(server_round, parameters, client_manager)
        for client, fit_ins in instructions:
            placement = self._placement_for_client(server_round, client.cid, training=True)
            fit_ins.config.update(
                self._autosplit_config(server_round, training=True, placement=placement)
            )
        return instructions

    def select_fit_clients(
        self,
        *,
        server_round: int,
        client_manager: ClientManager,
        sample_size: int,
        min_num_clients: int,
    ):
        if self.client_selector is None:
            return super().select_fit_clients(
                server_round=server_round,
                client_manager=client_manager,
                sample_size=sample_size,
                min_num_clients=min_num_clients,
            )

        try:
            available_clients = client_manager.all()
        except (AttributeError, NotImplementedError):
            return super().select_fit_clients(
                server_round=server_round,
                client_manager=client_manager,
                sample_size=sample_size,
                min_num_clients=min_num_clients,
            )
        client_map = self._normalize_client_map(available_clients)
        candidate_cids = list(client_map)
        if len(candidate_cids) < min_num_clients:
            return super().select_fit_clients(
                server_round=server_round,
                client_manager=client_manager,
                sample_size=sample_size,
                min_num_clients=min_num_clients,
            )

        result = self.client_selector.select(
            round_id=server_round,
            candidate_cids=candidate_cids,
            num_clients=sample_size,
        )
        self._last_selection_result = result
        clients = [
            client_map[cid]
            for cid in result.selected_cids
            if cid in client_map
        ]
        self._round_active_clients = [client.cid for client in clients]
        return clients

    @staticmethod
    def _normalize_client_map(available_clients):
        if isinstance(available_clients, dict):
            return {str(cid): client for cid, client in available_clients.items()}
        return {
            str(client.cid): client
            for client in available_clients
            if hasattr(client, "cid")
        }

    def configure_evaluate(self, server_round, parameters, client_manager):
        instructions = super().configure_evaluate(server_round, parameters, client_manager)
        for client, evaluate_ins in instructions:
            placement = self._placement_for_client(server_round, client.cid, training=False)
            evaluate_ins.config.update(
                self._autosplit_config(server_round, training=False, placement=placement)
            )
        return instructions

    def configure_server_fit(self, server_round, parameters, cids):
        self._round_initial_server_states[int(server_round)] = [
            np.array(value, copy=True) for value in parameters
        ]
        server_configs = super().configure_server_fit(server_round, parameters, cids)
        for config in server_configs:
            config.config["sid"] = config.sid
            cid = config.sid or cids[0]
            placement = self._placement_for_client(server_round, cid, training=True)
            config.config.update(
                self._autosplit_config(server_round, training=True, placement=placement)
            )
        return server_configs

    def configure_server_evaluate(self, server_round, parameters, cids):
        server_configs = super().configure_server_evaluate(server_round, parameters, cids)
        for config in server_configs:
            config.config["sid"] = config.sid
            cid = config.sid or cids[0]
            placement = self._placement_for_client(server_round, cid, training=False)
            config.config.update(
                self._autosplit_config(server_round, training=False, placement=placement)
            )
        return server_configs

    def aggregate_fit(self, server_round, results, failures):
        usable_results = [
            (client, fit_res)
            for client, fit_res in results
            if int(fit_res.num_examples) > 0
        ]
        ignored_zero_example_results = len(results) - len(usable_results)
        if ignored_zero_example_results:
            log(
                WARNING,
                "Ignoring %d zero-example client update(s) in round %d.",
                ignored_zero_example_results,
                int(server_round),
            )
        results = usable_results
        if self.client_selector is not None:
            self._update_client_selector_after_fit(server_round, results, failures)
        self._update_placement_policy_after_fit(server_round, results, failures)
        self.requests_state = {}
        self._round_active_clients = []
        for failure in failures:
            log(ERROR, str(failure))

        per_client = self.replica_scope_policy.resolve() == ReplicaScope.PER_CLIENT
        if not results:
            # Every selected client failed. Record the empty round so the
            # suffix replicas this round already created are discarded instead
            # of being mistaken for an out-of-order aggregation call.
            if per_client:
                self._pending_client_updates[int(server_round)] = {}
            return None, {}

        weights = self.aggregation_policy.reduce_weights(
            fit_res.num_examples for _, fit_res in results
        )
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)

        if per_client:
            # Each client update holds a fresh prefix and a suffix that is one
            # round stale, so there is no valid client-side global model until
            # `aggregate_server_fit` supplies the matching suffix halves.
            # `finalize_round` returns the reassembled model; returning it here
            # would mean publishing a half-stale one.
            self._pending_client_updates[int(server_round)] = {
                str(getattr(client, "cid")): (
                    parameters_to_ndarrays(fit_res.parameters),
                    float(weight),
                )
                for (client, fit_res), weight in zip(results, weights)
            }
            return None, aggregated_metrics

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), weight)
            for (_, fit_res), weight in zip(results, weights)
        ]
        return ndarrays_to_parameters(aggregate(weights_results)), aggregated_metrics

    def _update_placement_policy_after_fit(self, server_round, results, failures) -> None:
        """Feed round outcomes back into a stateful placement policy, if any."""

        policy = self.client_placement_fn
        observe_fit = getattr(policy, "observe_fit_metrics", None)
        observe_failure = getattr(policy, "observe_failure", None)
        if observe_fit is None and observe_failure is None:
            return
        if observe_fit is not None:
            for client, fit_res in results:
                cid = getattr(client, "cid", None)
                if cid is None:
                    continue
                metrics = dict(fit_res.metrics or {})
                metrics.setdefault("num_examples", fit_res.num_examples)
                observe_fit(
                    round_id=int(server_round),
                    cid=str(cid),
                    num_examples=fit_res.num_examples,
                    metrics=metrics,
                )
        if observe_failure is not None:
            for failure in failures:
                cid = self._failure_cid(failure)
                if cid is None:
                    continue
                observe_failure(round_id=int(server_round), cid=cid, reason=failure)

    def _update_client_selector_after_fit(self, server_round, results, failures) -> None:
        for client, fit_res in results:
            cid = getattr(client, "cid", None)
            if cid is None:
                continue
            metrics = dict(fit_res.metrics or {})
            metrics.setdefault("num_examples", fit_res.num_examples)
            self.client_selector.update_after_fit(
                round_id=server_round,
                cid=str(cid),
                num_examples=fit_res.num_examples,
                metrics=metrics,
            )
        for failure in failures:
            cid = self._failure_cid(failure)
            if cid is None:
                continue
            self.client_selector.update_after_failure(
                round_id=server_round,
                cid=cid,
                reason=failure,
            )

    def finalize_round(self, server_round, client_parameters, server_parameters):
        """Publish the reassembled logical model as this round's client model.

        Under `ReplicaScope.PER_CLIENT` the prefix and suffix halves of a round
        are aggregated by two different calls, so `aggregate_fit` returns `None`
        and the complete model is produced here. A round without a reassembled
        model keeps the previous global parameters rather than publishing a
        half-stale one.
        """

        reassembled = self._reassembled_client_states.pop(int(server_round), None)
        if reassembled is None:
            return client_parameters, server_parameters
        return ndarrays_to_parameters(reassembled), server_parameters

    def _logical_state_name_list(self) -> list[str]:
        """The state-dict names of the logical model, in manifest order."""

        if self._logical_state_names is None:
            manifest = self.backend_adapter.state_manifest(self.model)
            self._logical_state_names = [entry.name for entry in manifest.entries]
        return self._logical_state_names

    def _tied_state_groups(self) -> tuple[tuple[int, ...], ...]:
        if self._logical_tied_groups is None:
            self._logical_tied_groups = tuple(
                tuple(int(index) for index in group)
                for group in self.backend_adapter.tied_state_groups(self.model)
            )
        return self._logical_tied_groups

    @staticmethod
    def _failure_cid(failure) -> str | None:
        if isinstance(failure, tuple) and failure:
            cid = getattr(failure[0], "cid", None)
            return str(cid) if cid is not None else None
        for attr in ("cid", "client_id"):
            cid = getattr(failure, attr, None)
            if cid is not None:
                return str(cid)
        return None

    def aggregate_server_fit(self, server_round, results):
        per_client = self.replica_scope_policy.resolve() == ReplicaScope.PER_CLIENT
        if not results:
            if per_client and self._pending_client_updates.get(int(server_round)):
                log(
                    ERROR,
                    "Round %d has client updates but no suffix result; the logical "
                    "model cannot be reassembled and the round is skipped.",
                    int(server_round),
                )
            self._discard_round_state(int(server_round))
            return None
        if per_client:
            round_id = int(server_round)
            client_updates = self._pending_client_updates.pop(round_id, None)
            initial_client = self._round_initial_client_states.pop(round_id, None)
            initial_server = self._round_initial_server_states.pop(round_id, None)
            if client_updates is None:
                raise RuntimeError(
                    "Per-client suffix aggregation requires matching client updates first."
                )
            if not client_updates:
                # `aggregate_fit` saw no successful client, so these suffix
                # replicas have no prefix half to be reassembled with.
                log(
                    ERROR,
                    "Round %d produced %d suffix result(s) but no client update; "
                    "keeping the previous global parameters.",
                    round_id,
                    len(results),
                )
                return None
            if initial_client is None or initial_server is None:
                raise RuntimeError("Round initial state is unavailable for logical model reassembly.")
            server_updates = {str(res.sid): res for res in results if res.sid is not None}
            names = self._logical_state_name_list()
            tied_groups = self._tied_state_groups()
            logical_updates = []
            missing_suffix_cids = []
            for cid, (client_state, weight) in client_updates.items():
                server_result = server_updates.get(cid)
                if server_result is None:
                    # One device losing its suffix replica must not discard the
                    # rounds every other device just finished.
                    missing_suffix_cids.append(cid)
                    continue
                logical_updates.append(
                    (
                        _reassemble_named_state(
                            names,
                            initial_client,
                            initial_server,
                            client_state,
                            server_result.parameters,
                            tied_groups=tied_groups,
                            tied_weight_update_mode=self.tied_weight_update_mode,
                        ),
                        weight,
                    )
                )
            if missing_suffix_cids:
                log(
                    ERROR,
                    "Dropping %d client update(s) without a matching per-client suffix "
                    "result in round %d: %s",
                    len(missing_suffix_cids),
                    int(server_round),
                    sorted(missing_suffix_cids),
                )
            if not logical_updates:
                raise RuntimeError(
                    "No client update had a matching per-client suffix result."
                )
            aggregated = aggregate(logical_updates)
            # The same logical model is both halves of the split: `finalize_round`
            # hands it back as the client-side global model for this round.
            self._reassembled_client_states[round_id] = aggregated
            return aggregated
        self._discard_round_state(int(server_round))
        weights = self.aggregation_policy.reduce_weights(
            res.config.get("num_examples", 1) for res in results
        )
        return aggregate(
            [
                (res.parameters, weight)
                for res, weight in zip(results, weights)
            ]
        )


def _reassemble_named_state(
    names: Sequence[str],
    initial_client: Sequence[np.ndarray],
    initial_server: Sequence[np.ndarray],
    client_state: Sequence[np.ndarray],
    server_state: Sequence[np.ndarray],
    *,
    tied_groups: Sequence[Sequence[int]] = (),
    tied_weight_update_mode: str = "reject",
) -> list[np.ndarray]:
    lengths = {
        len(names), len(initial_client), len(initial_server), len(client_state), len(server_state)
    }
    if len(lengths) != 1:
        raise RuntimeError("Logical model state schemas differ during per-client reassembly.")
    tied_positions = {int(index) for group in tied_groups for index in group}
    merged = []
    for position, (name, client_initial, server_initial, client_value, server_value) in enumerate(
        zip(names, initial_client, initial_server, client_state, server_state)
    ):
        if not np.array_equal(client_initial, server_initial):
            raise RuntimeError(f"Client/server initial state diverged for parameter {name!r}.")
        client_changed = not np.array_equal(client_value, client_initial)
        server_changed = not np.array_equal(server_value, server_initial)
        if position in tied_positions:
            if client_changed and server_changed:
                if tied_weight_update_mode != "additive_sgd":
                    raise RuntimeError(
                        f"Tied parameter {name!r} changed on both split stages. "
                        "Set tied_weight_update_mode='additive_sgd' only when both "
                        "stages use identical stateless SGD without weight decay."
                    )
                # For identical stateless SGD, both stages evaluated their
                # gradients from the same initial tensor and the logical update
                # is exactly the sum of the two deltas.
                merged.append(
                    np.asarray(client_initial)
                    + (np.asarray(client_value) - np.asarray(client_initial))
                    + (np.asarray(server_value) - np.asarray(server_initial))
                )
            elif server_changed:
                merged.append(np.array(server_value, copy=True))
            elif client_changed:
                merged.append(np.array(client_value, copy=True))
            else:
                merged.append(np.array(client_initial, copy=True))
            continue
        if client_changed and server_changed and not np.array_equal(client_value, server_value):
            raise RuntimeError(
                f"Both prefix and suffix changed parameter {name!r}; ownership is ambiguous."
            )
        if server_changed:
            merged.append(np.array(server_value, copy=True))
        elif client_changed:
            merged.append(np.array(client_value, copy=True))
        else:
            merged.append(np.array(client_initial, copy=True))
    return merged
