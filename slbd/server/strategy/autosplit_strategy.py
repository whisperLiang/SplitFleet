"""Autosplit-aware strategy facade built on top of PlainSlStrategy."""

from __future__ import annotations

import json
from logging import ERROR
from typing import Any, Dict, Optional, Sequence

from flwr.common import log, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy.aggregate import aggregate

from slbd.autosplit import (
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
)
from slbd.common.constants import (
    AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_CUTOFFS_CONFIG_KEY,
    AUTOSPLIT_PLAN_ID_CONFIG_KEY,
    AUTOSPLIT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_STAGE_TO_WORKER_CONFIG_KEY,
)
from slbd.server.server_model.autosplit_server_model import AutoSplitServerModel
from slbd.server.server_model.autosplit_tail_server_model import AutoSplitTailServerModel
from slbd.server.strategy.plain_strategy import PlainSlStrategy


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
    """Strategy that computes and propagates autosplit placement metadata."""

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
        preferred_stage_count: Optional[int] = None,
        client_stage_count: int = 0,
        loss_fn=None,
        optimizer_fn=None,
        runtime_device: str = "cpu",
        **kwargs,
    ) -> None:
        self.model = model
        self.sample_inputs = sample_inputs
        self.sample_kwargs = dict(sample_kwargs or {})
        self.worker_specs = list(worker_specs or [WorkerSpec(worker_id="local", device="cpu")])
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
            and self.aggregation_policy.requires_per_client_replica_scope()
        ):
            self.replica_scope_policy = ReplicaScopePolicy(replica_scope=ReplicaScope.PER_CLIENT)
        if (
            self.aggregation_policy.requires_per_client_replica_scope()
            and self.replica_scope_policy.resolve() != ReplicaScope.PER_CLIENT
        ):
            raise ValueError(
                "SplitFed aggregation requires `ReplicaScope.PER_CLIENT` server replicas."
            )
        self.execution_schedule_policy = execution_schedule_policy or ExecutionSchedulePolicy()
        self.client_stage_count = max(0, int(client_stage_count))
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.runtime_device = runtime_device
        self._placement_plan = None
        self._runtime_manager = None

        if (
            self.client_stage_count > 0
            and self.partition_selection_policy.preferred_stage_count is None
        ):
            self.partition_selection_policy = PartitionSelectionPolicy(
                preferred_stage_count=max(2, self.client_stage_count + 1)
            )

        init_server_model_fn = kwargs.pop("init_server_model_fn", None) or self._make_server_model
        super().__init__(
            init_server_model_fn=init_server_model_fn,
            common_server_model=not self.replica_scope_policy.requires_independent_replicas(),
            process_clients_as_batch=self.execution_schedule_policy.should_batch(),
            **kwargs,
        )

    def bind_stage_runtime_manager(self, runtime_manager) -> None:
        self._runtime_manager = runtime_manager

    def initialize_parameters(self, client_manager):
        _ = client_manager
        if self.client_stage_count <= 0:
            return ndarrays_to_parameters([])
        return ndarrays_to_parameters(
            [tensor.detach().cpu().numpy() for tensor in self.model.state_dict().values()]
        )

    def initialize_server_parameters(self):
        return [tensor.detach().cpu().numpy() for tensor in self.model.state_dict().values()]

    def get_or_create_placement_plan(self):
        if self._placement_plan is None:
            self._placement_plan = self.autosplit_session.plan(
                self.model,
                self.sample_inputs,
                sample_kwargs=self.sample_kwargs or None,
                worker_specs=self.worker_specs,
                constraints=self.constraints,
                objective=self.objective,
                preferred_stage_count=self.partition_selection_policy.preferred_stage_count,
                model_name=self.model.__class__.__name__,
            )
            if self.client_stage_count > 0 and not self._tail_is_trainable(self._placement_plan):
                self._placement_plan = self._select_client_split_compatible_plan()
            if self._placement_plan.partition_plan.stage_count <= self.client_stage_count:
                raise RuntimeError(
                    "Autosplit planned too few stages for the requested client-local prefix. "
                    f"stage_count={self._placement_plan.partition_plan.stage_count}, "
                    f"client_stage_count={self.client_stage_count}"
                )
        return self._placement_plan

    def _tail_is_trainable(self, placement_plan) -> bool:
        tail_stages = placement_plan.partition_plan.stages[self.client_stage_count :]
        return any(stage.estimated_parameter_bytes > 0 for stage in tail_stages)

    def _select_client_split_compatible_plan(self):
        traced = self.autosplit_session.planner.tracer.trace(
            self.model,
            self.sample_inputs,
            sample_kwargs=self.sample_kwargs or None,
        )
        partition_plans = self.autosplit_session.planner.enumerate_partition_plans(
            traced,
            constraints=self.constraints,
            preferred_stage_count=self.partition_selection_policy.preferred_stage_count,
        )
        placements = []
        for partition_plan in partition_plans:
            placement = self.autosplit_session.planner.place_partition_plan(
                partition_plan,
                self.worker_specs,
                constraints=self.constraints,
                objective=self.objective,
            )
            if placement is None:
                continue
            if placement.partition_plan.stage_count <= self.client_stage_count:
                continue
            if not self._tail_is_trainable(placement):
                continue
            placements.append(placement)
        if not placements:
            raise RuntimeError(
                "No autosplit placement keeps a trainable tail for the requested client prefix."
            )
        return self.partition_selection_policy.choose(placements)

    def _make_server_model(self):
        if self._runtime_manager is None:
            raise RuntimeError(
                "AutoSplitStrategy has not been bound to a StageRuntimeManager yet."
            )
        if self.client_stage_count > 0:
            return AutoSplitTailServerModel(
                runtime_manager=self._runtime_manager,
                model=self.model,
                optimizer_fn=self.optimizer_fn,
                loss_fn=self.loss_fn,
                client_stage_count=self.client_stage_count,
                device=self.runtime_device,
            )
        return AutoSplitServerModel(
            runtime_manager=self._runtime_manager,
            model=self.model,
            optimizer_fn=self.optimizer_fn,
            loss_fn=self.loss_fn,
            device=self.runtime_device,
        )

    def _autosplit_config(self) -> Dict[str, Any]:
        placement = self.get_or_create_placement_plan()
        return {
            AUTOSPLIT_PLAN_ID_CONFIG_KEY: placement.plan_id,
            AUTOSPLIT_STAGE_COUNT_CONFIG_KEY: placement.partition_plan.stage_count,
            AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY: self.client_stage_count,
            AUTOSPLIT_STAGE_TO_WORKER_CONFIG_KEY: json.dumps(placement.stage_to_worker, sort_keys=True),
            AUTOSPLIT_CUTOFFS_CONFIG_KEY: json.dumps(
                placement.partition_plan.metadata.get("cutoffs", []),
                sort_keys=True,
            ),
        }

    def configure_fit(self, server_round, parameters, client_manager):
        instructions = super().configure_fit(server_round, parameters, client_manager)
        autosplit_config = self._autosplit_config()
        for _, fit_ins in instructions:
            fit_ins.config.update(autosplit_config)
        return instructions

    def configure_evaluate(self, server_round, parameters, client_manager):
        instructions = super().configure_evaluate(server_round, parameters, client_manager)
        autosplit_config = self._autosplit_config()
        for _, evaluate_ins in instructions:
            evaluate_ins.config.update(autosplit_config)
        return instructions

    def configure_server_fit(self, server_round, parameters, cids):
        server_configs = super().configure_server_fit(server_round, parameters, cids)
        autosplit_config = self._autosplit_config()
        for config in server_configs:
            config.config["sid"] = config.sid
            config.config.update(autosplit_config)
        return server_configs

    def configure_server_evaluate(self, server_round, parameters, cids):
        server_configs = super().configure_server_evaluate(server_round, parameters, cids)
        autosplit_config = self._autosplit_config()
        for config in server_configs:
            config.config["sid"] = config.sid
            config.config.update(autosplit_config)
        return server_configs

    def aggregate_fit(self, server_round, results, failures):
        _ = server_round
        self.requests_state = {}
        self._round_active_clients = []
        for failure in failures:
            log(ERROR, str(failure))

        if not results:
            return None, {}

        weights = self.aggregation_policy.reduce_weights(
            fit_res.num_examples for _, fit_res in results
        )
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), weight)
            for (_, fit_res), weight in zip(results, weights)
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        return parameters_aggregated, aggregated_metrics

    def aggregate_server_fit(self, server_round, results):
        _ = server_round
        if not results:
            return None
        weights = self.aggregation_policy.reduce_weights(
            res.config.get("num_examples", 1) for res in results
        )
        return aggregate(
            [
                (res.parameters, weight)
                for res, weight in zip(results, weights)
            ]
        )
