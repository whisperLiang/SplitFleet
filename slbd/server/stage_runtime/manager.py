"""Autosplit-aware runtime manager."""

from __future__ import annotations

import copy
import uuid
from typing import Any, Optional

from slbd.autosplit.runtime import AutoSplitSession
from slbd.autosplit.serde import dump_model_state
from slbd.autosplit.types import PlacementPlan, WorkerSpec
from slbd.server.server_model.manager.grpc_manager import GrpcServerModelManager
from slbd.server.server_model.manager.manager import ServerModelManager
from slbd.server.stage_runtime.registry import GLOBAL_WORKER_REGISTRY, WorkerRegistry


class StageRuntimeManager(ServerModelManager):
    """Bridge autosplit planning/execution into the existing server runtime."""

    def __init__(
        self,
        *,
        init_server_model_fn=None,
        autosplit_session: Optional[AutoSplitSession] = None,
        worker_registry: Optional[WorkerRegistry] = None,
    ) -> None:
        super().__init__()
        self.autosplit_session = autosplit_session or AutoSplitSession()
        self.worker_registry = worker_registry or GLOBAL_WORKER_REGISTRY
        self._placement_plan: Optional[PlacementPlan] = None
        self._delegate = (
            GrpcServerModelManager(init_server_model_fn=init_server_model_fn)
            if init_server_model_fn is not None
            else None
        )

    def set_placement_plan(self, placement_plan: PlacementPlan) -> None:
        self._placement_plan = placement_plan
        self._sync_assigned_workers(placement_plan)

    def get_placement_plan(self) -> Optional[PlacementPlan]:
        return self._placement_plan

    def register_worker(self, worker_spec: WorkerSpec) -> WorkerSpec:
        registered = self.worker_registry.register(worker_spec)
        if (
            self._placement_plan is not None
            and worker_spec.worker_id in self._assigned_worker_ids(self._placement_plan)
        ):
            self._sync_assigned_workers(
                self._placement_plan,
                worker_ids=[worker_spec.worker_id],
            )
        return registered

    def list_workers(self, *, online_only: bool = True) -> list[WorkerSpec]:
        return self.worker_registry.list_workers(online_only=online_only)

    def clone_placement_plan(
        self,
        *,
        model=None,
        plan_id_suffix: Optional[str] = None,
    ) -> PlacementPlan:
        """Clone the active placement plan and optionally bind a different model instance."""

        if self._placement_plan is None:
            raise RuntimeError("No autosplit placement plan is active.")

        base_plan = self._placement_plan
        cloned_execution_plan = copy.copy(base_plan.partition_plan.execution_plan)
        bound_model = model or base_plan.partition_plan.execution_plan.model
        if bound_model is not None:
            cloned_execution_plan.set_model(bound_model)

        cloned_partition_plan = copy.copy(base_plan.partition_plan)
        cloned_partition_plan.execution_plan = cloned_execution_plan
        if plan_id_suffix:
            cloned_partition_plan.plan_id = (
                f"{base_plan.partition_plan.plan_id}_{plan_id_suffix}"
            )

        cloned_plan = copy.copy(base_plan)
        cloned_plan.partition_plan = cloned_partition_plan
        cloned_plan.stage_to_worker = dict(base_plan.stage_to_worker)
        cloned_plan.worker_specs = dict(base_plan.worker_specs)
        cloned_plan.stage_scores = dict(base_plan.stage_scores)
        cloned_plan.metadata = dict(base_plan.metadata)
        return cloned_plan

    def run_eval(self, inputs: Any, *, input_kwargs: Optional[dict] = None) -> Any:
        if self._placement_plan is None:
            raise RuntimeError("No autosplit placement plan is active.")
        return self.run_eval_plan(
            self._placement_plan,
            inputs,
            input_kwargs=input_kwargs,
        )

    def run_eval_plan(
        self,
        placement_plan: PlacementPlan,
        inputs: Any,
        *,
        input_kwargs: Optional[dict] = None,
    ) -> Any:
        if not self._requires_remote_execution(placement_plan):
            return self.autosplit_session.run_eval(
                placement_plan,
                inputs,
                input_kwargs=input_kwargs,
            )

        self._sync_assigned_workers(placement_plan)
        execution_id = f"eval-{uuid.uuid4().hex}"
        context = self.autosplit_session.prepare_execution_context(
            placement_plan,
            inputs,
            input_kwargs=input_kwargs,
            differentiable=False,
        )
        last_stage_id = placement_plan.partition_plan.stages[-1].stage_id
        try:
            for stage in placement_plan.partition_plan.stages:
                worker_id = placement_plan.stage_to_worker[stage.stage_id]
                executor = self.worker_registry.get_executor(worker_id)
                seeded_values = {
                    index: context.available_values[index]
                    for index in set(stage.input_indices)
                    | set(placement_plan.partition_plan.execution_plan.input_node_indices)
                    if index in context.available_values
                }
                result = executor.run_stage_forward(
                    placement_plan,
                    stage.stage_id,
                    execution_id,
                    seeded_values,
                    differentiable=False,
                    detach_boundary=stage.stage_id != last_stage_id,
                )
                context.available_values.update(result.available_updates)
                context.stored_values.update(result.stored_updates)
            return self.autosplit_session.reconstruct_outputs(
                placement_plan,
                context.stored_values,
            )
        finally:
            self._clear_remote_execution(placement_plan, execution_id)

    def run_eval_tail_plan(
        self,
        placement_plan: PlacementPlan,
        seeded_values: dict[int, Any],
        *,
        start_stage_index: int,
    ) -> Any:
        tail_stages = placement_plan.partition_plan.stages[start_stage_index:]
        if not tail_stages:
            return self.autosplit_session.reconstruct_outputs(placement_plan, seeded_values)

        self._sync_assigned_workers(placement_plan)
        execution_id = f"eval-tail-{uuid.uuid4().hex}"
        context = self.autosplit_session.prepare_seeded_execution_context(
            placement_plan,
            seeded_values,
        )
        last_stage_id = tail_stages[-1].stage_id
        try:
            for stage in tail_stages:
                worker_id = placement_plan.stage_to_worker[stage.stage_id]
                executor = self.worker_registry.get_executor(worker_id)
                stage_seeded_values = {
                    index: context.available_values[index]
                    for index in set(stage.input_indices)
                    | set(placement_plan.partition_plan.execution_plan.input_node_indices)
                    if index in context.available_values
                }
                result = executor.run_stage_forward(
                    placement_plan,
                    stage.stage_id,
                    execution_id,
                    stage_seeded_values,
                    differentiable=False,
                    detach_boundary=stage.stage_id != last_stage_id,
                )
                context.available_values.update(result.available_updates)
                context.stored_values.update(result.stored_updates)
            return self.autosplit_session.reconstruct_outputs(
                placement_plan,
                context.stored_values,
            )
        finally:
            self._clear_remote_execution(placement_plan, execution_id)

    def run_train(
        self,
        inputs: Any,
        *,
        input_kwargs: Optional[dict] = None,
        targets: Any = None,
        loss_fn=None,
        optimizer=None,
        zero_grad: bool = True,
        step_optimizer: bool = True,
    ) -> dict:
        if self._placement_plan is None:
            raise RuntimeError("No autosplit placement plan is active.")
        return self.run_train_plan(
            self._placement_plan,
            inputs,
            input_kwargs=input_kwargs,
            targets=targets,
            loss_fn=loss_fn,
            optimizer=optimizer,
            zero_grad=zero_grad,
            step_optimizer=step_optimizer,
        )

    def run_train_plan(
        self,
        placement_plan: PlacementPlan,
        inputs: Any,
        *,
        input_kwargs: Optional[dict] = None,
        targets: Any = None,
        loss_fn=None,
        optimizer=None,
        zero_grad: bool = True,
        step_optimizer: bool = True,
    ) -> dict:
        if not self._requires_remote_execution(placement_plan):
            return self.autosplit_session.run_train(
                placement_plan,
                inputs,
                input_kwargs=input_kwargs,
                targets=targets,
                loss_fn=loss_fn,
                optimizer=optimizer,
                zero_grad=zero_grad,
                step_optimizer=step_optimizer,
            )

        canonical_model = placement_plan.partition_plan.execution_plan.model
        if canonical_model is None:
            raise RuntimeError("Remote autosplit training requires a live canonical model.")
        if zero_grad:
            canonical_model.zero_grad(set_to_none=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

        self._sync_assigned_workers(placement_plan)
        execution_id = f"train-{uuid.uuid4().hex}"
        context = self.autosplit_session.prepare_execution_context(
            placement_plan,
            inputs,
            input_kwargs=input_kwargs,
            differentiable=True,
        )
        accumulated_parameter_grads = {}
        last_stage_id = placement_plan.partition_plan.stages[-1].stage_id
        try:
            for stage in placement_plan.partition_plan.stages:
                worker_id = placement_plan.stage_to_worker[stage.stage_id]
                executor = self.worker_registry.get_executor(worker_id)
                seeded_values = {
                    index: context.available_values[index]
                    for index in set(stage.input_indices)
                    | set(placement_plan.partition_plan.execution_plan.input_node_indices)
                    if index in context.available_values
                }
                result = executor.run_stage_forward(
                    placement_plan,
                    stage.stage_id,
                    execution_id,
                    seeded_values,
                    differentiable=True,
                    detach_boundary=stage.stage_id != last_stage_id,
                )
                context.available_values.update(result.available_updates)
                context.stored_values.update(result.stored_updates)

            outputs = self.autosplit_session.reconstruct_outputs(
                placement_plan,
                context.stored_values,
            )
            loss = self.autosplit_session.compute_loss(outputs, targets, loss_fn)
            loss.backward()

            upstream_grads = self.autosplit_session.collect_tensor_grads(
                {
                    index: context.stored_values[index]
                    for index in self.autosplit_session.get_output_indices(placement_plan)
                    if index in context.stored_values
                }
            )
            for stage in reversed(placement_plan.partition_plan.stages):
                worker_id = placement_plan.stage_to_worker[stage.stage_id]
                executor = self.worker_registry.get_executor(worker_id)
                backward_result = executor.run_stage_backward(
                    placement_plan,
                    stage.stage_id,
                    execution_id,
                    upstream_grads,
                )
                for name, grad in backward_result.parameter_grads.items():
                    if name in accumulated_parameter_grads:
                        accumulated_parameter_grads[name] = accumulated_parameter_grads[name] + grad
                    else:
                        accumulated_parameter_grads[name] = grad
                upstream_grads = backward_result.input_grads

            for name, parameter in canonical_model.named_parameters():
                grad = accumulated_parameter_grads.get(name)
                if grad is None:
                    continue
                parameter.grad = grad.to(parameter.device)

            if optimizer is not None and step_optimizer:
                optimizer.step()

            return {
                "output": outputs,
                "loss": loss,
                "stage_count": placement_plan.partition_plan.stage_count,
                "stage_to_worker": dict(placement_plan.stage_to_worker),
            }
        finally:
            self._clear_remote_execution(placement_plan, execution_id)

    def run_train_tail_plan(
        self,
        placement_plan: PlacementPlan,
        seeded_values: dict[int, Any],
        *,
        start_stage_index: int,
        targets: Any = None,
        loss_fn=None,
        optimizer=None,
        zero_grad: bool = True,
        step_optimizer: bool = True,
    ) -> dict:
        tail_stages = placement_plan.partition_plan.stages[start_stage_index:]
        if not tail_stages:
            raise ValueError("Tail execution requires at least one tail stage.")

        canonical_model = placement_plan.partition_plan.execution_plan.model
        if canonical_model is None:
            raise RuntimeError("Tail autosplit training requires a live canonical model.")
        if zero_grad:
            canonical_model.zero_grad(set_to_none=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

        self._sync_assigned_workers(placement_plan)
        execution_id = f"train-tail-{uuid.uuid4().hex}"
        context = self.autosplit_session.prepare_seeded_execution_context(
            placement_plan,
            seeded_values,
        )
        accumulated_parameter_grads = {}
        last_stage_id = tail_stages[-1].stage_id
        first_stage_id = tail_stages[0].stage_id
        boundary_grads = {}
        try:
            for stage in tail_stages:
                worker_id = placement_plan.stage_to_worker[stage.stage_id]
                executor = self.worker_registry.get_executor(worker_id)
                stage_seeded_values = {
                    index: context.available_values[index]
                    for index in set(stage.input_indices)
                    | set(placement_plan.partition_plan.execution_plan.input_node_indices)
                    if index in context.available_values
                }
                result = executor.run_stage_forward(
                    placement_plan,
                    stage.stage_id,
                    execution_id,
                    stage_seeded_values,
                    differentiable=True,
                    detach_boundary=stage.stage_id != last_stage_id,
                )
                context.available_values.update(result.available_updates)
                context.stored_values.update(result.stored_updates)

            outputs = self.autosplit_session.reconstruct_outputs(
                placement_plan,
                context.stored_values,
            )
            loss = self.autosplit_session.compute_loss(outputs, targets, loss_fn)
            loss.backward()

            upstream_grads = self.autosplit_session.collect_tensor_grads(
                {
                    index: context.stored_values[index]
                    for index in self.autosplit_session.get_output_indices(placement_plan)
                    if index in context.stored_values
                }
            )
            for stage in reversed(tail_stages):
                worker_id = placement_plan.stage_to_worker[stage.stage_id]
                executor = self.worker_registry.get_executor(worker_id)
                backward_result = executor.run_stage_backward(
                    placement_plan,
                    stage.stage_id,
                    execution_id,
                    upstream_grads,
                )
                for name, grad in backward_result.parameter_grads.items():
                    if name in accumulated_parameter_grads:
                        accumulated_parameter_grads[name] = accumulated_parameter_grads[name] + grad
                    else:
                        accumulated_parameter_grads[name] = grad
                upstream_grads = backward_result.input_grads
                if stage.stage_id == first_stage_id:
                    boundary_grads = {
                        index: grad.detach().clone()
                        for index, grad in upstream_grads.items()
                    }

            for name, parameter in canonical_model.named_parameters():
                grad = accumulated_parameter_grads.get(name)
                if grad is None:
                    continue
                parameter.grad = grad.to(parameter.device)

            if optimizer is not None and step_optimizer:
                optimizer.step()

            return {
                "output": outputs,
                "loss": loss,
                "boundary_grads": boundary_grads,
                "stage_count": len(tail_stages),
                "stage_to_worker": {
                    stage.stage_id: placement_plan.stage_to_worker[stage.stage_id]
                    for stage in tail_stages
                },
            }
        finally:
            self._clear_remote_execution(placement_plan, execution_id)

    def _assigned_worker_ids(self, placement_plan: PlacementPlan) -> list[str]:
        return sorted(set(placement_plan.stage_to_worker.values()))

    def _requires_remote_execution(self, placement_plan: PlacementPlan) -> bool:
        worker_ids = self._assigned_worker_ids(placement_plan)
        if not worker_ids:
            return False
        remote_mask = [
            self.worker_registry.get_executor(worker_id).is_remote
            for worker_id in worker_ids
        ]
        if any(remote_mask) and not all(remote_mask):
            raise NotImplementedError(
                "Mixed local/remote stage execution is not implemented yet."
            )
        return all(remote_mask)

    def _sync_assigned_workers(
        self,
        placement_plan: PlacementPlan,
        worker_ids: Optional[list[str]] = None,
    ) -> None:
        model = placement_plan.partition_plan.execution_plan.model
        model_state = dump_model_state(model) if model is not None else b""
        target_ids = worker_ids or self._assigned_worker_ids(placement_plan)
        for worker_id in target_ids:
            self.worker_registry.get_executor(worker_id).load_placement_plan(
                placement_plan,
                model_state=model_state,
            )

    def _clear_remote_execution(
        self,
        placement_plan: PlacementPlan,
        execution_id: str,
    ) -> None:
        for worker_id in self._assigned_worker_ids(placement_plan):
            executor = self.worker_registry.get_executor(worker_id)
            if executor.is_remote:
                executor.clear_execution(execution_id)

    def get_server_model(self, sid):
        if self._delegate is None:
            raise RuntimeError("No classic server-model delegate is configured.")
        return self._delegate.get_server_model(sid)

    def collect_server_fit_results(self):
        if self._delegate is None:
            return []
        return self._delegate.collect_server_fit_results()

    def initialize_server_models(self, configs):
        if self._delegate is not None:
            self._delegate.initialize_server_models(configs)

    def get_server_model_ids(self):
        if self._delegate is None:
            return []
        return self._delegate.get_server_model_ids()

    def end_round(self):
        if self._delegate is not None and hasattr(self._delegate, "end_round"):
            self._delegate.end_round()
