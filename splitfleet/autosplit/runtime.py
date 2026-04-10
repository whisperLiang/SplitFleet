"""Local autosplit execution runtime.

Optimized with:
- Dictionary comprehensions for faster iteration
- Walrus operator for reduced attribute lookups
- Pre-computed set operations
- Memory-efficient gradient collection
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import torch

from torchlens.replay_engine import (
    _collect_output_indices,
    _execute_nodes,
    _map_tree,
    _move_tree_to_device,
    _prepare_runtime,
    _prepare_seed_map,
    _reconstruct_outputs,
)
from torchlens.replay_train import (
    _detach_tree_for_boundary,
    _prepare_differentiable_seed_map,
)
from torchlens.validation_replay import _compute_loss

from splitfleet.autosplit.cache import PlanCacheStore
from splitfleet.autosplit.planner import AutoSplitPlanner
from splitfleet.autosplit.types import PartitionPlan, PartitionStage, PlacementPlan


@dataclass(slots=True)
class StageTrace:
    """Forward-pass artefacts required for explicit split backward."""

    source_outputs: Dict[int, Any] = field(default_factory=dict)
    transport_outputs: Dict[int, Any] = field(default_factory=dict)
    external_inputs: Dict[int, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ForwardTrace:
    """Aggregate state across all stages."""

    outputs: Any
    stored_values: Dict[int, Any]
    stage_traces: Dict[str, StageTrace]


@dataclass(slots=True)
class PreparedExecutionContext:
    """Coordinator-side seed and output state for stage orchestration."""

    execution_plan: Any
    runtime_device: torch.device
    available_values: Dict[int, Any]
    stored_values: Dict[int, Any]
    output_indices: Set[int]


@dataclass(slots=True)
class StageForwardResult:
    """Portable outputs produced by a single stage."""

    available_updates: Dict[int, Any] = field(default_factory=dict)
    stored_updates: Dict[int, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageBackwardResult:
    """Backward outputs produced by a single stage."""

    input_grads: Dict[int, torch.Tensor] = field(default_factory=dict)
    parameter_grads: Dict[str, torch.Tensor] = field(default_factory=dict)


def _filter_tensor_grads(values: Dict[int, Any]) -> Dict[int, torch.Tensor]:
    """Collect gradients from tensors, optimized with dict comprehension and walrus operator."""
    return {
        index: grad.detach().clone()
        for index, value in values.items()
        if isinstance(grad := getattr(value, "grad", None), torch.Tensor)
    }


def _enable_grad_for_transport(value: Any) -> Any:
    """Enable gradient tracking for transport tensors."""
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        value.requires_grad_(True)
        value.retain_grad()
    return value


class AutoSplitSession:
    """High-level session that plans and executes autosplit workloads."""

    def __init__(
        self,
        planner: Optional[AutoSplitPlanner] = None,
        *,
        cache_store: Optional[PlanCacheStore] = None,
        device: str = "cpu",
    ) -> None:
        self.planner = planner or AutoSplitPlanner()
        self.cache_store = cache_store
        self.device = device

    def plan(
        self,
        model,
        sample_inputs: Any,
        *,
        sample_kwargs: Optional[Dict[str, Any]] = None,
        worker_specs=None,
        constraints=None,
        objective=None,
        preferred_stage_count: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> PlacementPlan:
        return self.planner.plan(
            model,
            sample_inputs,
            sample_kwargs=sample_kwargs,
            worker_specs=worker_specs,
            constraints=constraints,
            objective=objective,
            preferred_stage_count=preferred_stage_count,
            cache_store=self.cache_store,
            model_name=model_name,
        )

    def prepare_execution_context(
        self,
        placement_plan: PlacementPlan,
        inputs: Any,
        *,
        input_kwargs: Optional[Dict[str, Any]] = None,
        differentiable: bool,
    ) -> PreparedExecutionContext:
        """Prepare initial seeds for staged execution."""

        execution_plan = placement_plan.partition_plan.execution_plan
        _, runtime_device = _prepare_runtime(execution_plan, inputs, self.device)
        if differentiable:
            available = _prepare_differentiable_seed_map(
                execution_plan,
                inputs,
                input_kwargs,
                runtime_device,
            )
        else:
            available = _prepare_seed_map(
                execution_plan,
                inputs,
                input_kwargs,
                runtime_device,
            )
        output_indices = set(execution_plan.output_node_indices) | _collect_output_indices(
            execution_plan.output_specs
        )
        return PreparedExecutionContext(
            execution_plan=execution_plan,
            runtime_device=runtime_device,
            available_values=dict(available),
            stored_values=dict(available),
            output_indices=output_indices,
        )

    def prepare_seeded_execution_context(
        self,
        placement_plan: PlacementPlan,
        seeded_values: Dict[int, Any],
    ) -> PreparedExecutionContext:
        """Prepare execution state when upstream stages already produced boundary values."""

        execution_plan = placement_plan.partition_plan.execution_plan
        _, runtime_device = _prepare_runtime(
            execution_plan,
            tuple(seeded_values.values()),
            self.device,
        )
        available = {
            index: _move_tree_to_device(value, runtime_device)
            for index, value in seeded_values.items()
        }
        output_indices = set(execution_plan.output_node_indices) | _collect_output_indices(
            execution_plan.output_specs
        )
        return PreparedExecutionContext(
            execution_plan=execution_plan,
            runtime_device=runtime_device,
            available_values=dict(available),
            stored_values=dict(available),
            output_indices=output_indices,
        )

    def reconstruct_outputs(
        self,
        placement_plan: PlacementPlan,
        stored_values: Dict[int, Any],
    ) -> Any:
        """Rebuild model outputs from retained values."""

        return _reconstruct_outputs(
            placement_plan.partition_plan.execution_plan,
            stored_values,
        )

    def get_output_indices(self, placement_plan: PlacementPlan) -> Set[int]:
        """Return all retained indices required to reconstruct final outputs."""

        execution_plan = placement_plan.partition_plan.execution_plan
        return set(execution_plan.output_node_indices) | _collect_output_indices(
            execution_plan.output_specs
        )

    def compute_loss(self, outputs: Any, targets: Any = None, loss_fn=None):
        """Compute a task loss from outputs and targets."""

        return _compute_loss(outputs, targets, loss_fn)

    def collect_tensor_grads(self, values: Dict[int, Any]) -> Dict[int, torch.Tensor]:
        """Collect gradients from exported tensors."""

        return _filter_tensor_grads(values)

    def run_stage_forward(
        self,
        placement_plan: PlacementPlan,
        stage: PartitionStage,
        seeded_values: Dict[int, Any],
        *,
        differentiable: bool,
        detach_boundary: bool,
        output_indices: Optional[Set[int]] = None,
    ) -> tuple[StageTrace, StageForwardResult]:
        """Execute a single stage and export detached outputs."""

        execution_plan = placement_plan.partition_plan.execution_plan
        _, runtime_device = _prepare_runtime(execution_plan, seeded_values, self.device)
        stage_seeded_values = {
            index: _move_tree_to_device(value, runtime_device)
            for index, value in seeded_values.items()
        }
        retained_output_indices = set(output_indices or ())
        computed, _ = _execute_nodes(
            execution_plan,
            node_indices=stage.node_indices,
            seeded_values=stage_seeded_values,
            device=runtime_device,
            preserve_rng=True,
            differentiable=differentiable,
            retain_nodes=set(stage.output_indices) | retained_output_indices,
            return_intermediates=False,
            model=execution_plan.model,
        )
        stage_trace = StageTrace(
            external_inputs={
                index: stage_seeded_values[index]
                for index in stage.input_indices
                if index not in execution_plan.input_node_indices and index in stage_seeded_values
            }
        )
        is_last_stage = stage.stage_id == placement_plan.partition_plan.stages[-1].stage_id
        exported = StageForwardResult()
        indices_to_export = set(stage.output_indices) | retained_output_indices
        for index in indices_to_export:
            if index not in computed:
                continue
            live_value = computed[index]
            should_track_grad = differentiable and (
                (is_last_stage and index in retained_output_indices)
                or (detach_boundary and not is_last_stage and index in stage.output_indices)
            )
            detached_value = _detach_tree_for_boundary(live_value, runtime_device)
            if should_track_grad:
                detached_value = _map_tree(detached_value, _enable_grad_for_transport)
            stage_trace.source_outputs[index] = live_value
            if index in stage.output_indices:
                stage_trace.transport_outputs[index] = detached_value
                exported.available_updates[index] = detached_value
            if index in retained_output_indices:
                exported.stored_updates[index] = detached_value
        return stage_trace, exported

    def run_stage_backward(
        self,
        placement_plan: PlacementPlan,
        stage_trace: StageTrace,
        upstream_grads: Dict[int, torch.Tensor],
    ) -> StageBackwardResult:
        """Run explicit backward for a single stage."""

        boundary_tensors = []
        boundary_grads = []
        for index, tensor in stage_trace.source_outputs.items():
            if (
                index not in upstream_grads
                or not isinstance(tensor, torch.Tensor)
                or not tensor.requires_grad
            ):
                continue
            boundary_tensors.append(tensor)
            boundary_grads.append(upstream_grads[index].to(tensor.device))
        if boundary_tensors:
            torch.autograd.backward(boundary_tensors, boundary_grads)

        parameter_grads: Dict[str, torch.Tensor] = {}
        model = placement_plan.partition_plan.execution_plan.model
        if model is not None:
            for name, parameter in model.named_parameters():
                if isinstance(parameter.grad, torch.Tensor):
                    parameter_grads[name] = parameter.grad.detach().clone()
            model.zero_grad(set_to_none=True)

        input_grads = _filter_tensor_grads(stage_trace.external_inputs)
        for index, value in stage_trace.external_inputs.items():
            if index in input_grads or index not in upstream_grads:
                continue
            if isinstance(value, torch.Tensor):
                input_grads[index] = upstream_grads[index].detach().clone().to(value.device)

        return StageBackwardResult(
            input_grads=input_grads,
            parameter_grads=parameter_grads,
        )

    def _run_forward(
        self,
        placement_plan: PlacementPlan,
        inputs: Any,
        *,
        input_kwargs: Optional[Dict[str, Any]] = None,
        differentiable: bool,
        detach_boundaries: bool,
    ) -> ForwardTrace:
        execution_plan = placement_plan.partition_plan.execution_plan
        _, runtime_device = _prepare_runtime(execution_plan, inputs, self.device)
        if differentiable:
            available = _prepare_differentiable_seed_map(
                execution_plan,
                inputs,
                input_kwargs,
                runtime_device,
            )
        else:
            available = _prepare_seed_map(
                execution_plan,
                inputs,
                input_kwargs,
                runtime_device,
            )

        stored_values: Dict[int, Any] = dict(available)
        stage_traces: Dict[str, StageTrace] = {}
        output_indices = set(execution_plan.output_node_indices) | _collect_output_indices(
            execution_plan.output_specs
        )

        for stage in placement_plan.partition_plan.stages:
            seeded_values = {
                index: available[index]
                for index in set(stage.input_indices) | set(execution_plan.input_node_indices)
                if index in available
            }
            computed, _ = _execute_nodes(
                execution_plan,
                node_indices=stage.node_indices,
                seeded_values=seeded_values,
                device=runtime_device,
                preserve_rng=True,
                differentiable=differentiable,
                retain_nodes=set(stage.output_indices) | output_indices,
                return_intermediates=False,
                model=execution_plan.model,
            )
            stage_trace = StageTrace(
                external_inputs={
                    index: seeded_values[index]
                    for index in stage.input_indices
                    if index not in execution_plan.input_node_indices and index in seeded_values
                }
            )
            for index in output_indices:
                if index in computed:
                    stored_values[index] = computed[index]
            for index in stage.output_indices:
                value = computed[index]
                stage_trace.source_outputs[index] = value
                if (
                    detach_boundaries
                    and stage.stage_id != placement_plan.partition_plan.stages[-1].stage_id
                ):
                    transported = _detach_tree_for_boundary(value, runtime_device)
                    if isinstance(transported, torch.Tensor) and transported.is_floating_point():
                        transported.requires_grad_(True)
                        transported.retain_grad()
                    stage_trace.transport_outputs[index] = transported
                    available[index] = transported
                    stored_values[index] = transported
                else:
                    available[index] = value
                    stored_values[index] = value
            stage_traces[stage.stage_id] = stage_trace

        outputs = _reconstruct_outputs(execution_plan, stored_values)
        return ForwardTrace(outputs=outputs, stored_values=stored_values, stage_traces=stage_traces)

    def run_eval(
        self,
        placement_plan: PlacementPlan,
        inputs: Any,
        *,
        input_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Run evaluation with torch.no_grad() for memory efficiency."""
        with torch.no_grad():
            trace = self._run_forward(
                placement_plan,
                inputs,
                input_kwargs=input_kwargs,
                differentiable=False,
                detach_boundaries=False,
            )
            return trace.outputs

    def run_train(
        self,
        placement_plan: PlacementPlan,
        inputs: Any,
        *,
        input_kwargs: Optional[Dict[str, Any]] = None,
        targets: Any = None,
        loss_fn=None,
        optimizer=None,
        zero_grad: bool = True,
        step_optimizer: bool = True,
        clear_cuda_cache: bool = False,
    ) -> Dict[str, Any]:
        """Run training with optimized backward pass and optional memory cleanup.
        
        Args:
            placement_plan: The placement plan defining stage execution.
            inputs: Model inputs.
            input_kwargs: Optional keyword arguments for the model.
            targets: Target values for loss computation.
            loss_fn: Loss function to use.
            optimizer: Optimizer for parameter updates.
            zero_grad: Whether to zero gradients before forward pass.
            step_optimizer: Whether to step the optimizer after backward.
            clear_cuda_cache: Whether to clear CUDA cache after each stage.
        
        Returns:
            Dictionary with training results.
        """
        execution_plan = placement_plan.partition_plan.execution_plan
        model = execution_plan.model
        if model is None:
            raise ValueError("The execution plan must keep a live model reference for training.")
        
        # Efficient gradient zeroing
        if zero_grad:
            model.zero_grad(set_to_none=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

        trace = self._run_forward(
            placement_plan,
            inputs,
            input_kwargs=input_kwargs,
            differentiable=True,
            detach_boundaries=True,
        )
        loss = _compute_loss(trace.outputs, targets, loss_fn)
        loss.backward()

        stages = placement_plan.partition_plan.stages
        upstream_grads = _filter_tensor_grads(trace.stage_traces[stages[-1].stage_id].external_inputs)
        
        # Optimized backward pass through stages
        for stage in reversed(stages[:-1]):
            stage_trace = trace.stage_traces[stage.stage_id]
            
            # Pre-allocate lists with known size hint
            source_items = list(stage_trace.source_outputs.items())
            boundary_tensors = []
            boundary_grads = []
            
            for index, tensor in source_items:
                if index not in upstream_grads or not isinstance(tensor, torch.Tensor):
                    continue
                if tensor.requires_grad:
                    boundary_tensors.append(tensor)
                    boundary_grads.append(upstream_grads[index].to(tensor.device))
            
            if boundary_tensors:
                torch.autograd.backward(boundary_tensors, boundary_grads)
            
            upstream_grads = _filter_tensor_grads(stage_trace.external_inputs)
            
            # Optional memory cleanup
            if clear_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if optimizer is not None and step_optimizer:
            optimizer.step()

        return {
            "output": trace.outputs,
            "loss": loss,
            "stage_count": placement_plan.partition_plan.stage_count,
            "stage_to_worker": dict(placement_plan.stage_to_worker),
        }
