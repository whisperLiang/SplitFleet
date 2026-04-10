"""Client-side autosplit split-learning adapter with local prefix execution."""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch

from slbd.autosplit import AutoSplitSession, PlacementPlan
from slbd.autosplit.planner import build_partition_plan
from slbd.autosplit.serde import dumps_torch_object, loads_torch_object
from slbd.client.numpy_client import NumPyClient
from slbd.common import BatchData, ControlCode
from slbd.common.constants import (
    AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_CUTOFFS_CONFIG_KEY,
    AUTOSPLIT_PLAN_ID_CONFIG_KEY,
    AUTOSPLIT_STAGE_TO_WORKER_CONFIG_KEY,
)


def _model_to_ndarrays(model: torch.nn.Module) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy() for tensor in model.state_dict().values()]


def _load_model_from_ndarrays(model: torch.nn.Module, ndarrays: list[np.ndarray]) -> None:
    if not ndarrays:
        return
    state_dict = model.state_dict()
    if len(state_dict) != len(ndarrays):
        raise ValueError(
            "Autosplit split client parameter mismatch: "
            f"expected {len(state_dict)} tensors, received {len(ndarrays)}."
        )
    loaded_state = OrderedDict()
    for (name, reference), array in zip(state_dict.items(), ndarrays):
        loaded_state[name] = torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
    model.load_state_dict(loaded_state, strict=True)


def _move_to_device(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _batch_size(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.shape[0]) if value.ndim > 0 else 1
    if isinstance(value, np.ndarray):
        return int(value.shape[0]) if value.ndim > 0 else 1
    if isinstance(value, dict):
        for item in value.values():
            size = _batch_size(item)
            if size > 0:
                return size
    if isinstance(value, (list, tuple)) and value:
        return _batch_size(value[0])
    return 1


class AutoSplitSplitLearningClient(NumPyClient):
    """Run local prefix stages and delegate the autosplit tail to the server runtime."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_data: Iterable[Any],
        sample_inputs: Any,
        evaluate_data: Optional[Iterable[Any]] = None,
        sample_kwargs: Optional[dict] = None,
        batch_adapter: Optional[Callable[[Any], tuple[Any, Any]]] = None,
        optimizer_fn=None,
        autosplit_session: Optional[AutoSplitSession] = None,
        device: str = "cpu",
    ) -> None:
        self.model = copy.deepcopy(model).to(device)
        self.train_data = train_data
        self.evaluate_data = evaluate_data if evaluate_data is not None else train_data
        self.sample_inputs = sample_inputs
        self.sample_kwargs = dict(sample_kwargs or {})
        self.batch_adapter = batch_adapter or self._default_batch_adapter
        self.optimizer_fn = optimizer_fn
        self.autosplit_session = autosplit_session or AutoSplitSession(device=device)
        self.device = device
        self._plan_cache: dict[str, PlacementPlan] = {}
        self._execution_plan = self._compile_execution_plan()

    def get_parameters(self, config):
        _ = config
        return _model_to_ndarrays(self.model)

    def fit(self, parameters, config):
        placement_plan, client_stage_count = self._prepare_round(parameters, config)
        self.model.train()
        optimizer = self._build_optimizer()

        num_examples = 0
        weighted_loss = 0.0
        for batch in self.train_data:
            inputs, targets = self.batch_adapter(batch)
            torch_inputs = _move_to_device(inputs, self.device)
            torch_targets = _move_to_device(targets, self.device)

            self.model.zero_grad(set_to_none=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            _, stage_traces, seeded_values = self._run_local_prefix_forward(
                placement_plan,
                torch_inputs,
                client_stage_count=client_stage_count,
            )
            response = self._call_tail(
                method_name="train_tail",
                seeded_values=seeded_values,
                targets=torch_targets,
                num_examples=_batch_size(torch_inputs),
            )
            self._run_local_prefix_backward(
                placement_plan,
                stage_traces,
                response["boundary_grads"],
                client_stage_count=client_stage_count,
            )
            if optimizer is not None:
                optimizer.step()

            batch_examples = int(response["num_examples"])
            batch_loss = float(response["loss"])
            num_examples += batch_examples
            weighted_loss += batch_loss * batch_examples

        metrics = {}
        if num_examples > 0:
            metrics["loss"] = weighted_loss / num_examples
        return _model_to_ndarrays(self.model), num_examples, metrics

    def evaluate(self, parameters, config):
        placement_plan, client_stage_count = self._prepare_round(parameters, config)
        self.model.eval()

        num_examples = 0
        weighted_loss = 0.0
        with torch.no_grad():
            for batch in self.evaluate_data:
                inputs, targets = self.batch_adapter(batch)
                torch_inputs = _move_to_device(inputs, self.device)
                torch_targets = _move_to_device(targets, self.device)
                _, _, seeded_values = self._run_local_prefix_forward(
                    placement_plan,
                    torch_inputs,
                    client_stage_count=client_stage_count,
                    differentiable=False,
                )
                response = self._call_tail(
                    method_name="evaluate_tail",
                    seeded_values=seeded_values,
                    targets=torch_targets,
                    num_examples=_batch_size(torch_inputs),
                )
                batch_examples = int(response["num_examples"])
                batch_loss = float(response["loss"])
                num_examples += batch_examples
                weighted_loss += batch_loss * batch_examples

        average_loss = weighted_loss / max(num_examples, 1)
        return float(average_loss), num_examples, {"loss": average_loss}

    def _prepare_round(self, parameters, config) -> tuple[PlacementPlan, int]:
        _load_model_from_ndarrays(self.model, parameters)
        placement_plan = self._ensure_placement_plan(config)
        client_stage_count = int(config.get(AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY, 1))
        if client_stage_count <= 0:
            raise ValueError("AutoSplitSplitLearningClient requires at least one client-local stage.")
        if client_stage_count >= placement_plan.partition_plan.stage_count:
            raise ValueError("Client-local stage count must leave at least one tail stage.")
        return placement_plan, client_stage_count

    def _ensure_placement_plan(self, config) -> PlacementPlan:
        plan_id = str(config[AUTOSPLIT_PLAN_ID_CONFIG_KEY])
        if plan_id in self._plan_cache:
            cached = self._plan_cache[plan_id]
            cached.partition_plan.execution_plan.set_model(self.model)
            return cached

        cutoffs = json.loads(config[AUTOSPLIT_CUTOFFS_CONFIG_KEY])
        stage_to_worker = json.loads(config[AUTOSPLIT_STAGE_TO_WORKER_CONFIG_KEY])
        self._execution_plan.set_model(self.model)
        partition_plan = build_partition_plan(
            self._execution_plan,
            cutoffs,
            model_name=self._execution_plan.model_name,
        )
        placement_plan = PlacementPlan(
            partition_plan=partition_plan,
            stage_to_worker=stage_to_worker,
            worker_specs={},
            score=0.0,
            metadata={"plan_id": plan_id},
        )
        self._plan_cache[plan_id] = placement_plan
        return placement_plan

    def _compile_execution_plan(self):
        traced = self.autosplit_session.planner.tracer.trace(
            self.model,
            self.sample_inputs,
            sample_kwargs=self.sample_kwargs or None,
        )
        traced.execution_plan.set_model(self.model)
        return traced.execution_plan

    def _run_local_prefix_forward(
        self,
        placement_plan: PlacementPlan,
        inputs: Any,
        *,
        client_stage_count: int,
        differentiable: bool = True,
    ):
        context = self.autosplit_session.prepare_execution_context(
            placement_plan,
            inputs,
            differentiable=differentiable,
        )
        execution_plan = placement_plan.partition_plan.execution_plan
        local_stages = placement_plan.partition_plan.stages[:client_stage_count]
        stage_traces = {}
        for stage in local_stages:
            seeded_values = {
                index: context.available_values[index]
                for index in set(stage.input_indices) | set(execution_plan.input_node_indices)
                if index in context.available_values
            }
            stage_trace, result = self.autosplit_session.run_stage_forward(
                placement_plan,
                stage,
                seeded_values,
                differentiable=differentiable,
                detach_boundary=True,
                output_indices=set(),
            )
            stage_traces[stage.stage_id] = stage_trace
            context.available_values.update(result.available_updates)
            context.stored_values.update(result.stored_updates)

        first_remote_stage = placement_plan.partition_plan.stages[client_stage_count]
        remote_seeded_values = {
            index: context.available_values[index]
            for index in set(first_remote_stage.input_indices)
            | set(execution_plan.input_node_indices)
            if index in context.available_values
        }
        return context, stage_traces, remote_seeded_values

    def _run_local_prefix_backward(
        self,
        placement_plan: PlacementPlan,
        stage_traces,
        boundary_grads,
        *,
        client_stage_count: int,
    ) -> None:
        upstream_grads = {
            int(index): grad.to(self.device)
            for index, grad in boundary_grads.items()
        }
        local_stages = placement_plan.partition_plan.stages[:client_stage_count]
        accumulated_parameter_grads = {}
        for stage in reversed(local_stages):
            result = self.autosplit_session.run_stage_backward(
                placement_plan,
                stage_traces[stage.stage_id],
                upstream_grads,
            )
            for name, grad in result.parameter_grads.items():
                if name in accumulated_parameter_grads:
                    accumulated_parameter_grads[name] = accumulated_parameter_grads[name] + grad
                else:
                    accumulated_parameter_grads[name] = grad
            upstream_grads = result.input_grads

        for name, parameter in self.model.named_parameters():
            grad = accumulated_parameter_grads.get(name)
            if grad is None:
                continue
            parameter.grad = grad.to(parameter.device)

    def _call_tail(
        self,
        *,
        method_name: str,
        seeded_values,
        targets,
        num_examples: int,
    ):
        payload = {
            "seeded_values": seeded_values,
            "targets": targets,
            "num_examples": num_examples,
        }
        request = BatchData(
            data={"payload": dumps_torch_object(payload)},
            control_code=ControlCode.OK,
        )
        response = getattr(self._require_server_model_proxy(), method_name)(
            request,
            _streams_=False,
        )
        return loads_torch_object(response.data["payload"], map_location=self.device)

    def _build_optimizer(self):
        if self.optimizer_fn is not None:
            return self.optimizer_fn(self.model)
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        if not trainable:
            return None
        return torch.optim.SGD(trainable, lr=0.01)

    def _require_server_model_proxy(self):
        proxy = getattr(self, "server_model_proxy", None)
        if proxy is None:
            raise RuntimeError("AutoSplitSplitLearningClient requires a server_model_proxy.")
        return proxy

    @staticmethod
    def _default_batch_adapter(batch: Any) -> tuple[Any, Any]:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        raise ValueError(
            "Expected each batch to be a `(inputs, targets)` pair. "
            "Provide `batch_adapter` to customize parsing."
        )
