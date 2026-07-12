"""TorchLens native autosplit runtime facade."""

from __future__ import annotations

from typing import Any, Optional

import torch

from splitfleet.autosplit.cache import PlanCacheStore
from splitfleet.autosplit.planner import AutoSplitPlanner
from splitfleet.autosplit.torchlens_backend import (
    SplitRuntimeHandle,
    TorchLensRuntimeHandle,
    backward_prefix,
    prepare_torchlens_runtime,
)
from splitfleet.autosplit.torchlens_runtime import normalize_example_inputs, require_torchlens_231
from splitfleet.autosplit.types import SplitPlan


def normalize_inputs(inputs: Any) -> tuple[Any, ...]:
    return normalize_example_inputs(inputs)


def _nested_tensor_loss(value: Any) -> torch.Tensor:
    losses: list[torch.Tensor] = []

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if item.is_floating_point() or item.is_complex():
                losses.append(item.float().square().mean())
            else:
                losses.append(item.float().mean())
            return
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    if not losses:
        raise ValueError("Cannot compute a default loss for outputs without tensors.")
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total


def compute_loss(outputs: Any, targets: Any = None, loss_fn=None) -> torch.Tensor:
    if loss_fn is not None:
        if targets is None:
            return loss_fn(outputs)
        return loss_fn(outputs, targets)
    if targets is not None and isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor):
        return torch.nn.functional.mse_loss(outputs, targets)
    return _nested_tensor_loss(outputs)


class AutoSplitSession:
    """High-level session that prepares and executes TorchLens split runtimes."""

    def __init__(
        self,
        planner: Optional[AutoSplitPlanner] = None,
        *,
        cache_store: Optional[PlanCacheStore] = None,
        device: str = "cpu",
        backend: str = "torchlens",
    ) -> None:
        if backend != "torchlens":
            raise ValueError(f"Only the TorchLens autosplit backend is supported, got {backend!r}.")
        require_torchlens_231()
        self.planner = planner or AutoSplitPlanner()
        self.cache_store = cache_store
        self.device = device
        self.backend = backend
        self._runtime_handles: dict[str, SplitRuntimeHandle] = {}

    def plan(
        self,
        model,
        sample_inputs: Any,
        *,
        sample_kwargs: Optional[dict[str, Any]] = None,
        worker_specs=None,
        constraints=None,
        objective=None,
        preferred_stage_count: Optional[int] = None,
        client_stage_count: Optional[int] = 1,
        model_name: Optional[str] = None,
        boundary: str = "50%",
        mode: str = "generated_eager",
        trainable: bool = True,
        dynamic_batch: tuple[int, int] | None = None,
        trace_batch_mode: str | None = None,
        compile_options: Any = None,
    ) -> SplitPlan:
        placement = self.planner.plan(
            model,
            sample_inputs,
            sample_kwargs=sample_kwargs,
            worker_specs=worker_specs,
            constraints=constraints,
            objective=objective,
            preferred_stage_count=preferred_stage_count,
            client_stage_count=client_stage_count,
            cache_store=self.cache_store,
            model_name=model_name,
            boundary=boundary,
            mode=mode,
            trainable=trainable,
            dynamic_batch=dynamic_batch,
            trace_batch_mode=trace_batch_mode,
            compile_options=compile_options,
        )
        handle = placement.metadata.get("_runtime_handle")
        if isinstance(handle, TorchLensRuntimeHandle):
            self._runtime_handles[placement.plan_id] = handle
        return placement

    def prepare_runtime(
        self,
        model,
        sample_inputs: Any,
        *,
        boundary: str = "50%",
        mode: str = "generated_eager",
        trainable: bool = True,
        dynamic_batch: tuple[int, int] | None = None,
        trace_batch_mode: str | None = None,
        objective: Any = None,
        compile_options: Any = None,
    ) -> SplitRuntimeHandle:
        del objective, compile_options
        handle = prepare_torchlens_runtime(
            model,
            sample_inputs,
            boundary=boundary,
            mode=mode,
            trainable=trainable,
            dynamic_batch=dynamic_batch,
            trace_batch_mode=trace_batch_mode,
            model_name=model.__class__.__name__,
        )
        self._runtime_handles[handle.plan.plan_id] = handle
        return handle

    def get_runtime_handle(self, value: SplitRuntimeHandle | SplitPlan | str) -> SplitRuntimeHandle:
        if isinstance(value, TorchLensRuntimeHandle):
            return value
        if isinstance(value, SplitPlan):
            handle = value.metadata.get("_runtime_handle")
            if isinstance(handle, TorchLensRuntimeHandle):
                return handle
            value = value.plan_id
        try:
            return self._runtime_handles[str(value)]
        except KeyError as exc:
            raise RuntimeError(f"No TorchLens runtime handle is registered for plan {value!r}.") from exc

    @staticmethod
    def _runtime_context(handle: SplitRuntimeHandle) -> str:
        plan = getattr(handle, "plan", None)
        return (
            f"torchlens_version={getattr(handle, 'torchlens_version', getattr(plan, 'torchlens_version', ''))!r}, "
            f"boundary={getattr(plan, 'boundary', '')!r}, "
            f"candidate_id={getattr(plan, 'candidate_id', '')!r}, "
            f"feature_abi_id={getattr(handle, 'feature_abi_id', getattr(plan, 'feature_abi_id', ''))!r}"
        )

    def compute_loss(self, outputs: Any, targets: Any = None, loss_fn=None) -> torch.Tensor:
        return compute_loss(outputs, targets, loss_fn)

    def run_eval(self, runtime_handle: SplitRuntimeHandle | SplitPlan, inputs: Any) -> Any:
        handle = self.get_runtime_handle(runtime_handle)
        with torch.no_grad():
            try:
                boundary = handle.backend.run_prefix(*normalize_inputs(inputs))
                return handle.backend.run_suffix(boundary)
            except Exception as exc:
                raise RuntimeError(f"TorchLens split eval failed ({self._runtime_context(handle)}).") from exc

    def run_train(
        self,
        runtime_handle: SplitRuntimeHandle | SplitPlan,
        inputs: Any,
        targets: Any,
        *,
        loss_fn=None,
        prefix_optimizer=None,
        suffix_optimizer=None,
    ) -> dict[str, Any]:
        handle = self.get_runtime_handle(runtime_handle)
        try:
            boundary = handle.backend.run_prefix(*normalize_inputs(inputs), training=True)
            loss, boundary_grads = handle.backend.train_suffix(
                boundary,
                targets,
                loss_fn=loss_fn,
                optimizer=suffix_optimizer,
            )
            backward_prefix(
                handle,
                boundary,
                boundary_grads=boundary_grads,
                optimizer=prefix_optimizer,
            )
        except Exception as exc:
            raise RuntimeError(f"TorchLens split train failed ({self._runtime_context(handle)}).") from exc
        return {
            "loss": loss,
            "boundary_grads": boundary_grads,
            "output": None,
            "split_id": handle.plan.split_id,
            "graph_signature": handle.plan.graph_signature,
        }

    def run_suffix_eval(
        self,
        runtime_handle: SplitRuntimeHandle | SplitPlan,
        boundary,
    ) -> Any:
        handle = self.get_runtime_handle(runtime_handle)
        return handle.backend.run_suffix(boundary)

    def run_suffix_train(
        self,
        runtime_handle: SplitRuntimeHandle | SplitPlan,
        boundary,
        targets: Any,
        *,
        loss_fn=None,
        optimizer=None,
    ) -> dict[str, Any]:
        handle = self.get_runtime_handle(runtime_handle)
        loss, boundary_grads = handle.backend.train_suffix(
            boundary,
            targets,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        return {
            "loss": loss,
            "boundary_grads": boundary_grads,
        }
