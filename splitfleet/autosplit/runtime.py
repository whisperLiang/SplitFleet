"""TorchLens native autosplit runtime facade."""

from __future__ import annotations

from typing import Any, Optional

from splitfleet.autosplit.cache import PlanCacheStore
from splitfleet.autosplit.planner import AutoSplitPlanner
from splitfleet.autosplit.torchlens_backend import (
    TorchLensRuntimeHandle,
    backward_prefix,
    prepare_torchlens_runtime,
)
from splitfleet.autosplit.torchlens_runtime import require_torchlens_version
from splitfleet.autosplit.types import SplitPlacementPlan
from splitfleet.tasks import ModelInputs
from splitfleet.backends.utils import inference_context


def compute_loss(outputs: Any, targets: Any = None, loss_fn=None) -> Any:
    """Evaluate the caller-provided task objective."""
    if loss_fn is None:
        raise ValueError("An explicit loss_fn is required for task evaluation and training.")
    return loss_fn(outputs) if targets is None else loss_fn(outputs, targets)


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
        require_torchlens_version()
        self.planner = planner or AutoSplitPlanner()
        self.cache_store = cache_store
        self.device = device
        self.backend = backend
        self._runtime_handles: dict[str, TorchLensRuntimeHandle] = {}

    def plan(
        self,
        model,
        sample_inputs: Any,
        *,
        sample_kwargs: Optional[dict[str, Any]] = None,
        batch_axes: dict[str, int] | None = None,
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
    ) -> SplitPlacementPlan:
        placement = self.planner.plan(
            model,
            sample_inputs,
            sample_kwargs=sample_kwargs,
            batch_axes=batch_axes,
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
        sample_kwargs: dict[str, Any] | None = None,
        batch_axes: dict[str, int] | None = None,
        boundary: str = "50%",
        mode: str = "generated_eager",
        trainable: bool = True,
        dynamic_batch: tuple[int, int] | None = None,
        trace_batch_mode: str | None = None,
    ) -> TorchLensRuntimeHandle:
        call = ModelInputs.from_value(sample_inputs)
        handle = prepare_torchlens_runtime(
            model,
            call.args,
            sample_kwargs=dict(call.kwargs) if sample_kwargs is None else sample_kwargs,
            batch_axes=batch_axes,
            boundary=boundary,
            mode=mode,
            trainable=trainable,
            dynamic_batch=dynamic_batch,
            trace_batch_mode=trace_batch_mode,
            model_name=model.__class__.__name__,
        )
        self._runtime_handles[handle.plan.plan_id] = handle
        return handle

    def get_runtime_handle(self, value: TorchLensRuntimeHandle | SplitPlacementPlan | str) -> TorchLensRuntimeHandle:
        if isinstance(value, TorchLensRuntimeHandle):
            return value
        if isinstance(value, SplitPlacementPlan):
            handle = value.metadata.get("_runtime_handle")
            if isinstance(handle, TorchLensRuntimeHandle):
                return handle
            value = value.plan_id
        try:
            return self._runtime_handles[str(value)]
        except KeyError as exc:
            raise RuntimeError(f"No TorchLens runtime handle is registered for plan {value!r}.") from exc

    @staticmethod
    def _runtime_context(handle: TorchLensRuntimeHandle) -> str:
        plan = handle.plan
        return (
            f"torchlens_version={plan.torchlens_version!r}, "
            f"boundary={plan.boundary!r}, "
            f"candidate_id={plan.candidate_id!r}, "
            f"feature_abi_id={plan.feature_abi_id!r}"
        )

    def compute_loss(self, outputs: Any, targets: Any = None, loss_fn=None) -> Any:
        return compute_loss(outputs, targets, loss_fn)

    def run_eval(self, runtime_handle: TorchLensRuntimeHandle | SplitPlacementPlan, inputs: Any) -> Any:
        handle = self.get_runtime_handle(runtime_handle)
        call = ModelInputs.from_value(inputs)
        with inference_context(handle.backend.framework_backend):
            try:
                boundary = handle.backend.run_prefix(*call.args, input_kwargs=dict(call.kwargs))
                return handle.backend.run_suffix(boundary)
            except Exception as exc:
                raise RuntimeError(f"TorchLens split eval failed ({self._runtime_context(handle)}).") from exc

    def run_train(
        self,
        runtime_handle: TorchLensRuntimeHandle | SplitPlacementPlan,
        inputs: Any,
        targets: Any,
        *,
        loss_fn=None,
        prefix_optimizer=None,
        suffix_optimizer=None,
    ) -> dict[str, Any]:
        handle = self.get_runtime_handle(runtime_handle)
        if loss_fn is None:
            raise ValueError("Split training requires an explicit loss_fn.")
        call = ModelInputs.from_value(inputs)
        try:
            boundary = handle.backend.run_prefix(*call.args, training=True, input_kwargs=dict(call.kwargs))
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
        runtime_handle: TorchLensRuntimeHandle | SplitPlacementPlan,
        boundary,
    ) -> Any:
        handle = self.get_runtime_handle(runtime_handle)
        return handle.backend.run_suffix(boundary)

    def run_suffix_train(
        self,
        runtime_handle: TorchLensRuntimeHandle | SplitPlacementPlan,
        boundary,
        targets: Any,
        *,
        loss_fn=None,
        optimizer=None,
    ) -> dict[str, Any]:
        handle = self.get_runtime_handle(runtime_handle)
        if loss_fn is None:
            raise ValueError("Split training requires an explicit loss_fn.")
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
