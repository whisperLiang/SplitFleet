"""TorchLens runtime manager for SplitFleet server-side suffix execution."""

from __future__ import annotations

import copy
from typing import Any, Optional

from splitfleet.autosplit import SplitRuntimeHandle
from splitfleet.autosplit.runtime import AutoSplitSession
from splitfleet.autosplit.types import SplitPlan, WorkerSpec
from splitfleet.server.server_model.manager.grpc_manager import GrpcServerModelManager
from splitfleet.server.server_model.manager.manager import ServerModelManager
from splitfleet.server.stage_runtime.registry import GLOBAL_WORKER_REGISTRY, WorkerRegistry


REMOTE_STAGE_ERROR = (
    "TorchLens autosplit backend supports coordinator-local suffix execution only; "
    "node-level remote stage execution is not active."
)


class StageRuntimeManager(ServerModelManager):
    """Bridge TorchLens split runtimes into the existing server-model lifecycle."""

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
        self._placement_plan: Optional[SplitPlan] = None
        self._runtime_handle: Optional[SplitRuntimeHandle] = None
        self._delegate = (
            GrpcServerModelManager(init_server_model_fn=init_server_model_fn)
            if init_server_model_fn is not None
            else None
        )

    def set_placement_plan(self, placement_plan: SplitPlan) -> None:
        self._placement_plan = placement_plan
        handle = placement_plan.metadata.get("_runtime_handle")
        if isinstance(handle, SplitRuntimeHandle):
            self.bind_runtime_handle(handle)

    def get_placement_plan(self) -> Optional[SplitPlan]:
        return self._placement_plan

    def bind_runtime_handle(self, runtime_handle: SplitRuntimeHandle) -> None:
        self._runtime_handle = runtime_handle
        self.autosplit_session._runtime_handles[runtime_handle.plan.plan_id] = runtime_handle

    def register_worker(self, worker_spec: WorkerSpec) -> WorkerSpec:
        return self.worker_registry.register(worker_spec)

    def list_workers(self, *, online_only: bool = True) -> list[WorkerSpec]:
        return self.worker_registry.list_workers(online_only=online_only)

    def _require_runtime_handle(self) -> SplitRuntimeHandle:
        if self._runtime_handle is None:
            raise RuntimeError("No TorchLens runtime handle is active.")
        return self._runtime_handle

    def clone_runtime_for_model(
        self,
        model,
        *,
        suffix: str = "",
    ) -> SplitRuntimeHandle:
        base = self._require_runtime_handle()
        sample_inputs = base.plan.metadata.get("_example_inputs")
        if sample_inputs is None:
            raise RuntimeError("The active TorchLens runtime does not retain sample inputs.")
        handle = self.autosplit_session.prepare_runtime(
            model,
            sample_inputs,
            boundary=base.plan.boundary,
            mode=base.plan.mode,
            trainable=base.plan.trainable,
            dynamic_batch=base.plan.dynamic_batch,
            trace_batch_mode=base.plan.trace_batch_mode,
        )
        if suffix:
            original_plan_id = handle.plan.plan_id
            handle.plan.plan_id = f"{handle.plan.plan_id}_{suffix}"
            self.autosplit_session._runtime_handles.pop(original_plan_id, None)
            self.autosplit_session._runtime_handles[handle.plan.plan_id] = handle
        return handle

    def clone_placement_plan(
        self,
        *,
        model=None,
        plan_id_suffix: Optional[str] = None,
    ) -> SplitPlan:
        if self._placement_plan is None:
            raise RuntimeError("No autosplit placement plan is active.")
        cloned = copy.copy(self._placement_plan)
        cloned.metadata = dict(self._placement_plan.metadata)
        if model is not None:
            cloned.metadata["_runtime_handle"] = self.clone_runtime_for_model(
                model,
                suffix=plan_id_suffix or "",
            )
        if plan_id_suffix:
            cloned.plan_id = f"{self._placement_plan.plan_id}_{plan_id_suffix}"
        return cloned

    def run_eval(self, inputs: Any) -> Any:
        return self.autosplit_session.run_eval(self._require_runtime_handle(), inputs)

    def run_eval_plan(self, runtime_handle: SplitRuntimeHandle, inputs: Any) -> Any:
        return self.autosplit_session.run_eval(runtime_handle, inputs)

    def run_train(
        self,
        inputs: Any,
        *,
        targets: Any = None,
        loss_fn=None,
        prefix_optimizer=None,
        suffix_optimizer=None,
    ) -> dict:
        return self.autosplit_session.run_train(
            self._require_runtime_handle(),
            inputs,
            targets,
            loss_fn=loss_fn,
            prefix_optimizer=prefix_optimizer,
            suffix_optimizer=suffix_optimizer,
        )

    def run_train_plan(
        self,
        runtime_handle: SplitRuntimeHandle,
        inputs: Any,
        *,
        targets: Any = None,
        loss_fn=None,
        prefix_optimizer=None,
        suffix_optimizer=None,
    ) -> dict:
        return self.autosplit_session.run_train(
            runtime_handle,
            inputs,
            targets,
            loss_fn=loss_fn,
            prefix_optimizer=prefix_optimizer,
            suffix_optimizer=suffix_optimizer,
        )

    def run_eval_tail_plan(self, runtime_handle: SplitRuntimeHandle, boundary) -> Any:
        return self.autosplit_session.run_suffix_eval(runtime_handle, boundary)

    def run_train_tail_plan(
        self,
        runtime_handle: SplitRuntimeHandle,
        boundary,
        *,
        targets: Any = None,
        loss_fn=None,
        optimizer=None,
    ) -> dict:
        return self.autosplit_session.run_suffix_train(
            runtime_handle,
            boundary,
            targets,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

    def run_stage_forward(self, *args, **kwargs):
        _ = (args, kwargs)
        raise NotImplementedError(REMOTE_STAGE_ERROR)

    def run_stage_backward(self, *args, **kwargs):
        _ = (args, kwargs)
        raise NotImplementedError(REMOTE_STAGE_ERROR)

    def clear_execution(self, *args, **kwargs):
        _ = (args, kwargs)
        raise NotImplementedError(REMOTE_STAGE_ERROR)

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
