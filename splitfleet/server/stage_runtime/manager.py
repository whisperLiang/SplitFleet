"""TorchLens runtime manager for SplitFleet server-side suffix execution."""

from __future__ import annotations

import hashlib

from splitfleet.autosplit import TorchLensRuntimeHandle
from splitfleet.autosplit.runtime import AutoSplitSession
from splitfleet.autosplit.torchlens_clone import clone_runtime_handle
from splitfleet.autosplit.torchlens_contract import (
    classify_contract_compatibility,
    runtime_contract_digest,
    stable_json,
)
from splitfleet.autosplit.types import SplitPlacementPlan
from splitfleet.server.server_model.manager.grpc_manager import GrpcServerModelManager
from splitfleet.server.server_model.manager.manager import ServerModelManager


class StageRuntimeManager(ServerModelManager):
    """Bridge TorchLens split runtimes into the existing server-model lifecycle."""

    def __init__(
        self,
        *,
        init_server_model_fn=None,
        autosplit_session: AutoSplitSession | None = None,
        replica_capacity: int | None = None,
        state_store=None,
        reuse_completed_replicas: bool = False,
        per_client_models: bool = False,
        model_capacity: int | None = None,
        replica_rng_isolation: bool = True,
    ) -> None:
        super().__init__()
        self.autosplit_session = autosplit_session or AutoSplitSession()
        self.replica_rng_isolation = bool(replica_rng_isolation)
        self.before_collect = None
        self._placement_plan: SplitPlacementPlan | None = None
        self._runtime_handle: TorchLensRuntimeHandle | None = None
        self._placement_plans: dict[str, SplitPlacementPlan] = {}
        self._runtime_handles_by_plan_id: dict[str, TorchLensRuntimeHandle] = {}
        self._clone_runtime_cache: dict[str, TorchLensRuntimeHandle] = {}
        self._delegate = (
            GrpcServerModelManager(
                init_server_model_fn=init_server_model_fn,
                persistent_models=True,
            )
            if init_server_model_fn is not None
            else None
        )
        if replica_capacity is not None:
            if init_server_model_fn is None or state_store is None or replica_capacity < 1:
                raise ValueError("Leased replicas require a factory, state store and positive capacity")
            from .leased_manager import LeasedServerModelManager
            self._delegate = LeasedServerModelManager(
                init_server_model_fn, capacity=replica_capacity,
                state_store=state_store, release_model=(
                    self.offload_client_model if per_client_models else self.release_model_runtime),
                retire_model=self.retire_model_task if reuse_completed_replicas else None,
                per_client_models=per_client_models, model_capacity=model_capacity)

    def offload_client_model(self, model):
        self.release_model_runtime(model)
        model.offload_model()

    def retire_model_task(self, model):
        """Detach a completed task while retaining one private execution slot."""
        model.reset_task_state()

    def release_model_runtime(self, model):
        """Remove every manager/session reference to one inactive replica."""
        capture_rng = getattr(model, "capture_replica_rng_state", None)
        if capture_rng is not None:
            capture_rng()
        handles = tuple(model.runtime_handles())
        owned_ids = {id(h) for h in handles}
        for key, handle in list(self._clone_runtime_cache.items()):
            if id(handle) in owned_ids:
                self._clone_runtime_cache.pop(key, None)
        for key, handle in list(self.autosplit_session._runtime_handles.items()):
            if id(handle) in owned_ids:
                self.autosplit_session._runtime_handles.pop(key, None)
        model.close_runtime()

    def acquire_replica(self, sid):
        return self._require_delegate().acquire(sid)

    def finish_replica(self, sid, *, success, reason=""):
        return self._require_delegate().finish(sid, success=success, reason=reason)

    def release_placement_runtimes(self):
        """Only inactive managers may release placement templates between rounds."""
        if self._require_delegate().get_server_model_ids():
            raise RuntimeError("Cannot discard placement runtimes while suffix models are live")
        self._placement_plan = self._runtime_handle = None
        self._placement_plans.clear()
        self._runtime_handles_by_plan_id.clear()
        self._clone_runtime_cache.clear()
        self.autosplit_session.discard_runtime_handles()

    def set_placement_plan(self, placement_plan: SplitPlacementPlan) -> None:
        self._placement_plan = placement_plan
        self.register_placement_plan(placement_plan, make_default=True)

    def register_placement_plan(
        self,
        placement_plan: SplitPlacementPlan,
        *,
        make_default: bool = False,
    ) -> None:
        """Register a placement without evicting other clients' active ABIs."""

        self._placement_plans[placement_plan.plan_id] = placement_plan
        handle = placement_plan.metadata.get("_runtime_handle")
        if isinstance(handle, TorchLensRuntimeHandle):
            self.bind_runtime_handle(
                handle,
                plan_id=placement_plan.plan_id,
                make_default=make_default,
            )

    def get_placement_plan(self, plan_id: str | None = None) -> SplitPlacementPlan | None:
        if plan_id is None:
            return self._placement_plan
        return self._placement_plans.get(str(plan_id))

    def bind_runtime_handle(
        self,
        runtime_handle: TorchLensRuntimeHandle,
        *,
        plan_id: str | None = None,
        make_default: bool = True,
    ) -> None:
        registry_key = str(plan_id or runtime_handle.plan.plan_id)
        self._runtime_handles_by_plan_id[registry_key] = runtime_handle
        if make_default:
            self._runtime_handle = runtime_handle
            self._clone_runtime_cache.clear()
        self.autosplit_session._runtime_handles[runtime_handle.plan.plan_id] = runtime_handle

    def _require_runtime_handle(self, plan_id: str | None = None) -> TorchLensRuntimeHandle:
        if plan_id is not None:
            handle = self._runtime_handles_by_plan_id.get(str(plan_id))
            if handle is None:
                raise RuntimeError(f"No TorchLens runtime handle is registered for plan {plan_id!r}.")
            return handle
        if self._runtime_handle is None:
            raise RuntimeError("No TorchLens runtime handle is active.")
        return self._runtime_handle

    def clone_runtime_for_model(
        self,
        model,
        *,
        suffix: str = "",
        plan_id: str | None = None,
    ) -> TorchLensRuntimeHandle:
        base = self._require_runtime_handle(plan_id)
        cache_key = self._clone_runtime_cache_key(base, model=model, suffix=suffix)
        cached = self._clone_runtime_cache.get(cache_key)
        if cached is not None:
            return cached
        if base.backend.framework_backend == "torch":
            handle = clone_runtime_handle(base, model)
            if handle is None:
                raise RuntimeError(
                    "TorchLens runtime cannot bind this model replica to the captured graph."
                )
        else:
            handle = None
            if base.backend.framework_backend == 'tf':
                from splitfleet.autosplit.tensorflow_clone import clone_tensorflow_runtime_handle
                from splitfleet.runtime.tensorflow_capture_scope import tensorflow_execution_scope
                with tensorflow_execution_scope():
                    handle = clone_tensorflow_runtime_handle(base, model)
        if handle is None:
            sample_inputs = base.plan.metadata.get("_example_inputs")
            if sample_inputs is None:
                raise RuntimeError("The active TorchLens runtime does not retain sample inputs.")
            point = base.runtime.request.point
            boundary = (
                f"{point.target}%"
                if point.kind == "percent"
                else f"{point.kind}:{point.target}"
            )
            handle = self.autosplit_session.prepare_runtime(
                model,
                sample_inputs,
                sample_kwargs=base.plan.metadata.get("_example_kwargs"),
                batch_axes=base.runtime.request.features.batch_axes,
                boundary=boundary,
                mode=base.plan.mode,
                trainable=base.plan.trainable,
                dynamic_batch=base.plan.dynamic_batch,
                trace_batch_mode=base.plan.trace_batch_mode,
            )
        enable_replica_rng = getattr(handle.backend, "enable_replica_rng_isolation", None)
        if enable_replica_rng is not None:
            seed = int.from_bytes(
                hashlib.sha256(str(suffix).encode()).digest()[:8], "little"
            )
            enable_replica_rng(self.replica_rng_isolation, seed=seed)
        from splitfleet.split_engine import graph_contract_for_runtime_handle
        from splitfleet.split_engine.contracts import validate_contract

        validate_contract(graph_contract_for_runtime_handle(base), graph_contract_for_runtime_handle(handle))
        compatibility = classify_contract_compatibility(
            base.plan.runtime_contract,
            handle.plan.runtime_contract,
        )
        if not bool(compatibility.get("compatible")):
            raise RuntimeError(
                "TorchLens cloned runtime feature ABI is incompatible with the base runtime: "
                f"{compatibility}."
            )
        if suffix:
            original_plan_id = handle.plan.plan_id
            handle.plan.plan_id = f"{handle.plan.plan_id}_{suffix}"
            if self.autosplit_session._runtime_handles.get(original_plan_id) is handle:
                self.autosplit_session._runtime_handles.pop(original_plan_id, None)
            self.autosplit_session._runtime_handles[handle.plan.plan_id] = handle
        self._clone_runtime_cache[cache_key] = handle
        return handle

    def _clone_runtime_cache_key(
        self,
        runtime_handle: TorchLensRuntimeHandle,
        *,
        model,
        suffix: str = "",
    ) -> str:
        plan = runtime_handle.plan
        return stable_json(
            {
                "plan_id": plan.plan_id,
                "split_id": plan.split_id,
                "feature_abi_id": plan.feature_abi_id,
                "boundary": plan.boundary,
                "graph_signature": plan.graph_signature,
                "torchlens_version": plan.torchlens_version,
                "runtime_backend": plan.runtime_backend,
                "backend": plan.backend,
                "mode": plan.mode,
                "trace_batch_mode": plan.trace_batch_mode,
                "dynamic_batch": list(plan.dynamic_batch) if plan.dynamic_batch is not None else None,
                "module_mode": "train" if getattr(model, "training", False) else "eval",
                "runtime_contract_digest": runtime_contract_digest(plan.runtime_contract or {}),
                "model_id": id(model),
                "suffix": suffix,
            }
        )

    def get_server_model(self, sid):
        return self._require_delegate().get_server_model(sid)

    def collect_server_fit_results(self):
        if self.before_collect is not None:
            self.before_collect()
        return self._require_delegate().collect_server_fit_results()

    def initialize_server_models(self, configs):
        self._require_delegate().initialize_server_models(configs)

    def get_server_model_ids(self):
        return self._require_delegate().get_server_model_ids()

    def end_round(self):
        self._require_delegate().end_round()

    def _require_delegate(self) -> GrpcServerModelManager:
        if self._delegate is None:
            raise RuntimeError(
                "StageRuntimeManager is not connected to the Flower server-model lifecycle. "
                "Pass init_server_model_fn when creating the server."
            )
        return self._delegate
