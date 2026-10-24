"""Lazy suffix replica lifecycle, shared by physical and virtual runtimes."""

from __future__ import annotations

import time
from threading import RLock

from splitfleet.common import ServerModelFitIns, ServerModelFitRes
from splitfleet.server.server_model.manager.manager import ServerModelManager


class LeasedServerModelManager(ServerModelManager):
    """Pin active clients and persist completed suffix updates independently.

    A replica is private for the complete task. It is never evicted between
    forward and backward. Incomplete tasks are discarded rather than retried
    against a partially updated replica; a later round starts from its global
    version, just as configure_fit does in the physical runtime.
    Optional idle slots are bounded by capacity and dropped at the round end.
    """

    def __init__(self, init_server_model_fn, *, capacity, state_store, release_model,
                 retire_model=None, per_client_models=False, model_capacity=None):
        super().__init__()
        self.factory = init_server_model_fn
        self.capacity = capacity
        self.state_store = state_store
        self.release_model = release_model
        # An opt-in retirement hook may keep an inactive execution slot within
        # this round. configure_fit still replaces all state and the optimizer.
        self.retire_model = retire_model
        if per_client_models and retire_model is not None:
            raise ValueError("Per-client suffix models cannot reuse another client's slot")
        self.per_client_models = per_client_models
        self.model_capacity = model_capacity or capacity
        # Logical client state is held by the round configuration/result store;
        # this map contains only currently materialized native models.  It is
        # deliberately not populated for every selected client.
        self.client_models = {}
        # Number of logical suffix identities in the current round.  This is
        # metadata/configuration state and must not be confused with the
        # number of materialized native models in ``live``/``idle``.
        self.logical_model_count = 0
        self.peak_logical_model_count = 0
        self.peak_client_models = 0
        self.idle = []
        self.lock = RLock()
        self.configs = {}
        self.live = {}
        self.preparing = set()
        self.finishing = set()
        self.completed = {}
        self.attempted = set()
        self.failed = {}
        self.peak_live = 0
        self.total_created = 0
        self.total_reused = 0
        # ``id(model)`` is only unique while the object is alive.  Per-client
        # models are released after each task, so CPython may reuse that
        # address before the metrics for the round are consumed.  Keep a
        # manager-local monotonic identity on each materialized model instead.
        self._next_model_instance_id = 0
        self.records = {}

    def initialize_server_models(self, configs):
        with self.lock:
            self.end_round()
            self.configs = {str(c.sid): c for c in configs}
            if len(self.configs) != len(configs):
                raise ValueError("Duplicate suffix replica identities")
            if any(c.sid == "" for c in configs) and len(configs) != 1:
                raise ValueError("Shared suffix configuration must contain one model")
            self.logical_model_count = len(self.configs)
            self.peak_logical_model_count = max(
                self.peak_logical_model_count, self.logical_model_count
            )
            if self.per_client_models and self.logical_model_count > self.model_capacity:
                raise ValueError(
                    "The per-client logical suffix population exceeds its declared model count"
                )
            # Do not construct one native model per logical client here.  A
            # physical slot is acquired below and released after the complete
            # prefix/suffix transaction.  Logical client identities remain in
            # ``configs`` and the spilled result store.

    def acquire(self, sid):
        sid = str(sid)
        started = time.perf_counter()
        with self.lock:
            if sid in self.attempted:
                raise RuntimeError(f"Suffix task {sid!r} already attempted; replay is forbidden")
            if sid not in self.configs:
                raise KeyError(f"No suffix configuration for {sid!r}")
            if len(self.live) + len(self.preparing) + len(self.finishing) >= self.capacity:
                raise RuntimeError("Suffix capacity exceeded; acquire a runtime lease before execution")
            self.attempted.add(sid)
            self.preparing.add(sid)
            reused = bool(self.idle)
            model = self.idle.pop() if reused else None
            config = self.configs[sid]
        try:
            if model is None:
                model = self.factory().to_server_model()
                with self.lock:
                    self._next_model_instance_id += 1
                    model_instance_id = self._next_model_instance_id
                setattr(model, "_splitfleet_model_instance_id", model_instance_id)
                # Model construction happens outside the critical section so
                # a slow framework build cannot block unrelated leases.  The
                # peak counter is shared state, however, and must be updated
                # while holding the same lock used for the live/preparing
                # sets; otherwise concurrent acquisitions can lose an update.
                with self.lock:
                    self.peak_client_models = max(
                        self.peak_client_models,
                        len(self.live) + len(self.preparing),
                    )
            else:
                model_instance_id = getattr(model, "_splitfleet_model_instance_id", None)
                if model_instance_id is None:
                    raise RuntimeError("Suffix slot is missing its model instance identity")
            if isinstance(config, ServerModelFitIns):
                model.configure_fit(config)
            else:
                model.configure_evaluate(config)
            receipt = {"suffix_prepare_sec": time.perf_counter() - started,
                       "suffix_replica_reused": reused,
                       "suffix_capture_reused": bool(
                           getattr(model, 'runtime_handle', None)
                           and model.runtime_handle.plan.metadata.get('_reused_capture', False)),
                       "suffix_model_instance_id": model_instance_id,
                       "suffix_model_client_id": sid}
        except BaseException as error:
            try:
                if model is not None:
                    self.release_model(model)
            finally:
                with self.lock:
                    self.preparing.remove(sid)
                    self.client_models.pop(sid, None)
                    self.failed[sid] = repr(error)
            raise
        with self.lock:
            self.preparing.remove(sid)
            self.live[sid] = model
            self.client_models[sid] = model
            self.peak_live = max(self.peak_live, len(self.live))
            self.total_created += int(not reused)
            self.total_reused += int(reused)
            self.records[sid] = receipt
        return receipt

    def get_server_model(self, sid):
        with self.lock:
            if str(sid) not in self.live:
                raise RuntimeError("Suffix request has no active replica lease")
            return self.live[str(sid)]

    def finish(self, sid, *, success, reason=""):
        sid = str(sid)
        with self.lock:
            model = self.live.pop(sid, None)
            if model is None:
                return
            self.client_models.pop(sid, None)
            self.finishing.add(sid)
            initial = self.configs[sid].parameters
        stored = None
        idle = False
        try:
            if success:
                import hashlib
                import numpy as np
                from splitfleet.autosplit.state_ownership import validate_owned

                result = model.get_fit_result()
                if any(not np.isfinite(value).all() for value in result.parameters):
                    raise FloatingPointError("Suffix update contains nonfinite model state")
                ownership = getattr(model, "state_ownership", None)
                if ownership:
                    decoded = validate_owned(result.parameters, ownership, "suffix", initial)
                    result.config["suffix_changed_tensors"] = sum(
                        not np.array_equal(value, initial[index]) for index, value in decoded.items())
                    result.config["suffix_update_hash"] = hashlib.sha256(
                        b"".join(value.tobytes() for value in result.parameters)).hexdigest()
                result.config["suffix_optimization_steps"] = len(getattr(model, "_processed_steps", ()))
                values = self.state_store.put(result.parameters)
                stored = ServerModelFitRes(parameters=values, config=dict(result.config))
                stored.sid = sid
                if self.per_client_models:
                    model.offloaded_parameters = values
            if stored is not None and self.retire_model is not None:
                self.retire_model(model)
                idle = True
            else:
                self.release_model(model)
        except BaseException as error:
            try:
                self.release_model(model)
            finally:
                with self.lock:
                    self.failed[sid] = repr(error)
            raise
        else:
            with self.lock:
                if stored is not None:
                    self.completed[sid] = stored
                else:
                    self.failed[sid] = reason
                if idle and not self.per_client_models:
                    self.idle.append(model)
        finally:
            with self.lock:
                self.finishing.remove(sid)

    def collect_server_fit_results(self):
        with self.lock:
            pending = list(self.live)
        # Reconciliation follows RPC draining; no export holds the registry lock.
        for sid in pending:
            self.finish(sid, success=False, reason="task did not commit")
        with self.lock:
            if self.preparing or self.finishing:
                raise RuntimeError("Cannot collect suffix results during a lifecycle operation")
            return [self.completed[sid] for sid in sorted(self.completed)]

    def get_server_model_ids(self):
        with self.lock:
            return list(self.live) + list(self.preparing) + list(self.finishing)

    def end_round(self):
        with self.lock:
            if self.preparing or self.finishing:
                raise RuntimeError("Cannot end a suffix round during a lifecycle operation")
            models = tuple(self.client_models.values()) + tuple(self.live.values())
            seen = set()
            for model in models:
                if id(model) in seen:
                    continue
                seen.add(id(model))
                self.release_model(model)
            for model in self.idle:
                if id(model) not in seen:
                    seen.add(id(model))
                    self.release_model(model)
            self.live.clear()
            self.idle.clear()
            self.client_models.clear()
            self.completed.clear()
            self.configs.clear()
            self.attempted.clear()
            self.failed.clear()
            self.records.clear()
            self.logical_model_count = 0
