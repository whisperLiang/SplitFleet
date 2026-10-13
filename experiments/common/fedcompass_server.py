"""Run FedCompass over the existing Flower bidirectional client connections."""

from __future__ import annotations

import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import torch
from flwr.common import Code, FitIns, GetPropertiesIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.history import History
from flwr.server.server import Server

from experiments.baselines.fedcompass import FedCompass, FedCompassConfig, Update


class FedCompassServer(Server):
    def assignment_config(self, job):
        return {}

    def __init__(self, *, strategy, client_manager, partition_sizes, worker_ids,
                 config: FedCompassConfig, time_budget_sec: float | None,
                 evaluation_interval_sec: float, rpc_timeout_sec: float):
        super().__init__(strategy=strategy, client_manager=client_manager)
        self.controller = FedCompass(strategy.model, partition_sizes, config)
        self.worker_ids = worker_ids
        self.time_budget_sec = time_budget_sec
        self.evaluation_interval_sec = evaluation_interval_sec
        self.rpc_timeout_sec = rpc_timeout_sec
        if ((time_budget_sec is not None and (not 0 < time_budget_sec < float("inf")))
                or not 0 < evaluation_interval_sec < float("inf")
                or not 0 < rpc_timeout_sec < float("inf")):
            raise ValueError("FedCompass time limits must be finite and positive")
        self.dispatch_records = []
        self.completed_dispatches = 0
        self.last_arrivals = {}
        self.stop_reason = None

    def fit(self, num_rounds, timeout):
        started = self.strategy.measurement_started_ns / 1e9
        elapsed = lambda: time.monotonic() - started
        deadline = self.time_budget_sec
        self.controller.clock, self.controller.deadline = elapsed, deadline
        max_updates = None if deadline is not None else num_rounds
        if not self._client_manager.wait_for(len(self.worker_ids), timeout=self.rpc_timeout_sec):
            raise TimeoutError("FedCompass clients did not connect")
        clients = list(self._client_manager.all().values())
        if len(clients) != len(self.worker_ids):
            raise ValueError("FedCompass requires exactly the configured client population")
        pool = ThreadPoolExecutor(max_workers=len(clients))
        finished = queue.Queue()
        pending = {}
        proxies = {}
        dispatch_counts = {cid: 0 for cid in self.worker_ids}
        history = History()
        evaluation_index = 0

        def evaluate(final=False):
            nonlocal evaluation_index
            evaluation_index += 1
            snapshot_time = elapsed()
            arrays = [v.numpy() for v in self.controller.state().values()]
            _, metrics = self.strategy._evaluate_round(evaluation_index, arrays, {})
            record = self.strategy.evaluation_records[-1]
            record.update(model_version=self.controller.version, snapshot_elapsed_sec=snapshot_time,
                          within_time_budget=deadline is None or record["elapsed_sec"] <= deadline,
                          evaluation_kind="final_frozen_model" if final else "checkpoint")
            if final:
                record["model_committed_elapsed_sec"] = (self.controller.aggregations[-1]["elapsed_sec"]
                                                         if self.controller.aggregations else 0.0)
            history.add_metrics_centralized(evaluation_index, metrics)

        def dispatch(job):
            cid = job.client_id
            # Multiple groups can expire in one event-loop iteration. Always
            # stamp the version of the actual model sent, including updates
            # committed by another expired group before this dispatch.
            job.model_version = self.controller.version
            job.started = elapsed()
            dispatch_counts[cid] += 1
            dispatch_id = len(self.dispatch_records) + 1
            state = self.controller.state()
            parameters = ndarrays_to_parameters([v.numpy() for v in state.values()])
            config = {"round_id": dispatch_id, "local_steps": job.local_steps,
                      "model_version": job.model_version, "dispatch_index": dispatch_counts[cid],
                      "initial_dispatch": dispatch_counts[cid] == 1}
            config.update(self.assignment_config(job))
            receipt = {"dispatch_id": dispatch_id, "client_id": cid,
                       "model_version": job.model_version, "local_steps": job.local_steps,
                       "group_id": job.group_id, "elapsed_sec": elapsed(),
                       "response_start_elapsed_sec": job.started,
                       "client_idle_sec": max(0.0, job.started - self.last_arrivals.get(cid, job.started))}
            group = self.controller.groups.get(job.group_id)
            receipt.update(expected_arrival_elapsed_sec=group.expected if group else None,
                           group_deadline_elapsed_sec=group.deadline if group else None)
            self.dispatch_records.append(receipt)
            proxy = proxies[cid]
            self.strategy.state_download_records.append({"round_id": dispatch_id, "cid": str(proxy.cid),
                "model_state_download_bytes": sum(map(len, parameters.tensors))})
            future = pool.submit(proxy.fit, FitIns(parameters, config), self.rpc_timeout_sec, dispatch_id)
            pending[future] = (job, state, dispatch_id, proxy)
            future.add_done_callback(lambda f: finished.put((time.monotonic(), f)))

        def consume(future, receive_time, accepting):
            job, base, dispatch_id, proxy = pending.pop(future)
            try:
                res = future.result()
                if res.status.code != Code.OK:
                    raise RuntimeError(f"FedCompass client {job.client_id}: {res.status.message}")
                metrics = dict(res.metrics)
                if (metrics.get("logical_client_id") != job.client_id
                        or metrics.get("num_batches") != job.local_steps
                        or metrics.get("model_version") != job.model_version
                        or res.num_examples != metrics.get("num_examples")
                        or metrics.get("partition_hash") != self.strategy.partition_hash):
                    raise ValueError("FedCompass client returned an incompatible assignment receipt")
                names = list(base)
                arrays = parameters_to_ndarrays(res.parameters)
                trained = {name: torch.from_numpy(value.copy())
                           for name, value in zip(names, arrays, strict=True)}
                self.controller._check_state(trained)
                record = {"round_id": dispatch_id, "dispatch_id": dispatch_id, "cid": str(proxy.cid),
                          "num_examples": int(res.num_examples), "model_version": job.model_version,
                          "group_id": job.group_id, "coordinator_arrival_elapsed_sec": receive_time - started,
                          "metrics": {**metrics, "model_state_upload_bytes": sum(map(len, res.parameters.tensors))},
                          "update_status": "discarded_at_stop"}
                self.strategy.fit_records.append(record)
                self.last_arrivals[job.client_id] = receive_time - started
                self.completed_dispatches += 1
                if accepting:
                    return self.controller.receive(Update(job, base, trained, record), elapsed())
                return []
            except Exception as exc:
                self.strategy.fit_failures.append({"round_id": dispatch_id, "reason": str(exc)})
                raise

        try:
            # Metadata is obtained before any fit; client indices cannot depend
            # on the nondeterministic order of network registration.
            requests = [(proxy, pool.submit(proxy.get_properties, GetPropertiesIns({}),
                                           self.rpc_timeout_sec, 0)) for proxy in clients]
            for proxy, future in requests:
                response = future.result()
                props = response.properties
                cid = str(props.get("logical_client_id"))
                if (response.status.code != Code.OK or cid not in self.worker_ids or cid in proxies
                        or props.get("client_index") != self.worker_ids.index(cid)
                        or props.get("partition_size") != self.controller.partition_sizes[cid]
                        or props.get("partition_hash") != self.strategy.partition_hash
                        or props.get("initial_model_hash") != self.strategy.model_initial_hash):
                    raise ValueError("FedCompass client population or initialization mismatch")
                proxies[cid] = proxy
            evaluate()
            next_evaluation = elapsed() + self.evaluation_interval_sec
            if deadline is None or elapsed() < deadline:
                for job in self.controller.initialize(elapsed()):
                    dispatch(job)
            while pending or any(g.arrived for g in self.controller.groups.values()):
                now = elapsed()
                if deadline is not None and now >= deadline:
                    self.stop_reason = "time_budget"
                    break
                if max_updates is not None and self.controller.version >= max_updates:
                    self.stop_reason = "global_update_limit"
                    break
                jobs = self.controller.expire(now, max_updates)
                for job in jobs:
                    if max_updates is None or self.controller.version < max_updates:
                        dispatch(job)
                if max_updates is not None and self.controller.version >= max_updates:
                    continue
                if now >= next_evaluation:
                    evaluate()
                    next_evaluation = elapsed() + self.evaluation_interval_sec
                    continue
                wake = min(next_evaluation, deadline if deadline is not None else float("inf"),
                           self.controller.next_deadline() or float("inf"))
                try:
                    receive_time, future = finished.get(timeout=max(0.001, wake - elapsed()))
                except queue.Empty:
                    continue
                # Evaluation or queueing may have consumed the remaining budget.
                accepting = deadline is None or elapsed() < deadline
                jobs = consume(future, receive_time, accepting)
                for job in jobs:
                    if (deadline is None or elapsed() < deadline) and (
                            max_updates is None or self.controller.version < max_updates):
                        dispatch(job)
            self.stop_reason = self.stop_reason or ("time_budget" if deadline is not None and elapsed() >= deadline
                else "global_update_limit" if max_updates is not None and self.controller.version >= max_updates
                else "no_inflight_updates")
            self.controller.stop()
            stopped_at = elapsed()
            # Drain already dispatched RPCs without changing the timed model.
            # Every completed transfer/optimizer step remains in the receipts.
            while pending:
                receive_time, future = finished.get(timeout=self.rpc_timeout_sec + 1)
                consume(future, receive_time, False)
            self.training_stopped_elapsed_sec = stopped_at
            self.drain_duration_sec = elapsed() - stopped_at
            evaluate(final=True)
            self.parameters = ndarrays_to_parameters([v.numpy() for v in self.controller.state().values()])
            return history, elapsed()
        finally:
            self.controller.stop()
            pool.shutdown(wait=True, cancel_futures=True)

    def result_metadata(self):
        return {"aggregation_protocol": "semi_asynchronous", "fedcompass_config": asdict(self.controller.config),
                "fedcompass_client_weights": self.controller.weights, "global_updates": self.controller.version,
                "rounds": self.controller.version, "completed_dispatches": self.completed_dispatches,
                "local_epochs": None, "local_work_policy": "dynamic_local_steps",
                "dispatch_records": self.dispatch_records, "aggregation_records": self.controller.aggregations,
                "group_events": self.controller.group_events, "stop_reason": self.stop_reason,
                "client_seconds_per_step_estimates": self.controller.speeds,
                "total_optimizer_steps": sum(row["metrics"]["num_batches"] for row in self.strategy.fit_records),
                "total_examples_processed": sum(row["num_examples"] for row in self.strategy.fit_records),
                "time_budget_sec": self.time_budget_sec, "evaluation_interval_sec": self.evaluation_interval_sec,
                "training_stopped_elapsed_sec": self.training_stopped_elapsed_sec,
                "drain_duration_sec": self.drain_duration_sec, "fit_barrier": "initial_readiness_only",
                "last_committed_model_elapsed_sec": (self.controller.aggregations[-1]["elapsed_sec"]
                                                      if self.controller.aggregations else 0.0),
                "evaluation_policy": "initial, wall-clock cadence and final frozen model",
                "aggregation_order": "coordinator_arrival_and_group_deadline",
                "budget_interpretation": "wall_clock" if self.time_budget_sec is not None else "global_updates; not matched synchronous work"}
