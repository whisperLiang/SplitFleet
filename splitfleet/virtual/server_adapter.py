"""Virtual dispatch/telemetry, delegating every training/aggregation operation."""

from __future__ import annotations

import json
import math
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from flwr.common import (
    GetPropertiesIns,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from splitfleet.server.server import Server
from splitfleet.server.strategy import AutoSplitStrategy

from .resources import peak_rss_mb
from .backend_ops import FrameworkOps
from .state import client_seed


class VirtualClientFailure(RuntimeError):
    def __init__(self, cid, reason):
        self.cid = str(cid)
        super().__init__(reason)


class RuntimeConditions:
    """Applied resource/network conditions, separate from calibrated timing priors."""

    def __init__(self, strategy):
        self.strategy = strategy
        self.round_id = 1

    def client_context(self, cid):
        pid = self.strategy.client_properties[cid]["virtual_partition_id"]
        network = self.strategy.virtual_config.network(pid, self.round_id)
        return {
            "uplink_mbps": network.upload_mbps,
            "downlink_mbps": network.download_mbps,
            "rtt_ms": network.rtt_ms if network.constrained else None,
            "network_constraint_source": "emulated"
            if network.constrained
            else "measured",
        }

    def server_context(self):
        servicer = self.strategy.service.servicer
        return {
            "active_jobs": servicer.active_executions,
            "queue_length": servicer.queued_executions,
        }


def task_overlap(metrics):
    """Measured overlap only when all task clocks share one OS clock domain."""
    return task_concurrency(metrics)["peak_running_clients"]


def task_concurrency(metrics):
    """Integrate actual handler intervals, including their native/RPC waits."""
    unavailable = {"peak_running_clients": None, "mean_running_clients": None,
                   "client_handler_window_sec": None, "peak_computing_clients": None,
                   "mean_computing_clients": None}
    if (
        not metrics
        or len({(m.get("worker_boot_id"), m["worker_host"]) for m in metrics}) != 1
        or any(not m.get("worker_boot_id") for m in metrics)
    ):
        return unavailable
    if any(m["worker_task_end_ns"] <= m["worker_task_start_ns"] for m in metrics):
        raise ValueError("Worker task intervals must have a positive measured duration")
    events = [(m["worker_task_start_ns"], 1) for m in metrics]
    events += [(m["worker_task_end_ns"], -1) for m in metrics]
    active, peak = 0, 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    elapsed = max(m["worker_task_end_ns"] for m in metrics) - min(
        m["worker_task_start_ns"] for m in metrics)
    occupied = sum(m["worker_task_end_ns"] - m["worker_task_start_ns"] for m in metrics)
    compute_peak, compute_mean = None, None
    if all("native_training_intervals_ns" in m for m in metrics):
        intervals = [interval for m in metrics for interval in json.loads(m["native_training_intervals_ns"])]
        if any(len(row) != 2 or row[1] <= row[0] for row in intervals):
            raise ValueError("Native training intervals must be measured positive durations")
        compute_events = [(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals]
        active, compute_peak = 0, 0
        for _, delta in sorted(compute_events):
            active += delta
            compute_peak = max(compute_peak, active)
        compute_mean = sum(end - start for start, end in intervals) / elapsed
    return {"peak_running_clients": peak, "mean_running_clients": occupied / elapsed,
            "client_handler_window_sec": elapsed / 1e9, "peak_computing_clients": compute_peak,
            "mean_computing_clients": compute_mean}


def round_concurrency(rows):
    measured = [row for row in rows if row.get("mean_running_clients") is not None]
    duration = sum(row["client_handler_window_sec"] for row in measured)
    return {
        "peak_running_clients": max((row["peak_running_clients"] for row in measured), default=None),
        "mean_running_clients": (sum(row["mean_running_clients"] * row["client_handler_window_sec"]
                                     for row in measured) / duration if duration else None),
        "peak_computing_clients": max((row["peak_computing_clients"] for row in measured
                                        if row.get("peak_computing_clients") is not None), default=None),
        "mean_computing_clients": (sum(row["mean_computing_clients"] * row["client_handler_window_sec"]
            for row in measured) / duration
            if duration and all(row.get("mean_computing_clients") is not None for row in measured) else None),
        "concurrency_scope": "Active native client fit handlers; includes suffix/RPC waits. Suffix executions are reported separately.",
        "compute_concurrency_scope": "Native client training call intervals; excludes suffix RPC and lease waits. It measures client operations, not simultaneous GPU kernels.",
    }


class VirtualDispatchProxy(ClientProxy):
    """The native GridClientProxy still dispatches every task through Flower."""

    def __init__(self, delegate, *, properties, state_store, service):
        super().__init__(delegate.cid)
        self.delegate, self.properties = delegate, properties
        self.state_store, self.service = state_store, service

    def get_properties(self, ins, timeout=None, group_id=None):
        return self.delegate.get_properties(ins, timeout, group_id)

    def get_parameters(self, ins, timeout=None, group_id=None):
        return self.delegate.get_parameters(ins, timeout, group_id)

    def fit(self, ins, timeout=None, group_id=None):
        ins.config["virtual_dispatch_ns"] = time.perf_counter_ns()
        ins.config["virtual_dispatch_host"] = socket.gethostname()
        boot_id = Path("/proc/sys/kernel/random/boot_id")
        ins.config["virtual_dispatch_boot_id"] = (
            boot_id.read_text().strip() if boot_id.is_file() else ""
        )
        start = time.perf_counter()
        try:
            result = self.delegate.fit(ins, timeout, group_id)
        except Exception as exc:
            raise VirtualClientFailure(self.cid, str(exc)) from exc
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics = result.metrics
        if metrics.get("virtual_state_via_rpc"):
            uploaded = self.service.servicer.prefix_uploads.get(self.cid)
            if uploaded is None:
                raise VirtualClientFailure(
                    self.cid, "No prefix state arrived on the constrained RPC path"
                )
            result.parameters = uploaded
        else:
            result.parameters = Parameters(
                tensors=self.state_store.put(result.parameters.tensors),
                tensor_type=result.parameters.tensor_type,
            )
        metrics["coordinator_fit_rpc_ms"] = elapsed_ms
        # Local dispatch-to-handler includes Flower delivery and actor queue.
        # None is retained for separate-host clocks; no estimated time fills it.
        queue = metrics.get("virtual_worker_queue_sec")
        if queue is not None:
            overhead_ms = max(
                elapsed_ms - float(metrics["virtual_handler_ms"]) - float(queue) * 1000,
                0.0,
            )
            metrics["virtual_state_exchange_ms"] = (
                overhead_ms
                + float(metrics.get("model_sync_sec", 0.0)) * 1000
                + float(metrics.get("model_state_rpc_upload_sec", 0.0)) * 1000
            )
        return result

    def evaluate(self, ins, timeout=None, group_id=None):
        return self.delegate.evaluate(ins, timeout, group_id)

    def reconnect(self, ins, timeout=None, group_id=None):
        return self.delegate.reconnect(ins, timeout, group_id)


class VirtualStrategy(AutoSplitStrategy):
    def __init__(self, *, virtual_config, evidence, spec, partitions, **kwargs):
        self.virtual_config, self.evidence, self.spec = virtual_config, evidence, spec
        self.framework_ops = FrameworkOps.for_spec(spec)
        self.partitions = partitions
        self.service = None
        self.client_properties = {}
        self.round_clients = {}
        self.round_evaluations = []
        self.round_aggregation_sec = {}
        self.round_prepare_sec = {}
        self.identity_lookup_sec = 0.0
        self.observed_startup_ms = {}
        self.observed_prepare_ms = {}
        super().__init__(**kwargs)

    def _identify_clients(self, client_manager):
        available = client_manager.all()
        if len(available) != self.virtual_config.virtual_clients.total:
            raise RuntimeError(
                "Flower registered a different logical population than declared"
            )
        pending = [(cid, proxy) for cid, proxy in available.items() if str(cid) not in self.client_properties]

        def identify(entry):
            cid, proxy = entry
            result = proxy.get_properties(GetPropertiesIns({"virtual_identity_only": True}),
                                          timeout=300, group_id=0)
            return str(cid), dict(result.properties)

        started = time.perf_counter()
        workers = min(len(pending), self.virtual_config.resources.max_concurrent_clients)
        if workers <= 1:
            results = [identify(entry) for entry in pending]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(identify, pending))
        properties = {**self.client_properties, **dict(results)}
        ids = [v["virtual_partition_id"] for v in properties.values()]
        if len(set(ids)) != len(ids) or set(ids) != set(range(len(available))):
            raise RuntimeError("Flower node-to-partition mapping is not bijective")
        self.client_properties = properties
        self.identity_lookup_sec += time.perf_counter() - started
        if not (self.evidence.directory / "client_manifest.json").exists():
            self.evidence.write(
                "client_manifest.json",
                [
                    {
                        "node_id": cid,
                        **props,
                        "profile": self.virtual_config.profile(
                            props["virtual_partition_id"]
                        ).__dict__,
                    }
                    for cid, props in sorted(
                        self.client_properties.items(),
                        key=lambda row: row[1]["virtual_partition_id"],
                    )
                ],
            )
        return available

    def select_fit_clients(
        self, *, server_round, client_manager, sample_size, min_num_clients
    ):
        available = self._identify_clients(client_manager)
        population = self.virtual_config.virtual_clients
        online = [
            cid
            for cid in available
            if server_round
            not in population.offline_rounds.get(
                str(self.client_properties[cid]["virtual_partition_id"]), []
            )
        ]
        online.sort(key=lambda cid: self.client_properties[cid]["virtual_partition_id"])
        count = min(
            len(online),
            max(1, math.ceil(len(online) * population.participation_fraction)),
        )
        chosen = np.random.default_rng(
            client_seed(self.virtual_config.seed, 0, f"selection:{server_round}")
        )
        selected = (
            sorted(
                chosen.choice(online, count, replace=False).tolist(),
                key=lambda cid: self.client_properties[cid]["virtual_partition_id"],
            )
            if online
            else []
        )
        self.round_clients[server_round] = {
            "online": online,
            "selected": selected,
            "offline": sorted(set(available) - set(online)),
            "not_selected": sorted(set(online) - set(selected)),
        }
        if self.placement_policy is not None:
            self.placement_policy.solver.client_concurrency = (
                self.virtual_config.resources.max_concurrent_clients
            )
            self.placement_policy.solver.dispatch_order = tuple(selected)
            self.placement_policy.solver.startup_costs_ms = dict(
                self.observed_startup_ms
            )
            self.placement_policy.solver.cold_prepare_costs_ms = dict(
                self.observed_prepare_ms
            )
        return [
            VirtualDispatchProxy(
                available[cid],
                properties=self.client_properties[cid],
                state_store=self.round_state_store,
                service=self.service,
            )
            for cid in selected
        ]

    def configure_fit(self, server_round, parameters, client_manager):
        started = time.perf_counter()
        previous_identity_sec = self.identity_lookup_sec
        if self.spec.learning_rate_fn is not None:
            from .client_adapter import OptimizerFactory
            self.optimizer_fn = OptimizerFactory(
                self.virtual_config.optimizer, float(self.spec.learning_rate_fn(server_round)),
                self.framework_ops.backend,
            )
        if all(
            server_round
            in self.virtual_config.virtual_clients.offline_rounds.get(str(pid), [])
            for pid in range(self.virtual_config.virtual_clients.total)
        ):
            self._identify_clients(client_manager)
            self.round_clients[server_round] = {
                "online": [],
                "selected": [],
                "offline": list(self.client_properties),
                "not_selected": [],
            }
            return []
        instructions = super().configure_fit(server_round, parameters, client_manager)
        self.service.servicer.prepare_round(
            [proxy.cid for proxy, _ in instructions], server_round
        )
        global_hash = self.framework_ops.hash_arrays(self.model, parameters_to_ndarrays(parameters))
        checked_ownership = set()
        trainable = self.framework_ops.trainable_names(self.model)
        for proxy, ins in instructions:
            manifest = json.loads(ins.config["autosplit_state_ownership"])
            digest = manifest["state_ownership_digest"]
            if digest not in checked_ownership:
                for stage in ("prefix", "suffix"):
                    if not any(
                        name in trainable and owner == stage
                        for name, owner in zip(manifest["names"], manifest["owners"])
                    ):
                        raise ValueError(
                            f"Virtual SFL requires trainable parameters on both sides; {stage} is empty"
                        )
                checked_ownership.add(digest)
            partition = self.client_properties[proxy.cid]["virtual_partition_id"]
            ins.config.update(
                {
                    "virtual_suffix_address": self.service.address,
                    "virtual_task_token": self.service.servicer.tokens[proxy.cid],
                    "virtual_partition_id": partition,
                    "virtual_global_model_hash": global_hash,
                    "virtual_state_download_bytes": sum(
                        len(v) for v in ins.parameters.tensors
                    ),
                }
            )
            if self.virtual_config.network(partition, server_round).constrained:
                # Model synchronization uses the same real, paced TCP channel.
                ins.parameters = ndarrays_to_parameters([])
        if server_round == 1:
            telemetry = getattr(self.placement_policy, 'telemetry_provider', None)
            calibration = getattr(telemetry, 'receipt', None)
            if calibration is not None:
                self.evidence.write('calibration_receipt.json', calibration)
            self.evidence.write('first_configuration_timing.json', {
                'source': 'measured', 'round_id': server_round,
                'configuration_sec': time.perf_counter() - started,
                'identity_lookup_sec': self.identity_lookup_sec - previous_identity_sec,
                'calibration': calibration.get('timing') if calibration else None,
            })
        self.round_prepare_sec[server_round] = time.perf_counter() - started
        return instructions

    def configure_server_fit(self, server_round, parameters, cids):
        configs = super().configure_server_fit(server_round, parameters, cids)
        for config in configs:
            config.config["virtual_partition_id"] = self.client_properties[config.sid][
                "virtual_partition_id"
            ]
        return configs

    def aggregate_fit(self, server_round, results, failures):
        started = time.perf_counter()
        results = sorted(
            results,
            key=lambda item: self.client_properties[item[0].cid][
                "virtual_partition_id"
            ],
        )
        for proxy, result in results:
            props = self.client_properties[proxy.cid]
            self.observed_startup_ms[proxy.cid] = (
                float(result.metrics.get("virtual_client_build_sec", 0))
                + float(result.metrics.get("suffix_prepare_sec", 0))
            ) * 1000
            self.observed_prepare_ms[proxy.cid] = (
                float(result.metrics.get("runtime_prepare_sec", 0)) * 1000
            )
            self.evidence.append(
                "client_metrics.jsonl",
                {
                    "round_id": server_round,
                    "node_id": proxy.cid,
                    "logical_client_id": props["logical_client_id"],
                    "partition_id": props["virtual_partition_id"],
                    "status": "success",
                    "source": "measured",
                    "num_examples": result.num_examples,
                    "metrics": dict(result.metrics),
                },
            )
        for failure in failures:
            cid = self._failure_cid(failure)
            props = self.client_properties.get(cid, {})
            reason = str(failure)
            kind = (
                "timeout"
                if ("deadline" in reason.lower() or "timeout" in reason.lower())
                else "failure"
            )
            self.evidence.append(
                "client_metrics.jsonl",
                {
                    "round_id": server_round,
                    "node_id": cid,
                    "logical_client_id": props.get("logical_client_id"),
                    "partition_id": props.get("virtual_partition_id"),
                    "status": kind,
                    "source": "measured",
                    "reason": reason,
                },
            )
        if (self.spec.frozen_partitions is not None and self.spec.trainer is not None
                and (failures or len(results) != len(self.round_clients[server_round]["selected"]))):
            raise RuntimeError("Quality round is incomplete; partial client states cannot enter aggregation")
        value = super().aggregate_fit(server_round, results, failures)
        self.round_aggregation_sec[server_round] = time.perf_counter() - started
        return value

    def aggregate_server_fit(self, server_round, results):
        started = time.perf_counter()
        for result in results:
            self.evidence.append(
                "client_metrics.jsonl",
                {
                    "round_id": server_round,
                    "node_id": result.sid,
                    "partition_id": self.client_properties[result.sid][
                        "virtual_partition_id"
                    ],
                    "status": "suffix_committed",
                    "source": "measured",
                    "metrics": dict(result.config),
                },
            )
        value = super().aggregate_server_fit(server_round, results)
        self.round_aggregation_sec[server_round] = (
            self.round_aggregation_sec.get(server_round, 0.0)
            + time.perf_counter()
            - started
        )
        policy = self.placement_policy
        if policy is not None:
            self.evidence.append(
                "placement_trace.jsonl",
                {
                    "round_id": server_round,
                    "source": "estimated",
                    "policy": getattr(policy, "name", "cosplit_ucb"),
                    "diagnostics": policy.round_diagnostics[server_round],
                },
            )
        else:
            self.evidence.append(
                "placement_trace.jsonl",
                {
                    "round_id": server_round,
                    "source": "configured",
                    "policy": "static",
                    "boundary": self.boundary,
                },
            )
        return value

    def evaluate(self, server_round, client_parameters, server_parameters):
        if self.spec.evaluate_fn is None:
            return None
        if server_round == 0 and self.spec.frozen_partitions is not None and self.spec.trainer is not None:
            return None
        started = time.perf_counter()
        self.backend_adapter.load_ndarrays(
            self.model, parameters_to_ndarrays(client_parameters)
        )
        metrics = self.spec.evaluate_fn(self.model, self.virtual_config.server_device)
        self.round_evaluations.append(
            {
                "round_id": server_round,
                "metrics": metrics,
                "source": "measured",
                "evaluation_sec": time.perf_counter() - started,
            }
        )
        if self.spec.round_callback is not None:
            self.spec.round_callback(
                server_round, parameters_to_ndarrays(client_parameters),
                metrics, self.evidence.directory, phase="evaluation",
            )
        if "loss" in metrics:
            return float(metrics["loss"]), metrics
        return None


class VirtualServer(Server):
    """Observe the existing round loop; suffix collection and aggregation stay there."""

    def __init__(self, *, evidence, **kwargs):
        super().__init__(**kwargs)
        self.evidence = evidence
        self.rounds_finished = 0

    def fit_round(self, server_round, timeout):
        started = time.perf_counter()
        config = self.strategy.virtual_config
        try:
            service = self.strategy.service
            concurrency = int(
                config.resources.server_concurrency_by_round.get(
                    server_round, config.resources.server_concurrency
                )
            )

            async def change_capacity():
                import asyncio

                service.servicer._execution_slots = asyncio.Semaphore(concurrency)
                service.servicer.concurrency = concurrency

            import asyncio

            asyncio.run_coroutine_threadsafe(change_capacity(), service.loop).result(
                timeout=30
            )
            policy = self.strategy.placement_policy
            if policy is not None:
                policy.solver.server_concurrency = concurrency
                policy.runtime_context_provider.round_id = server_round
            timeout = self.strategy.spec.round_timeout_sec or timeout
            result = super().fit_round(server_round, timeout)
            selected = self.strategy.round_clients[server_round]
            success, failures = result[3] if result is not None else ([], [])
            parameters = (
                result[0]
                if result is not None and result[0] is not None
                else self.client_parameters
            )
            arrays = parameters_to_ndarrays(parameters)
            model_hash = self.strategy.framework_ops.hash_arrays(self.strategy.model, arrays)
            metrics = [r.metrics for _, r in success]
            samples = sum(r.num_examples for _, r in success)
            loss = (
                sum(float(r.metrics["loss"]) * r.num_examples for _, r in success)
                / samples
                if samples
                else None
            )
            self.evidence.append(
                "round_metrics.jsonl",
                {
                    "round_id": server_round,
                    "source": "measured",
                    "logical_clients": config.virtual_clients.total,
                    "online_clients": len(selected["online"]),
                    "participating_clients": len(selected["selected"]),
                    "executed_clients": len(success) + len(failures),
                    "successful_clients": len(success),
                    "failed_clients": len(failures),
                    "offline_clients": len(selected["offline"]),
                    "planned_nonparticipants": len(selected["not_selected"]),
                    "selected_partition_ids": [
                        self.strategy.client_properties[cid]["virtual_partition_id"]
                        for cid in selected["selected"]
                    ],
                    "round_wall_sec": time.perf_counter() - started,
                    "train_loss": loss,
                    "round_wall_scope": "fit configuration/placement/calibration, training and aggregation; evaluation reported separately",
                    "fit_configuration_sec": self.strategy.round_prepare_sec.get(
                        server_round, 0.0
                    ),
                    "aggregation_sec": self.strategy.round_aggregation_sec.get(
                        server_round, 0.0
                    ),
                    "model_version": server_round,
                    "model_hash": model_hash,
                    "worker_limit": config.resources.max_concurrent_clients,
                    "server_concurrency": concurrency,
                    "suffix_model_count": self.server_model_manager._delegate.peak_client_models,
                    "resident_suffix_limit": config.resources.resident_suffix_capacity,
                    "worker_processes_observed": len(
                        {(m["worker_host"], m["worker_pid"]) for m in metrics}
                    ),
                    **task_concurrency(metrics if not failures else []),
                    "running_client_scope": "successful fit handler intervals; unavailable on failures or unrelated host clocks",
                    "worker_queue_scope": "dispatch to ClientApp entry, includes Flower delivery and Ray queue; same-host monotonic clocks",
                    "worker_queue_sec": sum(
                        float(m.get("virtual_worker_queue_sec", 0.0)) for m in metrics
                    )
                    if metrics and all("virtual_worker_queue_sec" in m for m in metrics)
                    else None,
                    "suffix_server_queue_sec": sum(
                        float(m["suffix_server_queue_sec"]) for m in metrics
                    ),
                    "prefix_forward_sec": sum(
                        float(m.get("client_forward_sec", 0.0)) for m in metrics
                    ),
                    "prefix_backward_sec": sum(
                        float(m.get("client_backward_sec", 0.0)) for m in metrics
                    ),
                    "suffix_forward_sec": sum(
                        float(m.get("server_forward_sec", 0.0)) for m in metrics
                    ),
                    "suffix_backward_sec": sum(
                        float(m.get("server_backward_sec", 0.0)) for m in metrics
                    ),
                    "suffix_replica_peak": self.server_model_manager._delegate.peak_live,
                    "server_peak_memory_mb": peak_rss_mb(),
                    "prefix_optimization_steps": sum(
                        int(m["num_batches"]) for m in metrics
                    ),
                },
            )
            self.rounds_finished += 1
            if self.strategy.spec.round_callback is not None:
                self.strategy.spec.round_callback(
                    server_round, arrays, {"train_loss": loss, "num_examples": samples,
                                          "successful_clients": len(success),
                                          "optimization_steps": sum(int(m["num_batches"]) for m in metrics)},
                    self.evidence.directory, phase="training",
                )
            print(json.dumps({"event": "virtual_round_finished", "method": "splitfleet",
                              "round": server_round, "successful_clients": len(success),
                              "examples": samples, "steps": sum(int(m["num_batches"]) for m in metrics)},
                             sort_keys=True), flush=True)
            return result
        finally:
            self.server_model_manager.end_round()
            if self.strategy.placement_policy is not None:
                self.strategy.release_placement_runtimes()
            self.strategy.round_state_store.clear()
