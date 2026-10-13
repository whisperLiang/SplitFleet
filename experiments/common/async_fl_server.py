"""Actual Flower RPC execution of fixed-step FedAsync and FedBuff."""
from dataclasses import asdict

from experiments.baselines.async_fl import AsyncFL
from experiments.common.fedcompass_server import FedCompassServer


class AsyncFLServer(FedCompassServer):
    def __init__(self, *, config, **kwargs):
        # Reuse population binding, one RPC per client, arrival handling,
        # evaluation and frozen-deadline drain from the tested Compass harness.
        from experiments.baselines.fedcompass import FedCompassConfig
        super().__init__(config=FedCompassConfig(min_local_steps=config.local_steps,
            max_local_steps=config.local_steps), **kwargs)
        self.controller = AsyncFL(self.strategy.model, kwargs["partition_sizes"], config)

    def assignment_config(self, job):
        return {"proximal_mu": self.controller.config.proximal_mu
                if self.controller.config.method == "fedasync" else 0.0}

    def result_metadata(self):
        config = self.controller.config
        return {"aggregation_protocol": "asynchronous", "async_fl_config": asdict(config),
            "global_updates": self.controller.version, "rounds": self.controller.version,
            "completed_dispatches": self.completed_dispatches, "local_epochs": None,
            "local_work_policy": "fixed_local_steps", "dispatch_records": self.dispatch_records,
            "aggregation_records": self.controller.aggregations, "stop_reason": self.stop_reason,
            "total_optimizer_steps": sum(r["metrics"]["num_batches"] for r in self.strategy.fit_records),
            "total_examples_processed": sum(r["num_examples"] for r in self.strategy.fit_records),
            "time_budget_sec": self.time_budget_sec, "evaluation_interval_sec": self.evaluation_interval_sec,
            "training_stopped_elapsed_sec": self.training_stopped_elapsed_sec,
            "drain_duration_sec": self.drain_duration_sec, "fit_barrier": "initial_readiness_only",
            "last_committed_model_elapsed_sec": self.controller.aggregations[-1]["elapsed_sec"]
                                               if self.controller.aggregations else 0.0,
            "evaluation_policy": "initial, wall-clock cadence and final frozen model",
            "aggregation_order": "coordinator_arrival", "budget_interpretation": "wall_clock",
            "client_weighting": "uniform_per_arrival", "secure_aggregation": False,
            "differential_privacy": False}
