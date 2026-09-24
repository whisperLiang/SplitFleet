# CoSplit-UCB experiments

This package evaluates SplitFleet's production
`splitfleet.server.placement.cosplit_ucb` implementation. It does not contain a
second scheduler or analytical end-to-end latency model.

The deterministic smoke uses three clients, at least five operation-like
candidates, bounded server concurrency, and 20–40 rounds. Every round records
selected boundaries, learned mean/uncertainty, exploration state, actual
completion, makespan, offline-oracle makespan, instantaneous regret, and
cumulative dynamic regret. The oracle is evaluation-only.

```bash
uv run --no-sync python -m experiments.cosplit_ucb.experiment_runner \
  --method cosplit_ucb --seed 233 --run-id smoke
```

New output is written under `results/cosplit_ucb/`. Existing historical results
under `results/resource_adaptive_splitfed/` are not modified or deleted.

TorchLens discovers candidates, CoSplit-UCB selects them, and Flower performs
federated orchestration and aggregation. The smoke demonstrates mechanism
execution and boundary changes, not method superiority; formal claims require
real devices and multi-seed experiments.
