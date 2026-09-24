# Protocol amendment A2: counterbalanced execution order

- Date: 2026-08-10
- Timing: after the four-host orchestration validation and before any
  confirmatory training run
- Outcome access: no confirmatory outcome existed or was inspected

The four methods share the same server and clients and therefore cannot run in
parallel. Their complete execution order was fixed before confirmatory training
to reduce confounding by device temperature, background load, and temporal
drift. Seeds 1--4 use a Williams-style counterbalanced sequence; seed 5 uses a
predeclared additional sequence.

| Seed | Run 1 | Run 2 | Run 3 | Run 4 |
|---:|---|---|---|---|---|
| 1 | best_global_fixed | static_heterogeneous | cosplit_ucb | edge_local_adaptive |
| 2 | static_heterogeneous | edge_local_adaptive | best_global_fixed | cosplit_ucb |
| 3 | edge_local_adaptive | cosplit_ucb | static_heterogeneous | best_global_fixed |
| 4 | cosplit_ucb | best_global_fixed | edge_local_adaptive | static_heterogeneous |
| 5 | static_heterogeneous | cosplit_ucb | edge_local_adaptive | best_global_fixed |

No workload, hypothesis, threshold, pacing phase, exclusion, or analysis rule
changed in this amendment.
