# Protocol amendment A10: latest-implementation complete rerun

- Date: 2026-08-12
- Trigger: user requested that the physical experiment be rerun using the
  latest implementation after the A9 parameter-ownership, persistent-suffix,
  native-executor, validator, and native scheduler-cost fixes were complete.
- Timing: frozen before the A10 profile or any A10 formal training outcome.
- Classification: independent post hoc systems replication; A8 and A9 records
  remain immutable.

The rerun uses the same four physical clients, fy205 suffix server, selected
`wide_resnet50_2`, CIFAR-10 seed-1 partition, 100 rounds, batch size 16, four
batches per client per round, SGD configuration, centralized evaluation, and
resource phases as A9.

The source version is frozen before profiling. A fresh nine-round native
profile executes `stem,layer2,layer4` three times. Its validated report alone
selects A10 `best_global_fixed` and `static_heterogeneous` placements. The
native scheduler reads per-round telemetry once and includes observed Flower
coordination overhead. No A8/A9 placement selection may be reused.

After the profile passes, the following seed-1 runs execute sequentially on
the same devices and source version:

1. `fedavg_full_local`;
2. `best_global_fixed`;
3. `static_heterogeneous`;
4. `edge_local_adaptive`;
5. `cosplit_ucb`.

Split methods use `native_prefix` with `persistent_eager`. Clients synchronize
only current-prefix parameters; server suffix replicas persist across rounds;
TorchLens is absent from the timed path. The optional compiled executor is not
used because the retained A9 physical feasibility attempt failed before
`orin140` connected due to an incompatible user-level Triton package. No
fallback is allowed within a run.

Every run must contain 400 client updates, split runs must contain 400 suffix
results and 388 post-calibration decisions, centralized evaluation must contain
100 records, and any recorded failure invalidates the run. Failed attempts are
retained and never overwritten or imputed.

