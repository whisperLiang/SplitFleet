# Protocol amendment A3: bounded TorchLens runtime lifecycle

- Date: 2026-08-10
- Timing: after the first confirmatory attempt failed, before any complete
  confirmatory run existed
- Trigger: seed-1 `best_global_fixed` failed during round 81 with CUDA OOM

The failed attempt is retained in
`results/physical_ra_splitfed/confirm_best_global_fixed_seed1_20260810` and is
excluded from complete-run estimands under the predeclared missing-round rule.
It is not replaced or deleted.

Diagnosis showed that the server created four new suffix model objects every
round. Each object therefore missed its model-local TorchLens runtime cache,
while the manager retained every cloned runtime under a key containing the new
model identity. GPU memory grew cumulatively to the device limit.

The repair is limited to TorchLens stage-runtime execution: the same suffix
model object is reused for a stable SID across rounds. `configure_fit` still
loads that round's aggregated parameters, creates a fresh optimizer, resets
loss/example counters, and clears step identities. Ordinary non-TorchLens
server model managers remain round-scoped. Allocated, reserved, and peak server
CUDA memory are now recorded for every suffix result.

Because this repair changes runtime preparation overhead, the pre-repair
physical candidate profile is retained but invalidated for confirmatory policy
selection. A new physical profile must pass validation before restart. All four
methods and all seeds then restart from round 1 with the repaired source.

No workload, data partition, model, optimizer, pacing phase, scheduler,
hypothesis, threshold, exclusion, or statistical analysis rule changed.
