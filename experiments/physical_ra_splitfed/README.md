# Physical RA-SplitFed confirmatory experiment

This directory defines the multi-host experiment in which `192.168.66.136`,
`192.168.66.140`, `192.168.66.118`, and `192.168.66.238` are clients and
`192.168.66.205` is the suffix/Federated Learning server.  It is distinct from
the controlled in-process experiments under `resource_adaptive_splitfed`.

The protocol is frozen before confirmatory training.  Deployment checks,
candidate profiling, and pilot runs are exploratory and must use run IDs with
the `pilot_` or `profile_` prefix.  Their results cannot be pooled into the
confirmatory comparisons.

Amendment A7 adds `fedavg_full_local` as a prospective secondary benchmark
after partial seed-1 SplitFed outcomes had already been inspected. It is kept
outside the original primary hypothesis family. In this method every client
trains the complete model, while fy205 performs only Flower FedAvg aggregation
and centralized evaluation and creates no suffix runtime.

## Evidence layers

1. **Controlled mechanism experiments** use the existing in-process runner to
   isolate CPU, bandwidth, memory, and server-contention factors.  They must be
   reported as `controlled_in_process`, never as physical-device results.
2. **Physical system experiments** execute real CIFAR-10 batches on the four
   clients and a ResNet-18 suffix on the A6000 server over the LAN.  These are
   reported as `physical_multi_host`.

Every non-oracle method within a seed must share the same initial-model hash,
CIFAR-10 partition hash, client order, batch budget, optimizer, and evaluation
set.  A run is invalid if any identity differs, a round is missing, or a
failure is absent from the failure stream.

The machine-readable frozen protocol is in `protocol.yaml`.  Numeric targets
there are decision rules, not expected outcomes.  Results that fail to meet a
target remain in the evidence package and the corresponding hypothesis is
reported as unsupported.

## Execution stages

1. Capture immutable environment, repository, dataset, clock, GPU, memory,
   and link records from all five hosts.
2. Profile every candidate cut on every client without using confirmatory
   model updates.  Choose `best_global_fixed` and per-client static cuts only
   from these profile records.
3. Run a one-seed pilot to check contracts, runtime, and record validation.
4. Freeze any necessary implementation correction before confirmatory runs;
   do not tune thresholds using confirmatory outcomes.
5. Run all methods for seeds 1--5 and validate each run before starting the
   next seed.
6. Compute paired seed-level effects, confidence intervals, and the accuracy
   non-inferiority result.  Preserve raw JSONL, stdout/stderr, resolved config,
   hashes, and failures.
