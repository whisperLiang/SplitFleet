# Protocol amendment A7: secondary traditional FedAvg benchmark

- Date: 2026-08-10
- Timing: requested after seed-1 outcomes from three valid SplitFed methods and
  one invalid edge-local run had been inspected, but before any physical
  FedAvg outcome existed
- Classification: prospective secondary benchmark; not part of the original
  primary confirmatory hypothesis family

## Rationale

The original four-method matrix isolates the effect of split-placement policy,
but cannot answer the broader question of whether split federated execution is
preferable to conventional federated learning. A standard synchronous FedAvg
baseline is therefore added without changing or relabelling the original
primary hypotheses.

## Frozen implementation

`fedavg_full_local` uses the same four physical clients, seed-specific
Dirichlet partition, ResNet-18 with GroupNorm, initial model, SGD learning rate,
batch size, four batches per client per round, 100 rounds, and centralized
10,000-image CIFAR-10 evaluation as the SplitFed matrix. Every client trains
the complete model locally. The fy205 server performs only sample-weighted
Flower FedAvg aggregation and centralized evaluation; it creates no suffix
model and executes no training batch.

Rounds 1--3 are retained as common unreported system warm-up rounds. FedAvg
uses full-local training in those rounds because it has no split runtime.
System summaries use rounds 4--100, matching the original matrix.

The pre-existing constrained-link phase applies application-level pacing only
to split-boundary activation and gradient messages. Full-model Flower
synchronization uses the physical LAN without added pacing in every method.
FedAvg has no split-boundary messages, so its application-pacing delay is zero.
Consequently, comparisons over rounds 26--60 are algorithm-plus-communication-
pattern comparisons, not claims that every network byte experienced a shaped
NIC. Natural-LAN rounds 4--25 and 61--100 are also reported separately.

## Records and validity

Each client records full-model compute time, model download and upload bytes,
peak CUDA allocation when available, identities, partition hash, source hash,
batch count, and loss. The server records round time, centralized accuracy and
loss, process status, and failures. Absence of suffix-server records and
split-decision records is required for this method.

The original fail-closed rule is unchanged: any recorded fit failure, missing
round, missing client update, identity mismatch, partition mismatch, source
hash mismatch, or nonzero process exit invalidates the complete run. Failed
runs are retained and are not imputed.

Seeds 1--5 are planned. The secondary report includes mean and nearest-rank
P95 round time over rounds 4--100, natural-LAN round time, final accuracy,
accuracy versus wall-clock time, time to common accuracy thresholds, model
communication bytes, client compute time, peak client memory, failures, and
participation. These results cannot rescue a failed original primary
hypothesis or be described as part of the original preregistered comparison.
