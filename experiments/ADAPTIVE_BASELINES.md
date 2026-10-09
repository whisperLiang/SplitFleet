# Adaptive placement references on actual devices

`physical_adaptive_baselines` uses the existing real-data SFL worker, owned
prefix-state exchange, all-client barrier, measured common server lane and
actual TCP pacing. It changes placement controllers; it does not reproduce
the complete L2S or FedAdapt training systems.

The comparison has six schemes: production CoSplit-UCB over all admitted
operation cuts; the same policy over admitted semantic module endpoints;
Independent-UCB, L2S-LinUCB-E, Profile-Greedy and FedAdapt-PPO over that same
module subset. `LayerScopedProvider` retains original cut identities and
does not create unsupported cuts. Bootstrap must also use an admitted cut
from the selected domain.

L2S-LinUCB-E implements the latency lower-confidence decision and ridge
updates from Algorithm 1 of Zhang and Xu, *Learning the Optimal Partition
for Collaborative DNN Training With Privacy Requirements*, IoTJ 2022,
DOI `10.1109/JIOT.2021.3127715`. The adaptation makes one decision per
federated round. Captured convolution/linear MACs, miscellaneous elementwise
work, parameter counts and boundary-volume proxies form eight structural
features; three extra features represent owned upload, global download and
framing. Four measured prefix anchors estimate frontend costs. Targets use
actual complete fit-RPC duration, including model-state exchange, minus
measured frontend time. Features have a fixed admitted-domain scale. The
fully-local action is absent from this two-sided split catalog, so the
implemented escape rule does not activate in the physical study. No privacy
threshold, privacy guarantee or original per-batch regret result is claimed.
The reference's own latency ridge starts at the paper's zero prior; inherited
component statistics serve common calibration and receipt validation and do
not choose its training actions.

Profile-Greedy is a repository reference, not a published system. It chooses
the least predicted solo round cost using four-anchor MAC interpolation,
actual transport-echo slopes and subsequent observed component costs.
It uses neither confidence bounds nor a joint shared-server queue objective.
Captured gradient volume is a proxy; actual activation/gradient/model-state
buffers are recorded separately and are the communication endpoint.

FedAdapt-PPO loads the authors' unmodified `PPO.py` as a separately supplied
artifact. Source: <https://github.com/qub-blesson/FedAdapt>, accompanying
Wu et al., *FedAdapt: Adaptive Offloading for IoT Devices in Federated
Learning*, IoTJ 2022, DOI `10.1109/JIOT.2022.3176469`. The authors' actor and
critic have 64/32 hidden units, with Gaussian exploration and PPO clipping.
The authors' code learning rate `.0003`, discount `.9`, clipping `.2`,
standard deviation `.5`, ten-transition updates and fifty optimization
epochs are retained. This code differs from the article's stated learning
rate; the source artifact and choice are recorded.

The adaptation maps continuous workloads to the nearest admitted cumulative
MAC fraction, rather than to graph depth. Three clients form three singleton
groups (`K=G=3`). An actual latest-admitted-cut round establishes the
normalization reference instead of the unavailable pure-FL action. Fifty
subsequent actual truncated training rounds supply PPO transitions. The
frozen final actor is reused for the five primary seeds; test accuracy is
not a reward, stopping criterion or checkpoint-selection criterion. Report
the separate preparation cost and the fixed fifty-transition budget.
Deployment never updates the actor. This is a controller adaptation with a
different workload and training stack, not a full-paper replication.

Fit proxies record client dispatch/completion on the coordinator's monotonic
clock. For each round, `max(finish)-finish_i` measures coordinator-observed
synchronization waiting. Per-client fit duration, its max/median ratio and
duration spread are different endpoints. Wire client IDs remain absent in
server-lane receipts, so per-client server-queue waiting is unknown.

Every cohort must freeze source, data, initialization, parameter budgets,
policy domains, method order, network controls and quality thresholds before
training. Preserve failed attempts and use fresh output directories. Report
all methods' time and accuracy/F1, actual state traffic, unreachable quality
thresholds, calibration and PPO preparation; do not equate fixed-work time
with equal-quality acceleration.

## Reproduction

Prepare a new data directory with the recorded checkpoint and real dataset:

```bash
.venv/bin/python -m experiments.prepare_adaptive_reference_bundles \
  --output /tmp/splitfleet-adaptive-bundles-new \
  --data-root /home/whisperliang/SplitFleet/data \
  --checkpoint /home/whisperliang/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
.venv/bin/python -m experiments.prepare_adaptive_reference_layers \
  --bundles /tmp/splitfleet-adaptive-bundles-new --device cuda:0
```

Both commands refuse existing outputs. Layer derivation changes only admitted
calibration anchors and preserves data, weights, partition and cut identities.
Use the exact frozen source, dependency overlay, data hashes and protocol for
a source-matched replay. The study root
`results/www2027_final/adaptive_baselines_20261007_v4` contains the executable
`run_physical_cohort.py`, exact `protocol.json`, deployment/admission receipts,
and each actual server/client command. It continues the identical frozen V3
runtime after a recorded numerical training failure, running only previously
unattempted arms. Successful V3 receipts and the failed arm remain at their
original paths. A replay needs a fresh root and owned
remote endpoints; do not rerun into completed arms. The controllers live in
`experiments/baselines/adaptive_reference.py`; the production scheduler is
unchanged.

Completed model weights and historical input copies have been removed after
verification. Metrics, failed attempts, frozen sources and the actual-trained
PPO actor remain. Prepare new bundles for a training replay. Reanalysis below
uses retained raw receipts and explicitly recorded pre-deletion weight hashes;
it does not perform a fresh check of removed tensors. See
`paper/evidence/storage_cleanup_receipt.json` for storage provenance.

Current preparation writes v3 role descriptors with one shared weight copy per
distinct initialization and shared data assets. Layer descriptors reuse those
assets. For manual remote staging, send only the selected role's descriptor and
the files returned by `experiments.common.bundle_storage.bundle_dependencies`,
preserving their relative paths. The physical matrix stages these dependencies
once per host automatically. Copying only a descriptor is insufficient.

Current server entrypoints default to JSON results plus in-memory final-state
verification, without full task checkpoints. Add `--save-model` to retain those
weights for re-evaluation. This does not suppress the trained PPO actor.
Standalone prepared inputs remain available across controller arms; their owner
may remove them after the cohort finishes. `orchestrate_physical_multitask` owns
and removes its temporary inputs by default; `--keep-input-bundles` retains them.
Historical frozen runtimes retain their original behavior and source hashes.

Supply the verified authors' `PPO.py` artifact separately: commit
`25f4ba0b97582f5b75a992c5bd8c4f78b20e025c`, SHA256
`625c80bb782ae5b38b6796926894858ead108b03b48f6aa033b8a277eaaceb31`.
The formal `FedAdapt-PPO` arm requires the fixed actual-trained actor; an
untrained actor is not admitted as the published-controller adaptation.

Derive completed actual receipts into a fresh analysis directory:

```bash
.venv/bin/python -m experiments.summarize_adaptive_baselines \
  --study-root results/www2027_final/adaptive_baselines_20261007_v4 \
  --output /tmp/splitfleet-adaptive-analysis-new
```

The derivation verifies raw receipt hashes, actual server-lane steps, complete
fit populations, per-batch means/sample counts and clock order. Inference
requires all thirty planned arms to be terminal. Numerical failures remain
failures without retry or seed replacement. Means use completed jobs; paired
intervals use each contrast's jointly completed seeds and therefore condition
on completion. Completion/failure counts are always reported. Failed arms are
excluded from TTQ counts and timing rather than classified as unreached;
unreached targets in completed jobs are right-censored. The failure-aware
analysis change is recorded in `analysis_plan_after_failure.json`. Completed
PPO preparation and deployment use separate endpoints.
