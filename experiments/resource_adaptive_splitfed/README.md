# Resource-Adaptive SplitFed experiments

This package is the reproducible reference implementation for **RA-SplitFed**
(`resource_adaptive_splitfed`). It tests the following systems question with
real model execution and measured records:

> Can a server choose a different cut for each edge client, and change those
> cuts between synchronous rounds, to reduce stragglers, memory failures,
> communication, and server congestion without materially changing
> accuracy-versus-round?

No script in this package generates accuracy, loss, time, bandwidth, energy,
or missing rounds. Failed measurements are stored with `success=false` and a
reason. Energy and unavailable GPU fields are JSON `null`. Plotting skips
missing series and never inserts zeroes or interpolated observations.

## Research basis

- [SplitFed: When Federated Learning Meets Split Learning](https://doi.org/10.1609/aaai.v36i8.20825)
  supplies the synchronous split/federated training pattern: clients train in
  parallel, the two sides of each logical client model are updated, and model
  updates are federated at a round boundary.
- [Flower: A Friendly Federated Learning Research Framework](https://arxiv.org/abs/2007.14390)
  supplies the heterogeneous federated orchestration and weighted aggregation
  model. Flower is infrastructure in this experiment, **not a learning
  algorithm baseline**. The name-based aggregator applies Flower's weighted
  aggregation primitive to every original model-state name.
- [Towards a Unified Framework for Split Learning](https://doi.org/10.1145/3721146.3721936)
  motivates a common split runtime and heterogeneous placement. Its static
  heterogeneous deployment idea is represented by the
  `static_heterogeneous` baseline; static heterogeneous SplitFed is not the same
  method as dynamic resource-aware SplitFed.

The experiment builds on SplitFleet's current production architecture rather
than replacing it:

- `splitfleet.autosplit` prepares TorchLens 2.31 native two-stage runtimes and
  enforces graph, feature-ABI, dynamic-batch, and boundary contracts;
- `BoundaryPayload` is converted to SplitFleet's versioned, pickle-free wire
  envelope, encoded to bytes, decoded for the suffix, and paired with a
  similarly encoded gradient response;
- the client keeps the graph-connected prefix boundary locally while the
  server suffix operates on the decoded payload;
- `AutoSplitSplitLearningClient` and `AutoSplitTailServerModel` are the existing
  Flower client/server bridges; and
- `AutoSplitStrategy.client_placement_fn` can now select and switch a distinct
  placement for every client and round; the stage manager retains all active
  plan/feature ABIs and the strategy reassembles prefix/suffix updates by the
  original state manifest before sample-weighted FedAvg.

RA-SplitFed's scheduler and measurements live in an independent experiment
package. Production changes are limited to an optional suffix phase timing
dictionary plus the reusable per-client placement registry/routing and
name-manifest state reassembly needed by a real distributed deployment.

## Design

### Candidates

`split_candidates.py` captures the actual **training-mode** TorchLens graph and
enumerates legal, single-boundary, trainable candidates. It maps module paths
to stable keys in this order:

```text
stem, maxpool, layer1, layer2, layer3, layer4, full_local
```

If those module paths do not exist, graph-order quantiles provide an explicit
`graph_quantile_fallback` mapping. Every key except `full_local` consumes one
distinct boundary, so a graph exposing fewer than six trainable cuts is
rejected up front with the required count rather than failing part-way through
the fallback mapping. The saved descriptor contains the TorchLens
node ID and label, module path, boundary schema, feature layout, graph
signature, measured-at-trace payload size, and client/server parameter-name
ownership. `full_local` has no boundary and follows ordinary FedAvg local
training.

The earliest cut is not necessarily the fastest: early activations can be much
larger than later activations, and an early cut consumes more server time.

### One complete state per logical client

Every client update is represented by `LogicalClientModelState` with one full
state dictionary keyed by the original ResNet-18 names. The actual TorchLens
runtime updates prefix parameters before the cut and suffix parameters after
the cut. At round end the experiment captures the complete model once and
performs sample-weighted FedAvg independently for every name. Different cuts
therefore cannot shift positional indices, omit a parameter, or aggregate an
unrelated prefix with a suffix.

Non-floating state values are weighted in floating point and rounded back to
their original integer dtype. The default ResNet-18 uses trainable GroupNorm,
so a genuine batch-size-one profile remains defined without duplicating data
or freezing normalization. `normalization: batchnorm` remains available when
batch size is at least two. The initial implementation deliberately
requires SGD with `momentum=0`. Adam, AdamW, and momentum-SGD are rejected if
they contain state; their state is never silently discarded when ownership
moves across a cut.

### Round-boundary switching

A client can switch only before its local epochs begin. Runtime caches are
owned per logical client and keyed by the canonical split. Each cached runtime
retains its own TorchLens plan and feature ABI. Activating a cut records:

```text
runtime_prepare_ms
model_transfer_ms
optimizer_state_transfer_ms
total_switch_ms
```

Preparation, state loading, and optimizer construction happen inside measured
client/round wall time. Switching is never allowed inside a batch or local
epoch.

The distributed Flower control plane accepts the same round-boundary decision
through `AutoSplitStrategy(client_placement_fn=...)`. The callback receives
`(server_round, cid, training)` and returns a TorchLens boundary such as
`after:layer2.1.relu`; client and server instructions reuse the cached decision
for that client/round, so their plan IDs cannot diverge.

### Real measurements

`resource_monitor.py` uses monotonic nanosecond clocks, CUDA events with device
synchronization when CUDA is active, sampled process RSS, PyTorch peak CUDA
allocation, `psutil`, and `nvidia-smi` where available. The server semaphore
measures queue time from attempted submission until execution begins. Network
bytes are the actual serialized boundary/gradient wire lengths; throughput is
computed from bytes divided by measured transfer duration.

Energy fields remain `null` unless a dedicated host exposes readable RAPL
counters and the operator explicitly sets
`SPLITFLEET_RAPL_ATTRIBUTABLE=1`. A shared package counter cannot reliably
attribute client and server energy and is intentionally not reported.

Peak memory follows the same rule. Process RSS and the CUDA peak counter are
process- and device-global, so a measurement window that overlaps another
thread's window measures both clients. Overlapping windows report
`client_peak_memory_mb`/`server_peak_memory_mb` as `null` instead of
attributing a shared peak to one client; with `client_worker_threads: 1`, and
in the sequential profiling path the cost model is fitted from, the values are
real measurements. Windows nested inside one thread stay attributable: only the
outermost window resets the CUDA peak counter, so the suffix measurement no
longer clears the surrounding client measurement.

CPU background load is produced by host-wide busy processes and cannot be
confined to one client thread. Each client record carries
`overlapping_host_load_clients`: the number of *other* clients whose load
emulation overlapped its measured window, with the per-round maximum in
`max_overlapping_host_load_clients`. Any value above zero means that timing
carries another device profile's handicap and is not attributable to its own
profile alone — run with `client_worker_threads: 1` when per-device timings
must be clean. CPU affinity controls are per-thread on Linux and are not
affected.

Default resource controls are unprivileged, controlled experiments:

- encoded byte transfers are paced to configured uplink/downlink ceilings;
- configured RTT is an actual sleep in the paced transport;
- CPU contention is produced by real background work; and
- server contention reduces the real suffix semaphore concurrency and starts
  a measured background workload.

All such runs record `emulation_mode=controlled_in_process`. They must not be
described as physical device changes. Every context has `finally`-equivalent
cleanup for background processes and CPU affinity. No `tc`, cgroup, or
machine-wide affinity rule is left behind.

### Cost model and global scheduler

`SplitCostModel` fits only successful real profile rows. It uses exact lookup,
linear batch scaling when the requested batch was not profiled, current
measured throughput/RTT, measured memory, current server jobs/queue, and an
online exponential moving average. Interpolation/scaling is used only for
prediction, never to create an experiment result.

For candidate `s`, the prediction is:

```text
batches per round × (
    client compute(s)
    + actual boundary bytes / measured uplink
    + actual gradient bytes / measured downlink
    + measured RTT
    + server compute(s)
    + predicted bounded-concurrency queue(s)
)
+ measured/predicted switch cost(s)
```

Every profiled compute and network metric is per batch, while a prediction is
compared against a measured whole-round `completion_ms`.
`batches_per_round` (the client's sample count, batch size, local epochs, and
any per-epoch cap) converts the per-batch profile into the same unit; the
switch cost is a per-round cost and is added once. Without that conversion the
online average and the profile path are different units, and the scheduler
switches away from whichever cut a client just used.

Online observations update a multiplicative error calibration rather than
replacing the prediction with an absolute historical duration. Current CPU,
network, and server terms therefore continue to respond when a resource phase
changes.

Peak memory is neither batch-invariant nor linear: it is a fixed model and
optimizer footprint plus a per-sample activation footprint. When two or more
batch sizes were profiled for a split, memory at the requested batch is a
least-squares fit over those measurements, floored by the largest measurement
taken at a smaller batch. With one profiled batch the whole measurement is
scaled, which over-estimates the fixed part and keeps the feasibility test
conservative rather than optimistic. Leaving it at the profiled batch's value
would report a batch-8 footprint as a batch-32 prediction and pass a client
that then runs out of memory.

Configured `memory_limit_mb` values are enforced before dispatch from the
profile-backed peak prediction and checked again against attributable measured
peaks. The check applies to baselines and memory-constraint ablations as well as
the adaptive scheduler, so exceeding a simulated device limit is recorded as an
OOM instead of silently succeeding on the larger host.

`predicted_switch_ms` is taken from the candidate whenever the candidate
provides one, including an explicit `0.0`; only a missing value falls back to
the profiled `runtime_prepare_ms`. The `no_switch_cost` ablation depends on
that distinction.

Memory violations mark candidates infeasible. The server-global scheduler uses
greedy list scheduling followed by bounded coordinate improvement to minimize
the predicted synchronous makespan over real server lanes. Independent client
selection is not generally globally optimal: many clients can choose an early
cut simultaneously and overload the suffix server.

The default hysteresis is a 10% required relative improvement and a three-round
minimum hold. An optional per-round switch cap is also enforced.

## Methods

- `fedavg_full_local`: every client trains the complete model locally.
- `fixed_early`, `fixed_middle`, `fixed_late`: one uniform semantic cut.
- `best_global_fixed`: the fastest mean cut in the separate profile/calibration
  run; it is never selected from final training results.
- `static_heterogeneous`: weak→stem, medium→layer2, strong→layer4 for all rounds.
- `compute_only_adaptive`: dynamic client compute/memory choice, with network
  and server costs removed.
- `edge_local_adaptive`: each client minimizes its own feasible predicted time.
- `resource_adaptive_splitfed`: globally coordinated full RA-SplitFed.
- `oracle`: on the first round of each resource phase, independent model copies
  execute every candidate on a fixed real evaluation batch. Probe gradients
  never reach the formal model. The choice is cached for the phase.
- `fedavg_full_local_with_timeout` and `fedavg_strong_clients_only`: participation
  fairness baselines.
- `no_network`, `no_server_state`, `no_memory_constraint`, `no_hysteresis`,
  `no_switch_cost`, `offline_profile_only`, `full_resource_adaptive`: ablations.

All non-oracle methods for a seed use the same initialization hash, Dirichlet
partition hash, sampled-client order, local epochs, batch size, and optimizer.
The validator compares these identities across sibling method runs.

`client_profiles` counts must sum exactly to `num_clients`; an over-specified
mapping is rejected instead of being truncated into a run with no clients of
the last profile. The Dirichlet partition repair guarantees that no client is
left without samples: sharing a class that only one client holds trades a
sample back when the donor would otherwise be emptied, and the final partition
is checked before training starts. A class with a single sample in the whole
dataset cannot be shared; the repair then keeps every partition non-empty and
leaves that class where it is.

## Experiments

1. `profile_resnet18_cifar10.yaml`: real batches for each
   device×network×cut×batch tuple; Fig.1 resource curves and Fig.2 measured
   optimal-cut heatmap.
2. `static_heterogeneity.yaml`: fixed resources and non-IID clients; Fig.3 round
   time plus Fig.4 accuracy-versus-round and accuracy-versus-wall-clock.
3. `dynamic_resources.yaml`: normal, client CPU contention, poor uplink, server
   contention, and recovery phases; six-panel Fig.5.
4. `server_scaling.yaml`: 4/8/16/32/64 clients × 2/4/8/16 server concurrency;
   Fig.6 time and Fig.7 queue growth.
5. `participation_fairness.yaml`: weak clients receive a larger, but not
   exclusive, share of CIFAR-10 classes 8 and 9; Fig.8.
6. `ablation.yaml`: all required ablations; Fig.9.

The intended conclusion must be checked on both accuracy-versus-round and
accuracy-versus-wall-clock. A system speedup with degraded learning per round
is not evidence for the research claim. Target thresholds in the research plan
are evaluation goals, not code assertions and not hard-coded outcomes.

## Installation

```bash
uv sync --extra dev --extra experiment
```

Install the optional TensorFlow/JAX/Paddle/tinygrad validation matrix with:

```bash
uv sync --extra dev --extra experiment --extra multibackend
```

Python 3.11, Torch 2.11, TorchLens 2.31, torchvision, PyYAML, psutil, and
matplotlib are locked by `uv.lock`.

## Run

Profile first (the default is deliberately substantial):

```bash
uv run --no-sync python -m experiments.resource_adaptive_splitfed.resource_profile \
  --config experiments/resource_adaptive_splitfed/configs/profile_resnet18_cifar10.yaml \
  --run-id profile_resnet18_cifar10
```

Run one method/seed:

```bash
uv run --no-sync python -m experiments.resource_adaptive_splitfed.experiment_runner \
  --config experiments/resource_adaptive_splitfed/configs/static_heterogeneity.yaml \
  --method resource_adaptive_splitfed --seed 1 \
  --run-id static_ra_splitfed_seed1
```

```bash
uv run --no-sync python -m experiments.resource_adaptive_splitfed.experiment_runner \
  --config experiments/resource_adaptive_splitfed/configs/dynamic_resources.yaml \
  --method edge_local_adaptive --seed 1 \
  --run-id dynamic_edge_local_seed1
```

Run every configured method and seed. `run_suite` also expands the Cartesian
product under `sweep`, so it executes the server-scaling grid:

```bash
uv run --no-sync python -m experiments.resource_adaptive_splitfed.run_suite \
  --config experiments/resource_adaptive_splitfed/configs/server_scaling.yaml \
  --run-prefix scaling --resume
```

`--resume` skips only complete runs that pass strict validation. It stops on an
invalid partial directory instead of deleting, repairing, or overwriting real
records.

Validate, aggregate, and plot:

```bash
uv run --no-sync python -m experiments.resource_adaptive_splitfed.validate_results \
  --results-dir results/resource_adaptive_splitfed/static_ra_splitfed_seed1

uv run --no-sync python -m experiments.resource_adaptive_splitfed.aggregate_results \
  --results-root results/resource_adaptive_splitfed \
  --output results/resource_adaptive_splitfed/aggregated

uv run --no-sync python -m experiments.resource_adaptive_splitfed.plot_results \
  --results-dir results/resource_adaptive_splitfed/aggregated
```

Use `configs/smoke_profile.yaml` and `configs/smoke.yaml` for a small real-data
pipeline check. `smoke_profile_gpu.yaml` and `smoke_gpu.yaml` exercise real CUDA
timing, allocation, and `nvidia-smi` state. These reduce observations and
rounds; they are not substitutes for the five-seed experiments.

## Results and statistics

Every training run contains the requested metadata/config/environment,
assignments and hashes, candidates, five raw JSONL streams, a recomputable CSV
summary, validation report, and figure directory. Profile runs additionally
contain `profiles/profile_records.jsonl` and `profile_summary.csv`.

Aggregation computes mean, standard deviation, median, p95, number of valid
runs, and a normal 95% interval. Figures shade the interval only for at least
three real observations and label smaller samples as `n<3; no 95% CI`.
Paired method comparisons use matching seeds; the CSV labels its p-value as a
normal approximation rather than presenting it as an exact small-sample test.

Validation rejects missing/duplicate rounds, NaN/Inf, negative units, illegal
cuts, unrecorded failures, changed run identities, non-CIFAR test accuracy,
placeholder tokens, and summaries that cannot be recomputed. Curves with no
valid measurements are skipped with a warning.

## Real edge/cloud deployment and limitations

For physical devices, remove `device_profile_controls` and `network_profiles`,
set the appropriate `device`, run the existing SplitFleet Flower client and
tail server on those hosts, and feed their measured resource states and profile
JSONL into the same scheduler/result schema. Linux `tc`/cgroups or a hardware
lab controller should be applied outside this unprivileged reference runner
with guaranteed cleanup. A real RPC heartbeat should replace the in-process
RTT probe.

Current limitations are explicit:

- the reproducible reference runner is in-process; default heterogeneity is
  controlled emulation and can affect colocated client/server work;
- CPU RSS is process-wide on an in-process run and can include allocator reuse;
- GPU utilization requires `nvidia-smi`, and GPU memory is null without CUDA;
- separate client/server energy requires separate attributable meters;
- each client now receives a deterministic stratified 10% local holdout for
  `worst_client_accuracy`, but a physical deployment may prefer a naturally
  collected temporal test set; and
- full multi-host resource attribution still requires separate device meters
  and a real RPC heartbeat rather than the reference in-process probe.

The best cut is expected to change with device, network, and server state.
Whether RA-SplitFed actually meets the proposed 0.5-point accuracy, 15% wall
time, 20% p95, and 10% oracle-regret goals is determined only by completed
measurements. Results that miss those targets remain in the output unchanged.
