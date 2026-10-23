# Default measured Greedy placement

The implementation is `splitfleet.server.placement.greedy.GreedyPlacementPolicy`. `GreedyConfig()` uses `server_concurrency=1`. The physical adaptive runner and virtual runtime select Greedy by default. The generic `AutoSplitStrategy` also supports explicit fixed cuts and user-supplied policies.

Before each training round's placement, every selected client actively probes
the split RPC channel used by its training traffic. A small RPC measures the
application-level RTT. Independently enlarging the request or reply estimates
upload or download goodput after subtracting the small-RPC baseline. All
durations use one client monotonic clock, without cross-host subtraction.
The controller receives measured samples, not network-pacing settings.

For client i, candidate c and n local batches, the active score in seconds is:

```
boundary_rpc = measured_rtt
             + activation_bytes * 8 / (measured_upload_mbps * 1e6)
             + gradient_proxy_bytes * 8 / (measured_download_mbps * 1e6)
state_exchange = measured_rtt
               + prefix_upload_bytes * 8 / (measured_upload_mbps * 1e6)
               + model_download_bytes * 8 / (measured_download_mbps * 1e6)
score = n * (prefix_batch_compute + suffix_batch_service + boundary_rpc)
      + state_exchange
```

Each round warms all three payload shapes, followed by three repeats of
baseline (64 KiB / 64 KiB), upload-heavy (2 MiB / 64 KiB) and download-heavy
(64 KiB / 2 MiB) RPCs. The multi-packet baseline avoids the small-message
delivery stalls observed with 1 KiB RPCs on the physical SSH/HTTP2 path. Median
durations limit sensitivity to individual outliers. Clients probe sequentially
so their probes do not compete for the shared RPC service. A nonpositive
directional difference triggers up to two fresh sets of samples; rejected
samples remain in the receipt and all attempts count toward probe time.
Malformed replies fail immediately. The measured RTT includes
fixed RPC overhead, so bootstrap intercepts are not added a second time.
The estimates are application goodput and RPC latency, not NIC bandwidth or
one-way propagation delay. Probe traffic does not construct a suffix model,
take an optimizer step or change training RNG state.

Raw samples, round identity, source, derived context and probe duration are
recorded in `round_diagnostics.network_probes`. Probes finish before selection;
their overhead is included in configuration and complete job time. Old-round,
malformed or unresolved measurements fail that selection round. Resuming
saved compute/cost tables still requires fresh probes. The optional L2S
comparator retains its reference scoring. Custom telemetry without active
probes retains the calibrated affine-byte fallback; configured link rates
are never used as measurements. Historical paper results are not rerun by
this implementation change.

Boundary bytes combine activation size and the differentiable-gradient volume proxy. Round state bytes combine the actual downloaded model-state size and optimizer-owned prefix-parameter upload estimate. State exchange is paid once. The lowest-scoring feasible candidate is selected independently for each client, with a stable tie order. Training/ABI, memory and failure-cooldown checks exclude invalid cuts. The cut remains fixed for the local round.

Initial compute costs come from four admitted anchors nearest 0%, 25%, 75% and 100%. One warmup and one measured optimizer step per anchor use temporary optimizers; weights, buffers, gradients, module modes and random streams are restored. Compute estimates interpolate by captured prefix MAC fraction, with a graph-position fallback when MAC metadata is unavailable. Measured small/large echo and state RPCs initialize nonnegative affine link costs. Evaluation does not perform training calibration.

After a successful fit, measured mean prefix and suffix batch times replace the selected cut's estimates. Actual boundary-RPC and once-per-round state-exchange durations update that client's byte coefficients. No client's observations update another client's table. Failures update feasibility rather than supplying fabricated latency. There is no UCB exploration or cooperative assignment solver. Server load/queue samples remain diagnostics rather than explicit score features.

C controls concurrent suffix execution on the same GPU. C=1 is the default and keeps private suffix replicas while serializing their execution. It is not a claim of optimal throughput. The concurrency-capacity study is excluded from the current manuscript at the author’s request.

Persisted Greedy state identifies the schema, algorithm, graph, backend and C. Incompatible legacy controller states are rejected. A compatible state resumes cost tables by logical client identity and skips bootstrap. Start a fresh state file when deployment characteristics change.

Complete experiment job time includes candidate/runtime construction and calibration, communication, training, aggregation and evaluation. The paper retains actual original Greedy C=1 observations and their hashes; it does not claim a performance rerun of this standalone implementation. Frozen historical sources/results are retained for reproducibility.
