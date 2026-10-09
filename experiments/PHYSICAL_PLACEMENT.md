# Physical placement controls

`experiments.physical_placement` reuses the real-data SFL worker and server in
`experiments.physical_multitask`. It adds an experiment-only Independent-UCB
adapter, a measured shared training lane, and coordinator round receipts.
Production CoSplit-UCB and its defaults remain unchanged.

For a paired study, freeze source, wheel contents, protocol, data, initial
weights, worker order and seeds. Admit the actual CPU/GPU workers before the
first training arm. Each policy uses fresh processes, the same sample order,
optimizer, epoch budget, evaluation budget and complete admitted cut catalog.
Keep failed attempts and completed controls in their original directories.

The independent adapter measures the restored server calibration once, then
initializes each client's private edge/server/network/switch learners solely
from that client's own calibration and RPC probes. Subsequent observations
remain private. The common server calibration is a measurement of the same
server, not shared client learning. State download size and measured fit-RPC
feedback remain active for both policies.

Example server invocation, from the frozen source:

```bash
python -m experiments.physical_placement server \
  --bundle NEW_SERVER_BUNDLE --device cuda:0 \
  --policy CoSplit-UCB --output NEW_RESULT.json \
  --bind 127.0.0.1:NEW_PORT --rounds 10 \
  --network-control NEW_CONTROL.json --change-round 6 --degraded-mbps 50 \
  --optimizer sgd --learning-rate 0.005
```

Use `--policy Independent-UCB` for the private-learner comparator, or `--policy
Fixed --boundary CANONICAL_BOUNDARY` for a fixed control. The client role takes
the matching policy, device, bundle, server, client identity/index and readiness
barrier. The result's base SFL execution method is retained, while
`physical_placement_experiment.policy` identifies the experimental policy.

`experiments.tcp_pacing` forwards actual TCP traffic on dedicated experiment
ports. Use one proxy per physical host and route that host's SSH reverse tunnel
through its proxy. All Flower model-state and split RPC connections for the
host share one pacer in each direction. The round control file changes the cap
before the next round's instructions. No NIC, route or firewall changes are
required. Give each arm unique forwarding ports to avoid reusing sockets in
TIME_WAIT, and clean up only that arm's processes and SSH control connection.

Verify the cap with real concurrent endpoint transfers before training. Save
each host's raw elapsed duration before checking the start gate. A configured
cap is a ceiling; report measured throughput separately. Forwarded TCP bytes
include gRPC framing but exclude outer SSH/TCP/IP/NIC overhead, management
connections and the separately forwarded readiness barrier.

The actual shared server training semaphore has capacity one, matching the
declared solver resource setting. Its receipts contain coordinator monotonic
queue, acquisition and finish timestamps; validate that service intervals do
not overlap and that each round completes the expected number of batches.
The initial implementation records the envelope's client ID, which can be
blank in this transport. Fleet queue time is measured; per-client queue
attribution is unavailable for those receipts and must not be reconstructed.
The production RTT residual also includes queue waiting; this limitation is
recorded separately from the explicit queue observations.

Record both coordinator round time including placement/bootstrap and the
interval after configuration through aggregation, with evaluation separately.
Keep the first round, its calibration overhead, per-client component counters,
actual model buffers, accuracy/F1 and final model hash. A short pilot cannot
establish convergence, energy advantage or general policy superiority.

The older `experiments.placement_study` cost simulation sets neither actual
Flower model-download nor prefix-state-upload cost in `CandidateEstimate`.
Its communication total includes activation and gradient bytes only. This
omission can favor cuts near the end of the graph when activation traffic
shrinks, and its CPU/A6000 profiles do not measure Orin execution. Retain
those runs as simulation records; use the actual-device runner above to
assess placement and elapsed time. Report protocol changes explicitly when
comparing the simulated record with a new physical study.

`server_duration_sec` starts after loading the evaluation bundle and initial
model, before constructing the placement provider and strategy. It includes
their setup, calibration, physical communication, training and evaluation;
it excludes Python process startup, initial model loading and final result
file writes. Coordinator round receipts distinguish configuration/bootstrap
from the interval through aggregation. Controller arm timestamps cover the
launch-to-completion interval separately.
