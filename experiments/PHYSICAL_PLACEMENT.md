# Physical Greedy placement and timing

`experiments.physical_multitask` defaults to `--placement-controller Greedy --server-concurrency 1`. `experiments.physical_adaptive_baselines` offers the same runner with `--policy Greedy` or `L2S`. `experiments.physical_placement` contains common measured timing utilities.

```bash
.venv/bin/python -m experiments.physical_adaptive_baselines server \
  --bundle SERVER_BUNDLE --device cuda:0 --policy Greedy \
  --output NEW_RESULT.json --bind 127.0.0.1:PORT \
  --rounds 10 --optimizer sgd --learning-rate 0.005
```

The server suffix execution capacity defaults to one. `--server-concurrency 2` enables two suffix executions on the same GPU. Each client's suffix has private weights, optimizer and ordered requests; client prefixes can execute concurrently even when C=1. Capacity is distinct from the GPU count and from the number of selected clients.

Greedy uses restored calibration and private measured feedback as described in [GREEDY.md](GREEDY.md). Before each round's placement, selected clients actively measure upload/download goodput and RPC RTT on the training split channel. A network schedule only changes the pacer; its settings are never provided to the selector. Raw probe records and their duration are retained, and probing is included in complete job time. Greedy chooses one cut per client per round. Record coordinator queue/service spans for diagnosis; component spans can overlap and must not simply be added to reconstruct job time.

`server_duration_sec` starts after initial model loading and before placement-provider/runtime construction. It includes setup, calibration, model-state and boundary communication, training, aggregation and evaluation. It excludes input preparation, deployment, process startup, initial model loading and final result-file writing. Client preparation can overlap startup; only its overlap with this inclusive coordinator interval contributes to elapsed time.

`experiments.tcp_pacing` applies real stream pacing to dedicated experiment ports. Save the schedule, measured throughput and raw forwarding receipts. New comparisons must match data, initialization, optimizer, local work, evaluation and network settings, and retain failed attempts. Separate seed/protocol cohorts must not be pooled or substituted for one another.

Historical immutable outputs retain their controller identities. The paper's Greedy C=1 values come from actual Greedy jobs, not renamed UCB observations. The active refactor itself has no new performance run.
