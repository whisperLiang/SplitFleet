# Adaptive placement comparison

The current method set is CoSplit-UCB, Independent-UCB, Greedy and L2S. All use the default admitted training catalog and the same physical SFL runtime, task inputs, optimization, state exchange, aggregation and evaluation.

The completed comparison adds 60 Independent-UCB/Greedy/L2S jobs to the unchanged 100 P08 baseline jobs in Figure 3. Four tasks and five paired seeds use the same ten-round protocol. The completed binding is in `paper/placement_comparison_plan.json` and `results/figure3_placement_20261012/placement_protocol_audit.json`.

- CoSplit-UCB is the proposed cooperative placement policy.
- Independent-UCB is a repository control with private per-client learners and independent placement decisions.
- Greedy is a repository control selecting the lowest predicted solo round cost from measured calibration and subsequent feedback.
- L2S adapts LinUCB-E from Zhang and Xu, *Learning the Optimal Partition for Collaborative DNN Training With Privacy Requirements*, IoTJ 2022, DOI `10.1109/JIOT.2021.3127715`. The adaptation uses one decision per federated round, captured structural and state-transfer features, and actual complete fit-RPC feedback. The admitted two-sided catalog has no fully-local action, so the original escaping rule is inactive. It is a controller adaptation rather than a complete-system replication.

The entrypoint is `experiments.physical_adaptive_baselines`, with exactly four policy names: `CoSplit-UCB`, `Independent-UCB`, `Greedy`, `L2S`. Each controller receives the same provider. Input preparation uses the common `physical_multitask.prepare_bundle` interface with the same admitted catalog.

Before execution, bind the source identity, data/initialization hashes, paired seeds, method order, hardware, timing, network schedule and analysis plan. Use a fresh output directory and report task quality with duration, calibration, state traffic, cut changes and synchronization waiting. Failed attempts remain explicit in any new run; do not replace missing endpoints by zero.

The old 72-cut batch, five older single-method CoSplit-UCB runs and their separate preparation/summary entry points are removed. At the author's request, failed/excluded Figure 3 attempts are also removed. The 60 published raw successes and paired protocol audit remain unchanged. `paper/figures/generate_evaluation_figures.py` validates their hashes, matching settings and measured duration/quality before plotting; `paper/refresh_evidence.py` refreshes current evidence in a new directory.
