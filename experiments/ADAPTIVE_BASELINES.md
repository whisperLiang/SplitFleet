# Adaptive placement

SplitFleet defaults to measured Greedy placement with server execution capacity C=1. `experiments.physical_multitask` uses `--placement-controller Greedy`; `experiments.physical_adaptive_baselines` accepts `--policy Greedy` or `--policy L2S`. Both use the shared admitted training catalog, worker execution, state ownership, aggregation and evaluation. The default policy is implemented in `splitfleet/server/placement/greedy/`.

Greedy selects the lowest predicted standalone local-round cost from private calibration and actual fit feedback. It does not optimize a joint server assignment. [The guide](GREEDY.md) specifies units, bootstrap and timing.

L2S retains the LinUCB-E adaptation from Zhang and Xu, *Learning the Optimal Partition for Collaborative DNN Training With Privacy Requirements*, IoTJ 2022, DOI `10.1109/JIOT.2021.3127715`. It makes one decision per federated round using structural/state-transfer features and actual complete fit-RPC feedback. The admitted two-sided catalog has no fully-local escape action. This is a controller adaptation rather than a complete-system replication.

The current manuscript uses the original 20 Greedy and 20 L2S endpoints together with 80 FedAvg/fixed-cut controls. Source labels and raw hashes remain unchanged. Earlier UCB results and frozen implementations remain historical evidence only; the corresponding active controllers and command-line options have been removed. The current runtime refactor has not been performance-rerun.

For new comparisons, freeze source, initial weights, data partitions, optimizer, sample order, paired seeds, network and evaluation settings. Use fresh result directories and include initialization/calibration in complete job time. Preserve failures and report quality together with time.
