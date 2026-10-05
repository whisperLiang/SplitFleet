# SplitFleet experiments

The physical runner uses CoSplit-UCB with current-device calibration by default. It initializes edge and server learners from three structural anchors (earliest admitted cut, the cut nearest 25%, and the latest admitted cut), then updates them from real training feedback. Calibration reuses the captured graph, performs no optimizer steps, and restores weights, buffers, gradients, module modes and random streams. Network costs start unknown.

Exploration checks both predicted mean and confidence-upper round time against a 5% slowdown budget. Prediction and queue-simulation caches remain enabled. The active context schema is `cosplit_context_v1`: edge/server contexts have 12 features, network contexts 13, and switching contexts 9. There is one adaptive implementation and no initialization or offline-profile placement switch.

## Training

Install task dependencies:

```bash
uv sync --extra dev --extra experiment --extra integration
```

Run RF-DETR Nano using a deployment file for the three physical hosts and a new output directory:

```bash
.venv/bin/python -m experiments.orchestrate_physical_multitask \
  --deployment PATH_TO_DEPLOYMENT.json --run-id nano-default-v1 \
  --model rfdetr_nano --pretrain-weights PATH_TO_CHECKPOINT.pth \
  --tasks object_detection --methods splitfleet \
  --data-root data --train-samples 240 --test-samples 200 \
  --batch-size 1 --rounds 10 --optimizer adam --learning-rate 1e-5 \
  --dirichlet-alpha 10 --split-state-exchange owned
```

For a matched comparison, use `--methods fedavg fedprox splitfed_fixed splitfleet`. Fixed SFL runs all three 25%/50%/75% cuts against the same data partition, initialization and training budget. Calibration applies to SplitFleet. Its time is included in the server's inclusive elapsed time.

| Task | Model | Dataset | Typical worker set | Batch | Adam learning rate |
|---|---|---|---|---:|---:|
| Image classification | ResNet50 | CIFAR-10 | 6 CPU/GPU | 4 | 1e-4 |
| Text classification | BERT-base | AG News | 3 GPU | 1 | 2e-5 |
| Object detection | RF-DETR Nano | VOC2007 | 6 CPU/GPU | 1 | 1e-5 |
| Semantic segmentation | DeepLabV3-ResNet50 | Oxford-IIIT Pet | 6 CPU/GPU | 2 | 1e-4 |

The orchestrator freezes source, verifies remote copies, prepares private client partitions, synchronizes fit starts, and records failures. Each run uses a new directory; failed attempts are retained without automatic reruns or seed replacement. Full epochs retain final partial batches and evaluate after each training round.

## Entry points

| Module | Responsibility |
|---|---|
| `physical_multitask` | Prepare task bundles; run FL/SFL clients and servers |
| `orchestrate_physical_multitask` | Execute a physical training matrix and validate its records |
| `www2027_study` | Freeze source and plans; execute and analyze paired seed blocks |
| `run_edge_standard_study` | Device admission and serial model training |
| `edge_model_admission` | Full-model and native split numerical/resource checks |
| `online_calibration_admission` | Execute calibration anchors on a target device and verify restoration |
| `summarize_physical_multitask` | Validate and summarize completed physical runs |

Calibration and exploration live in `splitfleet/server/placement/cosplit_ucb/`, shared by the native strategy and the physical runner. Core code does not import experiment modules.

`baselines/` and `ablations/` retain the independent learner, bounded-cost oracle, cooperation, joint-solver, discounting and layer-candidate controls needed for research. `heterogeneity/` defines simulated bandwidth/load scenarios. These are experiment interfaces; only CoSplit-UCB is a production adaptive policy.

`analysis/` retains general communication accounting, time-to-quality, partition coverage, backend validation and oracle-gap analysis. Simulated measurements are labeled as such; unit tests do not constitute physical performance evidence.

Historical results, failed attempts, reports and frozen source remain unchanged in `results/` and `paper/evidence/`. Reproduce an older experiment with its frozen runtime and plan. The active tree removes the old profile generators and one-off implementations/comparison scripts for individual optimization stages.
