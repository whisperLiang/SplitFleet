# SplitFleet experiments

The physical runner uses the canonical CoSplit-UCB implementation by default. Four structural anchors, nearest 0%, 25%, 75% and 100% of the admitted catalog, initialize current-device costs. Each uses one warmup and one measured batch with temporary stage-owned optimizers. Calibration restores weights, buffers, gradients, module modes and random streams; temporary optimizer state is discarded.

The `cosplit_context` schema uses actual owned parameter bytes and actual payload sizes for each batch size. Edge and server contexts have 13 features, network contexts 14, switching contexts 9 and state exchange contexts 5. Current-deployment echo probes initialize transport priors; actual split-RPC and fit-RPC feedback continue updating them during training. State exchange is modeled once per client round with nonnegative coefficients.

Both predicted mean and confidence-upper round time must fit within 105% of the baseline mean before a probe is admitted. The joint solver searches the full admitted catalog from the first round. Prediction and queue-simulation caches remain enabled. Decisions remain per round; batch observations are aggregated and weighted by the actual batch count. See the [algorithm guide](COSPLIT_UCB.md).

## Training

Install task dependencies:

```bash
uv sync --extra dev --extra experiment --extra integration
```

Run RF-DETR Nano using a deployment file for the three physical hosts and a new output directory:

```bash
.venv/bin/python -m experiments.orchestrate_physical_multitask \
  --deployment PATH_TO_DEPLOYMENT.json --run-id nano-default \
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

RF-DETR uses gather-based bilinear sampling with zero padding and
`align_corners=False` on every device and for every comparison method. This is
the same interpolation used by its deformable attention, expressed as ordinary
PyTorch operations following RF-DETR's existing MPS sampler. CUDA
`grid_sample` backward failed native/native repeatability in admission;
strict deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8` make the
gather gradients reproducible. Boolean position-mask counts use integer
`cumsum` followed by FP32 conversion, preserving the sine encoding exactly
within the model's image sizes and supporting strict mode on NVIDIA PyTorch 2.5.
The builder preserves the caller's matmul precision across RF-DETR import.
Sampling remains on the requested CPU/GPU, and
the full detector, pretrained weights and optimizer are retained. The math
policy is recorded in each admission and physical result. Frozen historical
snapshots retain their original implementation; compare results within a source
cohort.

DeepLab uses equivalent separable bilinear interpolation with half-pixel
coordinates and border replication, expressed with `index_select` on every
device. This gives CPU and CUDA the same captured graph and a deterministic
backward under strict mode. Spatial sparse cross entropy uses flattened pixel
rows with the same ignored-label normalization. Independent native-kernel
tests compare both outputs and gradients. All model modules, weights, auxiliary
head, batches and optimizer settings are retained. All six methods share these
operations; the interpolation policy is recorded with the math settings.
See `dependency_patches/DEEPLAB_INTERPOLATION_NOTICE.md` for attribution.

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

## Reproducible correctness and placement evidence

All evidence entry points below require a new output directory and record source, configuration, dependency identities, logs, results and validation receipts.

Profile and simulation children check the frozen source, configuration and installed TorchLens content hashes before execution. Simulation also checks the copied measured profiles. Each child claims its output directory once; completed, failed and interrupted attempts cannot be rerun over their original records, including through internal `--frozen` entry points.

```bash
.venv/bin/python -m experiments.analysis.canonical_backend_correctness \
  --output results/rq1_canonical_new_run
.venv/bin/python -m experiments.analysis.backend_validation \
  --output results/rq1_gated_residual_new_run
.venv/bin/python -m experiments.analysis.partition_coverage \
  --bundle PATH_TO_REAL_TASK_BUNDLE.pt --device cuda:0 \
  --output results/rq2_structural_new_run
.venv/bin/python -m experiments.placement_study profile \
  --bundle PATH_TO_REAL_TASK_BUNDLE.pt --device cuda:0 --repeats 3 \
  --output results/rq2_measured_landscape_new_run
.venv/bin/python -m experiments.placement_study simulate \
  --config PATH_TO_FROZEN_SIMULATION_CONFIG.json \
  --output results/rq3_simulation_new_run
```

Omitting `--bundle` from `profile` executes a download-free MLP fixture. Profiles measure actual local prefix/suffix forward/backward and encoded activation/gradient sizes with model/RNG restoration and no optimizer steps. They do not measure distributed round latency. CPU/GPU profiles must share graph, ABI, checkpoint and numerical input identities; mismatches and missing cost rows are refused.

Simulation configurations declare clients (`id`, `profile`, `num_batches`, optional bandwidth multiplier), paired policy seeds, rounds, scenario phases, adaptation tolerance and sustained recovery window. Each phase supplies `starts_round`, network upload/download Mbps and RTT, plus optional client/server cost multipliers and server concurrency. The default matrix includes fixed 25/50/75%, Independent-UCB, CoSplit-UCB, exact bounded offline oracle, layer-only, no cooperation, no joint solver and gamma=1. Every arm uses the same declared operation catalog and conditions; layer-only keeps its original descriptors. An oracle domain restriction must be explicit, never silent.

Profiles are measured local costs; network transfers and phase changes in this runner are analytical simulation. `placement_trace.jsonl` records estimates, observed simulated costs, queue time, makespan, cut switches and oracle gap per client/round. Policy seeds on one timing profile do not count as independent physical repetitions. Oracle is not deployable and optimizes the declared table and lane simulator.

The canonical residual MLP checks six numerical properties with identical backend math, weights and inputs. The gated residual suite uses backend-specific models: JAX omits BatchNorm and tinygrad uses scalar affine residual blocks. Its passing replay/training checks do not establish complete native loss/gradient/update parity. Structural candidate characterization also remains distinct from numerical replay verification.

## Quality and communication analysis

`www2027_study` exports `time_to_quality.json` and `communication.json`. Freeze task-specific thresholds in `analysis_config.tasks[TASK]` with `quality_metric` and `time_to_quality_thresholds` before running the study. TTQ requires that configuration to match the execution-time `frozen_plan.json` and any recorded ledger digest to remain valid. Unreached thresholds remain censored. Failed, running, duplicate or incomplete seed attempts exclude the entire paired block even when successful result files are present. Primary claims also require the full supplied plan to match its frozen execution ledger; removing seeds or changing a budget after execution cannot establish a complete planned population. Historical records without that binding remain descriptive. Paired summaries keep tasks separate even when their primary metrics have the same name.

The frozen TTQ metric may differ from the final quality comparison metric. For example, text classification can use recorded accuracy for TTQ while final quality comparisons use macro-F1. The analyzer keeps both metrics distinct and requires the configured TTQ metric to exist in the recorded evaluations. Analyze an already-running frozen study into a new output directory when correcting an analysis bug; preserve its execution snapshot, plans, raw results and original reports.

New physical records distinguish activation, target, request metadata, gradient, response metadata, and actual serialized Flower model-state upload/download buffers. A total requires every declared client round and every category. The accounting domain is training client application buffers; initialization, calibration, evaluation, Flower config/metrics, protobuf headers, TLS and retries are outside it. Coordinator-local suffix replicas are not client network traffic. Historical records retain unknown categories and totals.

For historical reports, call `analyze(frozen_plan, OLD_ROOT)` read-only and `write_report(report, NEW_EXPORT_ROOT)`. The CLI `analyze --root` writes reports into that root, so do not point it at an immutable historical run. Older source cohorts without online-learning receipts are historical context, not evidence for the current CoSplit-UCB implementation.

Algorithm iteration snapshots, intermediate comparisons and process reports have been removed. Formal baseline records remain in `results/`; fresh experiments must use a new output directory and the current source.
