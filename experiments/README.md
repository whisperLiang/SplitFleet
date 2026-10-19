# SplitFleet experiments

The physical runner defaults to **Greedy** with **`--server-concurrency 1`**. Greedy independently minimizes each client's predicted complete-round cost. Four restored training anchors initialize computation estimates; measured bootstrap RPCs initialize affine communication costs. Actual training feedback updates private cost tables. State exchange is counted once per round, and calibration is included in complete server job time. See [Greedy](GREEDY.md).

The active adaptive implementations are Greedy and the optional L2S reference. Retired UCB controllers are absent from active code and command-line choices. Frozen result directories retain their original labels and source snapshots for provenance.

The [current paper](../paper/README.md) reports 120 fixed-ten-round endpoints across six methods, including 20 measured Greedy C=1 runs. It also retains the original 160-endpoint common-budget comparison with FedAsync, FedBuff and FedCompass. That cohort's SplitFleet arm uses the original frozen CoSplit-UCB controller and is labelled accordingly; it is not a measurement of the current Greedy default. The separate six-round dynamic-network check retains its own protocol. The concurrency-capacity experiment is removed from the current paper.

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

The default comparison is `--methods fedavg fedcompass splitfed_fixed splitfleet`. Fixed SFL runs all three 25%/50%/75% cuts. FedCompass trains the complete model on each client and uses dynamic local steps and semi-asynchronous arrival groups. Calibration applies to SplitFleet, with its time included in the server's inclusive elapsed time.

Use a common wall-clock budget when comparing FedCompass with synchronous methods, for example:

```bash
.venv/bin/python -m experiments.orchestrate_physical_multitask \
  --deployment PATH_TO_DEPLOYMENT.json --run-id resnet-fedcompass-pilot \
  --model resnet50_pretrained --pretrain-weights PATH_TO_CHECKPOINT.pth \
  --tasks image_classification --methods fedavg fedasync fedbuff fedcompass splitfed_fixed splitfleet \
  --data-root data --train-samples 240 --test-samples 200 --batch-size 4 \
  --optimizer adam --learning-rate 1e-4 --dirichlet-alpha 10 \
  --time-budget-sec 600 --evaluation-interval-sec 30 \
  --fedcompass-min-steps 20 --fedcompass-max-steps 100 \
  --split-state-exchange owned --timeout 1800
```

The time budget supersedes `--rounds` for every method. Without a time budget, `--rounds` limits FedCompass global model updates; those are not equivalent to synchronous rounds or equal work. Summaries omit total-duration speed ratios for mixed or timed comparisons. See [the FedCompass implementation and protocol](FEDCOMPASS_BASELINE.md) for source differences, defaults and receipts. Local RPC correctness checks have passed; a new physical performance cohort is still required before updating paper result tables. Historical FedProx measurements retain their labels, and `--methods fedprox` remains available for explicit legacy replay.

| Task | Model | Dataset | Typical worker set | Batch | Adam learning rate |
|---|---|---|---|---:|---:|
| Image classification | ResNet50 | CIFAR-10 | 6 CPU/GPU | 4 | 1e-4 |
| Text classification | BERT-base | AG News | 3 GPU | 1 | 2e-5 |
| Object detection | RF-DETR Nano | VOC2007 | 6 CPU/GPU | 1 | 1e-5 |
| Semantic segmentation | DeepLabV3-ResNet50 | Oxford-IIIT Pet | 6 CPU/GPU | 2 | 1e-4 |

The orchestrator freezes source, verifies remote copies, prepares private client partitions, and records failures. Each run uses a new directory; failed attempts are retained without automatic reruns or seed replacement. Synchronous methods synchronize each round and train full epochs, retaining partial batches. FedCompass uses one initial readiness barrier, then dispatches updates by arrival and group deadline; its exact optimizer-step assignments cycle through the private partition as needed. Timed runs evaluate initially, at a common wall-clock cadence and finally on the frozen model. Drained work and communication are recorded without changing the timed model.

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
| `physical_adaptive_baselines` | Greedy and L2S physical controls |
| `analysis.execution_coverage` | Capture and validate the seven PyTorch model architectures |
| `edge_model_admission` | Full-model and native split numerical/resource checks |
| `summarize_physical_multitask` | Validate and summarize completed physical runs |

Calibration and measured feedback live in `splitfleet/server/placement/greedy/`, shared by the native strategy and the physical runner. Core code does not import experiment modules.

`baselines/` retains the physical Greedy/L2S controls and the FedAsync/FedBuff/FedCompass mechanisms used by the current manuscript. Their calibration, training, model-state exchange and evaluation share the physical task runtime. Unused profile simulations, oracle/ablation experiments and separate five-backend performance runners have been removed; backend adapters and their functional regression tests remain maintained.

## Reproducible execution evidence

The seven-model PyTorch suite requires a fresh output directory and records source, input, dependency and validation identities:

```bash
.venv/bin/python -m experiments.analysis.execution_coverage \
  --suite-bundles paper/evidence/rq1_execution/bundle_catalog.json \
  --output results/execution_coverage_new_run --steps 10
```

The four original client-0 inputs named by the bundle catalog remain available. Capture coverage and numerical training equivalence are distinct; the current manuscript's Figure 6 reports computational topology. Core multi-backend correctness can still be exercised with `python -m splitfleet.validation` and the functional tests.

## Quality and communication analysis

`www2027_study` exports `time_to_quality.json` and `communication.json`. Freeze task-specific thresholds in `analysis_config.tasks[TASK]` with `quality_metric` and `time_to_quality_thresholds` before running the study. TTQ requires that configuration to match the execution-time `frozen_plan.json` and any recorded ledger digest to remain valid. Unreached thresholds remain censored. Failed, running, duplicate or incomplete seed attempts exclude the entire paired block even when successful result files are present. Primary claims also require the full supplied plan to match its frozen execution ledger; removing seeds or changing a budget after execution cannot establish a complete planned population. Historical records without that binding remain descriptive. Paired summaries keep tasks separate even when their primary metrics have the same name.

The frozen TTQ metric may differ from the final quality comparison metric. For example, text classification can use recorded accuracy for TTQ while final quality comparisons use macro-F1. The analyzer keeps both metrics distinct and requires the configured TTQ metric to exist in the recorded evaluations. Analyze an already-running frozen study into a new output directory when correcting an analysis bug; preserve its execution snapshot, plans, raw results and original reports.

New physical records distinguish activation, target, request metadata, gradient, response metadata, and actual serialized Flower model-state upload/download buffers. A total requires every declared client round and every category. The accounting domain is training client application buffers; initialization, calibration, evaluation, Flower config/metrics, protobuf headers, TLS and retries are outside it. Coordinator-local suffix replicas are not client network traffic. Historical records retain unknown categories and totals.

For historical reports, call `analyze(frozen_plan, OLD_ROOT)` read-only and `write_report(report, NEW_EXPORT_ROOT)`. The CLI `analyze --root` writes reports into that root, so do not point it at an immutable historical run. Older source cohorts without online-learning receipts are historical context, not evidence for the current Greedy implementation.

The current manuscript cohorts are indexed in [results/README.md](../results/README.md). Only M01, P08, P09, P10 and E01 remain, together with their required inputs, source snapshots and protocol support files. Unused older cohorts, experimental entry points and their regression cases are removed.

At the author's request, Figure 3/4 failed and excluded attempts were also deleted. Successful raw endpoints and published data remain unchanged. Current checks verify those successes against frozen publication evidence and use [the cleanup receipt](../paper/evidence/manuscript_cleanup_20261013/receipt.json) for deleted source hashes, explicitly without reconstructing deleted attempt populations. Fresh experiments continue to record failures and require a new output directory.
