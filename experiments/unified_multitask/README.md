# Unified four-task federated benchmark

This package executes the same client training and sample-weighted logical
model aggregation interface for image classification, text classification,
object detection, and semantic segmentation. Its methods are `fedavg`,
`fedprox`, `splitfed_fixed`, and `splitfleet` (the repository's capability-aware
per-client cut policy). The split methods execute TorchLens training prefixes
and suffixes, with encoded/decoded activation and gradient envelopes.

The runner is **sequential and in-process**. It verifies multi-task integration
and learning curves on real datasets. Its elapsed time includes TorchLens
capture on each client and local wire serialization; it does not measure a real
LAN, separate edge device memory, or a physical heterogeneous fleet. Use
`experiments/physical_ra_splitfed` for multi-host system claims and
`experiments/resource_adaptive_splitfed` for controlled resource sweeps.

## Data layout

| Task | Loader | Expected data |
| --- | --- | --- |
| Image classification | `torchvision.datasets.CIFAR10` | `data/cifar-10-batches-py/`; `--download` is supported |
| Text classification | local AG News CSV | `data/ag_news_csv/train.csv`, `test.csv`; each row is `label,title,description`, with labels 1–4 |
| Detection | `torchvision.datasets.VOCDetection` | Pascal VOC 2007 trainval/test under `data/VOCdevkit/`; `--download` is supported |
| Segmentation | `torchvision.datasets.OxfordIIITPet` | Oxford-IIIT Pet trainval/test under `data/oxford-iiit-pet/`; `--download` is supported |

For AG News, the local CSV format matches the dataset used by the
[Flair AGNEWS loader](https://github.com/flairNLP/flair/blob/master/flair/datasets/document_classification.py).
The VOC and Pet loaders use [torchvision's dataset APIs](https://docs.pytorch.org/vision/main/datasets.html).
Data are never downloaded unless `--download` is passed. The exact transformed
training and test samples used in a run are hashed and recorded in `metadata.json`.
For capped real-data pilots, samples are deterministically spread across the
entire source index range rather than drawn from the first rows; this avoids
the class-ordered AG News CSV producing a single-class pilot. The text and
image macro-F1 evaluators retain all four or ten classes even if a tiny pilot
test subset misses some classes.
The fixture source is deterministic and download-free, but is useful only for
mechanism checks.

To obtain the AG News CSV mirror used by Flair, run:

```bash
uv run --no-sync python -m experiments.unified_multitask.fetch_ag_news \
  --destination data/ag_news_csv
```

The fetcher checks 120,000 train and 7,600 test records, validates their
three-column structure, and records file SHA-256 values in `source_manifest.json`.

The reference architectures are intentionally small: a convolutional image
classifier, masked-mean text classifier, three-layer grid detector, and compact
segmentation network. The grid detector predicts one object per spatial cell;
if several ground-truth objects land in one cell, the loss uses the first and
the architecture's limitation must be disclosed in VOC analyses. The tasks can
replace their model factories and metric implementations through `TaskSpec`.
The real VOC loader preserves `difficult` flags. Its `map50` uses the VOC2007
11-recall-point AP rule; `map50_allpoint` is reported separately. Tiny pilot
subsets exclude classes without non-difficult ground truth, so their scores
are not comparable to the official full-test-set challenge leaderboard.

## Quick smoke

To keep peak RAM/VRAM bounded, use the serial matrix launcher for several
methods. It starts exactly one process at a time, validates that run before
starting the next, caps the pilot sample/batch budgets, and defaults to CPU.
It will not download datasets or overwrite an existing run directory. Do not
run a second matrix, heavy pytest suite, or dataset extraction concurrently.

```bash
uv run --no-sync python -m experiments.unified_multitask.run_matrix \
  --tasks image_classification text_classification \
  --methods fedavg fedprox splitfed_fixed splitfleet \
  --source real --run-prefix serial_pilot \
  --max-train-samples 64 --max-test-samples 32 \
  --batch-size 2 --max-batches-per-client 1 --device cpu
```

For Oxford Pet and VOC, download and verify the source archives separately
before running the serial matrix; keep download/extraction separate from
training and test execution.

```bash
uv run --no-sync python -m experiments.unified_multitask.run \
  --task object_detection --method splitfleet --source fixture \
  --rounds 2 --output results/unified_multitask/pilot_detection

uv run --no-sync python -m experiments.unified_multitask.validate \
  --run-dir results/unified_multitask/pilot_detection
```

The local CIFAR-10 package in this workspace supports a real-data pipeline
check without downloading:

```bash
uv run --no-sync python -m experiments.unified_multitask.run \
  --task image_classification --method splitfleet --source real \
  --max-train-samples 64 --max-test-samples 32 \
  --max-batches-per-client 1 \
  --output results/unified_multitask/pilot_cifar_splitfleet
```

Run matched methods with the same `--task`, `--source`, `--seed`, number of
rounds, batch and sample budgets. The validator checks the initial-model,
partition, and data-content hashes across matching sibling runs. It rejects
missing round records, missing client updates, impossible metric values,
unreported boundary traffic, and absent completion markers.

```bash
uv run --no-sync python -m experiments.unified_multitask.summarize \
  --results-root results/unified_multitask \
  --include-prefix pilot_ \
  --output results/unified_multitask_summary.json
```

The summary includes only validated runs. It computes paired seed effects
and Holm-adjusted sign-flip p-values only with at least two matched seeds;
single-seed pilot differences have no confidence interval or p-value. Even
multi-seed results remain exploratory until the frozen protocol's method,
dataset, round budget, and physical measurement requirements are met.
Paired timing also requires recorded host and execution-device identity.
Legacy runs without those fields are listed as excluded rather than being
silently paired; fixed-cut and FedProx hyperparameter variants stay separate.

## Current evidence

The 2026-09-23 implementation check completed all 16 method×task fixture
combinations at one seed and two rounds, 16 one-round real-data pilots across
CIFAR-10, AG News, VOC 2007 and Oxford-IIIT Pet, and 16 three-round real-data
pilots across the same tasks. All 48 selected run directories pass
`validate_run`. Every method comparison still has only one paired seed;
the three-round detector's mAP is zero, so none of these pilots establish
convergence or superiority. In a separate
numerical comparator, the four benchmark models' two-batch SplitFed updates
match full-local updates within `rtol=2e-4`, `atol=2e-6`; FedProx is confirmed
to change a multi-batch update. These checks do not show four-task statistical
superiority. Exact commands and limitations are recorded in
[`docs/experiment_validation_report.md`](../../docs/experiment_validation_report.md).
