# Experiment entry points

The physical runner compares FedAvg, FedProx, SplitFed at 25%, 50% and 75%,
and SplitFleet on three Jetson hosts, each running one CPU and one GPU worker.
Each worker trains its entire disjoint private partition for one local epoch
per round. Dataset, partition and initial model hashes are shared across schemes.
Runtime preparation errors stop the run, and incomplete rounds fail validation.

Install the experiment and real-model dependencies:

```bash
uv sync --extra dev --extra experiment --extra integration
```

## Physical training

Review `physical_multitask_205.deployment` for host addresses, Python environments,
coordinator device and barrier ports. Workers need passwordless SSH, synchronized
clocks, CUDA and the original `torchlens-2.34.1-py3-none-any.whl` installed.

Run a bounded real-data comparison for all four task families:

```bash
uv run --no-sync python -m experiments.orchestrate_physical_multitask \
  --deployment experiments/physical_multitask_205.deployment \
  --run-id four_tasks --train-samples 180 --test-samples 100 \
  --batch-size 4 --rounds 3 --fixed-boundaries 25% 50% 75%

uv run --no-sync python -m experiments.summarize_physical_multitask \
  --run-root results/physical_multitask/four_tasks
```

Select tasks or methods with `--tasks` and `--methods`. The runner freezes input
bundles, stages current code, validates all six workers, saves final models and
removes its private staging directories. The summary distinguishes fixed cuts
and rejects mixed runtime snapshots or training protocols. JSON and CSV outputs
include total elapsed time, training spans per round and learning curves.

## RF-DETR Nano and device profiles

RF-DETR uses the pretrained Nano checkpoint and batch size one. Prepare frozen
VOC07 bundles, profile the coordinator and six workers, then use those profiles
for adaptive placement. Replace the checkpoint path with your local weights.

```bash
uv run --no-sync python -m experiments.physical_multitask prepare \
  --task object_detection --image-model rfdetr_nano \
  --pretrain-weights /path/to/rf-detr-nano.pth \
  --train-samples 180 --test-samples 100 --batch-size 1 --dirichlet-alpha 10 \
  --output results/rfdetr_bundles/object_detection.pt

uv run --no-sync python -m experiments.profile_split_execution \
  --bundle results/rfdetr_bundles/object_detection.pt --device cuda:0 \
  --output results/rfdetr_profiles/server.json

uv run --no-sync python -m experiments.collect_device_split_profiles \
  --deployment experiments/physical_multitask_205.deployment \
  --bundles results/rfdetr_bundles --output results/rfdetr_profiles

uv run --no-sync python -m experiments.orchestrate_physical_multitask \
  --deployment experiments/physical_multitask_205.deployment --run-id rfdetr_six \
  --tasks object_detection --image-model rfdetr_nano \
  --pretrain-weights /path/to/rf-detr-nano.pth \
  --device-profiles results/rfdetr_profiles/device_profiles.json \
  --train-samples 180 --test-samples 100 --batch-size 1 --dirichlet-alpha 10 \
  --optimizer adam --learning-rate 0.00001 --rounds 3 \
  --fixed-boundaries 25% 50% 75%

uv run --no-sync python -m experiments.summarize_physical_multitask \
  --run-root results/physical_multitask/rfdetr_six
```

Profiles measure operation costs using synthetic inputs on real devices. The
training runner uses frozen real samples. The completed native-wheel comparison
is recorded in `results/reports/real_device_training_time_20261001.md`.

## Local checks

Use [unified_multitask](unified_multitask/README.md) for sequential task integration,
fixture checks and dataset learning curves. Its timing measures a single process.
Shared partition, aggregation, identity and statistical helpers live in `common`.

Measure local split training and compare outputs, losses, gradients and updated
parameters against the full model:

```bash
uv run --no-sync python -m experiments.benchmark_torchlens_training \
  --label native-wheel --devices cpu cuda:0 \
  --output results/torchlens_training/native-wheel.json
```

Historical reports, logs and frozen source snapshots remain in ignored result
directories. Historical runs use their recorded source snapshot for reproduction.
