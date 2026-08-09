# SplitFleet

The complete RA-SplitFed experiment package is documented in
[`experiments/resource_adaptive_splitfed/README.md`](experiments/resource_adaptive_splitfed/README.md).

SplitFleet is a TorchLens-backed split learning framework built on top of [Flower](https://flower.ai/).
The autosplit runtime is backed by the repository-local `torchlens-2.31.0-py3-none-any.whl` via `uv.sources`, so SplitFleet focuses on Flower strategy integration, client/server transport, server-tail replicas, and aggregation policy.

TorchLens-backed split replay and training can also be enabled for TensorFlow, JAX,
Paddle, and tinygrad through the corresponding optional dependency groups
(`tensorflow`, `jax`, `paddle`, `tinygrad`, or `multibackend`). MLX and ONNX are
not registered because TorchLens 2.31 does not provide training-capable split
adapters for them. JAX callers provide `functional_update_fn` on the split client
when model parameters require an external functional update.
The TorchLens tinygrad adapter is pinned to tinygrad 0.13 and SplitFleet uses
Python 3.11 so the adapter can be exercised alongside the other backends.

## What This Project Does

Given a PyTorch model and example positional inputs, SplitFleet prepares a TorchLens split runtime, runs a client-local prefix, sends a typed `BoundaryPayload` to the server suffix, and completes split inference or split training inside the Flower round loop.

Currently supported:

- client prefix and coordinator/server suffix
- split inference
- split training with suffix gradients returned to the prefix
- shared server tail
- per-client server tail
- per-client and per-round placement selection through
  `AutoSplitStrategy.client_placement_fn`
- capability-aware per-client placement through
  [`CapabilityAwarePlacementPolicy`](splitfleet/server/placement/capability_placement.py)
- a fleet-wide dynamic batch window negotiated by the strategy, so devices can
  train heterogeneous batch sizes against the same suffix runtime
- SplitFed-style client and server aggregation
- name-manifest reassembly of per-client prefix/suffix updates before FedAvg
- `BoundaryPayload` serialization through SplitFleet torch serde helpers

Currently not supported:

- arbitrary multi-stage worker placement
- old node-by-node remote stage execution
- non-contiguous client/server stage ownership
- keyword-input tracing in the SplitFleet adapter

## Installation

Python `3.11` is required.

```bash
uv sync --extra dev
```

When the repository-local TorchLens wheel or lockfile changes, force uv to forget any cached TorchLens install before validating runtime behavior:

```bash
uv lock
uv cache clean
uv sync --extra dev --reinstall-package torchlens
```

Install the real-model integration dependencies when validating the model matrix:

```bash
uv sync --extra dev --extra integration
```

Install every training backend when validating cross-framework split learning:

```bash
uv sync --extra dev --extra integration --extra multibackend --reinstall-package torchlens
```

## Quick Start

Run the TorchLens split training demo:

```bash
uv run --no-sync python examples/torchlens_split_training_demo.py
```

Run the coordinator-local suffix demo:

```bash
uv run --no-sync python examples/autosplit_remote_worker_demo.py
```

Run the default test suite:

```bash
uv run --no-sync pytest -q
```

Run the real-model task matrix:

```bash
uv run --no-sync pytest tests/integration/test_torchlens_real_task_matrix.py -q
```

The matrix actively validates timm Swin, Hugging Face BERT/DistilBERT/RoBERTa,
CNN classifiers, torchvision detection heads, and semantic-segmentation models.

Run exhaustive split-node training on the cross-backend task models (YOLO-style
detection, FCN segmentation, RetinaNet-style detection, OCR, and foreground-mask
segmentation):

```bash
uv run --no-sync pytest tests/integration/test_all_split_nodes_training.py -q
```

The gated ResNet-18 exhaustive check limits native numerical libraries to one
CPU thread per backend subprocess by default. Increase the limit explicitly only
on a suitable host:

```bash
SPLITFLEET_RUN_RESNET18_ALL_NODES=1 SPLITFLEET_RESNET18_THREADS=2 \
  uv run --no-sync pytest tests/integration/test_resnet18_all_backends_all_nodes.py -q -s
```

Run optional heavy detection checks:

```bash
SPLITFLEET_RUN_HEAVY_REAL_MODELS=1 uv run --no-sync pytest tests/integration/test_torchlens_real_detection_optional.py -q
```

The heavy checks validate YOLOv8 and RF-DETR in addition to the default
torchvision Faster R-CNN and RetinaNet coverage.

## Example: AutoSplit Strategy

The main entrypoint is [`AutoSplitStrategy`](splitfleet/server/strategy/autosplit_strategy.py).

```python
import torch
from torch import nn

from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.server.strategy import AutoSplitStrategy

model = MyModel()
sample_inputs = torch.randn(2, 3, 224, 224)

strategy = AutoSplitStrategy(
    model=model,
    sample_inputs=sample_inputs,
    boundary="50%",
    client_stage_count=1,
    aggregation_policy="splitfed",
    dynamic_batch=(1, 256),
    loss_fn=nn.CrossEntropyLoss(),
    optimizer_fn=lambda m: torch.optim.SGD(m.parameters(), lr=0.01),
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)

client = AutoSplitSplitLearningClient(
    model=model,
    train_data=train_loader,
    evaluate_data=val_loader,
    sample_inputs=sample_inputs,
    optimizer_fn=lambda m: torch.optim.SGD(m.parameters(), lr=0.01),
)
```

## Cross-Device, Cross-Batch Rounds

### One negotiated batch window for the whole fleet

`dynamic_batch=(min, max)` is planned once by the strategy, travels in the round
config, and is what every device prepares its prefix runtime with. Devices
therefore keep their own local batch sizes — including a short trailing batch —
without each one re-deriving a window from its own sample inputs. The window is
part of the feature ABI, so a device that disagrees is rejected at round setup
instead of at the first boundary upload.

When `dynamic_batch` is omitted, the window defaults to `(2, 64)` for a sample
batch greater than one and `(1, 64)` otherwise.

A batch outside the window is refused by the prefix before it is uploaded and by
the suffix before it is executed, naming the observed batch and the window. Set
`partial_batch_policy="skip"` on the split client to drop such batches instead;
the dropped batch and example counts are reported in the round metrics rather
than silently absorbed.

### Capability-aware placement across heterogeneous devices

`CapabilityAwarePlacementPolicy` is usable directly as `client_placement_fn`. It
moves a slow device toward a lighter client prefix and a fast device toward a
heavier one, using a boundary ladder ordered from the lightest to the heaviest
client prefix:

```python
from splitfleet.server.placement import CapabilityAwarePlacementPolicy

placement = CapabilityAwarePlacementPolicy(
    boundary_ladder=["after:layer1", "after:layer2", "after:layer3"],
)
strategy = AutoSplitStrategy(
    model=model,
    sample_inputs=sample_inputs,
    client_placement_fn=placement,
    aggregation_policy="splitfed",
)
```

The strategy feeds every fit result and failure back into the policy, decides
once per round so fit and evaluate agree, and applies hysteresis so a fleet of
similar devices stays on a stable cut. Split clients report the signals the
policy consumes — `fit_duration_sec`, `prefix_compute_sec`, `tail_wait_sec`,
`upload_bytes`, `download_bytes`, `num_batches`, `min_batch_size`,
`max_batch_size`, and the skipped-batch counters.

A device whose per-client suffix replica is missing at aggregation time is
logged and dropped from that round's SplitFed reassembly; the rounds the other
devices completed are still aggregated. A round in which *every* client failed
is logged and skipped — the suffix replicas it created have no prefix half to
be reassembled with — and training continues from the previous global model
instead of raising. The same applies in reverse: client updates with no suffix
result at all leave the global model untouched rather than publishing a
prefix-only update.

### Weight-tied models

A tensor shared by several state-dict names (tied embeddings, a decoder reusing
its encoder weight) can be written by both stages. SplitFed rejects a dual-stage
tied update by default: independently transformed Adam, momentum, weight-decay,
or mismatched-learning-rate updates cannot be combined into one correct logical
step. Callers that guarantee identical stateless SGD without weight decay on
both stages may opt into `tied_weight_update_mode="additive_sgd"`; only in that
mode are the two deltas summed. Shared buffers are not treated as tied.

## Main Components

- [`splitfleet/autosplit`](splitfleet/autosplit): TorchLens adapter, two-stage planner, runtime facade, serde, and cache.
- [`splitfleet/server/strategy/autosplit_strategy.py`](splitfleet/server/strategy/autosplit_strategy.py): Flower strategy metadata and aggregation policy.
- [`splitfleet/server/placement`](splitfleet/server/placement): per-client placement policies for heterogeneous devices.
- [`splitfleet/autosplit/batch_window.py`](splitfleet/autosplit/batch_window.py): the dynamic batch window contract shared by the prefix and the suffix.
- [`splitfleet/server/stage_runtime`](splitfleet/server/stage_runtime): active TorchLens runtime handle management for coordinator-local suffix execution.
- [`splitfleet/server/server_model/autosplit_tail_server_model.py`](splitfleet/server/server_model/autosplit_tail_server_model.py): server suffix model bridge.
- [`splitfleet/client/autosplit_split_client.py`](splitfleet/client/autosplit_split_client.py): client prefix execution and boundary payload exchange.

## Runtime Invariants

- Runtime validation must prove `torchlens.__version__ == "2.31.0"` from the installed package, not only from wheel metadata.
- Final runtimes are prepared through TorchLens `prepare_split` or `prepare_split_replay`; low-level TorchLens graph APIs are reserved for read-only candidate probing.
- `BoundaryPayload` serialization is self-contained: tensors plus a stable `BoundarySpec` must be enough to recover after cross-process transport, and the optional native TorchLens object is not required after serde.
- Feature ABI identifiers are schema-only. They include labels, dtype, symbolic shape, layout, passthrough/preprocessing schema, trace mode, dynamic batch, model/runtime identifiers, and TorchLens version, but exclude sample tensor values, concrete sample batch values, target values, device, temporary runtime ids, and validation inputs.
- Flower autosplit config is JSON-stable. Strategy, client, and server code exchange deterministic runtime contract JSON, contract digests, feature ABI ids, boundary labels, trace batch mode, dynamic batch, backend, and TorchLens version.
- Boundary uploads are rejected before suffix execution when runtime backend, TorchLens version, feature ABI id, runtime contract digest, boundary label order, batch size, trace batch mode, or dynamic batch do not match the prepared server runtime.
- The dynamic batch window is negotiated once per placement by the strategy. Clients adopt the broadcast window instead of inferring one, and a client whose prepared prefix does not reproduce the announced feature ABI id fails the round before uploading a boundary.
- `backward_prefix` delegates to TorchLens split training support. If neither the prepared runtime nor TorchLens exposes a real implementation, SplitFleet raises a clear `RuntimeError` instead of fabricating gradients.
- Under `ReplicaScope.PER_CLIENT`, the prefix and suffix halves of a round are aggregated by two different calls, so `aggregate_fit` returns `None` for the client-side model and `Strategy.finalize_round` returns the reassembled logical model. No call path can publish a model whose suffix half is a round stale: a caller that skips `finalize_round` keeps the previous global parameters.
- Suffix timing measurements always contain `server_total_ms`. `server_forward_ms` and `server_backward_ms` are reported only by backends that execute the suffix phase by phase; no backend fabricates a phase split.
- Per-round aggregation state is dropped when the next round is configured, so a round that never reaches `aggregate_server_fit` cannot retain a copy of the client and server models.

## Validation

TorchLens 2.31 wheel and API checks:

```bash
uv lock
uv cache clean
uv sync --extra dev --reinstall-package torchlens
uv run python -c "import torchlens as tl; print(tl.__file__); print(tl.__version__); assert tl.__version__ == '2.31.0'"
uv run python -c "from torchlens.split import ReplayBoundary, prepare; print('torchlens 2.31 split api ok')"
```

Default checks:

```bash
uv run --no-sync pytest -q
uv run --no-sync pytest tests/unit/test_torchlens_218_api.py -q
uv run --no-sync pytest tests/unit/test_torchlens_boundary_serde.py -q
uv run --no-sync pytest tests/unit/test_torchlens_candidate_contract.py -q
uv run --no-sync pytest tests/unit/test_stage_runtime_contract.py -q
uv run --no-sync pytest tests/unit/test_batch_window.py -q
uv run --no-sync pytest tests/unit/test_capability_placement.py -q
uv run --no-sync pytest tests/test_cross_device_cross_batch.py -q
uv run --no-sync pytest tests/test_splitfed_aggregation.py -q
uv run --no-sync pytest tests/test_server_round_finalization.py -q
```

`tests/test_cross_device_cross_batch.py` covers the cross-device and cross-batch
round behaviour: window broadcast and adoption, heterogeneous batch sizes in one
round, prefix and suffix rejection of out-of-window batches, the skip policy,
feature-ABI refusal, capability-aware placement through the strategy, and
SplitFed aggregation with a missing suffix replica.

Integration checks:

```bash
uv run --no-sync pytest tests/integration/test_torchlens_runtime_replay.py -q
uv run --no-sync pytest tests/integration/test_torchlens_real_task_matrix.py -q
SPLITFLEET_RUN_HEAVY_REAL_MODELS=1 uv run --no-sync pytest tests/integration/test_torchlens_real_detection_optional.py -q
```

ResNet18 training correctness check:

```bash
uv sync --extra dev --extra integration --reinstall-package torchlens
uv run --no-sync pytest tests/integration/test_torchlens_real_task_matrix.py -k "torchvision_resnet18" -q
```

The ResNet18 check verifies that a split training step matches the full-model step, including loss, BatchNorm state, and parameter updates after the optimizer step.

Cleanup checks:

```bash
rg "torchlens-2\.1[7]\.0|2\.1[7]\.0" splitfleet tests examples README.md pyproject.toml uv.lock
rg -i "[a]riadne" splitfleet tests examples README.md pyproject.toml uv.lock
```

## Paper

If you use the framework in research, the original SplitBud paper is still the best citation context for the project lineage:

```bibtex
@article{Radovic-SplitBud,
  author = {Radovic, Boris and Canini, Marco and Horvath, Samuel and Pejovic, Veljko and Vepakomma, Praneeth},
  maintitle = {EuroSys},
  booktitle = {EuroMLSys},
  title = {Towards a Unified Framework for Split Learning},
  year = {2025}
}
```
