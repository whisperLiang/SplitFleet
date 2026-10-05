# SplitFleet

SplitFleet is a unified split federated learning framework built on top of [Flower](https://flower.ai/) and TorchLens. It separates framework backends, operation-level model partitioning, task adapters, boundary transport, and federated aggregation.

The unified task interface covers image classification, text classification, object detection, semantic segmentation, and instance segmentation.
Architecture notes and study protocols are kept locally in `plans/` and excluded from Git.
Dataset-level integrations can be registered through `TaskSpec`/`TaskRegistry`, which bind a task adapter, model and dataset factories, metrics, and candidate cuts without coupling the split runtime to a dataset package.
CoSplit-UCB is the default adaptive policy. Physical PyTorch runs initialize its discounted learners from three current-device split calibrations, bound exploration by predicted mean and confidence-upper round time (5% by default), and reuse prediction and scheduling caches. Calibration performs no optimizer steps and restores model state, gradients and random streams. Costs continue to come from online training feedback.
Fresh calibration synchronizes Flower's current global weights and uses the server's captured batch contract. Evaluation binds worker metadata without calibrating. A compatible restored learner keeps its statistics and continues from online feedback without repeating initialization.

The active code uses the original `cosplit_context_v1` context schema. Historical timing comparisons and frozen source snapshots remain under `paper/evidence/` and `results/`; use those snapshots to reproduce an old run. See the [experiment guide](experiments/README.md) for the current training entry points.

The autosplit runtime is backed by the repository-local `torchlens-2.34.1-py3-none-any.whl` via `uv.sources`. SplitFleet uses its native split APIs and maintains no local TorchLens patches or wheel builder. Candidate catalogs use `analyze()` for plans and capability reports; selected training cuts use `at()` to build executable segments. SplitFleet handles Flower strategy integration, client/server transport, server-tail replicas, and aggregation policy.

Boundary features and gradients use typed tensor envelopes, graph contracts, SHA-256 checks and resource limits.

TorchLens-backed split replay and training can also be enabled for TensorFlow, JAX,
Paddle, and tinygrad through the corresponding optional dependency groups
(`tensorflow`, `jax`, `paddle`, `tinygrad`, or `multibackend`). This project registers these five training backends; MLX and ONNX are
not exposed as SplitFleet training backends. JAX callers provide `functional_update_fn` on the split client
when model parameters require an external functional update.
The TorchLens tinygrad adapter is pinned to tinygrad 0.13 and SplitFleet uses
Python 3.11 so the adapter can be exercised alongside the other backends.

## What This Project Does

Given a model in a registered backend and example positional/keyword inputs, SplitFleet prepares a TorchLens split runtime, runs a client-local prefix, sends a typed `BoundaryPayload` to the server suffix, and completes split inference or split training inside the Flower round loop.

Split points may be selected before/after captured operations or by percentage.
Support is determined for each captured graph, backend, input schema, and training
mode; an untraceable model or unsupported operation is reported explicitly.
This is not a promise that every possible model or Python execution path can be split.

Currently supported:

- positional and keyword model inputs, including text input mappings
- task-aware batches and losses, including list-of-images detection batches
- inspectable backend availability through `BACKEND_ADAPTERS.availability()`
- client prefix and coordinator/server suffix
- split inference
- split training with suffix gradients returned to the prefix
- shared server tail
- per-client server tail
- one joint per-round dynamic placement policy interface through
  `AutoSplitStrategy.placement_policy`
- cooperative online placement through
  [`CoSplitUCBPlacementPolicy`](splitfleet/server/placement/cosplit_ucb/policy.py)
- a fleet-wide dynamic batch window negotiated by the strategy, permitting
  heterogeneous batch sizes when native TorchLens batch validation succeeds
- SplitFed-style client and server aggregation
- ownership-based aggregation of per-client prefix/suffix updates in physical PyTorch SplitFleet runs
- `BoundaryPayload` serialization through SplitFleet's typed wire envelopes

Currently not supported:

- arbitrary multi-stage worker placement
- non-contiguous client/server stage ownership

## Installation

Python `3.11` is required.

```bash
uv sync --extra dev
```

When the repository-local TorchLens wheel or lockfile changes, force uv to forget any cached TorchLens install before validating runtime behavior:

```bash
uv lock
uv cache clean torchlens
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

Run the download-free task correctness matrix (image/text classification, detection,
semantic and instance segmentation; PyTorch and JAX):

```bash
uv run --no-sync python -m splitfleet.validation --backends torch jax --all-nodes \
  --output results/unified_task_validation.json
```

The JSON compares outputs, task losses, all parameter gradients and one SGD step
at every enumerated before/after boundary, including activation and gradient wire
round trips. Unsupported and failed cases have separate outcomes and nonzero exit
codes. These synthetic checks do not measure dataset accuracy or convergence.
Install the JAX extra with `uv sync --extra dev --extra jax` before this command,
or use `--backends torch` with the base installation. The full task equivalence
matrix currently covers PyTorch and JAX. TensorFlow, Paddle and tinygrad have
separate native replay/training, task-loss and split-node checks; those checks
do not establish a complete five-backend by five-task equivalence matrix.

Run the TorchLens split training demo:

```bash
uv run --no-sync python examples/torchlens_split_training_demo.py
```

Run the default test suite:

```bash
uv run --no-sync pytest
```

Run the real-model task matrix:

```bash
uv run --no-sync pytest tests/integration/test_torchlens_real_task_matrix.py -q
```

The matrix validates Hugging Face BERT/DistilBERT/RoBERTa, CNN classifiers,
torchvision detection heads, and semantic-segmentation models. Swin training
and DeepLab training use explicit fixed-shape captures where TorchLens refuses
dynamic batch replay; the tests preserve those refusals as part of the contract.

Run exhaustive split-node training on the cross-backend task models (YOLO-style
detection, FCN segmentation, RetinaNet-style detection, OCR, and foreground-mask
segmentation):

```bash
uv run --no-sync pytest tests/integration/test_all_split_nodes_training.py -q
```

This check accounts for terminal and non-differentiable nodes separately. When a
backend exposes no differentiable single-node boundary, it also tests the
multi-tensor frontier chosen by TorchLens; a replay-only node is not reported as
a successful training boundary.

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
can keep local batch sizes, including a short trailing batch, when both the
negotiated window and native TorchLens batch validation allow them. The window is
part of the feature ABI, so a device that disagrees is rejected at round setup
instead of at the first boundary upload.

For a runtime with inferred or declared batch axes, omitting `dynamic_batch`
defaults the window to `(2, 64)` for a sample batch greater than one and `(1, 64)`
otherwise. Fixed-shape captures do not infer a dynamic window.

The window is an admission constraint, not proof of shape generalization.
`batch_axes` declares the model-call inputs and dimensions that carry the batch;
for example, `{"/args/0": 1}` for a `[T, B, C]` input. A missing or failed native
batch probe still rejects changed batches even when they are inside the window.

Pass `batch_axes={}` to capture the exact example shape without dynamic batch
axes. Set it on both `AutoSplitStrategy` and `AutoSplitSplitLearningClient`, with
matching sample shapes, and keep the data-loader batch shape fixed. This is the
explicit training path used by the BatchNorm1d, DeepLab and Swin configurations
whose native dynamic batch probes cannot validate replay. Fixed-shape examples
and native limitations are described in the local architecture guide,
`plans/unified_framework.md`, under “细粒度分割”.
Changing a fixed input shape requires a separate capture; widening
`dynamic_batch` does not make that shape dynamic.

A batch outside the window is refused by the prefix before it is uploaded and by
the suffix before it is executed, naming the observed batch and the window. Set
`partial_batch_policy="skip"` on the split client to drop such batches instead;
the dropped batch and example counts are reported in the round metrics rather
than silently absorbed. This skip policy does not handle fixed-shape mismatches
or native batch-probe failures inside the admitted window.

### CoSplit-UCB placement across heterogeneous devices

SplitFleet has one built-in dynamic partition policy: CoSplit-UCB. TorchLens
discovers valid operation-level cuts, CoSplit-UCB learns component costs and
jointly assigns selected clients under bounded suffix-server concurrency, and
Flower performs client selection and aggregation. `boundary="auto"` only means
automatic boundary discovery; dynamic selection belongs to the policy.

The physical multi-task runner searches every valid `before`/`after` operation
boundary, including RF-DETR Nano, with no candidate-count limit. Training cuts
must retain trainable parameters in both stages and pass native capability and
state ownership checks. Selected cuts are materialized for training.
The catalog shares one capture; execution handles are retained only for selected
cuts. The retained four-task jobs record catalog sizes, selected boundaries and
rejection diagnostics in their result files.

```python
from splitfleet.tasks import ImageClassificationTask, ModelInputs, TaskBatch
from splitfleet.server.strategy import AutoSplitStrategy
from splitfleet.server.placement import CoSplitUCBPlacementPolicy, TorchLensCandidateProvider

# Use representative inputs and targets from the current deployment.
sample = TaskBatch(ModelInputs.from_value(sample_inputs), sample_targets)
task = ImageClassificationTask()
placement = CoSplitUCBPlacementPolicy(
    candidate_provider=TorchLensCandidateProvider(model=model, sample_inputs=sample.inputs),
)
strategy = AutoSplitStrategy(
    model=model,
    sample_inputs=sample,
    task=task,
    placement_policy=placement,
    aggregation_policy="splitfed",
)
# On each worker, pass its own representative TaskBatch and task to
# AutoSplitSplitLearningClient. Calibration runs before the first placement.
```

CoSplit-UCB treats fine-grained split selection as a cooperative,
non-stationary contextual-bandit problem. It learns client, network, server and
repartitioning costs from real split-training feedback, shares edge knowledge
by execution profile, keeps link learners per client, and uses a shared server
learner. A lane simulator derives queueing and minimizes synchronous round
makespan. Exploration must stay within both the predicted mean and confidence-upper
round-time budgets. The default parameters are starting points, not claimed optima.

The solver schedules each client's sequential training batches on shared server
lanes; the switch cost is paid once per round. It uses the latest observed batch
count for each client, or a count supplied by runtime telemetry. Exploration is
held back when a selected client's batch count is still unknown. Existing
learner observations decay by elapsed training rounds before the next placement
prediction. Client-reported backend, device, accelerator, and precision identify
which clients share edge-side learning.
Production diagnostics record round wall time from fit dispatch to result
collection separately from the longest client-reported fit duration.

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
- [`splitfleet/server/placement/cosplit_ucb`](splitfleet/server/placement/cosplit_ucb): the sole production dynamic policy, including cooperative learners, feasibility, joint solving, safe exploration, and state.
- [`splitfleet/autosplit/batch_window.py`](splitfleet/autosplit/batch_window.py): the dynamic batch window contract shared by the prefix and the suffix.
- [`splitfleet/server/stage_runtime`](splitfleet/server/stage_runtime): active TorchLens runtime handle management for coordinator-local suffix execution.
- [`splitfleet/server/server_model/autosplit_tail_server_model.py`](splitfleet/server/server_model/autosplit_tail_server_model.py): server suffix model bridge.
- [`splitfleet/client/autosplit_split_client.py`](splitfleet/client/autosplit_split_client.py): client prefix execution and boundary payload exchange.

## Runtime Invariants

- Runtime validation requires the installed `torchlens==2.34.1`. `uv.sources` and the wheel hash in `uv.lock` select the repository wheel. Runtime contracts identify adapter v8, so contracts from earlier adapters must be regenerated.
- Runtimes use TorchLens `prepare`; split-point discovery, analysis, and repartition use the public `split_points`, `analyze`, and `at` methods. Unsupported points remain visible in candidate diagnostics.
- `BoundaryPayload` serialization is self-contained: tensors, a stable `BoundarySpec`, and portable replay metadata survive cross-process transport. The optional native TorchLens object and local autograd state are not required on the receiving side.
- Feature ABI identifiers are schema-only. They include labels, dtype, symbolic shape, layout, passthrough/preprocessing schema, trace mode, dynamic batch, model/runtime identifiers, and TorchLens version, but exclude sample tensor values, concrete sample batch values, target values, device, temporary runtime ids, and validation inputs.
- Flower autosplit config is JSON-stable. Strategy, client, and server code exchange deterministic runtime contract JSON, contract digests, feature ABI ids, boundary labels, trace batch mode, dynamic batch, backend, and TorchLens version.
- Boundary uploads are rejected before suffix execution when runtime backend, TorchLens version, feature ABI id, runtime contract digest, boundary label order, batch size, trace batch mode, or dynamic batch do not match the prepared server runtime.
- The dynamic batch window is negotiated once per placement by the strategy. Clients adopt the broadcast window instead of inferring one, and a client whose prepared prefix does not reproduce the announced feature ABI id fails the round before uploading a boundary.
- `backward_prefix` calls the pinned TorchLens runtime's training API directly.
- Runtime preparation, optimizer construction and semantic split lookup failures propagate to the caller. Training does not continue with an unprepared runtime, missing optimizer implementation or guessed semantic boundary.
- Under `ReplicaScope.PER_CLIENT`, the prefix and suffix halves of a round are aggregated by two different calls, so `aggregate_fit` returns `None` for the client-side model and `Strategy.finalize_round` returns the reassembled logical model. No call path can publish a model whose suffix half is a round stale: a caller that skips `finalize_round` keeps the previous global parameters.
- Suffix timing measurements always contain `server_total_ms`. PyTorch measures its complete suffix-service wall span, including boundary roots, task loss/matching, backward and optimizer work, and reports `server_loss_ms` through client telemetry. Phase timings are reported only by backends that measure them.
- Per-round aggregation state is dropped when the next round is configured, so a round that never reaches `aggregate_server_fit` cannot retain a copy of the client and server models.

## Validation

The local architecture guide, `plans/unified_framework.md`, describes validation scopes and reproduction commands. Verify the retained physical study without rewriting its results:

```bash
.venv/bin/python paper/evidence/verify_four_task_study.py
```

TorchLens 2.34.1 wheel and API checks:

```bash
uv lock
uv cache clean torchlens
uv sync --extra dev --reinstall-package torchlens
uv run python -c "import torchlens as tl; print(tl.__file__); print(tl.__version__); assert tl.__version__ == '2.34.1'"
uv run python -c "from torchlens.split import ReplayBoundary, prepare; print('torchlens 2.34.1 split api ok')"
```

Default checks:

```bash
uv run --no-sync pytest -q
uv run --no-sync pytest tests/unit/test_torchlens_split_engine.py -q
uv run --no-sync pytest tests/unit/test_torchlens_boundary_serde.py -q
uv run --no-sync pytest tests/unit/test_torchlens_candidate_contract.py -q
uv run --no-sync pytest tests/unit/test_stage_runtime_contract.py -q
uv run --no-sync pytest tests/unit/test_batch_window.py -q
uv run --no-sync pytest tests/unit/test_cosplit_policy.py -q
uv run --no-sync pytest tests/unit/test_cosplit_solver.py -q
uv run --no-sync pytest tests/test_cross_device_cross_batch.py -q
uv run --no-sync pytest tests/test_splitfed_aggregation.py -q
uv run --no-sync pytest tests/test_server_round_finalization.py -q
```

`tests/test_cross_device_cross_batch.py` covers the cross-device and cross-batch
round behaviour: window broadcast and adoption, heterogeneous batch sizes in one
round, prefix and suffix rejection of out-of-window batches, the skip policy,
feature-ABI refusal, CoSplit-UCB placement through the strategy, and
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
rg 'torchlens-2\.31\.0|torchlens==2\.31\.0' splitfleet tests examples README.md pyproject.toml uv.lock
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
