# SplitFleet

SplitFleet also supports real virtual split learning through Flower/Ray. See
[Virtual SFL Runtime](experiments/virtual_sfl/README.md) for the CPU smoke entry,
100-client real-data training, state isolation, bounded suffix replicas, network
constraints and measured Greedy placement. The current paper retains the paired
100-client, 50-round PyTorch result; see the [paper evidence guide](paper/README.md).
Physical TensorFlow/Paddle training uses
[six CPU/GPU clients on three Orins](experiments/resnet50_backends/README.md).

SplitFleet combines a unified split federated learning (SFL) abstraction, SFL-grade fine-grained partitioning, and measured Greedy placement. Built on [Flower](https://flower.ai/), it separates backend, task, partition, aggregation, and placement contracts. Three backend adapters implement the same SFL execution contract.

The unified task interface covers image classification, text classification, object detection, semantic segmentation, and instance segmentation.
Maintained experiment protocols are in `experiments/`; the local manuscript and evidence guide is `paper/README.md`.
Dataset-level integrations can be registered through `TaskSpec`/`TaskRegistry`, which bind a task adapter, model and dataset factories, metrics, and candidate cuts without coupling the split runtime to a dataset package.
Greedy is the default adaptive policy, with one server suffix execution slot (`C=1`). Each client selects its lowest predicted complete-round cost from the admitted cuts. Restored training calibration supplies initial compute and communication costs; actual fit feedback updates private client cost tables. Per-batch costs are multiplied by the local batch count, and model-state exchange is counted once per round. Calibration time is included in complete job time.

Physical PyTorch calibration synchronizes current global weights and restores model state and random streams after temporary optimizer steps. Evaluation does not initialize training costs. Compatible saved Greedy cost tables can resume without repeating calibration. See the [Greedy guide](experiments/GREEDY.md) and [physical experiment guide](experiments/README.md).

The autosplit runtime is backed by the repository-local `torchlens-2.34.1-py3-none-any.whl` via `uv.sources`. SplitFleet uses its native split APIs and maintains no local TorchLens patches or wheel builder. Candidate catalogs use `analyze()` for plans and capability reports; selected training cuts use `at()` to build executable segments. SplitFleet handles Flower strategy integration, client/server transport, server-tail replicas, and aggregation policy.

The local wheel's native split extensions are distinct from the execution-capture capabilities documented by [upstream TorchLens](https://github.com/johnmarktaylor91/torchlens). Graph capture alone does not establish a valid training partition. Catalog admission checks the captured graph, training support, state ownership and boundary ABI; numerical correctness requires execution checks. Canonical split identities and graph/schema hashes identify boundaries across compatible devices, while runtime operation positions are cost features.

Boundary features and gradients use typed tensor envelopes, graph contracts, SHA-256 checks and resource limits.

TorchLens-backed split replay and training can also be enabled for TensorFlow and
Paddle through the corresponding optional dependency groups (`tensorflow` and
`paddle`). MLX and ONNX are not exposed as SplitFleet training backends.

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
- one per-round dynamic placement policy interface through
  `AutoSplitStrategy.placement_policy`
- measured placement through
  [`GreedyPlacementPolicy`](splitfleet/server/placement/greedy/policy.py)
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
semantic and instance segmentation; PyTorch):

```bash
uv run --no-sync python -m splitfleet.validation --backends torch --all-nodes \
  --output results/unified_task_validation.json
```

The JSON compares outputs, task losses, all parameter gradients and one SGD step
at every enumerated before/after boundary, including activation and gradient wire
round trips. Unsupported and failed cases have separate outcomes and nonzero exit
codes. These synthetic checks do not measure dataset accuracy or convergence.
The full task-equivalence matrix covers PyTorch. TensorFlow and Paddle have
separate native replay/training, task-loss and split-node checks.

Run the TorchLens split training demo:

```bash
uv run --no-sync python examples/torchlens_split_training_demo.py
```

Run the default test suite:

```bash
uv run --no-sync pytest
```

Run the complete suite, including the gated heavy detection checks, with all
backend and task dependencies installed:

```bash
SPLITFLEET_RUN_HEAVY_REAL_MODELS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  uv run --no-sync pytest tests --tb=short -ra
```

The heavy switches execute the gated tests; they keep the default suite smaller.

Run the real-model task matrix:

```bash
uv run --no-sync pytest tests/integration/test_torchlens_real_task_matrix.py -q
```

The matrix validates Hugging Face BERT/DistilBERT/RoBERTa, CNN classifiers,
torchvision detection heads, and semantic-segmentation models. Swin training
and DeepLab training use explicit fixed-shape captures where TorchLens refuses
dynamic batch replay; the tests preserve those refusals as part of the contract.

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
whose native dynamic batch probes cannot validate replay. Fixed-shape examples and native limitations are covered by the batch-window tests and the full-model protocol in `experiments/EXECUTION_COVERAGE.md`.
Changing a fixed input shape requires a separate capture; widening
`dynamic_batch` does not make that shape dynamic.

A batch outside the window is refused by the prefix before it is uploaded and by
the suffix before it is executed, naming the observed batch and the window. Set
`partial_batch_policy="skip"` on the split client to drop such batches instead;
the dropped batch and example counts are reported in the round metrics rather
than silently absorbed. This skip policy does not handle fixed-shape mismatches
or native batch-probe failures inside the admitted window.

### Greedy placement across heterogeneous devices

The default physical and virtual adaptive entrypoints use Greedy and `C=1`. TorchLens supplies admitted operation-level cuts; Greedy chooses a cut independently for each selected client once per federated round. `boundary="auto"` enables discovery in the lower-level strategy API; pass a placement policy to enable adaptation there.

```python
from splitfleet.tasks import ImageClassificationTask, ModelInputs, TaskBatch
from splitfleet.server.strategy import AutoSplitStrategy
from splitfleet.server.placement import GreedyConfig, GreedyPlacementPolicy, TorchLensCandidateProvider

sample = TaskBatch(ModelInputs.from_value(sample_inputs), sample_targets)
placement = GreedyPlacementPolicy(
    candidate_provider=TorchLensCandidateProvider(model=model, sample_inputs=sample.inputs),
    config=GreedyConfig(server_concurrency=1),
)
strategy = AutoSplitStrategy(
    model=model,
    sample_inputs=sample,
    task=ImageClassificationTask(),
    placement_policy=placement,
    aggregation_policy="splitfed",
    optimizer_fn=lambda parameters: torch.optim.SGD(parameters, lr=0.005),
)
```

Provide each worker's representative task batch and matching optimizer as well. Calibration measures four admitted structural anchors, restores state, and initializes private affine communication costs. Successful training feedback updates measured computation and link costs; failures update feasibility. The policy uses no confidence-bound exploration or joint cut solver. Recorded server load is diagnostic and is not a Greedy scoring feature.

Set `--server-concurrency N` to permit up to `N` suffix executions on the same GPU. This changes execution capacity, not the number of GPUs. Each client retains a private suffix model and optimizer. At `C=1` their execution is serialized while client prefixes may overlap. Higher capacities can reduce queue wait while increasing service spans and need not reduce complete job time.

SplitFed reconstructs each client's complete logical update from its owned prefix and suffix states and then performs sample-weighted aggregation. FedAvg equivalence requires matched local work, optimizer state, buffers, randomness and numerical execution. Missing prefix/suffix pairs are dropped together; a round with no complete update preserves the previous global model.

The current paper uses the original measured Greedy endpoints as SplitFleet (`C=1`), with original hashes and labels retained. Historical controller runs remain immutable evidence; the active Greedy refactor has not been performance-rerun. See [the paper and evidence guide](paper/README.md).

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
- [`splitfleet/server/placement/greedy`](splitfleet/server/placement/greedy): the default measured policy, including private cost tables, feasibility, calibration, feedback and state.
- [`splitfleet/autosplit/batch_window.py`](splitfleet/autosplit/batch_window.py): the dynamic batch window contract shared by the prefix and the suffix.
- [`splitfleet/server/stage_runtime`](splitfleet/server/stage_runtime): active TorchLens runtime handle management for coordinator-local suffix execution.
- [`splitfleet/server/server_model/autosplit_tail_server_model.py`](splitfleet/server/server_model/autosplit_tail_server_model.py): server suffix model bridge.
- [`splitfleet/client/autosplit_split_client.py`](splitfleet/client/autosplit_split_client.py): client prefix execution and boundary payload exchange.

## Runtime Invariants

- Runtime validation requires the installed `torchlens==2.34.1`. `uv.sources` and the wheel hash in `uv.lock` select the repository wheel. Runtime contracts record adapter identity; incompatible contracts must be regenerated.
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

The [execution coverage guide](experiments/EXECUTION_COVERAGE.md) describes complete cut catalogs and paired training checks. Audit the retained physical study without rewriting its results:

```bash
.venv/bin/python paper/refresh_evidence.py --output-dir /tmp/splitfleet_evidence_audit --use-checkpoint-receipts
```

Completed training weights and unused input copies have been pruned. This audit
recomputes statistics from retained results and uses the recorded pre-pruning
checkpoint checks; it does not reload deleted weights. The four RQ1 input bundles
remain. See [storage cleanup](paper/evidence/storage_cleanup_receipt.json).

The manuscript's retained cohorts and required support files are listed in
[results/README.md](results/README.md). Unused pilots, intermediate analyses,
unexecuted plans and duplicate experiment snapshots have been pruned. The
[experiment cleanup receipt](paper/evidence/unused_experiment_cleanup_receipt.json)
records removed paths and checks that cited raw results, failures, inputs and
figures retain their original hashes.

New experiment runs default to compact storage. Servers keep numerical results,
failures and a final in-memory finite/hash check; full task weights require
`--save-model`. Prepared server/client bundles reference shared weights and data,
and derived role descriptors reuse those tensors. The physical
matrix deletes its owned inputs after it stops its workers, including on failure;
use `--keep-input-bundles` to retain them for replay or debugging. Manifests,
input hashes, frozen source and logs remain. Standalone input preparation keeps
its shared assets for subsequent runs; it cannot delete inputs used by other jobs.

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
uv run --no-sync pytest tests/test_greedy_placement.py tests/test_greedy_calibration.py -q
uv run --no-sync pytest tests/test_cross_device_cross_batch.py -q
uv run --no-sync pytest tests/test_splitfed_aggregation.py -q
uv run --no-sync pytest tests/test_server_round_finalization.py -q
```

`tests/test_cross_device_cross_batch.py` covers the cross-device and cross-batch
round behaviour: window broadcast and adoption, heterogeneous batch sizes in one
round, prefix and suffix rejection of out-of-window batches, the skip policy,
feature-ABI refusal, Greedy placement through the strategy, and
SplitFed aggregation with a missing suffix replica.

Integration checks:

```bash
uv run --no-sync pytest tests/integration/test_torchlens_runtime_replay.py -q
uv run --no-sync pytest tests/integration/test_torchlens_real_task_matrix.py -q
SPLITFLEET_RUN_HEAVY_REAL_MODELS=1 uv run --no-sync pytest tests/integration/test_torchlens_real_detection_optional.py -q
```

To include the full native controls, install the optional backends and detection
dependencies, and select a separate GPU Paddle environment (the CPU and GPU
wheels share the same import name):

```bash
SPLITFLEET_PADDLE_PYTHON="$PWD/.venv-paddle-gpu/bin/python" \
SPLITFLEET_FULL_PADDLE_CONTROL=1 \
SPLITFLEET_FULL_OWNERSHIP_TESTS=1 \
SPLITFLEET_NATIVE_CALIBRATION_BACKEND=pytorch \
SPLITFLEET_NATIVE_CALIBRATION_DEVICE=cpu \
SPLITFLEET_RUN_HEAVY_REAL_MODELS=1 \
uv run --no-sync pytest -ra
```

The GPU controls require two real GPUs and the ResNet50 gate's CIFAR-10 data.
Native framework controls use fresh processes; TensorFlow test imports default
to growing GPU allocations so they can coexist with child controls.
Missing dependencies or devices remain explicit skips in the report.

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
@inproceedings{Radovic-SplitBud,
  author = {Radovič, Boris and Canini, Marco and Horváth, Samuel and Pejović, Veljko and Vepakomma, Praneeth},
  booktitle = {Proceedings of the 5th Workshop on Machine Learning and Systems (EuroMLSys '25)},
  title = {Towards a Unified Framework for Split Learning},
  year = {2025},
  pages = {183--191},
  doi = {10.1145/3721146.3721936},
  url = {https://euromlsys.eu/pdf/euromlsys25-7.pdf}
}
```
