# SplitFleet

SplitFleet is a PyTorch split learning framework built on top of [Flower](https://flower.ai/).
The autosplit runtime is backed by the repository-local `torchlens-2.31.0-py3-none-any.whl` via `uv.sources`, so SplitFleet focuses on Flower strategy integration, client/server transport, server-tail replicas, and aggregation policy.

## What This Project Does

Given a PyTorch model and example positional inputs, SplitFleet prepares a TorchLens split runtime, runs a client-local prefix, sends a typed `BoundaryPayload` to the server suffix, and completes split inference or split training inside the Flower round loop.

Currently supported:

- client prefix and coordinator/server suffix
- split inference
- split training with suffix gradients returned to the prefix
- shared server tail
- per-client server tail
- SplitFed-style client and server aggregation
- `BoundaryPayload` serialization through SplitFleet torch serde helpers

Currently not supported:

- arbitrary multi-stage worker placement
- old node-by-node remote stage execution
- non-contiguous client/server stage ownership
- keyword-input tracing in the SplitFleet adapter

## Installation

Python `3.10` is required.

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

Run optional heavy detection checks:

```bash
SPLITFLEET_RUN_HEAVY_REAL_MODELS=1 uv run --no-sync pytest tests/integration/test_torchlens_real_detection_optional.py -q
```

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

## Main Components

- [`splitfleet/autosplit`](splitfleet/autosplit): TorchLens adapter, two-stage planner, runtime facade, serde, and cache.
- [`splitfleet/server/strategy/autosplit_strategy.py`](splitfleet/server/strategy/autosplit_strategy.py): Flower strategy metadata and aggregation policy.
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
- `backward_prefix` delegates to TorchLens split training support. If neither the prepared runtime nor TorchLens exposes a real implementation, SplitFleet raises a clear `RuntimeError` instead of fabricating gradients.

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
```

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
