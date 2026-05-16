# SplitFleet

SplitFleet is a PyTorch split learning framework built on top of [Flower](https://flower.ai/).
The autosplit runtime is now backed by the published `ariadne-split` package, so SplitFleet focuses on Flower strategy integration, client/server transport, server-tail replicas, and aggregation policy.

## What This Project Does

Given a PyTorch model and example positional inputs, SplitFleet prepares an Ariadne split runtime, runs a client-local prefix, sends a typed `BoundaryPayload` to the server suffix, and completes split inference or split training inside the Flower round loop.

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

Install the real-model integration dependencies when validating the model matrix:

```bash
uv sync --extra dev --extra integration
```

## Quick Start

Run the Ariadne split training demo:

```bash
uv run --no-sync python examples/ariadne_split_training_demo.py
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
uv run --no-sync pytest tests/integration/test_ariadne_real_task_matrix.py -q
```

Run optional heavy detection checks:

```bash
SPLITFLEET_RUN_HEAVY_REAL_MODELS=1 uv run --no-sync pytest tests/integration/test_ariadne_real_detection_optional.py -q
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

- [`splitfleet/autosplit`](splitfleet/autosplit): Ariadne adapter, two-stage planner, runtime facade, serde, and cache.
- [`splitfleet/server/strategy/autosplit_strategy.py`](splitfleet/server/strategy/autosplit_strategy.py): Flower strategy metadata and aggregation policy.
- [`splitfleet/server/stage_runtime`](splitfleet/server/stage_runtime): active Ariadne runtime handle management for coordinator-local suffix execution.
- [`splitfleet/server/server_model/ariadne_tail_server_model.py`](splitfleet/server/server_model/ariadne_tail_server_model.py): server suffix model bridge.
- [`splitfleet/client/autosplit_split_client.py`](splitfleet/client/autosplit_split_client.py): client prefix execution and boundary payload exchange.

## Validation

Default checks:

```bash
uv run --no-sync pytest -q
uv run --no-sync pytest tests/unit/test_ariadne_boundary_serde.py -q
```

Integration checks:

```bash
uv run --no-sync pytest tests/integration/test_ariadne_real_task_matrix.py -q
SPLITFLEET_RUN_HEAVY_REAL_MODELS=1 uv run --no-sync pytest tests/integration/test_ariadne_real_detection_optional.py -q
```

Cleanup checks:

```bash
grep -R "<old-runtime-package>" -n splitfleet tests pyproject.toml README.md || true
grep -R "<old-model-package>" -n splitfleet tests pyproject.toml README.md || true
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
