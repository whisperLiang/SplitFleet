# SplitFleet

SplitFleet is a unified split learning framework for PyTorch models built on top of [Flower](https://flower.ai/).

It upgrades the original SplitBud codebase from a mostly 2-party split learning stack into a coordinator-centered framework with:

- automatic multi-stage model partitioning
- worker placement over ordered execution stages
- full Flower round-loop integration
- support for several split learning variants behind one strategy API

`SplitFleet` is the proposed repository name for this project. If you keep the current folder name for compatibility, this README still reflects the new project identity.

## What This Project Does

SplitFleet is designed for the following problem:

> given a PyTorch model and example inputs, automatically trace the execution graph, choose valid split points, place stages across available nodes, and run split learning end-to-end inside a Flower training loop.

Today, the implementation supports:

- PyTorch-only autosplitting over a traced execution DAG
- training and evaluation
- shared-tail split learning
- per-client tail split learning
- SplitFed-style aggregation
- more than one client-local stage, as long as those client stages are a contiguous prefix
- remote worker execution through the autosplit stage runtime

## Current Scope

The current public scope is intentionally narrow:

- PyTorch only
- centralized coordinator architecture
- Flower as the control plane
- ordered multi-stage partitioning over a single traced execution graph

Not implemented yet:

- arbitrary interleaved client/server ownership over non-contiguous stages
- decentralized scheduling
- cross-framework tracing
- general recurrent or dynamic control-flow graph partitioning guarantees

## Core Design

SplitFleet is organized around four layers:

1. `splitfleet.autosplit`
   Compiles execution plans, enumerates split candidates, builds partition plans, places stages onto workers, and replays forward/backward execution.
2. `AutoSplitStrategy`
   Injects autosplit plans into the Flower round loop and exposes variant knobs such as shared tail, per-client tail, and SplitFed-style aggregation.
3. `StageRuntimeManager`
   Owns the active placement plan and dispatches stage execution to local or remote workers.
4. Client and server adapters
   Bridge Flower clients/server-models to autosplit execution, including full remote execution and split learning with client-local prefixes.

## Supported Split Learning Variants

- `shared tail`
  All clients share one server-side tail replica.
- `per-client tail`
  Each client gets its own server-side tail replica during the round, then tails are aggregated at the end of the round.
- `SplitFed-style aggregation`
  Client-side and server-side replicas are both trained per client and aggregated with weighted averaging at round end.
- `multi-stage client prefix`
  The client may execute more than one local stage before handing boundary tensors to the remote tail.

Important limitation:

- client-local stages must currently be a contiguous prefix, not an arbitrary subset of stages

## Installation

Python `3.10` is required.

Create the environment and install dependencies with `uv`:

```bash
uv sync --extra dev
```

If you prefer creating the virtual environment explicitly first:

```bash
uv venv
uv sync --extra dev
```

## Quick Start

Run the autosplit remote worker demo:

```bash
uv run --no-sync python examples/autosplit_remote_worker_demo.py
```

Run the test suite:

```bash
uv run --no-sync pytest -q
```

## Example: AutoSplit Strategy

The main entrypoint for the unified framework is [`AutoSplitStrategy`](splitfleet/server/strategy/autosplit_strategy.py).

```python
import torch
from torch import nn
from flwr.server.app import ServerConfig
from flwr.server.client_manager import SimpleClientManager

from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.server.app import init_defaults
from splitfleet.server.strategy import AutoSplitStrategy
from splitfleet.worker import start_worker

model = MyModel()
sample_inputs = torch.randn(8, 3, 224, 224)

worker = start_worker(
    worker_id="worker-a",
    model=model,
    sample_inputs=(sample_inputs,),
    server_address="127.0.0.1:50071",
    register_with_registry=False,
)

strategy = AutoSplitStrategy(
    model=model,
    sample_inputs=sample_inputs,
    worker_specs=[worker.worker_spec],
    preferred_stage_count=3,
    client_stage_count=1,
    aggregation_policy="splitfed",
    loss_fn=nn.CrossEntropyLoss(),
    optimizer_fn=lambda m: torch.optim.SGD(m.parameters(), lr=0.01),
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1,
)

server, config = init_defaults(
    server=None,
    config=ServerConfig(num_rounds=1),
    strategy=strategy,
    client_manager=SimpleClientManager(),
)

client = AutoSplitSplitLearningClient(
    model=model,
    train_data=train_loader,
    evaluate_data=val_loader,
    sample_inputs=sample_inputs,
    optimizer_fn=lambda m: torch.optim.SGD(m.parameters(), lr=0.01),
)
```

For a runnable example, see [`examples/autosplit_remote_worker_demo.py`](examples/autosplit_remote_worker_demo.py).

## Main Components

- [`splitfleet/autosplit`](splitfleet/autosplit)
  Autosplit tracing, planning, serialization, caching, and replay runtime.
- [`splitfleet/server/strategy/autosplit_strategy.py`](splitfleet/server/strategy/autosplit_strategy.py)
  Unified Flower strategy for autosplit and split learning variants.
- [`splitfleet/server/stage_runtime`](splitfleet/server/stage_runtime)
  Stage runtime manager, worker registry, and executors.
- [`splitfleet/client/autosplit_client.py`](splitfleet/client/autosplit_client.py)
  Data-owning client that delegates model execution to autosplit runtimes.
- [`splitfleet/client/autosplit_split_client.py`](splitfleet/client/autosplit_split_client.py)
  Split learning client with client-local prefix execution.
- [`splitfleet/worker`](splitfleet/worker)
  Worker runtime and worker startup helpers.

## Validation Status

The repository currently includes:

- unit tests for autosplit planning and strategy metadata
- remote stage runtime tests
- end-to-end Flower round-loop tests for shared remote execution
- end-to-end Flower round-loop tests for shared-tail split learning
- end-to-end Flower round-loop tests for multi-stage client-local prefixes
- end-to-end Flower round-loop tests for SplitFed-style per-client tails

At the time of this update, the full test suite passes with:

```bash
uv run --no-sync pytest -q
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
