"""Run physical CoSplit-UCB server/clients or construct a custom deployment."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from flwr.common import parameters_to_ndarrays
from flwr.server import ServerConfig
from torch import nn

from experiments.cosplit_ucb.config_utils import set_reproducible_seed
from experiments.cosplit_ucb.model_data import (
    build_model,
    cifar10_datasets,
    dataset_targets,
    dirichlet_partition,
    make_loader,
    partition_manifest,
)
from splitfleet.client.app import start_client
from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.server.placement.cosplit_ucb import (
    CoSplitUCBConfig,
    CoSplitUCBPlacementPolicy,
    TorchLensCandidateProvider,
)
from splitfleet.server.app import start_server
from splitfleet.server.strategy import AutoSplitStrategy


def build_cosplit_policy(
    *,
    model: Any,
    sample_inputs: Any,
    sample_kwargs: dict[str, Any] | None = None,
    batch_axes: dict[str, int] | None = None,
    mode: str = "generated_eager",
    dynamic_batch: tuple[int, int] | None = None,
    trace_batch_mode: str | None = None,
    telemetry_provider: Any = None,
    config: CoSplitUCBConfig | None = None,
) -> CoSplitUCBPlacementPolicy:
    """Build the same policy used by ``AutoSplitStrategy`` for physical runs."""

    resolved = config or CoSplitUCBConfig()
    return CoSplitUCBPlacementPolicy(
        candidate_provider=TorchLensCandidateProvider(
            model=model,
            sample_inputs=sample_inputs,
            sample_kwargs=sample_kwargs,
            batch_axes=batch_axes,
            mode=mode,
            dynamic_batch=dynamic_batch,
            trace_batch_mode=trace_batch_mode,
            max_candidates=resolved.max_candidates,
        ),
        telemetry_provider=telemetry_provider,
        config=resolved,
    )


def build_cosplit_strategy(
    *,
    model: Any,
    sample_inputs: Any,
    sample_kwargs: dict[str, Any] | None = None,
    batch_axes: dict[str, int] | None = None,
    mode: str = "generated_eager",
    dynamic_batch: tuple[int, int] | None = None,
    trace_batch_mode: str | None = None,
    telemetry_provider: Any = None,
    config: CoSplitUCBConfig | None = None,
    strategy_kwargs: Mapping[str, Any] | None = None,
) -> AutoSplitStrategy:
    """Build the production Flower strategy used by physical deployments."""

    policy = build_cosplit_policy(
        model=model,
        sample_inputs=sample_inputs,
        sample_kwargs=sample_kwargs,
        batch_axes=batch_axes,
        mode=mode,
        dynamic_batch=dynamic_batch,
        trace_batch_mode=trace_batch_mode,
        telemetry_provider=telemetry_provider,
        config=config,
    )
    return AutoSplitStrategy(
        model=model,
        sample_inputs=sample_inputs,
        sample_kwargs=sample_kwargs,
        batch_axes=batch_axes,
        mode=mode,
        dynamic_batch=dynamic_batch,
        trace_batch_mode=trace_batch_mode,
        placement_policy=policy,
        **dict(strategy_kwargs or {}),
    )


def _load_factory(value: str) -> Callable[[], Mapping[str, Any]]:
    module_name, separator, attribute = str(value).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("factory must use the form 'module:callable'")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"physical factory {value!r} is not callable")
    return factory


class LocalEpochs:
    """Reiterate a client's complete local partition each training round."""

    def __init__(self, loader: Any, count: int) -> None:
        if count < 1:
            raise ValueError("local_epochs must be positive")
        self.loader = loader
        self.count = int(count)

    def __iter__(self):
        for _ in range(self.count):
            yield from self.loader


class TaggedPhysicalClient(AutoSplitSplitLearningClient):
    """Attach stable physical identity to production client feedback."""

    def __init__(self, *, logical_client_id: str, client_index: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.logical_client_id = str(logical_client_id)
        self.client_index = int(client_index)

    def fit(self, parameters, config):
        updated, examples, metrics = super().fit(parameters, config)
        return updated, examples, {
            **metrics,
            "logical_client_id": self.logical_client_id,
            "client_index": self.client_index,
            "hostname": socket.gethostname(),
        }


class RecordingCoSplitStrategy(AutoSplitStrategy):
    """Record physical participation while using the production placement path."""

    def __init__(self, *, test_loader: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.test_loader = test_loader
        self.fit_records: list[dict[str, Any]] = []
        self.fit_failures: list[dict[str, Any]] = []
        self.server_fit_records: list[dict[str, Any]] = []
        self.evaluation_records: list[dict[str, Any]] = []

    def aggregate_fit(self, server_round, results, failures):
        for proxy, fit_res in results:
            self.fit_records.append({
                "round_id": int(server_round),
                "flower_cid": str(proxy.cid),
                "logical_client_id": str((fit_res.metrics or {}).get("logical_client_id", "")),
                "num_examples": int(fit_res.num_examples),
                "metrics": dict(fit_res.metrics or {}),
            })
        self.fit_failures.extend({"round_id": int(server_round), "reason": str(value)} for value in failures)
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_server_fit(self, server_round, results):
        self.server_fit_records.extend({
            "round_id": int(server_round),
            "flower_cid": str(value.sid),
            "num_examples": int(value.config.get("num_examples", 0)),
        } for value in results)
        return super().aggregate_server_fit(server_round, results)

    def evaluate(self, server_round, client_parameters, server_parameters):
        _ = server_parameters
        self.backend_adapter.load_ndarrays(self.model, parameters_to_ndarrays(client_parameters))
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        with torch.inference_mode():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.runtime_device)
                targets = targets.to(self.runtime_device)
                outputs = self.model(inputs)
                total_loss += float(nn.functional.cross_entropy(outputs, targets, reduction="sum"))
                total_correct += int((outputs.argmax(dim=1) == targets).sum())
                total_examples += int(targets.numel())
        if total_examples == 0:
            raise RuntimeError("central evaluation dataset is empty")
        row = {
            "round_id": int(server_round),
            "num_examples": total_examples,
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
        }
        self.evaluation_records.append(row)
        return row["loss"], {"accuracy": row["accuracy"]}


def _resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} was requested but is unavailable")
    return value


def _environment(device: str) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
    }


def _partition(args: argparse.Namespace):
    train, test = cifar10_datasets(args.data_root, download=args.download)
    targets = dataset_targets(train)
    assignments = dirichlet_partition(
        targets, args.num_clients, alpha=args.dirichlet_alpha, seed=args.seed
    )
    return train, test, assignments, partition_manifest(assignments, targets)


def _make_model(args: argparse.Namespace, device: str):
    set_reproducible_seed(args.seed)
    return build_model(args.model, normalization="groupnorm").to(device)


def run_server(args: argparse.Namespace) -> Path:
    """Start the repository's physical CoSplit-UCB Flower/server-model process."""

    device = _resolve_device(args.device)
    model = _make_model(args, device)
    sample_inputs = torch.zeros(2, 3, 32, 32, device=device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    train, test, assignments, manifest = _partition(args)
    _ = train
    test_loader = make_loader(
        test, None, batch_size=args.evaluation_batch_size, shuffle=False, seed=args.seed
    )
    policy = build_cosplit_policy(
        model=model,
        sample_inputs=sample_inputs,
        dynamic_batch=(1, args.batch_size),
        config=CoSplitUCBConfig(
            server_concurrency=args.server_concurrency,
            max_candidates=args.max_candidates,
            state_path=str(output / "bandit_state.json"),
            seed=args.seed,
        ),
    )
    optimizer = lambda module: torch.optim.SGD(
        module.parameters(), lr=args.learning_rate,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    strategy = RecordingCoSplitStrategy(
        test_loader=test_loader,
        model=model,
        sample_inputs=sample_inputs,
        placement_policy=policy,
        aggregation_policy="splitfed",
        dynamic_batch=(1, args.batch_size),
        loss_fn=nn.CrossEntropyLoss(),
        optimizer_fn=optimizer,
        runtime_device=device,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=0,
        min_available_clients=args.num_clients,
    )
    # TorchLens graph capture belongs to the server's owner thread. Resolve
    # and cache every candidate before gRPC worker threads are started.
    candidates = policy.candidate_provider.get_candidates(training=True)
    for candidate in candidates:
        placement = strategy.get_or_create_placement_plan(candidate.boundary)
        strategy._autosplit_config(0, training=True, placement=placement)
    started = time.time()
    history = start_server(
        server_address=args.bind,
        config=ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    if history is None:
        raise RuntimeError("physical CoSplit-UCB server failed before completing the run")
    result = {
        "schema": "splitfleet.physical-cosplit-ucb.v1",
        "placement_policy": "cosplit_ucb",
        "run_id": args.run_id,
        "environment": _environment(device),
        "rounds": args.rounds,
        "expected_clients": args.num_clients,
        "model": args.model,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "local_epochs": args.local_epochs,
        "partition_hash": manifest["partition_hash"],
        "partition_manifest": manifest,
        "candidate_count": len(candidates),
        "fit_records": strategy.fit_records,
        "fit_failures": strategy.fit_failures,
        "server_fit_records": strategy.server_fit_records,
        "evaluation_records": strategy.evaluation_records,
        "round_diagnostics": policy.round_diagnostics,
        "history": str(history),
        "started_unix": started,
        "finished_unix": time.time(),
    }
    path = output / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def run_client(args: argparse.Namespace) -> None:
    """Join one physical host to the production CoSplit-UCB server."""

    if not 0 <= args.client_index < args.num_clients:
        raise ValueError("client_index must identify one partition")
    device = _resolve_device(args.device)
    train, _test, assignments, manifest = _partition(args)
    indices = assignments[str(args.client_index)]
    loader = make_loader(
        train, indices, batch_size=args.batch_size, shuffle=True,
        seed=args.seed * 1_000_003 + args.client_index,
    )
    model = _make_model(args, device)
    client = TaggedPhysicalClient(
        logical_client_id=args.client_id,
        client_index=args.client_index,
        model=model,
        train_data=LocalEpochs(loader, args.local_epochs),
        evaluate_data=loader,
        sample_inputs=torch.zeros(2, 3, 32, 32, device=device),
        optimizer_fn=lambda module: torch.optim.SGD(
            module.parameters(), lr=args.learning_rate,
            momentum=args.momentum, weight_decay=args.weight_decay,
        ),
        device=device,
    )
    print(json.dumps({
        "event": "client_start", "client_id": args.client_id,
        "partition_hash": manifest["partition_hash"],
        "environment": _environment(device),
    }, sort_keys=True), flush=True)
    start_client(
        server_address=args.server,
        client=client.to_client(),
        max_retries=args.max_retries,
        max_wait_time=args.max_wait_time,
    )


def build_physical_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="role", required=True)
    server = subparsers.add_parser("server", help="run the production CoSplit-UCB server")
    client = subparsers.add_parser("client", help="run one physical client")
    for item in (server, client):
        item.add_argument("--num-clients", type=int, default=4)
        item.add_argument("--model", default="resnet18")
        item.add_argument("--seed", type=int, default=233)
        item.add_argument("--batch-size", type=int, default=32)
        item.add_argument("--local-epochs", type=int, default=1)
        item.add_argument("--learning-rate", type=float, default=0.001)
        item.add_argument("--momentum", type=float, default=0.9)
        item.add_argument("--weight-decay", type=float, default=0.0005)
        item.add_argument("--dirichlet-alpha", type=float, default=0.5)
        item.add_argument("--data-root", default="data")
        item.add_argument("--download", action="store_true")
        item.add_argument("--device", default="auto")
    server.add_argument("--bind", default="0.0.0.0:18097")
    server.add_argument("--run-id", required=True)
    server.add_argument("--rounds", type=int, default=3)
    server.add_argument("--server-concurrency", type=int, default=1)
    server.add_argument("--max-candidates", type=int, default=8)
    server.add_argument("--evaluation-batch-size", type=int, default=256)
    server.add_argument("--output", required=True)
    server.set_defaults(run=run_server)
    client.add_argument("--server", required=True)
    client.add_argument("--client-id", required=True)
    client.add_argument("--client-index", type=int, required=True)
    client.add_argument("--max-retries", type=int, default=60)
    client.add_argument("--max-wait-time", type=float, default=900.0)
    client.set_defaults(run=run_client)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run a physical role or load an optional site-specific deployment factory.

    The factory returns a mapping containing ``model`` and ``sample_inputs``.
    It may additionally provide any builder keyword, ``cosplit_config`` as a
    config mapping, and a ``launcher(strategy)`` callback for the site's
    existing Flower/server/client orchestration.
    """

    if argv is None:
        import sys

        argv = sys.argv[1:]
    if argv and argv[0] in {"server", "client"}:
        args = build_physical_parser().parse_args(argv)
        result = args.run(args)
        if result is not None:
            print(result)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="module:callable deployment factory")
    parser.add_argument(
        "--inspect-candidates",
        action="store_true",
        help="trace/validate the operation catalog without launching hosts",
    )
    args = parser.parse_args(argv)
    payload = dict(_load_factory(args.factory)())
    config = CoSplitUCBConfig(**dict(payload.pop("cosplit_config", {})))
    launcher = payload.pop("launcher", None)
    strategy = build_cosplit_strategy(config=config, **payload)
    if args.inspect_candidates:
        candidates = strategy.placement_policy.candidate_provider.get_candidates(
            training=True
        )
        print(
            json.dumps(
                {
                    "placement_policy": "cosplit_ucb",
                    "candidate_count": len(candidates),
                    "graph_signature": candidates[0].graph_signature,
                    "boundaries": [candidate.boundary for candidate in candidates],
                },
                sort_keys=True,
            )
        )
        return
    if not callable(launcher):
        raise ValueError(
            "deployment factory must provide launcher(strategy), or use "
            "--inspect-candidates"
        )
    launcher(strategy)


if __name__ == "__main__":
    main()


__all__ = [
    "build_cosplit_policy", "build_cosplit_strategy", "build_physical_parser",
    "run_server", "run_client", "main",
]
