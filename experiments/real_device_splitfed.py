"""Run a small, auditable SplitFed round across physical devices.

The server owns the suffix replicas and the Flower control plane.  Each client
owns its data and executes a real TorchLens prefix before sending the encoded
boundary to the server.  The workload is intentionally small enough to serve
as a deployment smoke test before launching the full RA-SplitFed suite.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path
from typing import Any

import torch
from flwr.server import ServerConfig
from torch import nn

from splitfleet.client.app import start_client
from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.server.app import start_server
from splitfleet.server.strategy import AutoSplitStrategy


MODEL_SEED = 20260809
INPUT_DIM = 16
HIDDEN_DIM = 32
NUM_CLASSES = 4


class PhysicalDeviceNet(nn.Module):
    """Small model with an unambiguous, trainable middle boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.projector = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(inputs)
        projected = self.projector(encoded)
        return self.classifier(projected)


class TaggedSplitClient(AutoSplitSplitLearningClient):
    """Attach the physical-device identity to metrics returned to the server."""

    def __init__(self, *, logical_client_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.logical_client_id = str(logical_client_id)

    def fit(self, parameters, config):
        updated, examples, metrics = super().fit(parameters, config)
        metrics = dict(metrics)
        metrics["logical_client_id"] = self.logical_client_id
        metrics["hostname"] = socket.gethostname()
        metrics["torch_version"] = torch.__version__
        return updated, examples, metrics


class RecordingAutoSplitStrategy(AutoSplitStrategy):
    """Retain raw client fit records as proof of physical participation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fit_records: list[dict[str, Any]] = []
        self.fit_failures: list[str] = []
        self.server_fit_records: list[dict[str, Any]] = []

    def aggregate_fit(self, server_round, results, failures):
        for proxy, fit_res in results:
            self.fit_records.append(
                {
                    "server_round": int(server_round),
                    "flower_cid": str(proxy.cid),
                    "num_examples": int(fit_res.num_examples),
                    "metrics": _json_safe(dict(fit_res.metrics)),
                }
            )
        self.fit_failures.extend(str(failure) for failure in failures)
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_server_fit(self, server_round, results):
        round_records = [
            {
                "server_round": int(server_round),
                "sid": str(result.sid),
                "num_examples": int(result.config.get("num_examples", 0)),
            }
            for result in results
        ]
        self.server_fit_records.extend(round_records)
        print(
            json.dumps({"event": "server_fit_results", "records": round_records}, sort_keys=True),
            flush=True,
        )
        return super().aggregate_server_fit(server_round, results)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested but CUDA is unavailable")
    return requested


def _make_model(device: str) -> PhysicalDeviceNet:
    torch.manual_seed(MODEL_SEED)
    return PhysicalDeviceNet().to(device)


def _make_batches(client_index: int, batch_size: int, num_batches: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(10000 + int(client_index))
    teacher = torch.linspace(-1.0, 1.0, INPUT_DIM * NUM_CLASSES).reshape(
        INPUT_DIM, NUM_CLASSES
    )
    batches = []
    for _ in range(int(num_batches)):
        inputs = torch.randn(batch_size, INPUT_DIM, generator=generator)
        logits = inputs @ teacher + 0.05 * torch.randn(
            batch_size, NUM_CLASSES, generator=generator
        )
        targets = logits.argmax(dim=1)
        batches.append((inputs, targets))
    return batches


def _environment(device: str) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def run_server(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    model = _make_model(device)
    sample_inputs = torch.zeros(2, INPUT_DIM, device=device)
    strategy = RecordingAutoSplitStrategy(
        model=model,
        sample_inputs=sample_inputs,
        boundary=args.boundary,
        aggregation_policy="splitfed",
        dynamic_batch=(1, args.batch_size),
        loss_fn=nn.CrossEntropyLoss(),
        optimizer_fn=lambda module: torch.optim.SGD(module.parameters(), lr=args.learning_rate),
        runtime_device=device,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=0,
        min_available_clients=args.num_clients,
    )
    started = time.time()
    history = start_server(
        server_address=args.bind,
        config=ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    finished = time.time()
    logical_ids = sorted(
        {
            str(record["metrics"].get("logical_client_id", ""))
            for record in strategy.fit_records
            if record["metrics"].get("logical_client_id")
        }
    )
    result = {
        "schema": "splitfleet.real-device.v1",
        "role": "server",
        "environment": _environment(device),
        "bind": args.bind,
        "rounds": int(args.rounds),
        "expected_clients": int(args.num_clients),
        "observed_logical_clients": logical_ids,
        "all_clients_observed": len(logical_ids) == int(args.num_clients),
        "fit_records": strategy.fit_records,
        "fit_failures": strategy.fit_failures,
        "server_fit_records": strategy.server_fit_records,
        "history": _json_safe(getattr(history, "__dict__", history)),
        "started_unix": started,
        "finished_unix": finished,
        "elapsed_sec": finished - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


def run_client(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    model = _make_model(device)
    batches = _make_batches(args.client_index, args.batch_size, args.num_batches)
    sample_inputs = torch.zeros(2, INPUT_DIM, device=device)
    client = TaggedSplitClient(
        logical_client_id=args.client_id,
        model=model,
        train_data=batches,
        evaluate_data=batches,
        sample_inputs=sample_inputs,
        optimizer_fn=lambda module: torch.optim.SGD(module.parameters(), lr=args.learning_rate),
        device=device,
    )
    print(
        json.dumps(
            {
                "event": "client_start",
                "client_id": args.client_id,
                "client_index": int(args.client_index),
                "server": args.server,
                "environment": _environment(device),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    start_client(
        server_address=args.server,
        client=client.to_client(),
        max_retries=args.max_retries,
        max_wait_time=args.max_wait_time,
    )
    print(
        json.dumps({"event": "client_stop", "client_id": args.client_id}, sort_keys=True),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="role", required=True)

    server = subparsers.add_parser("server")
    server.add_argument("--bind", default="0.0.0.0:18091")
    server.add_argument("--num-clients", type=int, default=4)
    server.add_argument("--rounds", type=int, default=1)
    server.add_argument("--batch-size", type=int, default=2)
    server.add_argument("--learning-rate", type=float, default=0.02)
    server.add_argument("--boundary", default="50%")
    server.add_argument("--device", default="auto")
    server.add_argument("--output", default="results/real_device_splitfed/result.json")
    server.set_defaults(run=run_server)

    client = subparsers.add_parser("client")
    client.add_argument("--server", required=True)
    client.add_argument("--client-id", required=True)
    client.add_argument("--client-index", type=int, required=True)
    client.add_argument("--batch-size", type=int, default=2)
    client.add_argument("--num-batches", type=int, default=2)
    client.add_argument("--learning-rate", type=float, default=0.02)
    client.add_argument("--device", default="auto")
    client.add_argument("--max-retries", type=int, default=40)
    client.add_argument("--max-wait-time", type=float, default=300.0)
    client.set_defaults(run=run_client)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
