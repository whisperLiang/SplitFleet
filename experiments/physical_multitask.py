"""Six-worker, three-host real-data comparison for the four unified tasks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import socket
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from flwr.client import NumPyClient as FlowerNumPyClient, start_client as flower_start_client
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig, start_server as flower_start_server
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader, Dataset, Subset

from experiments.common.identity import stable_hash, tensor_state_hash
from experiments.common.partition import dirichlet_partition
from experiments.unified_multitask.data import VOC_CLASSES, Workload, _build_workload, load_workload
from experiments.unified_multitask.models import GridDetector, ImageClassifier, Segmenter, TextClassifier
from experiments.unified_multitask.run import METHODS, _batch, _evaluate, _train_client
from splitfleet.client.app import start_client as split_start_client
from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.common.constants import AUTOSPLIT_MODEL_VERSION_CONFIG_KEY
from splitfleet.server.app import start_server as split_start_server
from splitfleet.server.placement.cosplit_ucb import (
    CoSplitUCBConfig, CoSplitUCBPlacementPolicy, TorchLensCandidateProvider,
)
from splitfleet.server.strategy import AutoSplitStrategy


TASKS = ("image_classification", "text_classification", "object_detection", "semantic_segmentation")
FIXED_BOUNDARIES = ("25%", "50%", "75%")


def scheme_name(method: str, fixed_boundary: str | None) -> str:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if method == "splitfed_fixed":
        if fixed_boundary not in FIXED_BOUNDARIES:
            raise ValueError(f"unsupported fixed boundary: {fixed_boundary}")
        return f"{method}{fixed_boundary.rstrip('%')}"
    if fixed_boundary is not None:
        raise ValueError(f"{method} must not declare a fixed boundary")
    return method


def _batch_axes(task: str) -> dict[str, int]:
    if task == "text_classification":
        return {"/kwargs/input_ids": 0, "/kwargs/attention_mask": 0}
    return {"/args/0": 0}


def _make_candidate_provider(*, model, sample_inputs, bundle):
    common = dict(
        model=model,
        sample_inputs=sample_inputs,
        batch_axes=_batch_axes(bundle["task"]),
        dynamic_batch=(1, int(bundle["batch_size"])),
    )
    if bundle.get("image_model") == "rfdetr_nano":
        from experiments.rfdetr_nano_physical import RFDETRCandidateProvider

        return RFDETRCandidateProvider(**common)
    common["model"] = copy.deepcopy(model)
    return TorchLensCandidateProvider(
        **common, require_trainable_prefix=True,
    )


def _wait_for_round_release(address: str, round_id: int, identity: str) -> None:
    host, port = address.rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=10) as connection:
        connection.settimeout(300)
        connection.sendall(json.dumps({"round_id": round_id, "id": identity}).encode() + b"\n")
        if connection.recv(1) != b"1":
            raise RuntimeError("six-worker round barrier did not release this client")


class ItemDataset(Dataset):
    def __init__(self, items: list[Any], task: str) -> None:
        self.items = items
        if task == "image_classification":
            self.targets = [int(item[1]) for item in items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Any:
        return self.items[index]


def _synthetic_trace_item(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    if isinstance(value, dict):
        return {key: _synthetic_trace_item(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_synthetic_trace_item(item) for item in value)
    if isinstance(value, list):
        return [_synthetic_trace_item(item) for item in value]
    if isinstance(value, (int, float)):
        return type(value)(0)
    raise TypeError(f"cannot make a shape-only trace item from {type(value).__name__}")


def _trace_item(value: Any, task: str) -> Any:
    item = _synthetic_trace_item(value)
    if task == "text_classification":
        item["input_ids"] = torch.full_like(item["input_ids"], 2)
        item["attention_mask"] = torch.ones_like(item["attention_mask"])
    else:
        image, target = item
        item = (torch.full_like(image, 0.5), target)
    return item


def _model_factory(task: str, vocab_size: int, image_model: str = "small"):
    if task == "image_classification":
        if image_model == "resnet50":
            from experiments.unified_multitask.models import build_model
            return lambda: build_model("resnet50", normalization="groupnorm")
        if image_model != "small":
            raise ValueError(f"unknown image model {image_model!r}")
        return lambda: ImageClassifier(10)
    if task == "object_detection" and image_model == "rfdetr_nano":
        from experiments.rfdetr_nano_physical import RFDETRNanoDetector
        return RFDETRNanoDetector
    if image_model != "small":
        raise ValueError("resnet50 requires classification; rfdetr_nano requires detection")
    if task == "text_classification":
        return lambda: TextClassifier(vocab_size)
    if task == "object_detection":
        return lambda: GridDetector(len(VOC_CLASSES))
    return lambda: Segmenter(3)


def prepare_bundle(*, task: str, data_root: str, output: Path, seed: int,
                   train_samples: int | None, test_samples: int | None,
                   batch_size: int, alpha: float = 0.5,
                   image_model: str = "small",
                   pretrain_weights: str | None = None) -> dict[str, Any]:
    workload = load_workload(task, data_root=data_root, source="real",
                             max_train_samples=train_samples, max_test_samples=test_samples,
                             seed=seed,
                             detection_image_size=384 if image_model == "rfdetr_nano" else 96)
    if image_model != "small":
        workload = replace(workload, model_factory=_model_factory(task, 0, image_model))
    if image_model == "rfdetr_nano":
        from experiments.rfdetr_nano_physical import RFDETRDetectionTask
        workload = replace(workload, task=replace(workload.task,
                                                  adapter_factory=RFDETRDetectionTask))
    torch.manual_seed(seed)
    if image_model == "rfdetr_nano" and pretrain_weights:
        from experiments.rfdetr_nano_physical import RFDETRNanoDetector
        model = RFDETRNanoDetector(pretrain_weights=pretrain_weights)
    else:
        model = workload.model_factory()
    vocab_size = model.embedding.num_embeddings if task == "text_classification" else 0
    assignments = dirichlet_partition(
        workload.partition_labels, 6, alpha=alpha, seed=seed,
        min_partition_size=1,
    )
    flattened = [index for indices in assignments.values() for index in indices]
    if sorted(flattened) != list(range(len(workload.train_dataset))):
        raise RuntimeError("training partitions are not disjoint and exhaustive")
    initial_hash = tensor_state_hash(model.state_dict())
    common = {
        "schema": "splitfleet.physical-multitask-bundle.v2",
        "task": task, "source": "real", "seed": seed,
        "image_model": image_model,
        "pretrain_checkpoint_sha256": (
            hashlib.sha256(Path(pretrain_weights).read_bytes()).hexdigest()
            if pretrain_weights else None
        ),
        "dirichlet_alpha": alpha,
        "assignments": assignments,
        "partition_hash": stable_hash(assignments),
        "partition_sizes": {key: len(value) for key, value in assignments.items()},
        "partition_label_counts": {
            key: dict(sorted(Counter(workload.partition_labels[index]
                                     for index in indices).items()))
            for key, indices in assignments.items()
        },
        "edge_client_partition_sizes": {
            str(index): len(assignments[str(index * 2)]) + len(assignments[str(index * 2 + 1)])
            for index in range(3)
        },
        "train_size": len(workload.train_dataset),
        "test_size": len(workload.test_dataset),
        "data_content_hash": workload.data_content_hash,
        "initial_model_hash": initial_hash,
        "initial_model_state": {name: value.detach().cpu().clone()
                                for name, value in model.state_dict().items()},
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "local_epochs": 1,
        "server_trace_source": "synthetic_shape_matched",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    from experiments.unified_multitask.data import _dataset_pair_hash
    shape_example = _trace_item(workload.train_dataset[0], task)
    server_train = [shape_example for _ in range(batch_size)]
    server_test = [workload.test_dataset[index] for index in range(len(workload.test_dataset))]
    server_payload = {**common, "role": "server", "train_items": server_train,
                      "test_items": server_test,
                      "local_data_hash": _dataset_pair_hash(ItemDataset(server_train, task),
                                                            ItemDataset(server_test, task))}
    torch.save(server_payload, output)
    partition_hashes = {}
    original_indices = (list(workload.train_dataset.indices)
                        if isinstance(workload.train_dataset, Subset)
                        else list(range(len(workload.train_dataset))))
    for client_index in range(6):
        positions = assignments[str(client_index)]
        items = [workload.train_dataset[index] for index in positions]
        local_hash = _dataset_pair_hash(ItemDataset(items, task), ItemDataset([], task))
        client_payload = {**common, "role": "client", "client_index": client_index,
                          "partition_positions": positions,
                          "source_indices": [int(original_indices[index]) for index in positions],
                          "train_items": items,
                          "test_items": [], "local_data_hash": local_hash}
        torch.save(client_payload, output.with_name(f"{output.stem}.client_{client_index}.pt"))
        partition_hashes[str(client_index)] = local_hash
    metadata = {key: common[key] for key in ("task", "seed", "partition_hash",
                                               "data_content_hash", "initial_model_hash",
                                               "batch_size", "train_size", "test_size",
                                               "partition_sizes", "local_epochs",
                                               "dirichlet_alpha", "edge_client_partition_sizes",
                                               "server_trace_source")}
    metadata["image_model"] = image_model
    metadata["pretrain_checkpoint_sha256"] = common["pretrain_checkpoint_sha256"]
    metadata["client_data_hashes"] = partition_hashes
    metadata["partition_label_counts"] = common["partition_label_counts"]
    metadata["assignments"] = assignments
    output.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_bundle(path: str | Path) -> tuple[Workload, dict[str, Any]]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema") != "splitfleet.physical-multitask-bundle.v2":
        raise ValueError("unsupported physical bundle")
    factory = _model_factory(bundle["task"], int(bundle["vocab_size"]),
                             bundle.get("image_model", "small"))
    from experiments.unified_multitask.data import _detection_collate
    workload = _build_workload(
        bundle["task"], factory, ItemDataset(bundle["train_items"], bundle["task"]),
        ItemDataset(bundle["test_items"], bundle["task"]),
        _detection_collate if bundle["task"] == "object_detection" else None,
        source="real",
    )
    if bundle.get("image_model") == "rfdetr_nano":
        from experiments.rfdetr_nano_physical import RFDETRDetectionTask
        workload = replace(workload, task=replace(workload.task,
                                                  adapter_factory=RFDETRDetectionTask))
    if workload.data_content_hash != bundle["local_data_hash"]:
        raise ValueError("transferred data do not match the local content hash")
    if stable_hash(bundle["assignments"]) != bundle["partition_hash"]:
        raise ValueError("transferred assignments do not match the partition hash")
    if tensor_state_hash(bundle["initial_model_state"]) != bundle["initial_model_hash"]:
        raise ValueError("initial model state does not match the source hash")
    if bundle["role"] == "client":
        index = str(bundle["client_index"])
        if bundle["partition_positions"] != bundle["assignments"][index]:
            raise ValueError("client positions do not match the partition")
        if len(workload.train_dataset) != bundle["partition_sizes"][index]:
            raise ValueError("client partition size is inconsistent")
        if len(workload.test_dataset):
            raise ValueError("client bundle must not contain the test split")
    elif bundle["role"] == "server":
        if len(workload.test_dataset) != bundle["test_size"]:
            raise ValueError("server test split is incomplete")
    else:
        raise ValueError("unknown bundle role")
    return workload, bundle


class RoundBatches:
    def __init__(self, workload: Workload, *, seed: int,
                 client_index: int, batch_size: int) -> None:
        self.workload = workload
        self.seed = seed
        self.client_index = client_index
        self.batch_size = batch_size
        self.round_id = 1

    def __iter__(self):
        loader = DataLoader(
            self.workload.train_dataset,
            batch_size=self.batch_size, shuffle=True, drop_last=False,
            collate_fn=self.workload.collate_fn,
            generator=torch.Generator().manual_seed(
                self.seed * 1_000_003 + self.round_id * 10_007 + self.client_index
            ),
        )
        yield from loader


class FullClient(FlowerNumPyClient):
    def __init__(self, *, workload: Workload, bundle: dict[str, Any], identity: str,
                 client_index: int, method: str, device: str, learning_rate: float,
                 barrier: str, optimizer_name: str = "sgd") -> None:
        self.workload, self.bundle = workload, bundle
        self.identity, self.client_index = identity, client_index
        self.method, self.device, self.learning_rate = method, device, learning_rate
        self.optimizer_name = optimizer_name
        self.barrier = barrier
        torch.manual_seed(int(bundle["seed"]))
        self.model = workload.model_factory()
        self.model.load_state_dict(bundle["initial_model_state"])
        self.names = list(self.model.state_dict())

    def get_parameters(self, config):
        return [value.detach().cpu().numpy() for value in self.model.state_dict().values()]

    def fit(self, parameters, config):
        state = {name: torch.from_numpy(value.copy()) for name, value in zip(self.names, parameters, strict=True)}
        _wait_for_round_release(self.barrier, int(config["round_id"]), self.identity)
        started_ns = time.time_ns()
        updated, details = _train_client(
            self.workload, state, list(range(len(self.workload.train_dataset))),
            method=self.method, boundary=None, seed=int(self.bundle["seed"]),
            round_id=int(config["round_id"]), client_id=str(self.client_index),
            batch_size=int(self.bundle["batch_size"]), max_batches=None,
            learning_rate=self.learning_rate, proximal_mu=0.01,
            device=torch.device(self.device), drop_last=False,
            optimizer_name=self.optimizer_name,
        )
        finished_ns = time.time_ns()
        self.model.load_state_dict(updated)
        metrics = {key: value for key, value in details.items() if value is not None}
        metrics.update(logical_client_id=self.identity, device=self.device, pid=os.getpid(),
                       fit_started_unix_ns=started_ns, fit_finished_unix_ns=finished_ns,
                       partition_hash=self.bundle["partition_hash"])
        return [updated[name].numpy() for name in self.names], int(details["num_examples"]), metrics

    def evaluate(self, parameters, config):
        return 0.0, 0, {}


class SplitClient(AutoSplitSplitLearningClient):
    def __init__(self, *, identity: str, client_index: int, source: RoundBatches,
                 partition_hash: str, barrier: str, **kwargs: Any) -> None:
        super().__init__(train_data=source, **kwargs)
        self.identity = identity
        self.client_index = client_index
        self.source = source
        self.partition_hash = partition_hash
        self.barrier = barrier

    def fit(self, parameters, config):
        self.source.round_id = int(config[AUTOSPLIT_MODEL_VERSION_CONFIG_KEY])
        _wait_for_round_release(self.barrier, self.source.round_id, self.identity)
        started_ns = time.time_ns()
        updated, examples, metrics = super().fit(parameters, config)
        finished_ns = time.time_ns()
        metrics.update(logical_client_id=self.identity, client_index=self.client_index,
                       device=str(self.device), pid=os.getpid(),
                       fit_started_unix_ns=started_ns, fit_finished_unix_ns=finished_ns,
                       partition_hash=self.partition_hash)
        return updated, examples, metrics

    def get_properties(self, config):
        _ = config
        return {"logical_client_id": self.identity,
                "device_type": str(self.device).split(":", 1)[0],
                "num_batches": (len(self.source.workload.train_dataset)
                                + self.source.batch_size - 1) // self.source.batch_size}


def _evaluate_physical(workload: Workload, model: torch.nn.Module, *,
                       device: torch.device) -> dict[str, float]:
    from experiments.rfdetr_nano_physical import RFDETRNanoDetector

    if isinstance(model, RFDETRNanoDetector):
        from experiments.rfdetr_nano_physical import evaluate
        return evaluate(workload, model, device=device)
    return _evaluate(workload, model, batch_size=8, device=device)


class RecordingFedAvg(FedAvg):
    def __init__(self, *, workload: Workload, model: torch.nn.Module, device: str, **kwargs: Any) -> None:
        self.workload, self.model, self.device = workload, model, device
        self.fit_records: list[dict[str, Any]] = []
        self.fit_failures: list[dict[str, Any]] = []
        self.evaluation_records: list[dict[str, Any]] = []
        super().__init__(
            initial_parameters=ndarrays_to_parameters(
                [value.detach().cpu().numpy() for value in model.state_dict().values()]
            ),
            evaluate_fn=self._evaluate_round,
            on_fit_config_fn=lambda round_id: {"round_id": round_id},
            accept_failures=False,
            **kwargs,
        )

    def aggregate_fit(self, server_round, results, failures):
        self.fit_records.extend({"round_id": int(server_round), "cid": str(proxy.cid),
                                 "num_examples": int(res.num_examples), "metrics": dict(res.metrics)}
                                for proxy, res in results)
        self.fit_failures.extend({"round_id": int(server_round), "reason": str(value)} for value in failures)
        return super().aggregate_fit(server_round, results, failures)

    def _evaluate_round(self, round_id, arrays, config):
        _ = config
        self.model.load_state_dict({name: torch.from_numpy(value.copy())
                                    for name, value in zip(self.model.state_dict(), arrays, strict=True)})
        metrics = _evaluate_physical(self.workload, self.model, device=torch.device(self.device))
        if round_id > 0:
            self.evaluation_records.append({"round_id": round_id, "metrics": metrics})
        return 0.0, metrics


class RecordingSplit(AutoSplitStrategy):
    def __init__(self, *, workload: Workload, **kwargs: Any) -> None:
        self.workload = workload
        self.fit_records: list[dict[str, Any]] = []
        self.fit_failures: list[dict[str, Any]] = []
        self.server_fit_records: list[dict[str, Any]] = []
        self.evaluation_records: list[dict[str, Any]] = []
        super().__init__(**kwargs)

    def aggregate_fit(self, server_round, results, failures):
        self.fit_records.extend({"round_id": int(server_round), "cid": str(proxy.cid),
                                 "num_examples": int(res.num_examples), "metrics": dict(res.metrics)}
                                for proxy, res in results)
        self.fit_failures.extend({"round_id": int(server_round), "reason": str(value)} for value in failures)
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_server_fit(self, server_round, results):
        self.server_fit_records.extend({"round_id": int(server_round), "cid": str(res.sid),
                                        "num_examples": int(res.config.get("num_examples", 0)),
                                        "metrics": dict(res.config)}
                                       for res in results)
        return super().aggregate_server_fit(server_round, results)

    def evaluate(self, server_round, client_parameters, server_parameters):
        _ = server_parameters
        self.backend_adapter.load_ndarrays(self.model, parameters_to_ndarrays(client_parameters))
        metrics = _evaluate_physical(self.workload, self.model,
                                     device=torch.device(self.runtime_device))
        self.evaluation_records.append({"round_id": int(server_round), "metrics": metrics})
        return 0.0, metrics


def run_server(args: argparse.Namespace) -> None:
    workload, bundle = load_bundle(args.bundle)
    if bundle["role"] != "server":
        raise ValueError("the server requires its evaluation bundle")
    torch.set_num_threads(1)
    torch.manual_seed(int(bundle["seed"]))
    model = workload.model_factory().to(args.device)
    model.load_state_dict(bundle["initial_model_state"])
    if tensor_state_hash(model.state_dict()) != bundle["initial_model_hash"]:
        raise RuntimeError("server model initialization differs from the bundle")
    common = dict(fraction_fit=1.0, fraction_evaluate=0.0,
                  min_fit_clients=6, min_evaluate_clients=0, min_available_clients=6)
    optimizer_class = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[args.optimizer]
    start = time.time()
    policy = None
    if args.method in ("fedavg", "fedprox"):
        strategy = RecordingFedAvg(workload=workload, model=model, device=args.device, **common)
        history = flower_start_server(server_address=args.bind,
                                      config=ServerConfig(num_rounds=args.rounds), strategy=strategy)
    else:
        loader = DataLoader(workload.train_dataset, batch_size=int(bundle["batch_size"]),
                            collate_fn=workload.collate_fn)
        sample, _, _ = _batch(workload, next(iter(loader)), torch.device(args.device))
        if args.method == "splitfleet":
            provider = _make_candidate_provider(model=model, sample_inputs=sample, bundle=bundle)
            profile_path = os.environ.get("SPLITFLEET_DEVICE_PROFILES")
            from splitfleet.server.placement.cosplit_ucb.device_cost import DeviceCostPrior
            if bundle.get("image_model") == "rfdetr_nano" and not profile_path:
                raise ValueError("RF-DETR SplitFleet requires SPLITFLEET_DEVICE_PROFILES")
            prior = DeviceCostPrior(profile_path) if profile_path else None
            policy = CoSplitUCBPlacementPolicy(
                candidate_provider=provider,
                config=CoSplitUCBConfig(seed=int(bundle["seed"]),
                                        server_concurrency=1,
                                        safe_exploration_epsilon=0.0 if prior else 0.05,
                                        min_residence_rounds=args.min_residence_rounds,
                                        state_path=str(Path(args.output).with_suffix(".bandit.json"))),
                device_cost_prior=prior,
            )
        strategy = RecordingSplit(
            workload=workload, model=model, sample_inputs=sample,
            batch_axes=_batch_axes(bundle["task"]),
            boundary=args.fixed_boundary, placement_policy=policy,
            aggregation_policy="splitfed",
            dynamic_batch=(1, int(bundle["batch_size"])),
            task=workload.task.make_adapter(),
            optimizer_fn=lambda module: optimizer_class(module.parameters(), lr=args.learning_rate),
            runtime_device=args.device, **common,
            owned_state_exchange=args.method == "splitfleet",
        )
        # Validate the complete catalog once; construct executable plans lazily
        # for the cuts actually selected for this round.
        if policy is not None:
            policy.candidate_provider.get_candidates(training=True)
        boundaries = () if policy is not None else (args.fixed_boundary,)
        for boundary in boundaries:
            placement = strategy.get_or_create_placement_plan(boundary)
            strategy._autosplit_config(0, training=True, placement=placement)
        history = split_start_server(server_address=args.bind,
                                     config=ServerConfig(num_rounds=args.rounds), strategy=strategy)
    result = {
        "schema": "splitfleet.physical-multitask-result.v2", "task": workload.task.name,
        "method": args.method, "source": "real", "seed": bundle["seed"],
        "owned_state_exchange": args.method == "splitfleet",
        "device_cost_profiles": os.environ.get("SPLITFLEET_DEVICE_PROFILES"),
        "min_residence_rounds": args.min_residence_rounds,
        "image_model": bundle.get("image_model", "small"),
        "pretrain_checkpoint_sha256": bundle.get("pretrain_checkpoint_sha256"),
        "rounds": args.rounds, "expected_clients": 6,
        "batch_size": bundle["batch_size"], "local_epochs": bundle["local_epochs"],
        "dirichlet_alpha": bundle["dirichlet_alpha"],
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "fixed_boundary": args.fixed_boundary if args.method == "splitfed_fixed" else None,
        "server_trace_source": bundle["server_trace_source"],
        "train_size": bundle["train_size"], "test_size": bundle["test_size"],
        "partition_sizes": bundle["partition_sizes"],
        "edge_client_partition_sizes": bundle["edge_client_partition_sizes"],
        "data_content_hash": bundle["data_content_hash"],
        "partition_hash": bundle["partition_hash"],
        "initial_model_hash": bundle["initial_model_hash"],
        "final_model_hash": tensor_state_hash(strategy.model.state_dict()),
        "fit_records": strategy.fit_records, "fit_failures": strategy.fit_failures,
        "server_fit_records": getattr(strategy, "server_fit_records", []),
        "evaluation_records": strategy.evaluation_records,
        "round_diagnostics": policy.round_diagnostics if args.method == "splitfleet" else {},
        "candidate_catalog_diagnostics": (
            policy.candidate_provider.catalog_diagnostics[True] if policy is not None else {}
        ),
        "candidate_catalog": [
            {"boundary": candidate.boundary,
             "prefix_node_count": candidate.prefix_node_count,
             "suffix_node_count": candidate.suffix_node_count,
             "graph_position_ratio": candidate.graph_position_ratio,
             "boundary_tensor_count": candidate.boundary_tensor_count,
             "boundary_forward_bytes": candidate.boundary_forward_bytes,
             "boundary_gradient_bytes": candidate.boundary_gradient_bytes,
             "prefix_trainable_parameter_count":
                 candidate.metadata.get("prefix_trainable_parameter_count")}
            for candidate in policy.candidate_provider.get_candidates(training=True)
        ] if policy is not None else [],
        "history": str(history), "started_unix": start, "finished_unix": time.time(),
        "server_device": args.device,
        "fit_barrier": "all_six_ready",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task": workload.task.name, "method": args.method,
                "state_dict": {name: value.detach().cpu().clone()
                               for name, value in strategy.model.state_dict().items()}},
               output.with_suffix(".model.pt"))
    output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")


def run_client(args: argparse.Namespace) -> None:
    workload, bundle = load_bundle(args.bundle)
    if bundle["role"] != "client" or bundle["client_index"] != args.client_index:
        raise ValueError("client received the wrong private partition")
    torch.set_num_threads(1)
    torch.manual_seed(int(bundle["seed"]))
    model = workload.model_factory().to(args.device)
    model.load_state_dict(bundle["initial_model_state"])
    optimizer_class = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[args.optimizer]
    if tensor_state_hash(model.state_dict()) != bundle["initial_model_hash"]:
        raise RuntimeError("client model initialization differs from the bundle")
    if args.method in ("fedavg", "fedprox"):
        client = FullClient(workload=workload, bundle=bundle, identity=args.client_id,
                            client_index=args.client_index, method=args.method,
                            device=args.device, learning_rate=args.learning_rate,
                            barrier=args.barrier, optimizer_name=args.optimizer)
        launch = flower_start_client
    else:
        source = RoundBatches(workload, seed=int(bundle["seed"]),
                              client_index=args.client_index,
                              batch_size=int(bundle["batch_size"]))
        loader = DataLoader(workload.train_dataset, batch_size=int(bundle["batch_size"]),
                            collate_fn=workload.collate_fn)
        sample, _, _ = _batch(workload, next(iter(loader)), torch.device(args.device))
        client = SplitClient(
            identity=args.client_id, client_index=args.client_index, source=source,
            partition_hash=bundle["partition_hash"], barrier=args.barrier,
            model=model, sample_inputs=sample,
            batch_axes=_batch_axes(bundle["task"]), task=workload.task.make_adapter(),
            device=args.device,
            optimizer_fn=lambda module: optimizer_class(module.parameters(), lr=args.learning_rate),
            partial_batch_policy="error",
            max_cached_runtimes=1 if bundle.get("image_model") == "rfdetr_nano" else None,
        )
        # Prepare the first TorchLens graph while the server is booting.  The
        # server may choose another candidate; _ensure_round_runtime will
        # repartition this captured graph after the round config arrives.
        prewarm_boundary = "50%" if args.method == "splitfleet" else args.fixed_boundary
        prewarm_started = time.perf_counter()
        client.prewarm_runtime(
            boundary=prewarm_boundary,
            dynamic_batch=(1, int(bundle["batch_size"])),
        )
        print(json.dumps({"event": "client_runtime_prewarm", "id": args.client_id,
                          "boundary": prewarm_boundary,
                          "resolved_boundary": client._prewarmed_runtime.plan.boundary,
                          "elapsed_sec": time.perf_counter() - prewarm_started}), flush=True)
        launch = split_start_client
    print(json.dumps({"event": "client_start", "id": args.client_id, "device": args.device,
                      "pid": os.getpid(), "host": socket.gethostname(),
                      "task": bundle["task"], "method": args.method,
                      "partition_hash": bundle["partition_hash"],
                      "torch": torch.__version__, "python": platform.python_version()}), flush=True)
    launch(server_address=args.server, client=client.to_client(), max_retries=60, max_wait_time=600)
    print(json.dumps({"event": "client_stop", "id": args.client_id}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="role", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--task", choices=TASKS, required=True)
    prepare.add_argument("--data-root", default="data")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=2026)
    prepare.add_argument("--train-samples", type=int)
    prepare.add_argument("--test-samples", type=int)
    prepare.add_argument("--batch-size", type=int, default=32)
    prepare.add_argument("--dirichlet-alpha", type=float, default=0.5)
    prepare.add_argument("--image-model", choices=("small", "resnet50", "rfdetr_nano"), default="small")
    prepare.add_argument("--pretrain-weights")
    for role in ("server", "client"):
        item = sub.add_parser(role)
        item.add_argument("--bundle", type=Path, required=True)
        item.add_argument("--method", choices=METHODS, required=True)
        item.add_argument("--device", required=True)
        item.add_argument("--learning-rate", type=float, default=0.01)
        item.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
        item.add_argument("--fixed-boundary", choices=("25%", "50%", "75%"), default="50%")
        item.add_argument("--min-residence-rounds", type=int, default=2,
                          help="CoSplit-UCB residence; values above the run length hold the initial cut")
        if role == "server":
            item.add_argument("--bind", required=True)
            item.add_argument("--rounds", type=int, default=3)
            item.add_argument("--output", type=Path, required=True)
        else:
            item.add_argument("--server", required=True)
            item.add_argument("--client-id", required=True)
            item.add_argument("--client-index", type=int, required=True)
            item.add_argument("--barrier", required=True)
    args = parser.parse_args()
    if args.role == "prepare":
        print(json.dumps(prepare_bundle(task=args.task, data_root=args.data_root,
                                        output=args.output, seed=args.seed,
                                        train_samples=args.train_samples,
                                        test_samples=args.test_samples,
                                        batch_size=args.batch_size,
                                        alpha=args.dirichlet_alpha,
                                        image_model=args.image_model,
                                        pretrain_weights=args.pretrain_weights), indent=2))
    elif args.role == "server":
        run_server(args)
    else:
        run_client(args)


if __name__ == "__main__":
    main()
