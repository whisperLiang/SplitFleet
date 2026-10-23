"""Shared data and budget contracts for CPU/GPU clients on three Orins."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.resnet50_quality.data import keyed_seed, load_cifar
from experiments.resnet50_quality.model_contract import canonical_state, state_hash

BACKENDS = ("tensorflow", "jax", "paddle")
METHODS = ("splitfleet", "fedavg")
HOST_SUFFIXES = ("140", "118", "238")
LOGICAL_CLIENTS = tuple(
    {"id": f"orin{suffix}_{kind}", "host_id": f"orin{suffix}",
     "device": device, "index": 2 * host_index + device_index}
    for host_index, suffix in enumerate(HOST_SUFFIXES)
    for device_index, (kind, device) in enumerate((("cpu", "cpu"), ("gpu", "cuda:0")))
)
ATOL = 1e-5
RTOL = 1e-4


def validate_methods(methods):
    methods = list(methods)
    if not methods or len(methods) != len(set(methods)) or any(name not in METHODS for name in methods):
        raise ValueError("Choose distinct experiment methods from splitfleet and fedavg")
    return methods


def compare_tensor(actual, expected, *, name):
    """Strict FP32/integer comparison used by the native admission receipt."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    report = {"name": name, "passed": False, "shape": list(expected.shape),
              "dtype": str(expected.dtype), "actual_shape": list(actual.shape),
              "actual_dtype": str(actual.dtype)}
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        report["reason"] = "schema_mismatch"
        return report
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        report["reason"] = "nonfinite"
        return report
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    exact = expected.dtype.kind not in "fc"
    allowance = np.zeros_like(difference) if exact else ATOL + RTOL * np.abs(expected.astype(np.float64))
    mismatch = difference > allowance
    report.update(passed=not bool(np.any(mismatch)), exact=exact,
                  mismatched_elements=int(np.count_nonzero(mismatch)),
                  max_absolute_error=float(difference.max(initial=0)))
    return report


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def learning_rate(round_id, config):
    if not 1 <= round_id <= config["rounds"]:
        raise ValueError("Round lies outside the frozen training budget")
    fraction = (round_id - 1) / max(config["rounds"] - 1, 1)
    return config["minimum_learning_rate"] + .5 * (
        config["learning_rate"] - config["minimum_learning_rate"]) * (1 + math.cos(math.pi * fraction))


def training_batch_sizes(config):
    """Distinct minibatch sizes in one complete client epoch, including its tail."""
    samples = config["train_samples"] // config.get("clients", 3)
    full_batches, remainder = divmod(samples, config["batch_size"])
    return ((config["batch_size"],) if full_batches else ()) + ((remainder,) if remainder else ())


def prepare(output, *, data_root, train_samples=45000, validation_samples=5000,
            rounds=50, batch_size=16, seed=233, checkpoint_retention="all", test_samples=10000):
    if checkpoint_retention not in {"all", "latest_and_best"}:
        raise ValueError("Unknown checkpoint retention policy")
    clients = len(LOGICAL_CLIENTS)
    if (not clients <= train_samples <= 45000 or train_samples % (10 * clients)
            or not 10 <= validation_samples <= 5000 or validation_samples % 10
            or not 10 <= test_samples <= 10000 or test_samples % 10
            or rounds < 1 or batch_size < 2):
        raise ValueError("Use class-balanced budgets, six equal clients and batch size >=2")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = load_cifar(data_root)
    rng = np.random.default_rng(seed)
    partitions = [[] for _ in range(clients)]
    validation = []
    for label in range(10):
        ids = rng.permutation(data.train_indices[data.train_labels[data.train_indices] == label])
        for client, chunk in enumerate(np.split(ids[:train_samples // 10], clients)):
            partitions[client].extend(chunk.tolist())
        ids = data.validation_indices[data.train_labels[data.validation_indices] == label]
        validation.extend(ids[:validation_samples // 10].tolist())
    initial = canonical_state(seed)
    np.savez(output / "initial.npz", **initial)
    for client, ids in enumerate(partitions):
        ids = np.asarray(sorted(ids), np.int64)
        np.savez(output / f"client_{client}.npz", ids=ids,
                 images=data.train_images[ids], labels=data.train_labels[ids])
    ids = np.asarray(sorted(validation), np.int64)
    np.savez(output / "validation.npz", ids=ids,
             images=data.train_images[ids], labels=data.train_labels[ids])
    test_ids = np.sort(np.concatenate([np.flatnonzero(data.test_labels == label)[:test_samples // 10]
                                      for label in range(10)])).astype(np.int64)
    np.savez(output / "test.npz", ids=test_ids, images=data.test_images[test_ids], labels=data.test_labels[test_ids])
    files = {p.name: file_hash(p) for p in sorted(output.glob("*.npz"))}
    config = {"schema": "splitfleet.resnet50.three-orin.six-clients.v2", "clients": clients,
              "logical_clients": list(LOGICAL_CLIENTS), "physical_hosts": 3, "rounds": rounds,
              "seed": seed, "batch_size": batch_size, "local_epochs": 1,
              "train_samples": train_samples, "validation_samples": validation_samples,
              "test_samples": test_samples, "learning_rate": .05, "minimum_learning_rate": .0005,
              "momentum": 0, "weight_decay": 0, "initial_hash": state_hash(initial),
              "placement_max_payload_bytes": 256 * 1024 * 1024,
              "checkpoint_retention": checkpoint_retention,
              "mean": data.mean.tolist(), "std": data.std.tolist(), "files": files,
              "training_source": "CIFAR-10 stratified training split; no pretrained weights",
              "convergence": {"target_val_top1": .70, "sustained_rounds": 5},
              "aggregation": {"floating": "sample_weighted_mean", "integer": "maximum"},
              "preprocessing": "32x32 pad/crop/flip, Pillow bilinear 64x64, training-split normalization",
              "augmentation": "global sample/client/round/seed keyed, identical across methods/backends",
              "formal": rounds == 50 and train_samples == 45000 and validation_samples == 5000 and test_samples == 10000}
    save_json(output / "protocol.json", config)
    return config


class Images:
    def __init__(self, path, config):
        with np.load(path) as record:
            self.ids, self.raw, self.labels = (record[name].copy() for name in ("ids", "images", "labels"))
        self.config = config
        self.mean = np.asarray(config["mean"], np.float32)
        self.std = np.asarray(config["std"], np.float32)

    def images(self, positions, *, client=0, round_id=0, augment=False):
        values = []
        for position in positions:
            raw = self.raw[position]
            if augment:
                rng = np.random.default_rng(keyed_seed(self.config["seed"], client, round_id,
                                                       int(self.ids[position]), "augmentation"))
                padded = np.pad(raw, ((4, 4), (4, 4), (0, 0)))
                top, left = rng.integers(0, 9, size=2)
                raw = padded[top:top + 32, left:left + 32]
                if rng.random() < .5:
                    raw = raw[:, ::-1]
            raw = Image.fromarray(raw).resize((64, 64), Image.Resampling.BILINEAR)
            values.append((np.asarray(raw, np.float32) / np.float32(255) - self.mean) / self.std)
        return np.ascontiguousarray(np.stack(values).transpose(0, 3, 1, 2), dtype=np.float32)

    def batches(self, *, client, round_id):
        seed = keyed_seed(self.config["seed"], client, round_id, 0, "batch_order")
        order = np.random.default_rng(seed).permutation(len(self.labels))
        for start in range(0, len(order), self.config["batch_size"]):
            positions = order[start:start + self.config["batch_size"]]
            yield (self.ids[positions], self.images(positions, client=client, round_id=round_id, augment=True),
                   self.labels[positions])


def client_identities(config):
    if "logical_clients" in config:
        return tuple(row["id"] for row in config["logical_clients"])
    # Retained three-client attempts remain readable, never relabelled as six.
    return ("orin140", "orin118", "orin238")


def validate_client_budget(row, config):
    count = config.get("clients", 3)
    expected = config["train_samples"] // count
    steps = math.ceil(expected / config["batch_size"])
    metrics = row["metrics"]
    if (row["num_examples"] != expected or metrics["num_batches"] != steps
            or metrics.get("skipped_examples", 0) or metrics.get("skipped_batches", 0)
            or len(metrics["data_batch_hash"]) != 64):
        raise RuntimeError("A client omitted, repeated or skipped part of its frozen local epoch")
    if "logical_clients" in config:
        specification = config["logical_clients"][int(row["client_index"])]
        if (metrics.get("logical_client_id") != specification["id"]
                or metrics.get("device") != specification["device"]
                or metrics.get("physical_host") != specification["host_id"]
                or metrics.get("machine") != "aarch64"):
            raise RuntimeError("Client identity or actual CPU/GPU placement disagrees with the frozen protocol")


def validate_round_budget(clients, config):
    count = config.get("clients", 3)
    if len(clients) != count or {int(row["client_index"]) for row in clients} != set(range(count)):
        raise RuntimeError(f"A complete round requires all {count} distinct Orin clients")
    for row in clients:
        validate_client_budget(row, config)
    return {"examples": config["train_samples"], "optimization_steps": count * math.ceil(
        (config["train_samples"] // count) / config["batch_size"])}
