"""Frozen real CIFAR-10 splits, stratified IID partitions and shared transforms."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

SPLIT_SEED = 20261014


def digest_indices(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def keyed_seed(seed, client, round_id, sample_id, purpose):
    payload = f"{seed}:{client}:{round_id}:{sample_id}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


@dataclass
class CifarData:
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def partitions(self, clients, seed):
        if 4500 % clients:
            raise ValueError("The declared stratified IID protocol requires equal class counts")
        rng = np.random.default_rng(seed)
        partitions = [[] for _ in range(clients)]
        for label in range(10):
            indices = self.train_indices[self.train_labels[self.train_indices] == label]
            chunks = np.array_split(rng.permutation(indices), clients)
            for client, chunk in enumerate(chunks):
                partitions[client].extend(chunk.tolist())
        return {str(client): sorted(indices) for client, indices in enumerate(partitions)}

    def images(self, indices, *, seed=0, client=0, round_id=0, augment=False, test=False):
        source = self.test_images if test else self.train_images
        output = []
        for sample_id in indices:
            value = source[int(sample_id)]
            if augment:
                rng = np.random.default_rng(keyed_seed(seed, client, round_id, sample_id, "augmentation"))
                padded = np.pad(value, ((4, 4), (4, 4), (0, 0)))
                top, left = rng.integers(0, 9, size=2)
                value = padded[top:top + 32, left:left + 32]
                if rng.random() < .5:
                    value = value[:, ::-1]
            image = Image.fromarray(value).resize((64, 64), Image.Resampling.BILINEAR)
            output.append((np.asarray(image, dtype=np.float32) / np.float32(255) - self.mean) / self.std)
        return np.ascontiguousarray(np.stack(output).transpose(0, 3, 1, 2), dtype=np.float32)

    def batches(self, indices, *, seed, client, round_id, batch_size=16):
        rng = np.random.default_rng(keyed_seed(seed, client, round_id, 0, "batch_order"))
        ordered = rng.permutation(indices)
        for start in range(0, len(ordered), batch_size):
            ids = ordered[start:start + batch_size]
            yield ids, self.images(ids, seed=seed, client=client, round_id=round_id, augment=True), self.train_labels[ids]

    def manifest(self, clients, seed):
        partitions = self.partitions(clients, seed)
        return {
            "dataset": "CIFAR-10", "split_seed": SPLIT_SEED,
            "train_indices": self.train_indices.tolist(), "validation_indices": self.validation_indices.tolist(),
            "test_indices": list(range(10000)), "mean": self.mean.tolist(), "std": self.std.tolist(),
            "train_hash": digest_indices(self.train_indices), "validation_hash": digest_indices(self.validation_indices),
            "preprocessing": {"padding": 4, "crop": 32, "flip_probability": .5, "resize": [64, 64], "interpolation": "Pillow.BILINEAR", "layout": "NCHW"},
            "clients": {cid: {"indices": ids, "hash": digest_indices(ids), "num_samples": len(ids),
                               "class_counts": np.bincount(self.train_labels[ids], minlength=10).tolist()}
                        for cid, ids in partitions.items()},
        }


def load_cifar(path="data/cifar-10-batches-py"):
    root = Path(path)
    train_parts, labels = [], []
    for index in range(1, 6):
        with (root / f"data_batch_{index}").open("rb") as stream:
            record = pickle.load(stream, encoding="bytes")
        train_parts.append(record[b"data"])
        labels.extend(record[b"labels"])
    with (root / "test_batch").open("rb") as stream:
        test = pickle.load(stream, encoding="bytes")
    train_images = np.concatenate(train_parts).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_images = test[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels, test_labels = np.asarray(labels, dtype=np.int64), np.asarray(test[b"labels"], dtype=np.int64)
    if len(train_images) != 50000 or len(test_images) != 10000:
        raise ValueError("CIFAR-10 archive is incomplete")
    rng = np.random.default_rng(SPLIT_SEED)
    training, validation = [], []
    for label in range(10):
        ids = rng.permutation(np.flatnonzero(train_labels == label))
        validation.extend(ids[:500])
        training.extend(ids[500:])
    training, validation = np.sort(training), np.sort(validation)
    # Accumulate in float64 in fixed chunks without expanding all images to FP64.
    total, squares = np.zeros(3), np.zeros(3)
    for start in range(0, len(training), 1000):
        pixels = train_images[training[start:start + 1000]].astype(np.float64) / 255
        total += pixels.sum(axis=(0, 1, 2))
        squares += (pixels * pixels).sum(axis=(0, 1, 2))
    count = len(training) * 32 * 32
    mean = total / count
    std = np.sqrt(squares / count - mean * mean)
    return CifarData(train_images, train_labels, test_images, test_labels, training, validation, mean.astype(np.float32), std.astype(np.float32))
