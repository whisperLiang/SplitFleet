"""Synthetic inputs for configuration tests, independent of retained runs."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _synthetic_cifar():
    from experiments.resnet50_quality.data import CifarData

    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    labels = np.repeat(np.arange(10, dtype=np.int64), 5000)
    indices = np.arange(50_000).reshape(10, 5000)
    return CifarData(
        train_images=np.broadcast_to(image, (50_000, *image.shape)),
        train_labels=labels,
        test_images=np.broadcast_to(image, (10_000, *image.shape)),
        test_labels=np.repeat(np.arange(10, dtype=np.int64), 1000),
        train_indices=indices[:, :4500].reshape(-1),
        validation_indices=indices[:, 4500:].reshape(-1),
        mean=np.full(3, .5, dtype=np.float32),
        std=np.full(3, .25, dtype=np.float32),
    )


def _fixture_initial_state(seed):
    from experiments.resnet50_quality.model_contract import SCHEMA

    # Keep the complete schema and real state hashes. Constant arrays compress
    # cheaply; these configuration fixtures are never numerical training gates.
    return {
        row.name: np.broadcast_to(np.asarray(
            0 if row.dtype == "int64" else seed / 1000, dtype=row.dtype,
        ), row.shape)
        for row in SCHEMA
    }


@pytest.fixture(scope="session")
def resnet50_test_inputs(tmp_path_factory):
    from experiments.resnet50_quality import protocol

    root = tmp_path_factory.mktemp("resnet50-synthetic-inputs")
    data_root = root / "synthetic-cifar"
    data_root.mkdir()
    archive = root / "synthetic-archive"
    archive.write_bytes(b"Synthetic configuration-test data; not a CIFAR archive")
    data = _synthetic_cifar()
    hash_file = protocol.hash_file
    package_version = protocol.importlib.metadata.version

    def fixture_hash(path):
        if Path(path) == Path("data/cifar-10-python.tar.gz"):
            path = archive
        return hash_file(path)

    def available_version(name):
        try:
            return package_version(name)
        except protocol.importlib.metadata.PackageNotFoundError:
            return "not-installed-test-fixture"

    with pytest.MonkeyPatch.context() as patches:
        patches.chdir(Path(__file__).resolve().parents[2])
        patches.setattr(protocol, "load_cifar", lambda: data)
        patches.setattr(protocol, "canonical_state", _fixture_initial_state)
        patches.setattr(protocol, "hash_file", fixture_hash)
        patches.setattr(protocol.np, "savez", np.savez_compressed)
        patches.setattr(protocol.importlib.metadata, "version", available_version)
        protocol.freeze_inputs(root / "protocol")
    return SimpleNamespace(protocol=root / "protocol",
                           data_root=data_root, data=data)


@pytest.fixture
def quality_inputs(resnet50_test_inputs, monkeypatch):
    from experiments.resnet50_quality import workloads

    def synthetic_data(root):
        if Path(root).resolve() != resnet50_test_inputs.data_root:
            raise AssertionError("Configuration tests must use their synthetic data root")
        return resnet50_test_inputs.data

    monkeypatch.setattr(workloads, "cifar_data", synthetic_data)
    return resnet50_test_inputs
