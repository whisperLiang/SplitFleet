"""Small role descriptors referencing shared, content-addressed input assets."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from experiments.common.artifact_retention import file_digest
from experiments.common.identity import tensor_state_hash
from experiments.unified_multitask.data import _dataset_pair_hash

LEGACY_SCHEMA = "splitfleet.physical-multitask-bundle.v2"
SHARED_SCHEMA = "splitfleet.physical-multitask-bundle.v3"


def _store(directory: Path, kind: str, content_hash: str, value) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}-{content_hash}.pt"
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                torch.save(value, stream)
                stream.flush()
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    pass  # Another preparation published the same asset.
            finally:
                temporary.unlink(missing_ok=True)
    return path


def write_bundle(path: Path, payload: dict, *, asset_dir: Path | None = None) -> None:
    """Save weights/data once; every server/client bundle is a small descriptor."""
    path = Path(path)
    directory = Path(asset_dir) if asset_dir is not None else path.parent / "assets"
    state = payload["initial_model_state"]
    model_hash = tensor_state_hash(state)
    data = {key: payload[key] for key in ("train_items", "test_items")}
    data_hash = _dataset_pair_hash(data["train_items"], data["test_items"])
    if model_hash != payload["initial_model_hash"] or data_hash != payload["local_data_hash"]:
        raise ValueError("Bundle contents differ from their recorded hashes")
    assets = {"initial_model_state": _store(directory, "model", model_hash, state),
              "data": _store(directory, "data", data_hash, data)}
    descriptor = {key: value for key, value in payload.items()
                  if key not in ("initial_model_state", "train_items", "test_items", "asset_refs")}
    descriptor.update(schema=SHARED_SCHEMA, asset_refs={
        key: {"path": os.path.relpath(asset.resolve(), path.parent.resolve()),
              "sha256": file_digest(asset)} for key, asset in assets.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(descriptor, path)


def _descriptor(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") not in (LEGACY_SCHEMA, SHARED_SCHEMA):
        raise ValueError("unsupported physical bundle")
    return payload


def bundle_dependencies(path: Path) -> tuple[Path, ...]:
    payload = _descriptor(Path(path))
    return tuple((Path(path).parent / ref["path"]).resolve()
                 for ref in payload.get("asset_refs", {}).values())


def read_bundle(path: Path | str) -> dict:
    path = Path(path)
    payload = _descriptor(path)
    if payload["schema"] == SHARED_SCHEMA:
        refs = payload["asset_refs"]
        for ref in refs.values():
            asset = path.parent / ref["path"]
            if file_digest(asset) != ref["sha256"]:
                raise ValueError(f"Shared bundle asset changed: {asset}")
        payload["initial_model_state"] = torch.load(
            path.parent / refs["initial_model_state"]["path"], map_location="cpu", weights_only=True)
        payload.update(torch.load(path.parent / refs["data"]["path"],
                                  map_location="cpu", weights_only=False))
    return payload


def derive_bundle(source: Path, target: Path, overrides: dict) -> None:
    """Change calibration metadata while keeping the same input assets."""
    if set(overrides) & {"schema", "asset_refs", "initial_model_state", "train_items", "test_items"}:
        raise ValueError("Derived bundles may only override metadata")
    payload = _descriptor(source)
    payload.update(overrides)
    target.parent.mkdir(parents=True, exist_ok=True)
    if payload["schema"] == LEGACY_SCHEMA:
        write_bundle(target, payload, asset_dir=source.parent / "assets")
    else:
        payload["asset_refs"] = {key: {**ref, "path": os.path.relpath(
            (source.parent / ref["path"]).resolve(), target.parent.resolve())}
            for key, ref in payload["asset_refs"].items()}
        torch.save(payload, target)


def prune_unused_assets(bundle_root: Path, asset_dir: Path) -> None:
    """Remove preparation variants superseded before the inputs are frozen."""
    referenced = set()
    for path in bundle_root.rglob("*.pt"):
        if not path.is_relative_to(asset_dir):
            referenced.update(bundle_dependencies(path))
    for path in asset_dir.glob("*.pt"):
        if path.resolve() not in referenced:
            path.unlink()
