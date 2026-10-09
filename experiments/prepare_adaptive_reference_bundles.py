"""Freeze real CIFAR data, public calibration samples and pretrained models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.physical_multitask import ItemDataset, prepare_bundle
from experiments.common.artifact_retention import file_digest
from experiments.common.bundle_storage import read_bundle, write_bundle, prune_unused_assets
from experiments.unified_multitask.data import _dataset_pair_hash
from experiments.unified_multitask.edge_models import load_edge_workload


def prepare(output, *, seed, train_samples, test_samples, data_root, checkpoint, asset_dir=None):
    metadata = prepare_bundle(task="image_classification", data_root=str(data_root),
        output=output, seed=seed, train_samples=train_samples, test_samples=test_samples,
        batch_size=8, alpha=10., model_name="resnet50_pretrained", pretrain_weights=str(checkpoint),
        worker_ids=["orin140-cpu", "orin118-gpu", "orin238-gpu"], asset_dir=asset_dir)
    workload, _, _ = load_edge_workload("image_classification", name="resnet50_pretrained",
        data_root=str(data_root), max_train_samples=train_samples, max_test_samples=test_samples)
    public_samples = [workload.train_dataset[index] for index in range(8)]
    # Prepare actual public calibration images before the cohort is frozen.
    # These are never counted as client training or model updates.
    for path in [output, *sorted(output.parent.glob(output.stem + '.client_*.pt'))]:
        payload = read_bundle(path)
        payload["server_trace_source"] = "real_public_CIFAR10_training_calibration"
        if payload["role"] == "server":
            payload["train_items"] = public_samples
            payload["local_data_hash"] = _dataset_pair_hash(ItemDataset(public_samples, payload["task"]), ItemDataset(payload["test_items"], payload["task"]))
        write_bundle(path, payload, asset_dir=asset_dir)
    metadata["server_trace_source"] = "real_public_CIFAR10_training_calibration"
    metadata["public_server_calibration_samples"] = 8
    output.with_suffix('.manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh bundle directory')
    args.output.mkdir(parents=True)
    manifests = {}
    for seed in [9599, 9601, 9602, 9603, 9604, 9605]:
        train, test = (96, 16) if seed == 9599 else (512, 256)
        path = args.output / str(seed) / 'image_classification.pt'
        manifests[str(seed)] = prepare(path, seed=seed, train_samples=train, test_samples=test,
            data_root=args.data_root, checkpoint=args.checkpoint, asset_dir=args.output / 'assets')
        print(json.dumps({'seed':seed, 'partition_sizes':manifests[str(seed)]['partition_sizes'],
                          'initial_hash':manifests[str(seed)]['initial_model_hash']}), flush=True)
    prune_unused_assets(args.output, args.output / 'assets')
    hashes = {str(path.relative_to(args.output)): file_digest(path)
              for path in sorted(args.output.rglob('*')) if path.is_file()}
    (args.output / 'bundle_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')


if __name__ == '__main__':
    main()
