"""Derive admitted layer-scoped bundles from immutable full-scope bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.baselines.adaptive_reference import LayerScopedProvider
from experiments.common.workload_training import _batch
from experiments.common.bundle_storage import derive_bundle
from experiments.physical_multitask import _make_candidate_provider, load_bundle


def prepare_layers(bundles: Path, *, device: str) -> dict:
    sources = sorted(bundles.glob('*/image_classification.pt'))
    if not sources:
        raise FileNotFoundError('Prepare full-scope bundles first')
    if any((source.parent / 'layer').exists() for source in sources):
        raise FileExistsError('Layer outputs already exist; use fresh bundles')
    receipts = {}
    for source in sources:
        workload, bundle = load_bundle(source)
        model = workload.model_factory().to(device)
        model.load_state_dict(bundle['initial_model_state'])
        loader = DataLoader(workload.train_dataset, batch_size=bundle['batch_size'],
                            collate_fn=workload.collate_fn)
        inputs, _, _ = _batch(workload, next(iter(loader)), torch.device(device))
        provider = LayerScopedProvider(_make_candidate_provider(
            model=model, sample_inputs=inputs, bundle=bundle))
        catalog = tuple(provider.get_candidates(training=True))
        anchors = list(dict.fromkeys(min(catalog, key=lambda candidate: (
            abs(candidate.graph_position_ratio - target), candidate.boundary)).boundary
            for target in (0., .25, .75, 1.)))
        output = source.parent / 'layer'
        output.mkdir()
        originals = [source, *sorted(source.parent.glob('image_classification.client_*.pt'))]
        hashes = {}
        for original in originals:
            target = output / original.name
            derive_bundle(original, target, {'online_calibration_boundaries': anchors})
            hashes[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
        receipts[source.parent.name] = {
            'source_bundle_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'candidate_count': len(catalog), 'calibration_boundaries': anchors,
            'layer_bundle_sha256': hashes,
            'scope': 'admitted captured semantic module endpoints; original cut identities',
        }
        del model, provider
    receipt = bundles / 'layer_preparation.json'
    with receipt.open('x') as stream:
        json.dump(receipts, stream, indent=2)
        stream.write('\n')
    return receipts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundles', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(prepare_layers(args.bundles, device=args.device)))
