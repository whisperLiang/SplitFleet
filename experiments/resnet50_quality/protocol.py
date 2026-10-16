"""Freeze paired inputs and the authorized experiment budget before main runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import load_cifar
from .model_contract import SCHEMA, canonical_state, state_hash
from .native import BACKENDS, CLIENTS

SEEDS = (233, 234)
PROTOCOL_VERSION = "resnet50_pytorch50_convergence_v1"
DEFAULT_MAIN_ROUNDS = 100
TRAIN_EXAMPLES_PER_ROUND = 45_000


def validate_worker_protocol(frozen):
    if (frozen.get("protocol_version") != PROTOCOL_VERSION
            or frozen.get("backend") != "pytorch"
            or frozen.get("clients") != CLIENTS
            or frozen.get("backends") != BACKENDS
            or frozen.get("worker_count_policy") != "fixed_hardware"
            or frozen.get("logical_suffix_models") != CLIENTS):
        raise ValueError(
            "ResNet50 runs require the current PyTorch / 50-client protocol"
        )


def learning_rate(round_id):
    if not 1 <= round_id <= 100:
        raise ValueError("Global training round must be in 1..100")
    return .0005 + .5 * (.05 - .0005) * (1 + np.cos(np.pi * (round_id - 1) / 99))


def validate_main_protocol(frozen):
    """Read the frozen budget, including the original learning-rate prefix."""
    rounds = frozen.get("rounds")
    if type(rounds) is not int or rounds != DEFAULT_MAIN_ROUNDS:
        raise ValueError("The convergence experiment requires exactly 100 complete rounds")
    examples = TRAIN_EXAMPLES_PER_ROUND * rounds
    if (frozen.get("batch_size") != 16 or frozen.get("local_epochs") != 1
            or frozen.get("all_clients_participate") is not True or frozen.get("seeds") != list(SEEDS)
            or frozen.get("main_examples_per_run") != examples
            or frozen.get("optimizer", {}).get("learning_rates") != [float(learning_rate(r)) for r in range(1, rounds + 1)]):
        raise ValueError("Main protocol has an inconsistent frozen round/sample/learning-rate budget")
    convergence = frozen.get("convergence", {})
    target = convergence.get("target_val_top1")
    if (isinstance(target, bool) or not isinstance(target, (int, float))
            or not np.isfinite(target) or not 0 < target < 1
            or convergence.get("sustained_rounds") != 5):
        raise ValueError("Freeze a validation accuracy target and five-round convergence window")
    return rounds, examples


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_inputs(output, *, rounds=DEFAULT_MAIN_ROUNDS, target_accuracy=.70):
    if type(rounds) is not int or rounds != DEFAULT_MAIN_ROUNDS:
        raise ValueError("Freeze exactly 100 complete training rounds")
    if isinstance(target_accuracy, bool) or not np.isfinite(target_accuracy) or not 0 < target_accuracy < 1:
        raise ValueError("Target accuracy must be a finite fraction between zero and one")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = load_cifar()
    spec = {
        "protocol_version": PROTOCOL_VERSION,
        "rounds": rounds, "local_epochs": 1, "batch_size": 16, "all_clients_participate": True,
        "seeds": list(SEEDS), "backends": BACKENDS, "methods": ["fedavg", "splitfleet"],
        "backend": "pytorch", "clients": CLIENTS,
        "worker_count_policy": "fixed_hardware",
        "worker_execution_policy": "exact hardware worker counts; stop on failure without retry or resizing",
        "failed_output_policy": "delete partial checkpoints and logs; retain concise status",
        "logical_suffix_models": CLIENTS,
        "suffix_model_policy": "one distinct model and isolated state per logical client",
        "resident_suffix_policy": "admitted worker count; idle native tensors offloaded",
        "suffix_execution_policy": "one execution lane per admitted resident; distinct private state per client",
        "optimizer": {"name": "sgd", "momentum": 0, "weight_decay": 0, "rebuild_each_round": True,
                      "schedule_horizon_rounds": 100,
                      "learning_rates": [float(learning_rate(r)) for r in range(1, rounds + 1)]},
        "model": {"name": "ResNet50-V1.5", "trainable_parameters": 23_528_522, "blocks": [3,4,6,3],
                  "input_size": [3,64,64], "schema": [asdict(row) for row in SCHEMA]},
        "batchnorm": {"epsilon": 1e-5, "old_statistics_retention": .9, "train_variance": "population",
                      "running_variance": "unbiased", "counter_updates_per_batch": 1},
        "precision": {"dtype": "float32", "amp": False, "tf32": False, "NVIDIA_TF32_OVERRIDE": "0"},
        "aggregation": {"floating": "sample_weighted_mean", "integer": "maximum"},
        "correctness_tolerance": {"atol": 1e-5, "rtol": 1e-4},
        "main_examples_per_run": TRAIN_EXAMPLES_PER_ROUND * rounds, "validation_size": 5000, "test_size": 10000,
        "convergence": {"target_val_top1": float(target_accuracy), "sustained_rounds": 5,
                        "criterion": "last five validation accuracies reach target and mean CE falls versus first five rounds",
                        "time_to_target": "elapsed time at the end of the first five consecutive qualifying validations; no interpolation"},
        "archive_sha256": hash_file("data/cifar-10-python.tar.gz"),
        "status": "inputs_frozen_training_not_started", "pairs": [],
    }
    versions = {}
    for name in ("numpy", "pillow", "flwr", "ray", "torchlens"):
        versions[name] = importlib.metadata.version(name)
    spec["shared_versions"] = versions
    for index, seed in enumerate(SEEDS):
        initial = canonical_state(seed)
        initial_path = output / f"initial_seed_{seed}.npz"
        np.savez(initial_path, **initial)
        for backend, clients in BACKENDS.items():
            manifest = data.manifest(clients, seed)
            partition_file = f"partition_{backend}_seed_{seed}.json"
            (output / partition_file).write_text(json.dumps(manifest, indent=2) + "\n")
            order = ["fedavg", "splitfleet"] if index % 2 == 0 else ["splitfleet", "fedavg"]
            spec["pairs"].append({"id": f"{backend}_n{clients}_iid_seed{seed}", "backend": backend,
                                  "clients": clients, "seed": seed, "method_order": order,
                                  "initial_state": initial_path.name, "initial_hash": state_hash(initial),
                                  "partition_manifest": partition_file, "partition_sha256": hash_file(output / partition_file)})
    code = {str(path): hash_file(path) for path in sorted(Path("experiments/resnet50_quality").glob("*.py"))}
    spec["preflight_source_sha256"] = code
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    spec["base_commit"] = revision.stdout.strip()
    (output / "protocol.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Frozen {len(spec['pairs'])} pairs / {2 * len(spec['pairs'])} runs at {output.resolve()}", flush=True)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, choices=(100,), default=DEFAULT_MAIN_ROUNDS)
    parser.add_argument("--target-accuracy", type=float, default=.70,
                        help="Predeclared validation top-1 fraction, shared by both methods")
    args = parser.parse_args()
    freeze_inputs(args.output, rounds=args.rounds, target_accuracy=args.target_accuracy)


if __name__ == "__main__":
    main()
