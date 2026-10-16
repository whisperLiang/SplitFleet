"""Independent PyTorch autograd control at a residual-stage boundary."""

from __future__ import annotations

import numpy as np
import torch

from .model_contract import NAMES, TRAINABLE_NAMES, validate_state
from .native import build_model, export_state, numpy_value, optimizer


def stage_owner(name, stage):
    if name.startswith("layer"):
        return "prefix" if int(name[5]) <= stage else "suffix"
    return "suffix" if name.startswith("fc.") else "prefix"


def train_staged(backend, initial, images, labels, *, stage, client_device, server_device, learning_rate):
    if backend != "pytorch" or stage not in (1, 2, 3):
        raise ValueError("PyTorch reference requires residual stage 1, 2 or 3")
    models = {"prefix": build_model(backend, initial, client_device),
              "suffix": build_model(backend, initial, server_device)}
    optimizers = {owner: optimizer(backend, model, learning_rate) for owner, model in models.items()}
    for owner, model in models.items():
        model.train()
        optimizers[owner].zero_grad()
    prefix, suffix = models["prefix"], models["suffix"]
    value = prefix.maxpool(prefix.relu(prefix.bn1(prefix.conv1(images))))
    for index in range(1, stage + 1):
        value = getattr(prefix, f"layer{index}")(value)
    value = value.to(server_device)
    for index in range(stage + 1, 5):
        value = getattr(suffix, f"layer{index}")(value)
    logits = suffix.fc(torch.flatten(suffix.avgpool(value), 1))
    loss = torch.nn.functional.cross_entropy(logits, labels.to(server_device))
    loss.backward()
    parameters = {owner: dict(model.named_parameters()) for owner, model in models.items()}
    gradients = {name: numpy_value(parameters[stage_owner(name, stage)][name].grad) for name in TRAINABLE_NAMES}
    loss_value, logits_array = float(loss.detach()), numpy_value(logits)
    for opt in optimizers.values():
        opt.step()
    states = {owner: export_state(backend, model) for owner, model in models.items()}
    state = {name: states[stage_owner(name, stage)][name] for name in NAMES}
    return {"loss": loss_value, "logits": logits_array, "gradients": gradients, "state": state,
            "reference_kind": "native_staged_same_compute_placement",
            "native_devices": {"prefix": client_device, "suffix": server_device},
            "native_autograd_device_copy": True, "native_vjp_composition": False,
            "batch_size": int(images.shape[0]), "executed_trainable_tensors": len(gradients),
            "canonical_state_tensors": len(state)}


def placement_reference(backend, initial, images, labels, *, stage, client_device, server_device,
                        learning_rate, full_reference, ownership):
    """Validate direct native stage execution, then change only its device copy."""
    from .gate_torch import compare_tensor

    expected_owners = {name: stage_owner(name, stage) for name in NAMES}
    actual_owners = dict(zip(ownership["names"], ownership["owners"], strict=True))
    if actual_owners != expected_owners:
        raise RuntimeError("Captured split ownership differs from the independent native stage partition")
    control = train_staged(backend, initial, images, labels, stage=stage, client_device=client_device,
                           server_device=client_device, learning_rate=learning_rate)
    comparisons = {
        "logits": compare_tensor(control["logits"], full_reference["logits"], name="logits"),
        "ce": compare_tensor(np.asarray(control["loss"], np.float32), np.asarray(full_reference["loss"], np.float32), name="ce"),
        "gradients": {name: compare_tensor(control["gradients"][name], full_reference["gradients"][name], name=name)
                      for name in TRAINABLE_NAMES},
        "state": {name: compare_tensor(control["state"][name], full_reference["state"][name], name=name) for name in NAMES},
    }
    checks = [comparisons["logits"], comparisons["ce"], *comparisons["gradients"].values(), *comparisons["state"].values()]
    if not all(row["passed"] for row in checks):
        failed = [row["name"] for row in checks if not row["passed"]]
        raise RuntimeError(f"Same-device native stage control differs from the original full model: {failed}")
    reference = control if client_device == server_device else train_staged(backend, initial, images, labels,
        stage=stage, client_device=client_device, server_device=server_device, learning_rate=learning_rate)
    reference["same_device_native_control"] = {"status": "passed", "comparisons": comparisons}
    reference["full_client_device_reference_differences"] = {
        "scope": "Hardware-placement differences only; does not admit comparable final accuracy",
        "ce": compare_tensor(np.asarray(reference["loss"], np.float32), np.asarray(full_reference["loss"], np.float32), name="ce"),
        "gradients": {name: compare_tensor(reference["gradients"][name], full_reference["gradients"][name], name=name)
                      for name in TRAINABLE_NAMES},
    }
    return reference


def aggregate_native(updates):
    """Independent sample-weighted FP32 sum, with integer BN-counter maximum."""
    from .model_contract import NAMES, validate_state
    total = sum(count for _, count in updates)
    if total <= 0:
        raise ValueError("Native reference needs all positive-weight local updates")
    result = {}
    for name in NAMES:
        values = [state[name] for state, _ in updates]
        if values[0].dtype.kind not in "fc":
            result[name] = np.maximum.reduce(values)
        else:
            weighted = [value * np.float32(count) for value, (_, count) in zip(values, updates, strict=True)]
            result[name] = np.add.reduce(weighted) / np.float32(total)
    validate_state(result)
    return result
