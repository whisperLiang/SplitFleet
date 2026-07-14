from __future__ import annotations

import copy
import importlib.util
import os
import subprocess
import sys
from typing import Any

import pytest
import torch

from splitfleet.autosplit import prepare_torchlens_runtime
from splitfleet.split_engine import graph_contract_for_runtime_handle
from splitfleet.transport import decode_boundary, encode_boundary
from splitfleet.transport.split_wire import boundary_to_envelope, envelope_to_boundary


def run_real_model_test_isolated(request) -> bool:
    """Execute one real-model node in a fresh interpreter exactly once."""
    if os.environ.get("SPLITFLEET_REAL_MODEL_CHILD") == "1":
        return False
    env = dict(os.environ)
    env["SPLITFLEET_REAL_MODEL_CHILD"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", request.node.nodeid, "-q"],
        cwd=str(request.config.rootpath),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"isolated real-model test failed:\n{result.stdout}", pytrace=False)
    return True


def normalize_inputs(inputs) -> tuple[Any, ...]:
    if isinstance(inputs, tuple):
        return inputs
    if isinstance(inputs, list):
        return tuple(inputs)
    return (inputs,)


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def first_tensor_batch_size(inputs) -> int:
    for tensor in _iter_tensors(inputs):
        if tensor.ndim > 0:
            return int(tensor.shape[0])
    raise AssertionError("No batched tensor found.")


def assert_nested_structure_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        return
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_nested_structure_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for l_item, r_item in zip(left, right):
            assert_nested_structure_equal(l_item, r_item)


def assert_nested_shape_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert tuple(left.shape) == tuple(right.shape)
        return
    if isinstance(left, dict):
        for key in left:
            assert_nested_shape_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        for l_item, r_item in zip(left, right):
            assert_nested_shape_equal(l_item, r_item)


def assert_nested_close(left, right, rtol=1e-4, atol=1e-5) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.allclose(left, right, rtol=rtol, atol=atol)
        return
    if isinstance(left, dict):
        for key in left:
            assert_nested_close(left[key], right[key], rtol=rtol, atol=atol)
    elif isinstance(left, (list, tuple)):
        for l_item, r_item in zip(left, right):
            assert_nested_close(l_item, r_item, rtol=rtol, atol=atol)


def nested_tensor_loss(output) -> torch.Tensor:
    losses = []
    for tensor in _iter_tensors(output):
        losses.append(tensor.float().square().mean())
    if not losses:
        raise AssertionError("Output does not contain tensors.")
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total


def make_runtime(model, trace_inputs, boundary, dynamic_batch=(2, 3)):
    return prepare_torchlens_runtime(
        model,
        trace_inputs,
        boundary=boundary,
        trainable=True,
        dynamic_batch=dynamic_batch,
    )


def run_split_inference_equivalence(
    model, trace_inputs, runtime_inputs, boundary, *, dynamic_batch=(2, 3),
):
    model.eval()
    runtime = make_runtime(model, trace_inputs, boundary, dynamic_batch=dynamic_batch)
    with torch.no_grad():
        direct = model(*normalize_inputs(runtime_inputs))
        boundary_payload = runtime.backend.run_prefix(*normalize_inputs(runtime_inputs))
        split = runtime.backend.run_suffix(boundary_payload)
    assert boundary_payload.tensors
    assert first_tensor_batch_size(runtime_inputs) >= 2
    assert_nested_structure_equal(direct, split)
    assert_nested_shape_equal(direct, split)
    assert_nested_close(direct, split)
    run_boundary_serde_roundtrip(runtime, boundary_payload)
    return runtime


def run_split_training_smoke(
    model, trace_inputs, runtime_inputs, targets, loss_fn, boundary="50%", *,
    dynamic_batch=(2, 3),
):
    model.train()
    runtime = make_runtime(model, trace_inputs, boundary, dynamic_batch=dynamic_batch)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    before = clone_trainable_state(model)
    boundary = runtime.backend.run_prefix(*normalize_inputs(runtime_inputs), training=True)
    loss, boundary_grads = runtime.backend.train_suffix(
        boundary,
        targets,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )
    runtime.backend.backward_prefix(boundary, boundary_grads=boundary_grads, optimizer=optimizer)
    after = clone_trainable_state(model)
    assert torch.isfinite(loss)
    assert boundary.tensors
    assert boundary_grads
    assert has_any_parameter_grad(model) or parameter_delta_nonzero(before, after)
    return loss


def run_boundary_serde_roundtrip(runtime, boundary):
    contract = graph_contract_for_runtime_handle(runtime)
    envelope = boundary_to_envelope(
        boundary,
        round_id=1,
        client_id="real-model",
        step_id="roundtrip",
        plan_id=runtime.plan.plan_id,
        split_id=contract.split_id,
        canonical_graph_hash=contract.canonical_graph_hash,
        boundary_schema_hash=contract.boundary_schema_hash,
        model_version=1,
    )
    restored = envelope_to_boundary(
        decode_boundary(encode_boundary(envelope)), runtime.runtime, "cpu"
    )
    output1 = runtime.backend.run_suffix(boundary)
    output2 = runtime.backend.run_suffix(restored)
    assert restored.tensors
    assert hasattr(restored, "passthrough_inputs")
    assert_nested_shape_equal(output1, output2)
    assert_nested_close(output1, output2)
    return restored


def clone_trainable_state(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def has_any_parameter_grad(model) -> bool:
    return any(param.grad is not None for param in model.parameters() if param.requires_grad)


def parameter_delta_nonzero(before, after) -> bool:
    return any(
        name in after and not torch.allclose(tensor, after[name])
        for name, tensor in before.items()
    )


def skip_if_missing_dependency(package_name):
    if importlib.util.find_spec(package_name) is None:
        pytest.skip(f"{package_name} is not installed; install splitfleet[integration].")
