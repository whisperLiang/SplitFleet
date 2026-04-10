"""Validation helpers for replay, partitioning, and split backward."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch

from .replay_engine import _run_direct_model_forward, replay_forward, replay_partitioned
from .replay_plan import ExecutionPlan, FrontierSplit
from .replay_train import backward_prefix_from_boundary, train_partitioned
from .replay_utils import resolve_device, tree_allclose, tree_to_device


def validate_replay_equivalence(
    plan: ExecutionPlan,
    model: torch.nn.Module,
    raw_inputs: Any,
    *,
    input_kwargs: Optional[Dict[str, Any]] = None,
    device: Any = "auto",
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> bool:
    """Compare replayed outputs against a direct model forward."""

    runtime_device = resolve_device(
        device, model=model, inputs=raw_inputs, default=plan.default_device
    )
    model = model.to(runtime_device)
    replay_plan = _clone_plan_for_model(plan, model)
    direct_output = _run_direct_model_forward(
        replay_plan,
        model,
        raw_inputs,
        input_kwargs=input_kwargs,
        device=runtime_device,
        restore_rng=plan.meta.get("pre_forward_rng_states"),
    )
    replay_output = replay_forward(
        replay_plan,
        raw_inputs,
        input_kwargs=input_kwargs,
        device=runtime_device,
        validate=False,
    )
    return tree_allclose(direct_output, replay_output, atol=atol, rtol=rtol)


def validate_split_equivalence(
    plan: ExecutionPlan,
    split: FrontierSplit,
    model: torch.nn.Module,
    raw_inputs: Any,
    *,
    input_kwargs: Optional[Dict[str, Any]] = None,
    device: Any = "auto",
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> bool:
    """Compare prefix/boundary/suffix replay against a direct model forward."""

    runtime_device = resolve_device(
        device, model=model, inputs=raw_inputs, default=plan.default_device
    )
    model = model.to(runtime_device)
    replay_plan = _clone_plan_for_model(plan, model)
    direct_output = _run_direct_model_forward(
        replay_plan,
        model,
        raw_inputs,
        input_kwargs=input_kwargs,
        device=runtime_device,
        restore_rng=plan.meta.get("pre_forward_rng_states"),
    )

    partitioned_result = replay_partitioned(
        replay_plan,
        raw_inputs,
        split=split,
        input_kwargs=input_kwargs,
        input_mode="raw",
        device=runtime_device,
        return_boundary=True,
        validate=False,
    )
    raw_output = partitioned_result["output"]
    boundary_payload = partitioned_result["boundary"]
    suffix_output = replay_partitioned(
        replay_plan,
        _bundle_boundary_mode_inputs(
            split,
            boundary_payload,
            raw_inputs,
            input_kwargs,
        ),
        split=split,
        input_mode="boundary",
        device=runtime_device,
        validate=False,
    )

    return tree_allclose(direct_output, raw_output, atol=atol, rtol=rtol) and tree_allclose(
        direct_output,
        suffix_output,
        atol=atol,
        rtol=rtol,
    )


def validate_gradient_equivalence(
    plan: ExecutionPlan,
    split: Optional[FrontierSplit],
    model: torch.nn.Module,
    raw_inputs: Any,
    targets: Any = None,
    loss_fn: Optional[Any] = None,
    *,
    input_kwargs: Optional[Dict[str, Any]] = None,
    device: Any = "auto",
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> bool:
    """Compare direct gradients against replayed split gradients."""

    runtime_device = resolve_device(
        device, model=model, inputs=raw_inputs, default=plan.default_device
    )
    direct_model = model.to(runtime_device)
    replay_model = copy.deepcopy(model).to(runtime_device)
    direct_inputs = tree_to_device(copy.deepcopy(raw_inputs), runtime_device)
    replay_inputs = tree_to_device(copy.deepcopy(raw_inputs), runtime_device)
    direct_kwargs = tree_to_device(
        copy.deepcopy(input_kwargs) if input_kwargs is not None else {}, runtime_device
    )
    replay_kwargs = tree_to_device(
        copy.deepcopy(input_kwargs) if input_kwargs is not None else {}, runtime_device
    )
    direct_targets = tree_to_device(copy.deepcopy(targets), runtime_device)
    replay_targets = tree_to_device(copy.deepcopy(targets), runtime_device)
    replay_plan = _clone_plan_for_model(plan, replay_model)
    direct_plan = _clone_plan_for_model(plan, direct_model)

    direct_model.zero_grad(set_to_none=True)
    direct_output = _run_direct_model_forward(
        direct_plan,
        direct_model,
        direct_inputs,
        input_kwargs=direct_kwargs,
        device=runtime_device,
        restore_rng=plan.meta.get("pre_forward_rng_states"),
    )
    direct_loss = _compute_loss(direct_output, direct_targets, loss_fn)
    direct_loss.backward()
    direct_grads = {
        name: (param.grad.detach().clone() if param.grad is not None else None)
        for name, param in direct_model.named_parameters()
    }
    replay_model.zero_grad(set_to_none=True)

    if split is None:
        train_partitioned(
            replay_plan,
            replay_inputs,
            input_kwargs=replay_kwargs,
            split=None,
            device=runtime_device,
            targets=replay_targets,
            loss_fn=loss_fn,
            optimizer=None,
            zero_grad=False,
            step_optimizer=False,
        )
    else:
        boundary_result = replay_partitioned(
            replay_plan,
            replay_inputs,
            split=split,
            input_kwargs=replay_kwargs,
            input_mode="raw",
            device=runtime_device,
            return_boundary=True,
            validate=False,
        )
        suffix_result = train_partitioned(
            replay_plan,
            _bundle_boundary_mode_inputs(
                split,
                boundary_result["boundary"],
                replay_inputs,
                replay_kwargs,
            ),
            split=split,
            input_mode="boundary",
            device=runtime_device,
            targets=replay_targets,
            loss_fn=loss_fn,
            optimizer=None,
            return_boundary_grad=True,
            zero_grad=False,
            step_optimizer=False,
        )
        backward_prefix_from_boundary(
            replay_plan,
            replay_inputs,
            split,
            suffix_result["boundary_grad"],
            input_kwargs=replay_kwargs,
            device=runtime_device,
            optimizer=None,
            zero_grad=False,
            step_optimizer=False,
            match_boundary=True,
            cached_boundary=boundary_result["boundary"],
        )

    for name, param in replay_model.named_parameters():
        direct_grad = direct_grads[name]
        replay_grad = param.grad
        if direct_grad is None or replay_grad is None:
            if direct_grad is not None or replay_grad is not None:
                return False
            continue
        if not torch.allclose(direct_grad, replay_grad, atol=atol, rtol=rtol):
            return False

    direct_model.zero_grad(set_to_none=True)
    return True


def _clone_plan_for_model(plan: ExecutionPlan, model: torch.nn.Module) -> ExecutionPlan:
    cloned_plan = copy.copy(plan)
    cloned_plan.set_model(model)
    return cloned_plan


def _bundle_boundary_mode_inputs(
    split: FrontierSplit,
    boundary_inputs: Any,
    raw_inputs: Any,
    input_kwargs: Optional[Dict[str, Any]],
) -> Any:
    if not split.meta.get("passthrough_input_indices"):
        return boundary_inputs
    return {
        "boundary": boundary_inputs,
        "raw_inputs": raw_inputs,
        "input_kwargs": input_kwargs,
    }


def _compute_loss(output: Any, targets: Any, loss_fn: Optional[Any]) -> torch.Tensor:
    if loss_fn is not None:
        return loss_fn(output, targets) if targets is not None else loss_fn(output)
    preferred = _extract_preferred_loss(output)
    if preferred is not None:
        return preferred
    return _reduce_output_to_scalar(output)


def _reduce_output_to_scalar(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        if output.is_floating_point() or output.is_complex():
            return output.sum()
        raise ValueError("Could not infer a scalar loss from a non-floating tensor output.")

    if isinstance(output, dict):
        terms = [_reduce_output_to_scalar(value) for value in output.values()]
    elif isinstance(output, (list, tuple)):
        terms = [_reduce_output_to_scalar(value) for value in output]
    else:
        raise ValueError("Could not infer a scalar loss from the replay output.")

    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def _extract_preferred_loss(output: Any) -> Optional[torch.Tensor]:
    if hasattr(output, "loss"):
        loss_value = getattr(output, "loss")
        if isinstance(loss_value, torch.Tensor) and (
            loss_value.is_floating_point() or loss_value.is_complex()
        ):
            return loss_value.sum()

    if isinstance(output, dict):
        loss_value = output.get("loss")
        if isinstance(loss_value, torch.Tensor) and (
            loss_value.is_floating_point() or loss_value.is_complex()
        ):
            return loss_value.sum()
        loss_dict = output.get("loss_dict")
        if isinstance(loss_dict, dict):
            return _reduce_output_to_scalar(loss_dict)

    if (
        isinstance(output, (list, tuple))
        and output
        and isinstance(output[0], torch.Tensor)
        and (output[0].is_floating_point() or output[0].is_complex())
    ):
        return output[0].sum()

    return None
