"""Training helpers built on top of execution-plan replay."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .replay_engine import (
    _collect_output_indices,
    _execute_nodes,
    _looks_like_boundary_payload,
    _pack_boundary_payload,
    _prepare_passthrough_seed_map,
    _prepare_runtime,
    _reconstruct_outputs,
    _split_boundary_mode_inputs,
)
from .replay_plan import ExecutionPlan, FrontierSplit
from .replay_utils import build_input_seed_map, tree_allclose


def train_partitioned(
    plan: ExecutionPlan,
    inputs: Any,
    *,
    split: Optional[FrontierSplit] = None,
    input_kwargs: Optional[Dict[str, Any]] = None,
    input_mode: str = "raw",
    device: Any = "auto",
    targets: Any = None,
    loss_fn: Optional[Any] = None,
    optimizer: Optional[Any] = None,
    return_boundary: bool = False,
    return_boundary_grad: bool = False,
    zero_grad: bool = True,
    step_optimizer: bool = True,
    preserve_rng: bool = True,
) -> Dict[str, Any]:
    """Run full-graph or partitioned replay with backward propagation.

    Args:
        plan: Compiled execution plan.
        inputs: Raw inputs or boundary tensors, depending on ``input_mode``.
        split: Optional frontier split. ``None`` means full-graph replay.
        input_kwargs: Optional replay keyword inputs for raw mode.
        input_mode: ``"raw"`` or ``"boundary"``.
        device: Target execution device selector.
        targets: Optional targets passed to ``loss_fn``.
        loss_fn: Optional custom loss function.
        optimizer: Optional optimizer stepped after backward.
        return_boundary: If True, include the boundary payload in the result.
        return_boundary_grad: If True, include boundary gradients in the result.
        zero_grad: Whether to zero the optimizer/model gradients before replay.
        step_optimizer: Whether to call ``optimizer.step()`` after backward.
        preserve_rng: Whether to restore per-node RNG states during replay.

    Returns:
        Dict containing at least ``output`` and ``loss``.
    """

    model, runtime_device = _prepare_runtime(plan, inputs, device)
    if zero_grad:
        _zero_grad(model, optimizer)

    if split is None:
        seed_map = _prepare_differentiable_seed_map(plan, inputs, input_kwargs, runtime_device)
        computed, _ = _execute_nodes(
            plan,
            node_indices=[node.idx for node in plan.nodes],
            seeded_values=seed_map,
            device=runtime_device,
            preserve_rng=preserve_rng,
            differentiable=True,
            retain_nodes=set(plan.output_node_indices) | _collect_output_indices(plan.output_specs),
            return_intermediates=False,
            model=model,
        )
        output = _reconstruct_outputs(plan, computed)
        loss = _compute_loss(output, targets=targets, loss_fn=loss_fn)
        loss.backward()
        if optimizer is not None and step_optimizer:
            optimizer.step()
        return {"output": output, "loss": loss}

    if input_mode == "raw":
        prefix_seed_map = _prepare_differentiable_seed_map(
            plan, inputs, input_kwargs, runtime_device
        )
        prefix_computed, _ = _execute_nodes(
            plan,
            node_indices=split.prefix_node_indices,
            seeded_values=prefix_seed_map,
            device=runtime_device,
            preserve_rng=preserve_rng,
            differentiable=True,
            retain_nodes=set(split.boundary_indices),
            return_intermediates=False,
            model=model,
        )
        boundary_tensors = {
            label: prefix_computed[idx]
            for label, idx in zip(split.boundary_labels, split.boundary_indices)
        }
        for tensor in boundary_tensors.values():
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.retain_grad()
        boundary_payload = (
            _pack_boundary_payload(plan, split, boundary_tensors, runtime_device)
            if return_boundary
            else None
        )

        suffix_seed_map = {idx: prefix_computed[idx] for idx in split.boundary_indices}
        suffix_seed_map.update(
            _prepare_passthrough_seed_map(
                plan,
                split,
                raw_inputs=inputs,
                input_kwargs=input_kwargs,
                device=runtime_device,
            )
        )
        suffix_computed, _ = _execute_nodes(
            plan,
            node_indices=split.suffix_node_indices,
            seeded_values=suffix_seed_map,
            device=runtime_device,
            preserve_rng=preserve_rng,
            differentiable=True,
            retain_nodes=set(plan.output_node_indices) | _collect_output_indices(plan.output_specs),
            return_intermediates=False,
            model=model,
        )
        prefix_computed.update(suffix_computed)
        output = _reconstruct_outputs(plan, prefix_computed)
        loss = _compute_loss(output, targets=targets, loss_fn=loss_fn)
        loss.backward()
        if optimizer is not None and step_optimizer:
            optimizer.step()

        result = {"output": output, "loss": loss}
        if return_boundary:
            result["boundary"] = boundary_payload
        if return_boundary_grad:
            result["boundary_grad"] = {
                label: tensor.grad
                for label, tensor in boundary_tensors.items()
                if isinstance(tensor, torch.Tensor)
            }
        return result

    if input_mode == "boundary":
        boundary_inputs, raw_inputs, boundary_input_kwargs = _split_boundary_mode_inputs(inputs)
        boundary_seed_map, boundary_tensors = _prepare_boundary_seed_map_differentiable(
            plan,
            split,
            boundary_inputs,
            runtime_device,
        )
        boundary_payload = (
            _pack_boundary_payload(plan, split, boundary_tensors, runtime_device)
            if return_boundary
            else None
        )
        boundary_seed_map.update(
            _prepare_passthrough_seed_map(
                plan,
                split,
                raw_inputs=raw_inputs,
                input_kwargs=boundary_input_kwargs,
                device=runtime_device,
            )
        )
        suffix_computed, _ = _execute_nodes(
            plan,
            node_indices=split.suffix_node_indices,
            seeded_values=boundary_seed_map,
            device=runtime_device,
            preserve_rng=preserve_rng,
            differentiable=True,
            retain_nodes=set(plan.output_node_indices) | _collect_output_indices(plan.output_specs),
            return_intermediates=False,
            model=model,
        )
        output = _reconstruct_outputs(plan, suffix_computed)
        loss = _compute_loss(output, targets=targets, loss_fn=loss_fn)
        loss.backward()
        if optimizer is not None and step_optimizer:
            optimizer.step()

        result = {"output": output, "loss": loss}
        if return_boundary:
            result["boundary"] = boundary_payload
        if return_boundary_grad:
            result["boundary_grad"] = {
                label: tensor.grad
                for label, tensor in boundary_tensors.items()
                if isinstance(tensor, torch.Tensor)
            }
        return result

    raise ValueError(f"Unsupported train_partitioned input_mode {input_mode!r}.")


def backward_prefix_from_boundary(
    plan: ExecutionPlan,
    raw_inputs: Any,
    split: FrontierSplit,
    boundary_grads: Any,
    *,
    input_kwargs: Optional[Dict[str, Any]] = None,
    device: Any = "auto",
    optimizer: Optional[Any] = None,
    zero_grad: bool = True,
    step_optimizer: bool = True,
    match_boundary: bool = False,
    cached_boundary: Any = None,
) -> Dict[str, Any]:
    """Recompute the prefix and backpropagate externally supplied boundary gradients."""

    model, runtime_device = _prepare_runtime(plan, raw_inputs, device)
    if zero_grad:
        _zero_grad(model, optimizer)

    prefix_seed_map = _prepare_differentiable_seed_map(
        plan, raw_inputs, input_kwargs, runtime_device
    )
    prefix_computed, _ = _execute_nodes(
        plan,
        node_indices=split.prefix_node_indices,
        seeded_values=prefix_seed_map,
        device=runtime_device,
        preserve_rng=True,
        differentiable=True,
        retain_nodes=set(split.boundary_indices),
        return_intermediates=False,
        model=model,
    )
    boundary_tensors = {
        label: prefix_computed[idx]
        for label, idx in zip(split.boundary_labels, split.boundary_indices)
    }
    boundary_grad_map = _normalize_boundary_grad_map(
        split,
        boundary_grads,
        boundary_tensors,
        runtime_device,
    )

    if match_boundary and cached_boundary is not None:
        cached_boundary_map = _coerce_boundary_label_map(split, cached_boundary)
        if not tree_allclose(boundary_tensors, cached_boundary_map):
            raise ValueError(
                "Recomputed boundary tensors did not match the cached boundary payload."
            )

    backprop_labels = [
        label
        for label in split.boundary_labels
        if isinstance(boundary_tensors[label], torch.Tensor) and boundary_tensors[label].requires_grad
    ]
    if backprop_labels:
        boundary_values = [boundary_tensors[label] for label in backprop_labels]
        grad_values = [boundary_grad_map[label] for label in backprop_labels]
        torch.autograd.backward(boundary_values, grad_tensors=grad_values)
    if optimizer is not None and step_optimizer:
        optimizer.step()

    return {
        "boundary": _pack_boundary_payload(plan, split, boundary_tensors, runtime_device),
        "boundary_grad": boundary_grad_map,
    }


def _prepare_differentiable_seed_map(
    plan: ExecutionPlan,
    inputs: Any,
    input_kwargs: Optional[Dict[str, Any]],
    device: torch.device,
) -> Dict[int, Any]:
    raw_seed_map = build_input_seed_map(plan.input_specs, inputs, input_kwargs)
    return {idx: _move_tree_for_grad(value, device) for idx, value in raw_seed_map.items()}


def _prepare_boundary_seed_map_differentiable(
    plan: ExecutionPlan,
    split: FrontierSplit,
    boundary_inputs: Any,
    device: torch.device,
) -> Tuple[Dict[int, Any], Dict[str, Any]]:
    boundary_map = _coerce_boundary_label_map(split, boundary_inputs)
    differentiable_boundary = {
        label: _detach_tree_for_boundary(value, device) for label, value in boundary_map.items()
    }
    for value in differentiable_boundary.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            value.requires_grad_(True)
            value.retain_grad()
    seed_map = {
        node_idx: differentiable_boundary[label]
        for label, node_idx in zip(split.boundary_labels, split.boundary_indices)
    }
    return seed_map, differentiable_boundary


def _move_tree_for_grad(tree: Any, device: torch.device) -> Any:
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        kwargs = {
            field.name: _move_tree_for_grad(getattr(tree, field.name), device)
            for field in dataclasses.fields(tree)
        }
        return type(tree)(**kwargs)
    if isinstance(tree, list):
        return [_move_tree_for_grad(item, device) for item in tree]
    if isinstance(tree, tuple):
        return type(tree)(_move_tree_for_grad(item, device) for item in tree)
    if isinstance(tree, dict):
        return {key: _move_tree_for_grad(value, device) for key, value in tree.items()}
    if isinstance(tree, torch.Tensor):
        if tree.device == device:
            return tree
        moved = tree.detach().to(device)
        if tree.requires_grad:
            moved.requires_grad_(True)
        return moved
    return tree


def _detach_tree_for_boundary(tree: Any, device: torch.device) -> Any:
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        kwargs = {
            field.name: _detach_tree_for_boundary(getattr(tree, field.name), device)
            for field in dataclasses.fields(tree)
        }
        return type(tree)(**kwargs)
    if isinstance(tree, list):
        return [_detach_tree_for_boundary(item, device) for item in tree]
    if isinstance(tree, tuple):
        return type(tree)(_detach_tree_for_boundary(item, device) for item in tree)
    if isinstance(tree, dict):
        return {key: _detach_tree_for_boundary(value, device) for key, value in tree.items()}
    if isinstance(tree, torch.Tensor):
        return tree.detach().clone().to(device)
    return tree


def _compute_loss(output: Any, *, targets: Any, loss_fn: Optional[Any]) -> torch.Tensor:
    if loss_fn is not None:
        return loss_fn(output, targets) if targets is not None else loss_fn(output)

    preferred = _extract_preferred_loss(output)
    if preferred is not None:
        return preferred

    reduced = _reduce_output_to_scalar(output)
    if reduced is None:
        raise ValueError("Could not infer a scalar loss from the replay output.")
    return reduced


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
            reduced = _reduce_output_to_scalar(loss_dict)
            if reduced is not None:
                return reduced

    if (
        isinstance(output, (list, tuple))
        and output
        and isinstance(output[0], torch.Tensor)
        and (output[0].is_floating_point() or output[0].is_complex())
    ):
        return output[0].sum()

    return None


def _reduce_output_to_scalar(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        if output.is_floating_point() or output.is_complex():
            return output.sum()
        return None

    if isinstance(output, dict):
        terms = [_reduce_output_to_scalar(value) for value in output.values()]
    elif isinstance(output, (list, tuple)):
        terms = [_reduce_output_to_scalar(value) for value in output]
    else:
        return None

    valid_terms = [term for term in terms if term is not None]
    if not valid_terms:
        return None

    total = valid_terms[0]
    for term in valid_terms[1:]:
        total = total + term
    return total


def _coerce_boundary_label_map(split: FrontierSplit, boundary_inputs: Any) -> Dict[str, Any]:
    if _looks_like_boundary_payload(boundary_inputs):
        payload = boundary_inputs
        if payload["cut_id"] != split.split_id:
            raise ValueError(
                f"Boundary payload cut_id {payload['cut_id']!r} does not match split {split.split_id!r}."
            )
        boundary_map = payload["tensors"]
    elif isinstance(boundary_inputs, dict):
        boundary_map = boundary_inputs
    elif isinstance(boundary_inputs, (list, tuple)):
        if len(boundary_inputs) != len(split.boundary_labels):
            raise ValueError(
                "Boundary input length does not match split boundary size: "
                f"expected {len(split.boundary_labels)}, got {len(boundary_inputs)}."
            )
        boundary_map = dict(zip(split.boundary_labels, boundary_inputs))
    elif len(split.boundary_labels) == 1:
        boundary_map = {split.boundary_labels[0]: boundary_inputs}
    else:
        raise ValueError(
            "Boundary tensors must be provided as a payload, dict, or ordered sequence."
        )

    missing = [label for label in split.boundary_labels if label not in boundary_map]
    if missing:
        raise ValueError(f"Boundary tensors are missing labels required by the split: {missing}.")
    return {label: boundary_map[label] for label in split.boundary_labels}


def _normalize_boundary_grad_map(
    split: FrontierSplit,
    boundary_grads: Any,
    boundary_tensors: Dict[str, Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    grad_map = _coerce_boundary_label_map(split, boundary_grads)
    normalized = {}
    for label, grad in grad_map.items():
        if isinstance(grad, torch.Tensor):
            normalized[label] = grad.to(device)
            continue
        if grad is None and isinstance(boundary_tensors[label], torch.Tensor):
            normalized[label] = torch.zeros_like(boundary_tensors[label], device=device)
            continue
        if not isinstance(grad, torch.Tensor):
            raise ValueError(f"Boundary gradient for {label!r} must be a tensor.")
    return normalized


def _zero_grad(model: Optional[torch.nn.Module], optimizer: Optional[Any]) -> None:
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    elif model is not None:
        model.zero_grad(set_to_none=True)
