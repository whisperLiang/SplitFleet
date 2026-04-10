"""Shared utilities for execution-plan compilation, replay, and validation."""

from __future__ import annotations

import collections.abc
import copy
import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .utils.collections import assign_to_sequence_or_dict

OUTPUT_REF_TAG = "__tl_output_ref__"


@dataclass(frozen=True, slots=True)
class TreeSpec:
    """Container schema returned by :func:`tree_flatten`."""

    kind: str
    type_: Any = None
    context: Any = None
    children: Tuple["TreeSpec", ...] = ()


def tree_flatten(tree: Any) -> Tuple[List[Any], TreeSpec]:
    """Flatten a nested structure into leaves plus a reconstruction spec."""

    leaves: List[Any] = []
    spec = _tree_flatten_into(tree, leaves)
    return leaves, spec


def _tree_flatten_into(tree: Any, leaves: List[Any]) -> TreeSpec:
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        dataclass_children: List[TreeSpec] = []
        field_names: List[str] = []
        for field in dataclasses.fields(tree):
            field_names.append(field.name)
            dataclass_children.append(_tree_flatten_into(getattr(tree, field.name), leaves))
        return TreeSpec("dataclass", type(tree), tuple(field_names), tuple(dataclass_children))

    if isinstance(tree, list):
        children = tuple(_tree_flatten_into(item, leaves) for item in tree)
        return TreeSpec("list", list, None, children)

    if isinstance(tree, tuple):
        children = tuple(_tree_flatten_into(item, leaves) for item in tree)
        if hasattr(type(tree), "_fields"):
            return TreeSpec(
                "namedtuple", type(tree), tuple(getattr(type(tree), "_fields")), children
            )
        return TreeSpec("tuple", tuple, None, children)

    if isinstance(tree, collections.abc.Mapping):
        keys = tuple(tree.keys())
        children = tuple(_tree_flatten_into(tree[key], leaves) for key in keys)
        return TreeSpec("dict", type(tree), keys, children)

    leaves.append(tree)
    return TreeSpec("leaf")


def tree_unflatten(spec: TreeSpec, leaves: Sequence[Any]) -> Any:
    """Reconstruct a nested structure from leaves and a :class:`TreeSpec`."""

    iterator = iter(leaves)
    return _tree_unflatten_from_spec(spec, iterator)


def _tree_unflatten_from_spec(spec: TreeSpec, leaves: Iterator[Any]) -> Any:
    if spec.kind == "leaf":
        return next(leaves)

    child_values = [_tree_unflatten_from_spec(child_spec, leaves) for child_spec in spec.children]

    if spec.kind == "list":
        return child_values

    if spec.kind == "tuple":
        return tuple(child_values)

    if spec.kind == "namedtuple":
        return spec.type_(*child_values)

    if spec.kind == "dict":
        items = {key: value for key, value in zip(spec.context, child_values)}
        if spec.type_ is dict:
            return items
        try:
            return spec.type_(items)
        except Exception:
            return items

    if spec.kind == "dataclass":
        kwargs = {key: value for key, value in zip(spec.context, child_values)}
        return spec.type_(**kwargs)

    raise ValueError(f"Unsupported TreeSpec kind: {spec.kind}")


def tree_map(fn: Callable[[Any], Any], tree: Any) -> Any:
    """Apply ``fn`` to every leaf in ``tree`` and rebuild the structure."""

    leaves, spec = tree_flatten(tree)
    mapped = [fn(leaf) for leaf in leaves]
    return tree_unflatten(spec, mapped)


def tree_to_device(tree: Any, device: torch.device) -> Any:
    """Move every tensor leaf in ``tree`` to ``device``."""

    def _move(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            return leaf.to(device)
        return leaf

    return tree_map(_move, tree)


def tree_allclose(
    left: Any,
    right: Any,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> bool:
    """Recursively compare nested outputs, including tensor leaves."""

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if left.dtype.is_floating_point or left.dtype.is_complex:
            return bool(torch.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True))
        return bool(torch.equal(left, right))

    if dataclasses.is_dataclass(left) and dataclasses.is_dataclass(right):
        return all(
            tree_allclose(
                getattr(left, field.name), getattr(right, field.name), atol=atol, rtol=rtol
            )
            for field in dataclasses.fields(left)
        )

    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            tree_allclose(l_item, r_item, atol=atol, rtol=rtol)
            for l_item, r_item in zip(left, right)
        )

    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            tree_allclose(l_item, r_item, atol=atol, rtol=rtol)
            for l_item, r_item in zip(left, right)
        )

    if isinstance(left, collections.abc.Mapping) and isinstance(right, collections.abc.Mapping):
        if tuple(left.keys()) != tuple(right.keys()):
            return False
        return all(
            tree_allclose(left[key], right[key], atol=atol, rtol=rtol) for key in left.keys()
        )

    return left == right


def clone_tree_tensors(
    tree: Any,
    *,
    device: Optional[torch.device] = None,
    detach: bool,
) -> Any:
    """Clone all tensors in a tree, optionally detaching and moving them."""

    memo: Dict[int, Any] = {}
    return _clone_tree_tensors_with_memo(tree, memo, device=device, detach=detach)


def _clone_tree_tensors_with_memo(
    tree: Any,
    memo: Dict[int, Any],
    *,
    device: Optional[torch.device],
    detach: bool,
) -> Any:
    tree_id = id(tree)
    if tree_id in memo:
        return memo[tree_id]

    if isinstance(tree, torch.Tensor):
        cloned = tree.detach().clone() if detach else tree.clone()
        if device is not None:
            cloned = cloned.to(device)
        memo[tree_id] = cloned
        return cloned

    if isinstance(tree, list):
        cloned_list: List[Any] = []
        memo[tree_id] = cloned_list
        cloned_list.extend(
            _clone_tree_tensors_with_memo(item, memo, device=device, detach=detach) for item in tree
        )
        return cloned_list

    if isinstance(tree, tuple):
        placeholder: List[Any] = []
        memo[tree_id] = placeholder
        cloned_tuple = type(tree)(
            _clone_tree_tensors_with_memo(item, memo, device=device, detach=detach) for item in tree
        )
        memo[tree_id] = cloned_tuple
        return cloned_tuple

    if isinstance(tree, dict):
        cloned_dict: Dict[Any, Any] = {}
        memo[tree_id] = cloned_dict
        cloned_dict.update(
            {
                key: _clone_tree_tensors_with_memo(value, memo, device=device, detach=detach)
                for key, value in tree.items()
            }
        )
        return cloned_dict

    memo[tree_id] = copy.copy(tree)
    return memo[tree_id]


def clone_constant_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Clone a tensor constant into CPU memory for device-agnostic plan storage."""

    cloned = tensor.detach().clone()
    if cloned.device.type != "cpu":
        cloned = cloned.cpu()
    return cloned


def get_first_tensor(tree: Any) -> Optional[torch.Tensor]:
    """Return the first tensor leaf found in ``tree``."""

    if isinstance(tree, torch.Tensor):
        return tree

    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        for field in dataclasses.fields(tree):
            tensor = get_first_tensor(getattr(tree, field.name))
            if tensor is not None:
                return tensor
        return None

    if isinstance(tree, collections.abc.Mapping):
        for value in tree.values():
            tensor = get_first_tensor(value)
            if tensor is not None:
                return tensor
        return None

    if isinstance(tree, (list, tuple)):
        for value in tree:
            tensor = get_first_tensor(value)
            if tensor is not None:
                return tensor
        return None

    return None


def get_model_device(model: nn.Module) -> torch.device:
    """Infer a model's current device from its parameters or buffers."""

    first_param = next(model.parameters(), None)
    if first_param is not None:
        return first_param.device
    first_buffer = next(model.buffers(), None)
    if first_buffer is not None:
        return first_buffer.device
    return torch.device("cpu")


def resolve_device(
    device: Any,
    *,
    model: Optional[nn.Module] = None,
    inputs: Any = None,
    default: Optional[Any] = None,
) -> torch.device:
    """Resolve a public device selector into a concrete :class:`torch.device`."""

    if isinstance(device, torch.device):
        return device

    if device is None:
        device = default if default is not None else "auto"

    if device == "same_as_model":
        if model is not None:
            return get_model_device(model)
        return torch.device("cpu")

    if device == "same_as_input":
        first_tensor = get_first_tensor(inputs)
        if first_tensor is not None:
            return first_tensor.device
        if model is not None:
            return get_model_device(model)
        return torch.device("cpu")

    if device == "auto":
        if model is not None:
            return get_model_device(model)
        first_tensor = get_first_tensor(inputs)
        if first_tensor is not None:
            return first_tensor.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def ensure_model_device(model: nn.Module, device: torch.device) -> nn.Module:
    """Move a model to ``device`` if it is not already there."""

    if get_model_device(model) != device:
        model.to(device)
    return model


def build_output_structure_template(
    outputs: Any,
    *,
    tensor_encoder: Optional[Callable[[torch.Tensor], Any]] = None,
) -> Any:
    """Capture a nested output structure while replacing tensor leaves with references."""

    if isinstance(outputs, torch.Tensor):
        if tensor_encoder is None:
            return outputs
        return tensor_encoder(outputs)

    if dataclasses.is_dataclass(outputs) and not isinstance(outputs, type):
        return {
            "__tl_dataclass_type__": type(outputs),
            "items": [
                (
                    field.name,
                    build_output_structure_template(
                        getattr(outputs, field.name), tensor_encoder=tensor_encoder
                    ),
                )
                for field in dataclasses.fields(outputs)
            ],
        }

    if isinstance(outputs, list):
        return [
            build_output_structure_template(item, tensor_encoder=tensor_encoder) for item in outputs
        ]

    if isinstance(outputs, tuple):
        return {
            "__tl_tuple_type__": type(outputs),
            "items": [
                build_output_structure_template(item, tensor_encoder=tensor_encoder)
                for item in outputs
            ],
        }

    if isinstance(outputs, collections.abc.Mapping):
        return {
            "__tl_dict_type__": type(outputs),
            "items": [
                (key, build_output_structure_template(value, tensor_encoder=tensor_encoder))
                for key, value in outputs.items()
            ],
        }

    return outputs


def reconstruct_from_template(template: Any, leaf_resolver: Callable[[Any], Any]) -> Any:
    """Rebuild a nested structure from a stored template."""

    if isinstance(template, tuple) and len(template) == 2 and template[0] == OUTPUT_REF_TAG:
        return leaf_resolver(template[1])

    if isinstance(template, list):
        return [reconstruct_from_template(item, leaf_resolver) for item in template]

    if isinstance(template, dict):
        if "__tl_tuple_type__" in template:
            items = [reconstruct_from_template(item, leaf_resolver) for item in template["items"]]
            tuple_type = template["__tl_tuple_type__"]
            if tuple_type is tuple:
                return tuple(items)
            return tuple_type(*items)

        if "__tl_dict_type__" in template:
            rebuilt = {
                key: reconstruct_from_template(value, leaf_resolver)
                for key, value in template["items"]
            }
            dict_type = template["__tl_dict_type__"]
            if dict_type is dict:
                return rebuilt
            try:
                return dict_type(rebuilt)
            except Exception:
                return rebuilt

        if "__tl_dataclass_type__" in template:
            dataclass_type = template["__tl_dataclass_type__"]
            kwargs = {
                key: reconstruct_from_template(value, leaf_resolver)
                for key, value in template["items"]
            }
            return dataclass_type(**kwargs)

        return {
            key: reconstruct_from_template(value, leaf_resolver) for key, value in template.items()
        }

    return template


def translate_output_template(template: Any, label_to_value: Dict[str, Any]) -> Any:
    """Reconstruct an output structure by label lookup."""

    return reconstruct_from_template(template, lambda label: label_to_value[label])


def hash_graph_signature(payload: Any) -> str:
    """Return a stable SHA-1 hash for a JSON-serializable payload."""

    payload_repr = repr(payload).encode("utf-8")
    return hashlib.sha1(payload_repr).hexdigest()


def canonicalize_batch_agnostic_shape(
    shape: Optional[Sequence[int]],
) -> Optional[Tuple[Any, ...]]:
    """Normalize a tensor shape so the leading batch dimension is shape-agnostic."""

    if shape is None:
        return None

    shape_tuple = tuple(shape)
    if not shape_tuple:
        return shape_tuple

    return ("B",) + shape_tuple[1:]


def parse_input_io_role(io_role: str) -> Tuple[str, Tuple[Any, ...]]:
    """Parse an input-layer role like ``input.x.0`` into root name and nested path."""

    if not io_role.startswith("input."):
        raise ValueError(f"Expected an input io_role, got {io_role!r}.")

    parts = io_role.split(".")[1:]
    root_name = parts[0]
    nested_path = tuple(int(part) if part.isdigit() else part for part in parts[1:])
    return root_name, nested_path


def normalize_positional_inputs(
    inputs: Any,
    expected_num_positional_args: int,
) -> List[Any]:
    """Normalize raw replay inputs into positional root values."""

    if isinstance(inputs, tuple):
        if expected_num_positional_args == 1 and len(inputs) != 1:
            return [inputs]
        return list(inputs)

    if isinstance(inputs, list):
        if expected_num_positional_args == 1 and len(inputs) != 1:
            return [inputs]
        return list(inputs)

    if inputs is None:
        return []

    return [inputs]


def index_nested(obj: Any, path: Sequence[Any]) -> Any:
    """Index into a nested structure using ``path``."""

    output = obj
    for key in path:
        output = output[key]
    return output


def apply_value_at_location(container: Any, location: Any, value: Any) -> Any:
    """Assign ``value`` at ``location`` in a possibly nested container."""

    if not isinstance(location, tuple):
        return assign_to_sequence_or_dict(container, location, value)

    updated_child = apply_value_at_location(container[location[0]], location[1], value)
    return assign_to_sequence_or_dict(container, location[0], updated_child)


def build_input_seed_map(
    input_specs: Sequence[Dict[str, Any]],
    inputs: Any,
    input_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[int, Any]:
    """Map replay-time raw inputs onto compiled input-node indices."""

    if input_kwargs is None:
        input_kwargs = {}

    ordered_root_names: List[str] = []
    seen_root_names = set()
    for spec in input_specs:
        io_role = spec.get("io_role")
        if io_role is None:
            continue
        root_name, _ = parse_input_io_role(io_role)
        if root_name not in seen_root_names:
            seen_root_names.add(root_name)
            ordered_root_names.append(root_name)

    kwarg_root_names = {name for name in ordered_root_names if name in input_kwargs}
    positional_root_names = [name for name in ordered_root_names if name not in kwarg_root_names]
    positional_inputs = normalize_positional_inputs(inputs, len(positional_root_names))

    if len(positional_inputs) < len(positional_root_names):
        raise ValueError(
            "Not enough replay inputs provided: "
            f"expected {len(positional_root_names)} positional roots, got {len(positional_inputs)}."
        )

    positional_by_name = {
        root_name: positional_inputs[index] for index, root_name in enumerate(positional_root_names)
    }

    seed_map: Dict[int, Any] = {}
    for spec in input_specs:
        io_role = spec.get("io_role")
        if io_role is None:
            continue
        root_name, nested_path = parse_input_io_role(io_role)
        if root_name in input_kwargs:
            root_value = input_kwargs[root_name]
        elif root_name in positional_by_name:
            root_value = positional_by_name[root_name]
        else:
            raise ValueError(
                f"Missing replay input value for input root {root_name!r} required by {spec['label']!r}."
            )

        seed_map[spec["idx"]] = index_nested(root_value, nested_path) if nested_path else root_value

    return seed_map
