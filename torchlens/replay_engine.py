"""Execution-plan compilation and lightweight forward replay."""

from __future__ import annotations

import copy
import dataclasses
import inspect
import weakref
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

import torch
from torch import nn

from .data_classes.layer_pass_log import LayerPassLog
from .replay_plan import (
    BoundaryPayload,
    BufferRef,
    ExecNode,
    ExecutionPlan,
    FrontierSplit,
    ParamRef,
    ParentRef,
    TensorConst,
)
from .replay_utils import (
    OUTPUT_REF_TAG,
    apply_value_at_location,
    build_input_seed_map,
    canonicalize_batch_agnostic_shape,
    clone_constant_tensor,
    clone_tree_tensors,
    hash_graph_signature,
    index_nested,
    normalize_positional_inputs,
    reconstruct_from_template,
    resolve_device,
    tree_allclose,
)
from .utils.rng import AutocastRestore, log_current_rng_states, set_rng_from_saved_states

_BATCH_DYNAMIC_RESHAPE_FUNCS = {"_reshape_alias", "reshape", "view"}
_BATCH_DYNAMIC_FACTORY_FUNCS = {
    "arange",
    "empty",
    "new_empty",
    "new_ones",
    "new_zeros",
    "ones",
    "rand",
    "randn",
    "zeros",
}
_RNG_SENSITIVE_FUNC_NAMES = {
    "bernoulli",
    "bernoulli_",
    "dropout",
    "dropout_",
    "dropout1d",
    "dropout2d",
    "dropout3d",
    "alpha_dropout",
    "alpha_dropout_",
    "alphadropout",
    "feature_alpha_dropout",
    "feature_alpha_dropout_",
    "featurealphadropout",
    "feature_dropout",
    "feature_dropout_",
    "gumbel_softmax",
    "multinomial",
    "native_dropout",
    "normal",
    "normal_",
    "poisson",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
    "rrelu",
    "rrelu_",
}
_TRAINING_FLAG_CACHE: Dict[Any, Tuple[Optional[str], Optional[int]]] = {}
_MISSING = object()
_LIVE_STATE_MAP_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class _ComputedStore(Protocol):
    def __contains__(self, idx: object) -> bool: ...

    def __getitem__(self, idx: int) -> Any: ...

    def get(self, idx: int, default: Any = None) -> Any: ...

    def pop(self, idx: int, default: Any = None) -> Any: ...

    def values(self) -> Iterable[Any]: ...


class _NodeValueStore:
    __slots__ = ("_values",)

    def __init__(self, size: int, seeded_values: Optional[Dict[int, Any]] = None) -> None:
        self._values = [_MISSING] * size
        if seeded_values is not None:
            for idx, value in seeded_values.items():
                self._values[idx] = value

    def __contains__(self, idx: object) -> bool:
        return isinstance(idx, int) and 0 <= idx < len(self._values) and self._values[idx] is not _MISSING

    def __getitem__(self, idx: int) -> Any:
        value = self._values[idx]
        if value is _MISSING:
            raise KeyError(idx)
        return value

    def __setitem__(self, idx: int, value: Any) -> None:
        self._values[idx] = value

    def get(self, idx: int, default: Any = None) -> Any:
        value = self._values[idx]
        return default if value is _MISSING else value

    def pop(self, idx: int, default: Any = None) -> Any:
        value = self._values[idx]
        if value is _MISSING:
            return default
        self._values[idx] = _MISSING
        return value

    def values(self) -> Iterable[Any]:
        return (value for value in self._values if value is not _MISSING)

    def materialize(self, indices: Iterable[int]) -> Dict[int, Any]:
        return {
            idx: self._values[idx]
            for idx in indices
            if 0 <= idx < len(self._values) and self._values[idx] is not _MISSING
        }


def compile_execution_plan(
    model: nn.Module,
    example_inputs: Any,
    *,
    input_kwargs: Optional[Dict[str, Any]] = None,
    device: Any = "auto",
    trace_on_device: bool = True,
    retrace_if_needed: bool = True,
    preserve_rng: bool = True,
    store_minimal_metadata: bool = True,
    strict: bool = True,
) -> ExecutionPlan:
    """Trace one concrete execution graph and compile a lightweight replay plan."""

    from .user_funcs import log_forward_pass

    trace_device = resolve_device(device, model=model, inputs=example_inputs)
    if trace_on_device:
        model.to(trace_device)
        example_inputs = _move_tree_to_device(example_inputs, trace_device)
        if input_kwargs is not None:
            input_kwargs = _move_tree_to_device(input_kwargs, trace_device)

    model_log = log_forward_pass(
        model,
        example_inputs,
        input_kwargs=input_kwargs,
        layers_to_save="all",
        save_function_args=True,
        save_rng_states=preserve_rng,
    )

    try:
        label_to_idx = {
            layer.layer_label: index for index, layer in enumerate(model_log.layer_list)
        }
        raw_to_final = getattr(model_log, "_raw_to_final_layer_labels", {})
        nodes: List[ExecNode] = []
        input_specs: List[Dict[str, Any]] = []
        output_node_indices: List[int] = []
        buffer_node_indices: List[int] = []

        for index, layer in enumerate(model_log.layer_list):
            parents = [label_to_idx[parent_label] for parent_label in layer.parent_layers]
            parent_refs, parent_arg_locs = _build_parent_refs(layer, label_to_idx)
            args_template = _convert_args_to_template(layer.captured_args)
            kwargs_template = _convert_args_to_template(layer.captured_kwargs or {})

            for location, parent_ref in parent_refs["args"].items():
                args_template = _apply_placeholder(args_template, location, parent_ref)
            for location, parent_ref in parent_refs["kwargs"].items():
                kwargs_template = _apply_placeholder(kwargs_template, location, parent_ref)
            inferred_parent_refs = _infer_missing_parent_refs(
                layer=layer,
                layer_index=index,
                previous_layers=model_log.layer_list[:index],
                label_to_idx=label_to_idx,
                args_template=args_template,
                kwargs_template=kwargs_template,
            )
            for location, parent_ref in inferred_parent_refs["args"].items():
                args_template = _apply_placeholder(args_template, location, parent_ref)
                parents.append(parent_ref.node_idx)
                parent_arg_locs.append(
                    {
                        "kind": "args",
                        "location": location,
                        "parent_idx": parent_ref.node_idx,
                        "path": parent_ref.path,
                        "inferred": True,
                    }
                )
            for location, parent_ref in inferred_parent_refs["kwargs"].items():
                kwargs_template = _apply_placeholder(kwargs_template, location, parent_ref)
                parents.append(parent_ref.node_idx)
                parent_arg_locs.append(
                    {
                        "kind": "kwargs",
                        "location": location,
                        "parent_idx": parent_ref.node_idx,
                        "path": parent_ref.path,
                        "inferred": True,
                    }
                )
            parents = sorted(set(parents))

            num_users = sum(
                1 for other in model_log.layer_list if layer.layer_label in other.parent_layers
            )
            node_meta = _build_node_meta(layer, store_minimal_metadata=store_minimal_metadata)
            exec_node = ExecNode(
                idx=index,
                label=layer.layer_label,
                op=layer.func_applied if callable(layer.func_applied) else None,
                parents=parents,
                parent_arg_locs=parent_arg_locs,
                const_args_template=args_template,
                const_kwargs_template=kwargs_template,
                is_input=layer.is_input_layer,
                is_output=layer.is_output_layer,
                is_buffer=layer.is_buffer_layer,
                is_internal_init=layer.is_internally_initialized,
                is_inplace=layer.func_is_inplace,
                rng_state=layer.func_rng_states if preserve_rng else None,
                output_selector=layer.iterable_output_index,
                num_users=num_users,
                meta=node_meta,
            )
            nodes.append(exec_node)

            if layer.is_input_layer:
                input_specs.append(
                    {
                        "idx": index,
                        "label": layer.layer_label,
                        "io_role": layer.io_role,
                        "shape": tuple(layer.tensor_shape)
                        if layer.tensor_shape is not None
                        else None,
                        "dtype": str(layer.tensor_dtype)
                        if layer.tensor_dtype is not None
                        else None,
                    }
                )

            if layer.is_output_layer:
                output_node_indices.append(index)

            if layer.is_buffer_layer:
                buffer_node_indices.append(index)

        output_template = getattr(model_log, "_output_structure_template", None)
        if output_template is None:
            output_specs = _default_output_specs(model_log, label_to_idx)
        else:
            output_specs = _translate_output_template_to_indices(
                output_template,
                label_to_idx,
                raw_to_final,
            )

        plan = ExecutionPlan(
            nodes=nodes,
            node_to_index=label_to_idx,
            input_specs=input_specs,
            output_specs=output_specs,
            graph_signature="",
            model_name=model_log.model_name,
            default_device=str(trace_device),
            shape_constraints={
                spec["label"]: spec["shape"]
                for spec in input_specs
                if spec.get("shape") is not None
            },
            dtype_constraints={
                spec["label"]: spec["dtype"]
                for spec in input_specs
                if spec.get("dtype") is not None
            },
            meta={
                "trace_device": str(trace_device),
                "has_rng_nodes": any(bool(node.meta.get("uses_rng")) for node in nodes),
                "preserve_rng": preserve_rng,
                "retrace_if_needed": retrace_if_needed,
                "strict": strict,
                "pre_forward_rng_states": getattr(model_log, "_pre_forward_rng_states", None),
                "raw_output_layers": list(model_log.output_layers),
            },
            input_node_indices=[spec["idx"] for spec in input_specs],
            output_node_indices=output_node_indices,
            buffer_node_indices=buffer_node_indices,
        )
        plan.set_model(model)
        plan.graph_signature = _build_graph_signature(plan)
        return plan
    finally:
        model_log.cleanup()


def _move_tree_to_device(tree: Any, device: torch.device) -> Any:
    return _map_tree(tree, lambda leaf: leaf.to(device) if isinstance(leaf, torch.Tensor) else leaf)


def _map_tree(tree: Any, fn) -> Any:
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        kwargs = {
            field.name: _map_tree(getattr(tree, field.name), fn)
            for field in dataclasses.fields(tree)
        }
        return type(tree)(**kwargs)
    if isinstance(tree, list):
        return [_map_tree(item, fn) for item in tree]
    if isinstance(tree, tuple):
        return type(tree)(_map_tree(item, fn) for item in tree)
    if isinstance(tree, dict):
        return {key: _map_tree(value, fn) for key, value in tree.items()}
    return fn(tree)


def _build_parent_refs(
    layer: LayerPassLog,
    label_to_idx: Dict[str, int],
) -> Tuple[Dict[str, Dict[Any, ParentRef]], List[Dict[str, Any]]]:
    parent_refs: Dict[str, Dict[Any, ParentRef]] = {"args": {}, "kwargs": {}}
    parent_arg_locs: List[Dict[str, Any]] = []

    for kind in ("args", "kwargs"):
        for location, parent_label in layer.parent_layer_arg_locs[kind].items():
            parent_layer = layer.source_model_log[parent_label]
            parent_idx = label_to_idx[parent_label]
            path = _infer_parent_output_path(parent_layer, layer.layer_label)
            frozen_value = _infer_child_specific_frozen_value(
                parent_layer=parent_layer,
                child_label=layer.layer_label,
                captured_container=layer.captured_args if kind == "args" else layer.captured_kwargs,
                location=location,
            )
            parent_ref = ParentRef(parent_idx, path, frozen_value)
            parent_refs[kind][location] = parent_ref
            parent_arg_locs.append(
                {
                    "kind": kind,
                    "location": location,
                    "parent_idx": parent_idx,
                    "path": path,
                    "has_frozen_value": frozen_value is not None,
                }
            )

    return parent_refs, parent_arg_locs


def _find_tensor_path(container: Any, target: torch.Tensor) -> Optional[Tuple[Any, ...]]:
    if isinstance(container, torch.Tensor):
        if (
            container.shape == target.shape
            and container.dtype == target.dtype
            and torch.equal(container, target)
        ):
            return ()
        return None

    if isinstance(container, (list, tuple)):
        for index, value in enumerate(container):
            subpath = _find_tensor_path(value, target)
            if subpath is not None:
                return (index,) + subpath
        return None

    if isinstance(container, dict):
        for key, value in container.items():
            subpath = _find_tensor_path(value, target)
            if subpath is not None:
                return (key,) + subpath
        return None

    return None


def _infer_parent_output_path(
    parent_layer: LayerPassLog, child_label: str
) -> Optional[Tuple[Any, ...]]:
    if child_label not in parent_layer.children_tensor_versions:
        return None
    child_saved_value = parent_layer.children_tensor_versions[child_label]
    if not isinstance(child_saved_value, torch.Tensor) or parent_layer.activation is None:
        return None
    return _find_tensor_path(parent_layer.activation, child_saved_value)


def _location_to_path(location: Any) -> Tuple[Any, ...]:
    if isinstance(location, tuple):
        return (location[0],) + _location_to_path(location[1])
    return (location,)


def _infer_child_specific_frozen_value(
    *,
    parent_layer: LayerPassLog,
    child_label: str,
    captured_container: Any,
    location: Any,
) -> Optional[Any]:
    child_saved_value = parent_layer.children_tensor_versions.get(child_label)
    if not isinstance(child_saved_value, torch.Tensor):
        return None
    if child_saved_value.requires_grad:
        return None
    if _infer_parent_output_path(parent_layer, child_label) is not None:
        return None

    try:
        captured_value = index_nested(captured_container, _location_to_path(location))
    except Exception:
        return None

    if not _tensor_exact_match(child_saved_value, captured_value):
        return None

    return clone_constant_tensor(child_saved_value)


def _convert_args_to_template(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        if hasattr(value, "tl_param_address"):
            return ParamRef(getattr(value, "tl_param_address"))
        if hasattr(value, "tl_buffer_address"):
            return BufferRef(getattr(value, "tl_buffer_address"))
        return TensorConst(clone_constant_tensor(value))

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        kwargs = {
            field.name: _convert_args_to_template(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
        return type(value)(**kwargs)

    if isinstance(value, list):
        return [_convert_args_to_template(item) for item in value]

    if isinstance(value, tuple):
        return type(value)(_convert_args_to_template(item) for item in value)

    if isinstance(value, dict):
        return {key: _convert_args_to_template(item) for key, item in value.items()}

    return copy.copy(value)


def _apply_placeholder(template: Any, location: Any, placeholder: ParentRef) -> Any:
    if template is None:
        return template
    return apply_value_at_location(template, location, placeholder)


def _infer_missing_parent_refs(
    *,
    layer: LayerPassLog,
    layer_index: int,
    previous_layers: Sequence[LayerPassLog],
    label_to_idx: Dict[str, int],
    args_template: Any,
    kwargs_template: Any,
) -> Dict[str, Dict[Any, ParentRef]]:
    inferred: Dict[str, Dict[Any, ParentRef]] = {"args": {}, "kwargs": {}}
    current_label = layer.layer_label

    for kind, template in (("args", args_template), ("kwargs", kwargs_template)):
        for path, tensor_const in _iter_tensor_const_paths(template):
            parent_ref = _find_matching_parent_ref(
                tensor_const.tensor,
                current_label=current_label,
                previous_layers=previous_layers,
                label_to_idx=label_to_idx,
            )
            if parent_ref is None:
                continue
            inferred[kind][_path_to_location(path)] = parent_ref

    return inferred


def _iter_tensor_const_paths(
    value: Any,
    path: Tuple[Any, ...] = (),
) -> List[Tuple[Tuple[Any, ...], TensorConst]]:
    matches: List[Tuple[Tuple[Any, ...], TensorConst]] = []

    if isinstance(value, TensorConst):
        matches.append((path, value))
        return matches

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            matches.extend(
                _iter_tensor_const_paths(
                    getattr(value, field.name),
                    path + (field.name,),
                )
            )
        return matches

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            matches.extend(_iter_tensor_const_paths(item, path + (index,)))
        return matches

    if isinstance(value, dict):
        for key, item in value.items():
            matches.extend(_iter_tensor_const_paths(item, path + (key,)))

    return matches


def _path_to_location(path: Tuple[Any, ...]) -> Any:
    if not path:
        raise ValueError("Tensor placeholder paths must not be empty.")
    if len(path) == 1:
        return path[0]
    return (path[0], _path_to_location(path[1:]))


def _find_matching_parent_ref(
    target: torch.Tensor,
    *,
    current_label: str,
    previous_layers: Sequence[LayerPassLog],
    label_to_idx: Dict[str, int],
) -> Optional[ParentRef]:
    for parent_layer in reversed(previous_layers):
        child_version = parent_layer.children_tensor_versions.get(current_label)
        if _tensor_exact_match(child_version, target):
            path = _find_tensor_path(parent_layer.activation, child_version)
            frozen_value = None
            if path is None and not child_version.requires_grad:
                frozen_value = clone_constant_tensor(child_version)
            return ParentRef(label_to_idx[parent_layer.layer_label], path or None, frozen_value)

        path = _find_tensor_path(parent_layer.activation, target)
        if path is not None:
            return ParentRef(label_to_idx[parent_layer.layer_label], path or None)

    return None


def _tensor_exact_match(candidate: Any, target: torch.Tensor) -> bool:
    return (
        isinstance(candidate, torch.Tensor)
        and candidate.shape == target.shape
        and candidate.dtype == target.dtype
        and torch.equal(candidate, target)
    )


def _func_uses_rng(func_name: Optional[str]) -> bool:
    if not isinstance(func_name, str):
        return False
    if func_name in _RNG_SENSITIVE_FUNC_NAMES:
        return True
    if func_name.startswith("rand"):
        return True
    return "dropout" in func_name


def _resolve_training_flag_binding(func: Any) -> Tuple[Optional[str], Optional[int]]:
    if func is None or not callable(func):
        return None, None

    cached = _TRAINING_FLAG_CACHE.get(func)
    if cached is not None:
        return cached

    training_binding: Tuple[Optional[str], Optional[int]] = (None, None)
    try:
        signature = inspect.signature(func)
    except Exception:
        _TRAINING_FLAG_CACHE[func] = training_binding
        return training_binding

    param_names = list(signature.parameters.keys())
    for training_name in ("training", "train"):
        if training_name in param_names:
            training_binding = (training_name, param_names.index(training_name))
            break

    _TRAINING_FLAG_CACHE[func] = training_binding
    return training_binding


def _build_node_meta(
    layer: LayerPassLog,
    *,
    store_minimal_metadata: bool,
) -> Dict[str, Any]:
    training_flag_name, training_flag_index = _resolve_training_flag_binding(layer.func_applied)
    meta = {
        "func_name": layer.func_name,
        "tensor_shape": tuple(layer.tensor_shape) if layer.tensor_shape is not None else None,
        "tensor_dtype": str(layer.tensor_dtype) if layer.tensor_dtype is not None else None,
        "io_role": layer.io_role,
        "buffer_address": layer.buffer_address,
        "buffer_parent": layer.buffer_parent,
        "func_autocast_state": layer.func_autocast_state,
        "func_config": layer.func_config,
        "uses_rng": _func_uses_rng(layer.func_name),
        "training_flag_name": training_flag_name,
        "training_flag_index": training_flag_index,
    }
    if not store_minimal_metadata:
        meta.update(
            {
                "operation_equivalence_type": layer.operation_equivalence_type,
                "containing_module": layer.containing_module,
                "containing_modules": list(layer.containing_modules),
                "parent_param_shapes": list(layer.parent_param_shapes),
                "parent_param_addresses": [
                    param_log.address for param_log in layer.parent_param_logs
                ],
            }
        )
    return meta


def _translate_output_template_to_indices(
    template: Any,
    label_to_idx: Dict[str, int],
    raw_to_final: Dict[str, str],
) -> Any:
    if isinstance(template, tuple) and len(template) == 2 and template[0] == OUTPUT_REF_TAG:
        raw_label = template[1]
        if not isinstance(raw_label, str):
            raise ValueError("Output template references must resolve to layer-label strings.")
        final_label = raw_to_final.get(raw_label, raw_label)
        return (OUTPUT_REF_TAG, label_to_idx[final_label])

    if isinstance(template, list):
        return [
            _translate_output_template_to_indices(item, label_to_idx, raw_to_final)
            for item in template
        ]

    if isinstance(template, dict):
        if "__tl_tuple_type__" in template:
            return {
                "__tl_tuple_type__": template["__tl_tuple_type__"],
                "items": [
                    _translate_output_template_to_indices(item, label_to_idx, raw_to_final)
                    for item in template["items"]
                ],
            }
        if "__tl_dict_type__" in template:
            return {
                "__tl_dict_type__": template["__tl_dict_type__"],
                "items": [
                    (
                        key,
                        _translate_output_template_to_indices(value, label_to_idx, raw_to_final),
                    )
                    for key, value in template["items"]
                ],
            }
        if "__tl_dataclass_type__" in template:
            return {
                "__tl_dataclass_type__": template["__tl_dataclass_type__"],
                "items": [
                    (
                        key,
                        _translate_output_template_to_indices(value, label_to_idx, raw_to_final),
                    )
                    for key, value in template["items"]
                ],
            }
        return {
            key: _translate_output_template_to_indices(value, label_to_idx, raw_to_final)
            for key, value in template.items()
        }

    return template


def _default_output_specs(model_log, label_to_idx: Dict[str, int]) -> Any:
    if len(model_log.output_layers) == 1:
        return (OUTPUT_REF_TAG, label_to_idx[model_log.output_layers[0]])
    return [(OUTPUT_REF_TAG, label_to_idx[label]) for label in model_log.output_layers]


def _build_graph_signature(plan: ExecutionPlan) -> str:
    payload = [
        {
            "idx": node.idx,
            "label": node.label,
            "func_name": node.meta.get("func_name"),
            "parents": node.parents,
            "is_input": node.is_input,
            "is_output": node.is_output,
            "is_buffer": node.is_buffer,
            "is_internal_init": node.is_internal_init,
            "output_selector": node.output_selector,
            "shape": canonicalize_batch_agnostic_shape(node.meta.get("tensor_shape")),
            "dtype": node.meta.get("tensor_dtype"),
            "io_role": node.meta.get("io_role"),
        }
        for node in plan.nodes
    ]
    return hash_graph_signature(payload)


def replay_forward(
    plan: ExecutionPlan,
    inputs: Any,
    *,
    input_kwargs: Optional[Dict[str, Any]] = None,
    device: Any = "auto",
    input_mode: str = "raw",
    preserve_rng: bool = True,
    return_intermediates: bool = False,
    validate: bool = False,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> Any:
    """Replay a compiled plan node by node and return the final output."""

    if input_mode != "raw":
        raise ValueError("replay_forward only supports input_mode='raw'.")

    model, runtime_device = _prepare_runtime(plan, inputs, device)
    seed_map = _prepare_seed_map(plan, inputs, input_kwargs, runtime_device)
    retain_nodes = set(plan.output_node_indices) | _collect_output_indices(plan.output_specs)
    computed, intermediates = _execute_nodes(
        plan,
        node_indices=[node.idx for node in plan.nodes],
        seeded_values=seed_map,
        device=runtime_device,
        preserve_rng=preserve_rng,
        differentiable=False,
        retain_nodes=retain_nodes,
        return_intermediates=return_intermediates,
        model=model,
    )
    outputs = _reconstruct_outputs(plan, computed)

    if validate:
        if model is None:
            raise ValueError("Replay validation requires the plan to keep a live model reference.")
        direct_output = _run_direct_model_forward(
            plan,
            model,
            inputs,
            input_kwargs=input_kwargs,
            device=runtime_device,
            restore_rng=plan.meta.get("pre_forward_rng_states") if preserve_rng else None,
        )
        if not tree_allclose(outputs, direct_output, atol=atol, rtol=rtol):
            raise ValueError(
                "Replay output did not match the direct model output within tolerance."
            )

    if return_intermediates:
        return {"output": outputs, "intermediates": intermediates}
    return outputs


def replay_partitioned(
    plan: ExecutionPlan,
    inputs: Any,
    *,
    split: Optional[FrontierSplit] = None,
    input_kwargs: Optional[Dict[str, Any]] = None,
    input_mode: str = "auto",
    device: Any = "auto",
    return_boundary: bool = False,
    pack_boundary: bool = True,
    validate: bool = False,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> Any:
    """Replay the full graph or a prefix/boundary/suffix partition."""

    if split is None:
        return replay_forward(
            plan,
            inputs,
            input_kwargs=input_kwargs,
            device=device,
            input_mode="raw",
            preserve_rng=True,
            return_intermediates=False,
            validate=validate,
            atol=atol,
            rtol=rtol,
        )

    if input_mode == "auto":
        input_mode = "boundary" if _looks_like_boundary_payload(inputs) else "raw"

    model, runtime_device = _prepare_runtime(plan, inputs, device)

    if input_mode == "raw":
        prefix_seed_map = _prepare_seed_map(plan, inputs, input_kwargs, runtime_device)
        prefix_computed, _ = _execute_nodes(
            plan,
            node_indices=split.prefix_node_indices,
            seeded_values=prefix_seed_map,
            device=runtime_device,
            preserve_rng=True,
            differentiable=False,
            retain_nodes=set(split.boundary_indices),
            return_intermediates=False,
            model=model,
        )
        boundary_tensors = {
            label: prefix_computed[idx]
            for label, idx in zip(split.boundary_labels, split.boundary_indices)
        }
        boundary_payload = (
            _pack_boundary_payload(plan, split, boundary_tensors, runtime_device)
            if pack_boundary
            else boundary_tensors
        )

        if not split.suffix_node_indices:
            return {"boundary": boundary_payload} if return_boundary else boundary_payload

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
            preserve_rng=True,
            differentiable=False,
            retain_nodes=set(plan.output_node_indices) | _collect_output_indices(plan.output_specs),
            return_intermediates=False,
            model=model,
        )
        prefix_computed.update(suffix_computed)
        outputs = _reconstruct_outputs(plan, prefix_computed)

        if validate:
            if model is None:
                raise ValueError("Partition replay validation requires a live model reference.")
            direct_output = _run_direct_model_forward(
                plan,
                model,
                inputs,
                input_kwargs=input_kwargs,
                device=runtime_device,
                restore_rng=plan.meta.get("pre_forward_rng_states"),
            )
            if not tree_allclose(outputs, direct_output, atol=atol, rtol=rtol):
                raise ValueError("Partitioned replay output did not match direct model output.")

        if return_boundary:
            return {"output": outputs, "boundary": boundary_payload}
        return outputs

    if input_mode == "boundary":
        boundary_inputs, raw_inputs, boundary_input_kwargs = _split_boundary_mode_inputs(inputs)
        boundary_seed_map = _normalize_boundary_inputs(
            plan,
            split,
            boundary_inputs,
            runtime_device,
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
            preserve_rng=True,
            differentiable=False,
            retain_nodes=set(plan.output_node_indices) | _collect_output_indices(plan.output_specs),
            return_intermediates=False,
            model=model,
        )
        outputs = _reconstruct_outputs(plan, suffix_computed)
        if return_boundary:
            payload = (
                boundary_inputs
                if _looks_like_boundary_payload(boundary_inputs)
                else _pack_boundary_payload(
                    plan,
                    split,
                    {
                        label: boundary_seed_map[idx]
                        for label, idx in zip(split.boundary_labels, split.boundary_indices)
                    },
                    runtime_device,
                )
            )
            return {"output": outputs, "boundary": payload}
        return outputs

    raise ValueError(f"Unsupported replay input_mode {input_mode!r}.")


def _prepare_runtime(
    plan: ExecutionPlan,
    inputs: Any,
    device: Any,
) -> Tuple[Optional[nn.Module], torch.device]:
    model = plan.model
    runtime_device = resolve_device(device, model=model, inputs=inputs, default=plan.default_device)
    if model is not None:
        model_device = _peek_model_device(model)
        if model_device is not None and model_device != runtime_device:
            model.to(runtime_device)
    return model, runtime_device


def _prepare_seed_map(
    plan: ExecutionPlan,
    inputs: Any,
    input_kwargs: Optional[Dict[str, Any]],
    device: torch.device,
) -> Dict[int, Any]:
    seed_map = build_input_seed_map(plan.input_specs, inputs, input_kwargs)
    return {idx: _move_tree_to_device(value, device) for idx, value in seed_map.items()}


def _peek_model_device(model: nn.Module) -> Optional[torch.device]:
    for param in model.parameters():
        return param.device
    for buffer in model.buffers():
        return buffer.device
    return None


def _build_live_state_maps(
    model: Optional[nn.Module],
) -> Tuple[Dict[str, torch.nn.Parameter], Dict[str, torch.Tensor]]:
    param_map: Dict[str, torch.nn.Parameter] = {}
    buffer_map: Dict[str, torch.Tensor] = {}

    if model is None:
        return param_map, buffer_map

    cached_maps = _LIVE_STATE_MAP_CACHE.get(model)
    if cached_maps is not None:
        cached_param_map, cached_buffer_map = cached_maps
        return cached_param_map, cached_buffer_map

    for name, param in model.named_parameters():
        param_map[name] = param
        tl_addr = getattr(param, "tl_param_address", None)
        if tl_addr is not None:
            param_map[tl_addr] = param

    for name, buffer in model.named_buffers():
        buffer_map[name] = buffer
        tl_addr = getattr(buffer, "tl_buffer_address", None)
        if tl_addr is not None:
            buffer_map[tl_addr] = buffer

    _LIVE_STATE_MAP_CACHE[model] = (param_map, buffer_map)
    return param_map, buffer_map


def _resolve_tensor_address(model: Optional[nn.Module], address: str) -> Optional[torch.Tensor]:
    if model is None:
        return None

    current: Any = model
    for part in address.split("."):
        if isinstance(current, (list, tuple)) and part.isdigit():
            current = current[int(part)]
            continue
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
            continue
        if not hasattr(current, part):
            return None
        current = getattr(current, part)

    return current if isinstance(current, torch.Tensor) else None


def _materialize_template(
    template: Any,
    *,
    computed: _ComputedStore,
    device: torch.device,
    param_map: Dict[str, torch.nn.Parameter],
    buffer_map: Dict[str, torch.Tensor],
    model: Optional[nn.Module],
    differentiable: bool,
    parent_clone_indices: Set[int],
    parent_clone_cache: Dict[int, Any],
) -> Any:
    if isinstance(template, ParentRef):
        if template.frozen_value is not None:
            return clone_tree_tensors(
                template.frozen_value,
                device=device,
                detach=not differentiable,
            )
        parent_value = computed[template.node_idx]
        if template.path is not None:
            parent_value = index_nested(parent_value, template.path)
        if template.node_idx not in parent_clone_indices:
            return parent_value
        if template.node_idx not in parent_clone_cache:
            parent_clone_cache[template.node_idx] = clone_tree_tensors(
                parent_value,
                device=device,
                detach=not differentiable,
            )
        return parent_clone_cache[template.node_idx]

    if isinstance(template, ParamRef):
        if template.address not in param_map:
            raise ValueError(
                f"Parameter {template.address!r} is not available on the runtime model."
            )
        return param_map[template.address]

    if isinstance(template, BufferRef):
        if template.address in buffer_map:
            return buffer_map[template.address]
        resolved_buffer = _resolve_tensor_address(model, template.address)
        if resolved_buffer is None:
            raise ValueError(f"Buffer {template.address!r} is not available on the runtime model.")
        buffer_map[template.address] = resolved_buffer
        return resolved_buffer

    if isinstance(template, TensorConst):
        return template.tensor.clone().to(device)

    if dataclasses.is_dataclass(template) and not isinstance(template, type):
        kwargs = {
            field.name: _materialize_template(
                getattr(template, field.name),
                computed=computed,
                device=device,
                param_map=param_map,
                buffer_map=buffer_map,
                model=model,
                differentiable=differentiable,
                parent_clone_indices=parent_clone_indices,
                parent_clone_cache=parent_clone_cache,
            )
            for field in dataclasses.fields(template)
        }
        return type(template)(**kwargs)

    if isinstance(template, list):
        return [
            _materialize_template(
                item,
                computed=computed,
                device=device,
                param_map=param_map,
                buffer_map=buffer_map,
                model=model,
                differentiable=differentiable,
                parent_clone_indices=parent_clone_indices,
                parent_clone_cache=parent_clone_cache,
            )
            for item in template
        ]

    if isinstance(template, tuple):
        return type(template)(
            _materialize_template(
                item,
                computed=computed,
                device=device,
                param_map=param_map,
                buffer_map=buffer_map,
                model=model,
                differentiable=differentiable,
                parent_clone_indices=parent_clone_indices,
                parent_clone_cache=parent_clone_cache,
            )
            for item in template
        )

    if isinstance(template, dict):
        return {
            key: _materialize_template(
                value,
                computed=computed,
                device=device,
                param_map=param_map,
                buffer_map=buffer_map,
                model=model,
                differentiable=differentiable,
                parent_clone_indices=parent_clone_indices,
                parent_clone_cache=parent_clone_cache,
            )
            for key, value in template.items()
        }

    return copy.copy(template)


def _sync_runtime_training_flag(
    node: ExecNode,
    args_list: List[Any],
    kwargs_dict: Dict[str, Any],
    model_training: Optional[bool],
) -> None:
    if model_training is None:
        return

    training_name = node.meta.get("training_flag_name")
    training_idx = node.meta.get("training_flag_index")
    if not isinstance(training_name, str) or not isinstance(training_idx, int):
        return

    if training_name in kwargs_dict and isinstance(kwargs_dict[training_name], bool):
        kwargs_dict[training_name] = model_training
        return

    if training_idx < len(args_list) and isinstance(args_list[training_idx], bool):
        args_list[training_idx] = model_training


def _maybe_retry_getitem_with_safe_indexing(
    func: Any,
    args_list: List[Any],
    kwargs_dict: Dict[str, Any],
) -> Optional[Any]:
    if getattr(func, "__name__", None) != "__getitem__":
        return None
    if kwargs_dict or len(args_list) < 2:
        return None

    source, index = args_list[0], args_list[1]
    if not isinstance(source, torch.Tensor) or source.ndim == 0:
        return None
    if not isinstance(index, torch.Tensor):
        return None
    if index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        return None

    dim0 = source.shape[0]
    safe_index = index[(index >= -dim0) & (index < dim0)]
    return func(source, safe_index)


def _execute_nodes(
    plan: ExecutionPlan,
    *,
    node_indices: Sequence[int],
    seeded_values: Dict[int, Any],
    device: torch.device,
    preserve_rng: bool,
    differentiable: bool,
    retain_nodes: Set[int],
    return_intermediates: bool,
    model: Optional[nn.Module],
) -> Tuple[Dict[int, Any], Dict[str, Any]]:
    node_set = set(node_indices)
    remaining_users = _compute_remaining_users(plan, node_indices)
    computed = _NodeValueStore(len(plan.nodes), seeded_values=seeded_values)
    intermediates: Dict[str, Any] = {}
    saved_values: Dict[int, Any] = {}
    param_map, buffer_map = _build_live_state_maps(model)
    needs_rng_restore = preserve_rng and bool(plan.meta.get("has_rng_nodes", True))
    initial_rng_state = log_current_rng_states() if needs_rng_restore else None

    try:
        for idx in retain_nodes:
            if idx in computed and remaining_users.get(idx, 0) > 0:
                saved_values[idx] = clone_tree_tensors(
                    computed[idx],
                    device=device,
                    detach=not differentiable,
                )

        for idx in node_indices:
            if idx in computed:
                if return_intermediates:
                    intermediates[plan.nodes[idx].label] = clone_tree_tensors(
                        computed[idx],
                        device=device,
                        detach=not differentiable,
                    )
                continue

            node = plan.nodes[idx]
            parent_clone_indices = {
                parent_idx
                for parent_idx in node.parents
                if node.is_inplace
                and (
                    remaining_users.get(parent_idx, 0) > 1
                    or (
                        differentiable
                        and parent_idx in computed
                        and _tree_has_leaf_grad_tensors(computed[parent_idx])
                    )
                )
            }
            parent_clone_cache: Dict[int, Any] = {}

            args = _materialize_template(
                node.const_args_template if node.const_args_template is not None else [],
                computed=computed,
                device=device,
                param_map=param_map,
                buffer_map=buffer_map,
                model=model,
                differentiable=differentiable,
                parent_clone_indices=parent_clone_indices,
                parent_clone_cache=parent_clone_cache,
            )
            kwargs = _materialize_template(
                node.const_kwargs_template if node.const_kwargs_template is not None else {},
                computed=computed,
                device=device,
                param_map=param_map,
                buffer_map=buffer_map,
                model=model,
                differentiable=differentiable,
                parent_clone_indices=parent_clone_indices,
                parent_clone_cache=parent_clone_cache,
            )
            try:
                output = _execute_single_node(
                    plan,
                    node,
                    args,
                    kwargs,
                    computed=computed,
                    preserve_rng=preserve_rng,
                    model=model,
                    buffer_map=buffer_map,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Replay failed at node {node.idx} ({node.label}, func={node.meta.get('func_name')})."
                ) from exc
            computed[idx] = output
            if idx in retain_nodes and remaining_users.get(idx, 0) > 0:
                saved_values[idx] = clone_tree_tensors(
                    output,
                    device=device,
                    detach=not differentiable,
                )
            if return_intermediates:
                intermediates[node.label] = clone_tree_tensors(
                    output,
                    device=device,
                    detach=not differentiable,
                )

            for parent_idx in node.parents:
                if parent_idx not in remaining_users or parent_idx not in node_set:
                    continue
                remaining_users[parent_idx] -= 1
                if (
                    remaining_users[parent_idx] <= 0
                    and not return_intermediates
                    and parent_idx not in retain_nodes
                ):
                    computed.pop(parent_idx, None)
    finally:
        if initial_rng_state is not None:
            set_rng_from_saved_states(initial_rng_state)

    for idx, value in saved_values.items():
        computed[idx] = value
    return computed.materialize(retain_nodes), intermediates


def _execute_single_node(
    plan: ExecutionPlan,
    node: ExecNode,
    args: Any,
    kwargs: Any,
    *,
    computed: _ComputedStore,
    preserve_rng: bool,
    model: Optional[nn.Module],
    buffer_map: Dict[str, torch.Tensor],
) -> Any:
    if node.op is None:
        if node.is_input:
            return computed[node.idx]
        if node.is_buffer:
            buffer_address = node.meta.get("buffer_address")
            if buffer_address is not None and buffer_address in buffer_map:
                return buffer_map[buffer_address]
            if buffer_address is not None:
                resolved_buffer = _resolve_tensor_address(model, buffer_address)
                if resolved_buffer is not None:
                    buffer_map[buffer_address] = resolved_buffer
                    return resolved_buffer
        if node.parents:
            return computed[node.parents[0]]
        raise ValueError(f"Cannot execute source-less node {node.label!r}.")

    if preserve_rng and node.meta.get("uses_rng") and node.rng_state is not None:
        set_rng_from_saved_states(node.rng_state)

    autocast_state = node.meta.get("func_autocast_state") or {}
    model_training = model.training if model is not None else None
    args_list = list(args) if isinstance(args, tuple) else list(args)
    kwargs_dict = dict(kwargs)
    _sync_runtime_training_flag(node, args_list, kwargs_dict, model_training)
    _adapt_batch_sensitive_call_args(plan, node, args_list, kwargs_dict, computed)

    with AutocastRestore(autocast_state):
        try:
            output = node.op(*args_list, **kwargs_dict)
        except IndexError:
            output = _maybe_retry_getitem_with_safe_indexing(node.op, args_list, kwargs_dict)
            if output is None:
                raise

    if output is None:
        if node.is_inplace and len(args_list) > 0:
            output = args_list[0]
        elif node.parents:
            output = computed[node.parents[0]]
        else:
            raise ValueError(f"Node {node.label!r} returned None and cannot be recovered.")

    if node.output_selector is not None:
        selector = (
            [node.output_selector]
            if not isinstance(node.output_selector, (list, tuple))
            else node.output_selector
        )
        output = index_nested(output, selector)

    return output


def _adapt_batch_sensitive_call_args(
    plan: ExecutionPlan,
    node: ExecNode,
    args_list: List[Any],
    kwargs_dict: Dict[str, Any],
    computed: _ComputedStore,
) -> None:
    """Adjust replay-time args for ops whose leading size dim tracks batch."""

    func_name = node.meta.get("func_name")
    if not isinstance(func_name, str):
        return

    if func_name in _BATCH_DYNAMIC_RESHAPE_FUNCS:
        _adapt_batch_sensitive_reshape_args(plan, node, args_list)
        return

    if func_name in _BATCH_DYNAMIC_FACTORY_FUNCS:
        _adapt_batch_sensitive_factory_args(plan, node, args_list, kwargs_dict, computed)


def _adapt_batch_sensitive_reshape_args(
    plan: ExecutionPlan,
    node: ExecNode,
    args_list: List[Any],
) -> None:
    """Rewrite the leading reshape dim when only the source batch changed."""

    if not args_list or not isinstance(args_list[0], torch.Tensor) or not node.parents:
        return

    traced_source_shape = plan.nodes[node.parents[0]].meta.get("tensor_shape")
    source_tensor = args_list[0]
    batch_context = _resolve_batch_context_from_shapes(traced_source_shape, tuple(source_tensor.shape))
    if batch_context is None:
        return

    shape_spec, update_shape_spec = _extract_shape_spec(args_list, start_index=1)
    if shape_spec is None or not shape_spec:
        return

    updated_first_dim = _scale_batch_dependent_dim(shape_spec[0], *batch_context)
    if updated_first_dim is None or updated_first_dim == shape_spec[0]:
        return

    updated_shape = [updated_first_dim, *shape_spec[1:]]
    if not _shape_spec_matches_numel(source_tensor.numel(), updated_shape):
        return

    update_shape_spec(updated_shape)


def _adapt_batch_sensitive_factory_args(
    plan: ExecutionPlan,
    node: ExecNode,
    args_list: List[Any],
    kwargs_dict: Dict[str, Any],
    computed: _ComputedStore,
) -> None:
    """Rewrite factory size args whose first dim equals the traced batch."""

    func_name = node.meta.get("func_name")
    if func_name == "arange":
        _adapt_batch_sensitive_arange_args(plan, args_list, computed)
        return

    if func_name in {"new_empty", "new_ones", "new_zeros"}:
        if not args_list or not isinstance(args_list[0], torch.Tensor) or not node.parents:
            return
        traced_source_shape = plan.nodes[node.parents[0]].meta.get("tensor_shape")
        batch_context = _resolve_batch_context_from_shapes(
            traced_source_shape,
            tuple(args_list[0].shape),
        )
        shape_spec, update_shape_spec = _extract_shape_spec(args_list, start_index=1)
    else:
        batch_context = _resolve_primary_input_batch_context(plan, computed)
        shape_spec, update_shape_spec = _extract_shape_spec(args_list, start_index=0)

    if batch_context is None or shape_spec is None or not shape_spec:
        return

    old_batch, new_batch = batch_context
    if shape_spec[0] != old_batch:
        return

    update_shape_spec([new_batch, *shape_spec[1:]])


def _adapt_batch_sensitive_arange_args(
    plan: ExecutionPlan,
    args_list: List[Any],
    computed: _ComputedStore,
) -> None:
    """Rewrite ``arange`` bounds when they encode the traced batch length."""

    batch_context = _resolve_primary_input_batch_context(plan, computed)
    if batch_context is None:
        return

    old_batch, new_batch = batch_context
    if len(args_list) == 1 and args_list[0] == old_batch:
        args_list[0] = new_batch
        return

    if len(args_list) >= 2 and args_list[0] == 0 and args_list[1] == old_batch:
        args_list[1] = new_batch


def _extract_shape_spec(
    args_list: List[Any],
    *,
    start_index: int,
) -> Tuple[Optional[List[Any]], Any]:
    """Return a mutable size spec plus a callback that writes it back."""

    if len(args_list) <= start_index:
        return None, lambda updated: None

    first_shape_arg = args_list[start_index]
    if isinstance(first_shape_arg, (list, tuple)):

        def _update_container(updated_shape: List[Any]) -> None:
            args_list[start_index] = type(first_shape_arg)(updated_shape)

        return list(first_shape_arg), _update_container

    shape_values = list(args_list[start_index:])

    def _update_varargs(updated_shape: List[Any]) -> None:
        args_list[start_index:] = updated_shape

    return shape_values, _update_varargs


def _resolve_primary_input_batch_context(
    plan: ExecutionPlan,
    computed: _ComputedStore,
) -> Optional[Tuple[int, int]]:
    """Return traced/current batch sizes for the first replay input tensor."""

    if not plan.input_node_indices:
        return None

    input_idx = plan.input_node_indices[0]
    traced_shape = plan.nodes[input_idx].meta.get("tensor_shape")
    current_value = computed.get(input_idx)
    if isinstance(current_value, torch.Tensor):
        return _resolve_batch_context_from_shapes(traced_shape, tuple(current_value.shape))

    fallback_tensor = next(
        (value for value in computed.values() if isinstance(value, torch.Tensor) and value.ndim > 0),
        None,
    )
    if fallback_tensor is None or traced_shape is None:
        return None

    traced_shape_tuple = tuple(traced_shape)
    if not traced_shape_tuple or traced_shape_tuple[0] <= 0:
        return None

    return traced_shape_tuple[0], int(fallback_tensor.shape[0])


def _resolve_batch_context_from_shapes(
    traced_shape: Optional[Sequence[int]],
    current_shape: Tuple[int, ...],
) -> Optional[Tuple[int, int]]:
    """Return traced/current batch sizes when only dim0 changed."""

    if traced_shape is None:
        return None

    traced_shape_tuple = tuple(traced_shape)
    if not traced_shape_tuple or not current_shape:
        return None
    if len(traced_shape_tuple) != len(current_shape):
        return None
    if traced_shape_tuple[0] <= 0:
        return None
    if traced_shape_tuple[1:] != current_shape[1:]:
        return None

    return traced_shape_tuple[0], current_shape[0]


def _scale_batch_dependent_dim(
    value: Any,
    old_batch: int,
    new_batch: int,
) -> Optional[int]:
    """Scale a traced size value by the runtime batch ratio when possible."""

    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if value <= 0:
        return None
    if value % old_batch != 0:
        return None

    scaled_value = value * new_batch
    if scaled_value % old_batch != 0:
        return None

    return scaled_value // old_batch


def _shape_spec_matches_numel(
    source_numel: int,
    shape_spec: Sequence[Any],
) -> bool:
    """Return whether a proposed reshape target is compatible with ``source_numel``."""

    unknown_dims = 0
    known_product = 1
    for dim in shape_spec:
        if not isinstance(dim, int) or isinstance(dim, bool):
            return False
        if dim == -1:
            unknown_dims += 1
            continue
        if dim < 0:
            return False
        known_product *= dim

    if unknown_dims > 1:
        return False
    if unknown_dims == 1:
        return known_product > 0 and source_numel % known_product == 0

    return known_product == source_numel


def _compute_remaining_users(plan: ExecutionPlan, node_indices: Sequence[int]) -> Dict[int, int]:
    node_set = set(node_indices)
    counts: Dict[int, int] = {idx: 0 for idx in node_set}
    for idx in node_indices:
        for parent_idx in plan.nodes[idx].parents:
            if parent_idx in node_set:
                counts[parent_idx] += 1
    return counts


def _tree_has_leaf_grad_tensors(tree: Any) -> bool:
    if isinstance(tree, torch.Tensor):
        return tree.requires_grad and (
            tree.is_leaf or getattr(tree, "_base", None) is not None
        )
    if isinstance(tree, list):
        return any(_tree_has_leaf_grad_tensors(item) for item in tree)
    if isinstance(tree, tuple):
        return any(_tree_has_leaf_grad_tensors(item) for item in tree)
    if isinstance(tree, dict):
        return any(_tree_has_leaf_grad_tensors(item) for item in tree.values())
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        return any(
            _tree_has_leaf_grad_tensors(getattr(tree, field.name))
            for field in dataclasses.fields(tree)
        )
    return False


def _reconstruct_outputs(plan: ExecutionPlan, computed: Dict[int, Any]) -> Any:
    return reconstruct_from_template(plan.output_specs, lambda idx: computed[idx])


def _collect_output_indices(template: Any) -> Set[int]:
    if isinstance(template, tuple) and len(template) == 2 and template[0] == OUTPUT_REF_TAG:
        return {template[1]}
    if isinstance(template, list):
        list_output_indices: Set[int] = set()
        for item in template:
            list_output_indices.update(_collect_output_indices(item))
        return list_output_indices
    if isinstance(template, dict):
        if "__tl_tuple_type__" in template:
            tuple_output_indices: Set[int] = set()
            for item in template["items"]:
                tuple_output_indices.update(_collect_output_indices(item))
            return tuple_output_indices
        if "__tl_dict_type__" in template or "__tl_dataclass_type__" in template:
            mapping_output_indices: Set[int] = set()
            for _, value in template["items"]:
                mapping_output_indices.update(_collect_output_indices(value))
            return mapping_output_indices
        plain_output_indices: Set[int] = set()
        for value in template.values():
            plain_output_indices.update(_collect_output_indices(value))
        return plain_output_indices
    return set()


def _run_direct_model_forward(
    plan: ExecutionPlan,
    model: nn.Module,
    inputs: Any,
    *,
    input_kwargs: Optional[Dict[str, Any]],
    device: torch.device,
    restore_rng: Optional[Dict[str, Any]],
) -> Any:
    call_args, call_kwargs = _build_model_call_roots(plan, inputs, input_kwargs, device)
    current_rng_state = log_current_rng_states() if restore_rng else None
    if restore_rng:
        set_rng_from_saved_states(restore_rng)
    try:
        return model(*call_args, **call_kwargs)
    finally:
        if current_rng_state is not None:
            set_rng_from_saved_states(current_rng_state)


def _build_model_call_roots(
    plan: ExecutionPlan,
    inputs: Any,
    input_kwargs: Optional[Dict[str, Any]],
    device: torch.device,
) -> Tuple[List[Any], Dict[str, Any]]:
    if input_kwargs is None:
        input_kwargs = {}

    ordered_root_names: List[str] = []
    seen_root_names = set()
    for spec in plan.input_specs:
        io_role = spec.get("io_role")
        if io_role is None:
            continue
        root_name = io_role.split(".")[1]
        if root_name not in seen_root_names:
            ordered_root_names.append(root_name)
            seen_root_names.add(root_name)

    kwarg_root_names = {name for name in ordered_root_names if name in input_kwargs}
    positional_root_names = [name for name in ordered_root_names if name not in kwarg_root_names]
    positional_inputs = normalize_positional_inputs(inputs, len(positional_root_names))
    call_args = [
        _move_tree_to_device(positional_inputs[index], device)
        for index in range(len(positional_root_names))
    ]
    call_kwargs = {key: _move_tree_to_device(value, device) for key, value in input_kwargs.items()}
    return call_args, call_kwargs


def _pack_boundary_payload(
    plan: ExecutionPlan,
    split: FrontierSplit,
    boundary_tensors: Dict[str, Any],
    device: torch.device,
) -> BoundaryPayload:
    # Snapshot the boundary immediately so later in-place suffix ops cannot mutate
    # the payload that callers feed back into boundary-mode replay/training.
    frozen_tensors = clone_tree_tensors(boundary_tensors, device=device, detach=True)
    return {
        "cut_id": split.split_id,
        "labels": list(split.boundary_labels),
        "tensors": frozen_tensors,
        "meta": {
            "graph_signature": plan.graph_signature,
            "device": str(device),
            "dtype": {
                label: str(tensor.dtype)
                if isinstance(tensor, torch.Tensor)
                else type(tensor).__name__
                for label, tensor in frozen_tensors.items()
            },
            "shapes": {
                label: tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None
                for label, tensor in frozen_tensors.items()
            },
        },
    }


def _looks_like_boundary_payload(inputs: Any) -> bool:
    return isinstance(inputs, dict) and "tensors" in inputs and "labels" in inputs


def _split_boundary_mode_inputs(
    inputs: Any,
) -> Tuple[Any, Any, Optional[Dict[str, Any]]]:
    if isinstance(inputs, dict) and "boundary" in inputs:
        return (
            inputs["boundary"],
            inputs.get("raw_inputs"),
            inputs.get("input_kwargs"),
        )
    return inputs, None, None


def _required_passthrough_input_indices(split: FrontierSplit) -> List[int]:
    passthrough_indices = split.meta.get("passthrough_input_indices", [])
    return [int(idx) for idx in passthrough_indices]


def _prepare_passthrough_seed_map(
    plan: ExecutionPlan,
    split: FrontierSplit,
    *,
    raw_inputs: Any,
    input_kwargs: Optional[Dict[str, Any]],
    device: torch.device,
) -> Dict[int, Any]:
    required_indices = _required_passthrough_input_indices(split)
    if not required_indices:
        return {}

    if raw_inputs is None and not input_kwargs:
        required_labels = [plan.nodes[idx].label for idx in required_indices]
        raise ValueError(
            "Boundary-mode replay requires raw passthrough inputs for suffix execution: "
            f"{required_labels}."
        )

    raw_seed_map = _prepare_seed_map(plan, raw_inputs, input_kwargs, device)
    missing = [idx for idx in required_indices if idx not in raw_seed_map]
    if missing:
        missing_labels = [plan.nodes[idx].label for idx in missing]
        raise ValueError(
            "Raw passthrough inputs did not provide all suffix-required input roots: "
            f"{missing_labels}."
        )

    return {idx: raw_seed_map[idx] for idx in required_indices}


def _normalize_boundary_inputs(
    plan: ExecutionPlan,
    split: FrontierSplit,
    boundary_inputs: Any,
    device: torch.device,
) -> Dict[int, Any]:
    if _looks_like_boundary_payload(boundary_inputs):
        payload = boundary_inputs
        if payload["cut_id"] != split.split_id:
            raise ValueError(
                f"Boundary payload cut_id {payload['cut_id']!r} does not match split {split.split_id!r}."
            )
        boundary_tensors = payload["tensors"]
    elif isinstance(boundary_inputs, dict):
        boundary_tensors = boundary_inputs
    elif isinstance(boundary_inputs, (list, tuple)):
        if len(boundary_inputs) != len(split.boundary_labels):
            raise ValueError(
                "Boundary input length does not match split boundary size: "
                f"expected {len(split.boundary_labels)}, got {len(boundary_inputs)}."
            )
        boundary_tensors = dict(zip(split.boundary_labels, boundary_inputs))
    elif len(split.boundary_labels) == 1:
        boundary_tensors = {split.boundary_labels[0]: boundary_inputs}
    else:
        raise ValueError("Boundary inputs must be a payload dict, label dict, or ordered sequence.")

    missing_labels = [label for label in split.boundary_labels if label not in boundary_tensors]
    if missing_labels:
        raise ValueError(
            f"Boundary inputs are missing labels required by the split: {missing_labels}."
        )

    return {
        node_idx: clone_tree_tensors(boundary_tensors[label], device=device, detach=True)
        for label, node_idx in zip(split.boundary_labels, split.boundary_indices)
    }
