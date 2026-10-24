"""Explicit state ownership for split training and streaming aggregation."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping

import numpy as np

OWNERSHIP_KEY = "autosplit_state_ownership"
DIGEST_KEY = "state_ownership_digest"


def _attribute(value, name, default=None):
    try:
        return getattr(value, name, default)
    except (AttributeError, KeyError):
        return default


def _native_identities(value):
    """Include a Keras wrapper, its native variable and its resource handle."""
    values, identities = [value], set()
    for _ in range(3):
        following = []
        for item in values:
            if item is None or id(item) in identities:
                continue
            identities.add(id(item))
            for name in ("value", "handle", "_param_ref", "_buffer_ref"):
                native = _attribute(item, name)
                if native is not None and not callable(native):
                    following.append(native)
        values = following
    return identities


def _runtime_adapter(handle):
    from splitfleet.backends import BACKEND_ADAPTERS
    return BACKEND_ADAPTERS.create(str(handle.backend.framework_backend))

def _is_state_source(node):
    kind = str(node.op_type).lower().replace("_", "")
    return bool(node.is_input or node.is_param_source or node.is_buffer_only_source
                or kind in {"buffer", "parameter", "getattr", "readvariableop"})


def state_ownership(handle, schema_hash: str, *, native_plan=None) -> dict:
    """Prove disjoint ownership from native consumers and declared associations."""
    runtime = handle.runtime
    plan = runtime.plan if native_plan is None else native_plan
    adapter = _runtime_adapter(handle)
    backend = adapter.backend_name
    graph = runtime.trace_graph
    if backend == "torch" and runtime.retains_trace:
        from torchlens.split._torch_compact import compact_torch_graph

        graph = compact_torch_graph(graph)
    state = (runtime.model.state_dict(keep_vars=True) if backend == "torch"
             else adapter._state(runtime.model))
    names = list(state)
    identities = defaultdict(set)
    for name, value in state.items():
        for identity in _native_identities(value):
            identities[identity].add(name)
    source_names = {}
    if backend == "torch":
        trainables = {name for name, parameter in runtime.model.named_parameters(remove_duplicate=False)
                      if parameter.requires_grad}
    else:
        trainables = {name for name, value in state.items() if (
            bool(_attribute(value, "trainable", False)) if backend == "tf"
            else not bool(_attribute(value, "stop_gradient", True)) if backend == "paddle"
            else bool(_attribute(value, "requires_grad", False))
        )}
    stages = {"prefix": set(plan.prefix_node_ids), "suffix": set(plan.suffix_node_ids)}
    if stages["prefix"] & stages["suffix"]:
        raise ValueError("Native split plan has overlapping stage nodes")
    used = {name: set() for name in names}
    children = defaultdict(set)
    alias_nodes = defaultdict(set)
    for node in graph.nodes:
        for alias in (node.canonical_id, node.label, node.raw_label):
            if alias is not None:
                alias_nodes[alias].add(node.canonical_id)
    aliases = {alias: next(iter(node_ids)) for alias, node_ids in alias_nodes.items() if len(node_ids) == 1}
    for node in graph.nodes:
        for parent in node.parents:
            if parent not in aliases:
                raise ValueError(f"Native ownership has an unresolved graph parent {parent!r}")
            children[aliases[parent]].add(node.canonical_id)

    def consume(referenced, node_id):
        for stage, node_ids in stages.items():
            if node_id in node_ids:
                for name in referenced:
                    used[name].add(stage)

    for node in graph.nodes:
        referenced = set(source_names.get(node.canonical_id, ()))
        refs = list(node.param_refs) + list(node.buffer_refs)
        # Torch module buffers are explicit live declarations even when the
        # batch counter lacks a standalone captured arithmetic source node.
        for ref in node.param_refs:
            module = _attribute(ref, "module")
            refs.extend((_attribute(module, "buffers", {}) or {}).values())
        for ref in refs:
            matched = set().union(*(identities.get(identity, set()) for identity in _native_identities(ref)))
            if not matched and bool(_attribute(ref, "is_trainable", False)):
                raise ValueError("Native trainable parameter reference cannot be bound to model state")
            referenced.update(matched)
        # TF primitive capture can expose only ReadVariableOp resource inputs.
        for value in _attribute(node.target, "inputs", ()) or ():
            if _attribute(value, "source_kind") == "resource":
                for identity in _native_identities(_attribute(value, "tensor")):
                    referenced.update(identities.get(identity, ()))
        if _is_state_source(node) or node.canonical_id in source_names:
            frontier, visited = list(children[node.canonical_id]), set()
            while frontier:
                child_id = frontier.pop()
                if child_id in visited:
                    continue
                visited.add(child_id)
                child = graph.node_by_id[child_id]
                if _is_state_source(child) or child_id in source_names:
                    frontier.extend(children[child_id])
                else:
                    consume(referenced, child_id)
        else:
            consume(referenced, node.canonical_id)
    owner_by_name = {}
    for name, consumed in used.items():
        if len(consumed) > 1:
            raise ValueError(f"State {name!r} is shared across split stages; disjoint ownership required")
        owner_by_name[name] = next(iter(consumed), "initial")
    declaration = _attribute(runtime.model, "state_associations")
    declared = set()
    if callable(declaration):
        associations = declaration()
        if not isinstance(associations, Mapping):
            raise TypeError("state_associations() must map buffer names to trainable names")
        for name, anchors in associations.items():
            anchors = (anchors,) if isinstance(anchors, str) else tuple(anchors)
            if name not in state or name in trainables or not anchors or not set(anchors).issubset(trainables):
                raise ValueError(f"Invalid declared state association for {name!r}")
            owners = {owner_by_name[anchor] for anchor in anchors}
            if len(owners) != 1:
                raise ValueError(f"Declared state {name!r} has ambiguous ownership")
            owner = next(iter(owners))
            if owner_by_name[name] not in {"initial", owner}:
                raise ValueError(f"Declared state {name!r} contradicts native ownership")
            owner_by_name[name] = owner
            declared.add(name)
    if callable(_attribute(runtime.model, "apply_statistics")) and bool(_attribute(runtime.model, "defer_statistics", True)):
        unproven = {name for name in names if name not in trainables and name not in declared
                    and owner_by_name[name] == "initial"}
        if unproven:
            raise ValueError(f"Deferred state has no declared ownership association: {sorted(unproven)}")
    result = {"version": 1, "schema_hash": schema_hash, "backend": backend,
              "split_id": handle.plan.split_id, "names": names,
              "owners": [owner_by_name[name] for name in names]}
    result[DIGEST_KEY] = ownership_digest(result)
    return result


def ownership_digest(manifest: dict) -> str:
    body = {key: value for key, value in manifest.items() if key != DIGEST_KEY}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_ownership(config, handle, schema_hash: str):
    raw = config.get(OWNERSHIP_KEY)
    if not raw:
        return None
    manifest = json.loads(raw)
    if manifest != state_ownership(handle, schema_hash):
        raise RuntimeError("Split state ownership contract mismatch")
    return manifest


def export_owned(model, manifest: dict, stage: str) -> list[np.ndarray]:
    from splitfleet.backends import BACKEND_ADAPTERS
    from splitfleet.backends.utils import adapter_for

    backend = manifest.get("backend")
    adapter = BACKEND_ADAPTERS.create(backend) if backend is not None else adapter_for(
        model, (_attribute(model, "params", {}),),
    )
    if adapter.backend_name != "torch":
        state = adapter._state(model)
        if set(state) != set(manifest["names"]):
            raise RuntimeError("Owned state names differ from model state")
        return [adapter._to_numpy(state[name])
                for name, owner in zip(manifest["names"], manifest["owners"], strict=True) if owner == stage]
    state = model.state_dict()
    # CPU exports are views. Flower serializes them before the next model
    # mutation; local suffix replicas remain idle through aggregation.
    return [state[name].detach().cpu().numpy()
            for name, owner in zip(manifest["names"], manifest["owners"]) if owner == stage]


def validate_owned(values, manifest, stage, initial):
    if (len(manifest["names"]) != len(initial) or len(manifest["owners"]) != len(initial)
            or any(owner not in {"prefix", "suffix", "initial"} for owner in manifest["owners"])):
        raise RuntimeError("Incorrect state ownership schema")
    indices = [i for i, owner in enumerate(manifest["owners"]) if owner == stage]
    if len(values) != len(indices):
        raise RuntimeError(f"Incorrect {stage} state tensor count")
    for index, value in zip(indices, values):
        if value.shape != initial[index].shape or value.dtype != initial[index].dtype:
            raise RuntimeError(f"Incorrect owned state schema for {manifest['names'][index]}")
    return dict(zip(indices, values))


def aggregate_owned(initial, updates, *, integer_policy="weighted_round"):
    """One output and one scratch tensor per state; no reconstructed replicas.

    updates contains (manifest, prefix arrays, suffix arrays, weight).
    Validation checks metadata, shapes and dtypes, never parameter values.
    """
    if integer_policy not in {"weighted_round", "maximum"}:
        raise ValueError(f"Unsupported integer aggregation policy {integer_policy!r}")
    result, total = None, 0.
    for manifest, prefix_values, suffix_values, raw_weight in updates:
        prefix = validate_owned(prefix_values, manifest, "prefix", initial)
        suffix = validate_owned(suffix_values, manifest, "suffix", initial)
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight <= 0:
            raise RuntimeError("State aggregation requires finite positive weights")
        if result is None:
            result = [np.array(base, dtype=base.dtype if base.dtype.kind in "fc" or integer_policy == "maximum" else np.float64,
                               copy=True) for base in initial]
            for index, accumulator in enumerate(result):
                value = prefix.get(index, suffix.get(index, initial[index]))
                if integer_policy == "maximum" and initial[index].dtype.kind not in "fc":
                    np.copyto(accumulator, value)
                else:
                    np.multiply(value, weight, out=accumulator)
        else:
            for index, accumulator in enumerate(result):
                value = prefix.get(index, suffix.get(index, initial[index]))
                if integer_policy == "maximum" and initial[index].dtype.kind not in "fc":
                    np.maximum(accumulator, value, out=accumulator)
                else:
                    scratch = np.multiply(np.asarray(value, dtype=accumulator.dtype), weight)
                    np.add(accumulator, scratch, out=accumulator)
        total += weight
    if result is None:
        raise RuntimeError("No matching prefix/suffix update")
    for base, accumulator in zip(initial, result):
        if integer_policy == "maximum" and base.dtype.kind not in "fc":
            continue
        np.divide(accumulator, total, out=accumulator)
        if base.dtype.kind not in "fc":
            np.rint(accumulator, out=accumulator)
    return [value.astype(base.dtype, copy=False) for base, value in zip(initial, result)]
