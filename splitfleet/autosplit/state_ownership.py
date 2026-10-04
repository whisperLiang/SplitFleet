"""Explicit state ownership for split training and streaming aggregation."""
from __future__ import annotations

import hashlib
import json
import numpy as np

OWNERSHIP_KEY = "autosplit_state_ownership"
DIGEST_KEY = "state_ownership_digest"


def state_ownership(handle, schema_hash: str) -> dict:
    runtime = handle.runtime
    from torchlens.split._torch_compact import compact_torch_graph
    graph = compact_torch_graph(runtime.trace_graph) if runtime.retains_trace else runtime.trace_graph
    state = runtime.model.state_dict(keep_vars=True)
    names = list(state)
    prefix, suffix = set(runtime.plan.prefix_node_ids), set(runtime.plan.suffix_node_ids)
    used = {"prefix": set(), "suffix": set()}
    for node in graph.nodes:
        refs = list(node.param_refs) + list(node.buffer_refs)
        # Module buffer bindings can be used without a standalone buffer node.
        for ref in node.param_refs:
            module = getattr(ref, "module", None)
            if module is not None:
                refs.extend(module.buffers.values())
        identities = {id(ref.handle) for ref in refs if ref.handle is not None}
        for stage, nodes in (("prefix", prefix), ("suffix", suffix)):
            if node.canonical_id in nodes:
                used[stage].update(identities)
    owners = []
    for name, value in state.items():
        stages = [stage for stage in used if id(value) in used[stage]]
        if len(stages) > 1:
            raise ValueError(f"State {name!r} is shared across split stages; disjoint ownership required")
        owners.append(stages[0] if stages else "initial")
    result = {"version": 1, "schema_hash": schema_hash,
              "split_id": handle.plan.split_id, "names": names, "owners": owners}
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
    state = model.state_dict()
    # CPU exports are views. Flower serializes them before the next model
    # mutation; local suffix replicas remain idle through aggregation.
    return [state[name].detach().cpu().numpy()
            for name, owner in zip(manifest["names"], manifest["owners"]) if owner == stage]


def validate_owned(values, manifest, stage, initial):
    indices = [i for i, owner in enumerate(manifest["owners"]) if owner == stage]
    if len(values) != len(indices):
        raise RuntimeError(f"Incorrect {stage} state tensor count")
    for index, value in zip(indices, values):
        if value.shape != initial[index].shape or value.dtype != initial[index].dtype:
            raise RuntimeError(f"Incorrect owned state schema for {manifest['names'][index]}")
    return dict(zip(indices, values))


def aggregate_owned(initial, updates):
    """One output and one scratch tensor per state; no reconstructed replicas.

    updates contains (manifest, prefix arrays, suffix arrays, weight).
    Validation checks metadata, shapes and dtypes, never parameter values.
    """
    if not updates:
        raise RuntimeError("No matching prefix/suffix update")
    decoded = [(validate_owned(p, m, "prefix", initial),
                validate_owned(s, m, "suffix", initial), float(w)) for m, p, s, w in updates]
    if any(not np.isfinite(w) or w <= 0 for _, _, w in decoded):
        raise RuntimeError("State aggregation requires finite positive weights")
    total = sum(w for _, _, w in decoded)
    if total <= 0:
        raise RuntimeError("State aggregation requires positive weights")
    result = []
    for index, base in enumerate(initial):
        first_p, first_s, weight = decoded[0]
        value = first_p.get(index, first_s.get(index, base))
        # Multiplication can turn a zero-dimensional ndarray into a NumPy
        # scalar, which cannot be an ufunc output. Preserve ndarray rank and
        # the model schema; integer buffers use a floating accumulator.
        working_dtype = base.dtype if base.dtype.kind in "fc" else np.float64
        accumulator = np.array(value, dtype=working_dtype, copy=True)
        np.multiply(accumulator, weight, out=accumulator)
        scratch = np.empty_like(accumulator) if len(decoded) > 1 else None
        for prefix, suffix, weight in decoded[1:]:
            value = prefix.get(index, suffix.get(index, base))
            np.multiply(value, weight, out=scratch)
            np.add(accumulator, scratch, out=accumulator)
        np.divide(accumulator, total, out=accumulator)
        if base.dtype.kind not in "fc":
            np.rint(accumulator, out=accumulator)
        result.append(accumulator.astype(base.dtype, copy=False))
    return result
