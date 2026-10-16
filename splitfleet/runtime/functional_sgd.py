"""Stage-owned SGD around native JAX split gradients and private suffix state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np


class OwnedFunctionalSGD:
    """Opt-in functional SGD; all derivatives still come from native TorchLens."""

    def __init__(self, model, learning_rate):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Functional SGD needs a finite positive learning rate")
        self.model, self.learning_rate = model, float(learning_rate)
        self.ownership = None
        self.last_gradients = {}

    def _parameters(self):
        from splitfleet.autosplit.state_ownership import _jax_parameter_bindings
        from splitfleet.backends import JaxBackendAdapter

        return _jax_parameter_bindings(JaxBackendAdapter(), self.model)

    def prefix_update(self, model, result):
        if self.ownership is None:
            raise RuntimeError("Functional prefix SGD requires verified ownership")
        if model is not self.model:
            raise RuntimeError("Functional SGD was bound to another model")
        import jax

        gradients = dict(jax.tree_util.tree_flatten_with_path(result["inputs"][0])[0])
        owners = dict(zip(self.ownership["names"], self.ownership["owners"], strict=True))
        selected = [(path, name) for path, name, _ in self._parameters().values() if owners[name] == "prefix"]
        if any(path not in gradients for path, _ in selected):
            raise RuntimeError("Native prefix returned no gradients for an owned parameter path")
        self.last_gradients = {name: gradients[path] for path, name in selected}
        self._step(self.last_gradients)

    def suffix_update(self, bindings, gradients):
        missing = set(bindings) - set(gradients)
        if missing:
            raise RuntimeError(f"Native suffix returned no gradients for resident roots: {sorted(missing)}")
        self.last_gradients = {name: gradients[key] for key, name in bindings.items()}
        self._step(self.last_gradients)
        return {key: value for key, value in gradients.items() if key not in bindings}

    def _step(self, gradients):
        import jax

        bindings = self._parameters()
        parameters = {name: value for _, name, value in bindings.values()}
        named_gradients = {}
        for name, gradient in gradients.items():
            # Accept native addresses from existing full-model SGD callers,
            # while stage-owned updates use the adapter's state names.
            canonical = name if name in parameters else bindings[name][1] if name in bindings else None
            if canonical is None or canonical in named_gradients:
                raise RuntimeError(f"Unknown or duplicate functional SGD parameter {name!r}")
            parameter = parameters[canonical]
            if parameter.shape != getattr(gradient, "shape", None) or parameter.dtype != getattr(gradient, "dtype", None):
                raise RuntimeError(f"Native functional SGD gradient schema mismatch for {name}")
            named_gradients[canonical] = gradient
        by_path = {path: named_gradients[name] for path, name, _ in bindings.values() if name in named_gradients}
        values = jax.tree_util.tree_map_with_path(
            lambda path, parameter: parameter - np.float32(self.learning_rate) * by_path[path]
            if path in by_path else parameter, self.model.params)
        self.model.set_params(values)


def resident_parameter_bindings(handle, manifest, *, native_plan=None):
    """Resolve direct suffix parameter roots from audited native input bindings."""
    from splitfleet.autosplit.state_ownership import _jax_bindings, _runtime_adapter

    adapter = _runtime_adapter(handle)
    if adapter.backend_name != "jax" or not hasattr(handle.model, "params"):
        raise TypeError("Resident functional parameters require a declared JAX model")
    state = adapter._state(handle.model)
    parameter_names, sources = _jax_bindings(adapter, handle.model, state, handle.runtime.trace_graph)
    owners = dict(zip(manifest["names"], manifest["owners"], strict=True))
    suffix_names = {name for name in parameter_names.values() if owners[name] == "suffix"}
    bindings = {}
    plan = handle.runtime.plan if native_plan is None else native_plan
    for key, source in plan.boundary_bindings.items():
        names = sources.get(source, set()) & suffix_names
        if len(names) > 1:
            raise RuntimeError("A resident parameter boundary root has ambiguous state bindings")
        if names:
            bindings[key] = next(iter(names))
    if set(bindings.values()) != suffix_names or len(bindings) != len(suffix_names):
        raise RuntimeError("Native split does not expose every owned suffix parameter as a unique direct root")
    return bindings


def resident_contract(handle, manifest, bindings):
    body = {"protocol": "splitfleet.functional-state.v1", "ownership": manifest["state_ownership_digest"],
            "split_id": handle.plan.split_id, "bindings": sorted(bindings.items())}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def hydrate_resident_boundary(boundary, handle, manifest, *, contract):
    bindings = resident_parameter_bindings(handle, manifest)
    if contract != resident_contract(handle, manifest, bindings):
        raise RuntimeError("Functional resident state contract mismatch")
    required = set(handle.runtime.boundary_spec)
    if set(boundary.tensors) != required - set(bindings):
        raise RuntimeError("Functional boundary is incomplete or includes unnegotiated resident state")
    tensors = dict(boundary.tensors)
    from splitfleet.autosplit.state_ownership import _runtime_adapter
    state = _runtime_adapter(handle)._state(handle.model)
    tensors.update({key: state[name] for key, name in bindings.items()})
    hydrated = replace(boundary, tensors=tensors)
    # The complete native ABI is validated after restoring private state.
    from splitfleet.autosplit.boundary import to_torchlens_boundary
    handle.runtime.validate_boundary(to_torchlens_boundary(hydrated))
    return hydrated, bindings
