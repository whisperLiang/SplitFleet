"""Restrict an existing valid catalog to captured semantic module endpoints."""

from __future__ import annotations


def module_end_boundaries(graph):
    """Map after-boundaries at each module path's final captured compute node.

    Endpoints include descendants of a module/block path. For a reused or
    recurrent module, only its final captured invocation is retained. This is
    deliberately a conservative semantic subset, not a depth-percentile rule.
    """
    last = {}
    for node in graph.nodes:
        path = getattr(node, "module_path", None)
        if not path or any(getattr(node, name, False) for name in
                           ("is_input", "is_output", "is_buffer", "is_buffer_only_source")):
            continue
        parts = path.split(".")
        for index in range(1, len(parts) + 1):
            last[".".join(parts[:index])] = node.canonical_id
    mapping = {}
    for path, node_id in sorted(last.items()):
        mapping.setdefault("after:" + node_id, []).append(path)
    return mapping


class LayerCandidateProvider:
    """Return original descriptors whose semantic endpoints exist in the catalog."""

    def __init__(self, full_provider, boundary_modules):
        self.full_provider = full_provider
        self.boundary_modules = {cut: tuple(paths) for cut, paths in boundary_modules.items() if paths}
        self.catalog_diagnostics = {}
        if not self.boundary_modules:
            raise ValueError("Layer-only ablation needs captured module endpoint metadata")

    @classmethod
    def from_torchlens(cls, provider, *, training=True):
        provider.get_candidates(training=training)
        graph = provider._backends[bool(training)].runtime.trace_graph
        return cls(provider, module_end_boundaries(graph))

    def get_candidates(self, *, training=True):
        full = self.full_provider.get_candidates(training=training)
        retained = tuple(candidate for candidate in full if candidate.boundary in self.boundary_modules)
        if not retained:
            raise ValueError("No captured module endpoint is an admitted candidate")
        self.catalog_diagnostics[bool(training)] = {
            "scope": "captured_semantic_module_endpoints",
            "full_catalog_size": len(full), "catalog_size": len(retained),
            "boundary_modules": {value.boundary: self.boundary_modules[value.boundary] for value in retained},
            "definition": "After the final captured compute node of a module path and its descendants; only existing valid cuts are retained",
        }
        return retained

    def get_placement_plan(self, boundary, *, training=True, **kwargs):
        if boundary not in {value.boundary for value in self.get_candidates(training=training)}:
            raise ValueError("Selected cut is outside the semantic layer candidate domain")
        return self.full_provider.get_placement_plan(boundary, training=training, **kwargs)
