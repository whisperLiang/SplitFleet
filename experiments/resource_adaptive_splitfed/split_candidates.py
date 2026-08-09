"""Automatic TorchLens candidate discovery with stable semantic split keys."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend


SEMANTIC_SPLIT_ORDER = (
    "stem",
    "maxpool",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "full_local",
)


@dataclass
class ExperimentSplitCandidate:
    split_key: str
    boundary: str | None
    graph_node_id: str | None
    graph_node_label: str | None
    module_path: str | None
    node_index: int
    boundary_tensor_labels: list[str]
    boundary_schema: dict[str, Any]
    feature_layout: dict[str, Any]
    boundary_forward_bytes: int
    boundary_gradient_bytes: int
    client_parameter_names: list[str]
    server_parameter_names: list[str]
    graph_signature: str
    mapping_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_split_candidates(
    model: torch.nn.Module,
    sample_inputs: Any,
    *,
    dynamic_batch: tuple[int, int] = (1, 8),
) -> list[ExperimentSplitCandidate]:
    was_training = model.training
    # Training and evaluation captures can have different BatchNorm graph
    # labels. Experiment cuts are therefore discovered from the actual training
    # graph; the caller's original mode is restored after capture.
    model.train()
    backend = TorchLensSplitBackend(model_name=model.__class__.__name__)
    try:
        backend.trace(
            model,
            sample_inputs,
            boundary="50%",
            trainable=True,
            dynamic_batch=(max(2, dynamic_batch[0]), max(2, dynamic_batch[1])),
            trace_batch_mode="batch_gt1",
        )
    finally:
        model.train(was_training)
    raw = sorted(
        backend.enumerate_candidates(max_boundary_count=1),
        key=lambda candidate: int(candidate.node_index or 0),
    )
    # Every semantic key except ``full_local`` consumes one distinct boundary, so
    # the graph must expose at least that many trainable cuts.
    required = len(SEMANTIC_SPLIT_ORDER) - 1
    if len(raw) < required:
        raise RuntimeError(
            f"TorchLens exposed {len(raw)} trainable split candidates; "
            f"{required} distinct cuts are required for {list(SEMANTIC_SPLIT_ORDER[:-1])}."
        )
    graph = backend.runtime.trace_graph
    node_for = {int(candidate.node_index): graph.nodes[int(candidate.node_index)] for candidate in raw}

    def matching(prefixes: Sequence[str], exact: Sequence[str] = ()):
        matches = []
        for candidate in raw:
            node = node_for[int(candidate.node_index)]
            path = str(getattr(node, "module_path", "") or "")
            if path in exact or any(path == prefix or path.startswith(prefix + ".") for prefix in prefixes):
                matches.append(candidate)
        return matches

    semantic_matches = {
        "stem": matching((), ("conv1", "bn1", "relu")),
        "maxpool": matching((), ("maxpool",)),
        "layer1": matching(("layer1",)),
        "layer2": matching(("layer2",)),
        "layer3": matching(("layer3",)),
        "layer4": matching(("layer4",)),
    }
    quantiles = {"stem": 0.05, "maxpool": 0.12, "layer1": 0.25, "layer2": 0.45, "layer3": 0.65, "layer4": 0.88}
    selected: dict[str, tuple[Any, str]] = {}
    used: set[str] = set()
    for key in SEMANTIC_SPLIT_ORDER[:-1]:
        matches = [candidate for candidate in semantic_matches[key] if candidate.boundary not in used]
        if matches:
            chosen, source = max(matches, key=lambda candidate: int(candidate.node_index or 0)), "module_path"
        else:
            target = round(quantiles[key] * (len(raw) - 1))
            ordered = sorted(raw, key=lambda candidate: abs(raw.index(candidate) - target))
            chosen = next(
                (candidate for candidate in ordered if candidate.boundary not in used), None
            )
            if chosen is None:
                raise RuntimeError(
                    f"No unused trainable split candidate remains for {key!r}; "
                    f"{len(raw)} candidates were exposed and {len(used)} are already assigned."
                )
            source = "graph_quantile_fallback"
        selected[key] = (chosen, source)
        used.add(chosen.boundary)

    all_parameters = [name for name, _ in model.named_parameters()]
    descriptors: list[ExperimentSplitCandidate] = []
    for key in SEMANTIC_SPLIT_ORDER[:-1]:
        candidate, mapping_source = selected[key]
        handle = backend.repartition(candidate.boundary)
        node = handle.runtime.trace_graph.nodes[int(candidate.node_index)]
        prefix_names = _parameter_names_for_node_ids(
            handle.runtime,
            list(getattr(handle.runtime.plan, "prefix_node_ids", ()) or ()),
            all_parameters,
        )
        schema = dict(candidate.descriptor.get("boundary_schema") or {})
        descriptors.append(
            ExperimentSplitCandidate(
                split_key=key,
                boundary=candidate.boundary,
                graph_node_id=str(getattr(node, "canonical_id", "") or ""),
                graph_node_label=str(getattr(node, "label", "") or candidate.split_label),
                module_path=str(getattr(node, "module_path", "") or ""),
                node_index=int(candidate.node_index),
                boundary_tensor_labels=list(candidate.boundary_tensor_labels),
                boundary_schema=schema,
                feature_layout=dict(candidate.descriptor.get("feature_layout") or {}),
                boundary_forward_bytes=int(candidate.estimated_payload_bytes),
                boundary_gradient_bytes=int(candidate.estimated_payload_bytes),
                client_parameter_names=prefix_names,
                server_parameter_names=[name for name in all_parameters if name not in prefix_names],
                graph_signature=candidate.graph_signature,
                mapping_source=mapping_source,
            )
        )
    descriptors.append(
        ExperimentSplitCandidate(
            split_key="full_local",
            boundary=None,
            graph_node_id=None,
            graph_node_label=None,
            module_path=None,
            node_index=len(graph.nodes),
            boundary_tensor_labels=[],
            boundary_schema={},
            feature_layout={},
            boundary_forward_bytes=0,
            boundary_gradient_bytes=0,
            client_parameter_names=all_parameters,
            server_parameter_names=[],
            graph_signature=descriptors[0].graph_signature,
            mapping_source="full_model",
        )
    )
    _validate_descriptors(descriptors, all_parameters)
    return descriptors


def _parameter_names_for_node_ids(runtime: Any, node_ids: Sequence[str], known: Sequence[str]) -> list[str]:
    selected = set(str(node_id) for node_id in node_ids)
    names: set[str] = set()
    for node in runtime.trace_graph.nodes:
        if str(getattr(node, "canonical_id", "")) not in selected:
            continue
        refs = list(getattr(node, "param_refs", ()) or ())
        for ref in refs:
            address = str(getattr(ref, "address", "") or "")
            if address in known:
                names.add(address)
    return [name for name in known if name in names]


def _validate_descriptors(
    descriptors: Sequence[ExperimentSplitCandidate], all_parameters: Sequence[str]
) -> None:
    if [item.split_key for item in descriptors] != list(SEMANTIC_SPLIT_ORDER):
        raise RuntimeError("Candidate keys are not in canonical early-to-late order.")
    for item in descriptors:
        owned = item.client_parameter_names + item.server_parameter_names
        if len(owned) != len(set(owned)) or set(owned) != set(all_parameters):
            raise RuntimeError(f"Parameter ownership is incomplete for {item.split_key!r}.")
