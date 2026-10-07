"""Candidate discovery adapters for CoSplit-UCB."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, replace
from typing import Any, Protocol, Sequence

from .types import SplitCandidateDescriptor

LOGGER = logging.getLogger(__name__)


class CandidateProvider(Protocol):
    """Return a stable catalog of valid backend-neutral split descriptors."""

    def get_candidates(self, *, training: bool = True) -> Sequence[SplitCandidateDescriptor]:
        """Enumerate candidates, using a provider cache after the first trace."""


@dataclass
class StaticCandidateProvider:
    """Explicit catalog provider for deterministic tests and offline experiments."""

    candidates: Sequence[SplitCandidateDescriptor]

    def get_candidates(self, *, training: bool = True) -> Sequence[SplitCandidateDescriptor]:
        _ = training
        return tuple(self.candidates)


class TorchLensCandidateProvider:
    """Trace the real TorchLens graph and expose every valid operation-level cut."""

    def __init__(
        self,
        *,
        model: Any,
        sample_inputs: Any,
        sample_kwargs: dict[str, Any] | None = None,
        batch_axes: dict[str, int] | None = None,
        mode: str = "generated_eager",
        dynamic_batch: tuple[int, int] | None = None,
        trace_batch_mode: str | None = None,
        kinds: tuple[str, ...] = ("before", "after"),
        require_trainable_prefix: bool = False,
    ) -> None:
        self.model = model
        self.sample_inputs = sample_inputs
        self.sample_kwargs = dict(sample_kwargs or {})
        self.batch_axes = batch_axes
        self.mode = mode
        self.dynamic_batch = dynamic_batch
        self.trace_batch_mode = trace_batch_mode
        self.kinds = kinds
        self.require_trainable_prefix = bool(require_trainable_prefix)
        self.framework_backend = "unknown"
        self.runtime_backend = "torchlens_native"
        self._catalogs: dict[bool, tuple[SplitCandidateDescriptor, ...]] = {}
        self._backends: dict[bool, Any] = {}
        self._candidates: dict[bool, dict[str, Any]] = {}
        self._validations: dict[bool, dict[str, dict[str, Any]]] = {}
        self._prepared_handles: dict[tuple[bool, str], Any] = {}
        self.catalog_diagnostics: dict[bool, dict[str, Any]] = {}

    def get_candidates(self, *, training: bool = True) -> Sequence[SplitCandidateDescriptor]:
        cached = self._catalogs.get(bool(training))
        if cached is not None:
            return cached
        # Backend-specific imports intentionally remain in this adapter. The
        # learner and solver modules have no Torch/PyTorch dependency.
        from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend
        from splitfleet.backends.utils import adapter_for
        from splitfleet.tasks import ModelInputs

        call = ModelInputs.from_value(self.sample_inputs)
        model = self.model
        if not training:
            adapter = adapter_for(model, call.args if call.args else call.kwargs)
            model = adapter.clone_model(model)
            adapter.set_training(model, False)
        backend = TorchLensSplitBackend(model_name=self.model.__class__.__name__)
        backend.trace(
            model,
            call.args,
            sample_kwargs=dict(call.kwargs) if not self.sample_kwargs else self.sample_kwargs,
            batch_axes=self.batch_axes,
            boundary="auto",
            mode=self.mode,
            trainable=bool(training),
            dynamic_batch=self.dynamic_batch,
            trace_batch_mode=self.trace_batch_mode,
            model_name=self.model.__class__.__name__,
        )
        if training:
            # Broadcast the captured contract, including values inferred from
            # the server sample, rather than letting each worker infer its own.
            self.dynamic_batch = backend.dynamic_batch
            self.trace_batch_mode = backend.trace_batch_mode
        self.framework_backend = str(backend.framework_backend)
        cross_stage_spans = ()
        if self.framework_backend == "torch":
            from torchlens.split._torch_compact import compact_torch_graph
            graph = (compact_torch_graph(backend.runtime.trace_graph)
                     if backend.runtime.retains_trace else backend.runtime.trace_graph)
            locations: dict[int, list[int]] = {}
            for position, node in enumerate(graph.nodes):
                refs = list(node.param_refs) + list(node.buffer_refs)
                for ref in node.param_refs:
                    module = getattr(ref, "module", None)
                    if module is not None:
                        refs.extend(module.buffers.values())
                for ref in refs:
                    value = ref.handle
                    if value is not None:
                        locations.setdefault(id(value), []).append(position)
            cross_stage_spans = tuple((min(positions), max(positions))
                                      for positions in locations.values())
        pre_rejected: dict[str, str] = {}
        parameter_bytes = {}
        payload_sizes = {}
        if self.framework_backend == "torch":
            from splitfleet.autosplit.torchlens_candidate import ParameterCountIndex, ParameterByteIndex, payload_bytes_from_plan
            from torchlens.split.errors import SplitRequestError, SplitUnsupportedError, SplitBoundaryError
            from torchlens.split.planner import plan_split

            runtime = backend.runtime
            parameter_index = ParameterCountIndex.from_runtime(runtime)
            byte_index = ParameterByteIndex.from_runtime(runtime)
            low, high = self.dynamic_batch or (runtime.traced_batch_size, runtime.traced_batch_size)
            batches = set(range(low, high + 1)) if high - low < 64 else {low, high}
            batches.add(runtime.traced_batch_size)
            for site in runtime.split_points(diagnose=False).candidates:
                if site.kind not in self.kinds:
                    continue
                try:
                    plan = plan_split(runtime.trace_graph, replace(runtime.request, point=site.point))
                except (SplitRequestError, SplitUnsupportedError):
                    # Leave structural failures to TorchLens's normal report.
                    continue
                boundary = f"{plan.boundary_kind}:{plan.target_node_id}"
                parameter_bytes[boundary] = (byte_index.count(plan.prefix_node_ids),
                                             byte_index.count(plan.suffix_node_ids))
                sizes = {}
                for batch_size in sorted(batches):
                    try:
                        sizes[batch_size] = payload_bytes_from_plan(runtime, plan, batch_size=batch_size)
                    except SplitBoundaryError:
                        # Missing shape proofs remain unknown cost features;
                        # they do not silently change the existing catalog.
                        sizes[batch_size] = None
                payload_sizes[boundary] = sizes
                prefix_nodes = len(plan.prefix_node_ids)
                if training and not parameter_index.count(plan.suffix_node_ids, trainable_only=True):
                    pre_rejected[boundary] = "suffix_not_trainable"
                elif training and self.require_trainable_prefix and not parameter_index.count(
                    plan.prefix_node_ids, trainable_only=True
                ):
                    pre_rejected[boundary] = "prefix_not_trainable"
                elif any(first < prefix_nodes <= last for first, last in cross_stage_spans):
                    pre_rejected[boundary] = "state_shared_across_stages"
        LOGGER.info("TorchLens candidate capture ready; checking every requested operation boundary")
        # Capability and descriptor checks need no executable segments. Only
        # selected cuts are materialized with normal state inheritance.
        raw = backend.iter_candidates(kinds=self.kinds, excluded_boundaries=pre_rejected)
        raw_count = 0
        descriptors: list[SplitCandidateDescriptor] = []
        accepted_candidates = {}
        validations = {}
        rejected = dict(pre_rejected)
        for candidate in raw:
            raw_count += 1
            if raw_count % 200 == 0:
                LOGGER.info("TorchLens candidate check progress: supported=%d accepted=%d",
                            raw_count, len(descriptors))
            if training and candidate.descriptor.get("capabilities", {}).get("training_supported") is False:
                rejected[candidate.boundary] = "native_training_unsupported"
                continue
            if training and not bool(candidate.is_trainable_tail):
                rejected[candidate.boundary] = "suffix_not_trainable"
                continue
            if training and self.require_trainable_prefix:
                prefix_parameters = candidate.descriptor.get("trainable_prefix_parameter_count")
                if prefix_parameters is None or int(prefix_parameters) < 1:
                    rejected[candidate.boundary] = "prefix_not_trainable"
                    continue
            prefix_nodes = int(candidate.descriptor.get("prefix_node_count", 0))
            if any(first < prefix_nodes <= last for first, last in cross_stage_spans):
                rejected[candidate.boundary] = "state_shared_across_stages"
                continue
            # Native strict capability analysis and the state ownership checks
            # above establish catalog eligibility. Numerical replay is omitted;
            # the selected cut is rebuilt for actual training.
            validation = {
                "success": True,
                "verification": "native_capability_only",
                "replay_performed": False,
                "candidate_id": candidate.candidate_id,
                "split_id": candidate.boundary,
                "runtime": "torchlens_native",
                "tail_trainability": bool(candidate.is_trainable_tail),
                "error": None,
            }
            runtime_plan = backend.make_plan()
            accepted_candidates[candidate.boundary] = candidate
            validations[candidate.boundary] = dict(validation)
            total_nodes = max(
                int(candidate.descriptor.get("prefix_node_count", 0))
                + int(candidate.descriptor.get("suffix_node_count", 0)),
                1,
            )
            descriptors.append(
                SplitCandidateDescriptor(
                    boundary=str(candidate.boundary),
                    split_id=str(runtime_plan.split_id),
                    graph_position_ratio=prefix_nodes / total_nodes,
                    prefix_node_count=prefix_nodes,
                    suffix_node_count=int(candidate.descriptor.get("suffix_node_count", 0)),
                    total_node_count=total_nodes,
                    boundary_forward_bytes=int(candidate.estimated_payload_bytes),
                    # The activation payload is known from the captured shape
                    # program. Gradient envelopes may omit non-differentiable
                    # boundary values, so no byte count is invented here.
                    boundary_gradient_bytes=None,
                    boundary_tensor_count=int(candidate.boundary_count),
                    # TorchLens currently exposes parameter counts but not a
                    # backend-neutral byte size for every framework dtype.
                    prefix_parameter_bytes=None,
                    suffix_parameter_bytes=None,
                    client_memory_bytes=None,
                    server_memory_bytes=None,
                    trainable=bool(candidate.is_trainable_tail),
                    feature_abi_id=str(runtime_plan.feature_abi_id),
                    graph_signature=str(runtime_plan.graph_signature),
                    framework_backend=self.framework_backend,
                    runtime_backend=self.runtime_backend,
                    valid=True,
                    runtime_contract=dict(runtime_plan.runtime_contract),
                    metadata={
                        "optimizer_prefix_parameter_bytes": parameter_bytes.get(candidate.boundary, (None, None))[0],
                        "optimizer_suffix_parameter_bytes": parameter_bytes.get(candidate.boundary, (None, None))[1],
                        "boundary_forward_bytes_by_batch_size": payload_sizes.get(candidate.boundary, {}),
                        "payload_batch_size": candidate.descriptor.get("payload_batch_size"),
                        "candidate_id": candidate.candidate_id,
                        "node_index": candidate.node_index,
                        "boundary_tensor_labels": tuple(candidate.boundary_tensor_labels),
                        "prefix_trainable_parameter_count": (
                            int(candidate.descriptor["trainable_prefix_parameter_count"])
                            if training and self.require_trainable_prefix else None
                        ),
                    },
                )
            )
        descriptors.sort(
            key=lambda value: (
                value.graph_position_ratio,
                value.prefix_node_count,
                value.boundary,
            )
        )
        if not descriptors:
            raise RuntimeError("TorchLens did not expose any valid trainable split candidates")
        catalog = tuple(descriptors)
        key = bool(training)
        self._catalogs[key] = catalog
        self._backends[key] = backend
        self._candidates[key] = {value.boundary: accepted_candidates[value.boundary] for value in catalog}
        self._validations[key] = {value.boundary: validations[value.boundary] for value in catalog}
        report = backend.candidate_report or {}
        self.catalog_diagnostics[key] = {
            "scope": "all_valid_operation_boundaries",
            "enumerated_boundaries": report.get("total", raw_count),
            "unsupported_boundaries": max(0, report.get("unsupported", 0) - len(pre_rejected)),
            "prechecked_training_exclusions": len(pre_rejected),
            "unsupported_candidates": {
                f"{value['point']['kind']}:{value['point']['target']}": value["unsupported_reasons"]
                for value in report.get("candidates", [])
                if not value["replay_supported"] and not any(
                    str(reason).startswith("training_catalog_rejection:")
                    for reason in value["unsupported_reasons"]
                )
            },
            "replay_supported_boundaries": raw_count,
            "structurally_supported_training_candidates": len(accepted_candidates),
            "numeric_replay_validations": 0,
            "catalog_size": len(catalog),
            "rejected_candidates": rejected,
        }
        return catalog

    def get_placement_plan(
        self,
        boundary: str,
        *,
        worker_specs,
        constraints,
        objective,
        model_name: str | None = None,
        training: bool = True,
    ):
        """Prepare selected cuts from the catalog capture, without retracing.

        Keep descriptors for every valid cut, but retain execution handles
        only for cuts actually requested by the placement policy.
        """

        key = bool(training)
        self.get_candidates(training=key)
        requested = str(boundary)
        candidate = self._candidates[key].get(requested)
        if candidate is None:
            raise ValueError(f"Split boundary {requested!r} is not in the supported catalog")
        cache_key = (key, requested)
        handle = self._prepared_handles.get(cache_key)
        if handle is None:
            backend = copy.copy(self._backends[key])
            backend.split(candidate)
            handle = backend.make_handle()
            self._prepared_handles[cache_key] = handle
        from splitfleet.autosplit.planner import (
            _build_placement, _candidate_satisfies_constraints, _rejection_reason,
        )

        effective_constraints = constraints if key else replace(constraints, require_trainable_tail=False)
        validation = self._validations[key][requested]
        if not _candidate_satisfies_constraints(candidate, validation, effective_constraints):
            reason = _rejection_reason(candidate, validation, effective_constraints)
            raise RuntimeError(f"Split boundary {requested!r} violates placement constraints: {reason}")

        return _build_placement(
            handle,
            candidate=candidate,
            validation=validation,
            worker_specs=worker_specs,
            constraints=effective_constraints,
            objective=objective,
            model_name=model_name or self.model.__class__.__name__,
        )


__all__ = ["CandidateProvider", "StaticCandidateProvider", "TorchLensCandidateProvider"]
