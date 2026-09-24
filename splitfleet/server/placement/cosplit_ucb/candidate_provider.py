"""Candidate discovery adapters for CoSplit-UCB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .types import SplitCandidateDescriptor


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
        max_candidates: int | None = None,
        kinds: tuple[str, ...] = ("before", "after"),
    ) -> None:
        if max_candidates is not None and max_candidates < 1:
            raise ValueError("max_candidates must be positive or None")
        self.model = model
        self.sample_inputs = sample_inputs
        self.sample_kwargs = dict(sample_kwargs or {})
        self.batch_axes = batch_axes
        self.mode = mode
        self.dynamic_batch = dynamic_batch
        self.trace_batch_mode = trace_batch_mode
        self.max_candidates = max_candidates
        self.kinds = kinds
        self.framework_backend = "unknown"
        self.runtime_backend = "torchlens_native"
        self._catalogs: dict[bool, tuple[SplitCandidateDescriptor, ...]] = {}

    def get_candidates(self, *, training: bool = True) -> Sequence[SplitCandidateDescriptor]:
        cached = self._catalogs.get(bool(training))
        if cached is not None:
            return cached
        # Backend-specific imports intentionally remain in this adapter. The
        # learner and solver modules have no Torch/PyTorch dependency.
        from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend
        from splitfleet.tasks import ModelInputs

        call = ModelInputs.from_value(self.sample_inputs)
        backend = TorchLensSplitBackend(model_name=self.model.__class__.__name__)
        backend.trace(
            self.model,
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
        self.framework_backend = str(backend.framework_backend)
        raw = backend.enumerate_candidates(max_candidates=None, kinds=self.kinds)
        descriptors: list[SplitCandidateDescriptor] = []
        for candidate in raw:
            validation = backend.validate_candidate(candidate)
            if not bool(validation.get("success")):
                continue
            handle = backend.make_handle()
            total_nodes = max(
                int(candidate.descriptor.get("prefix_node_count", 0))
                + int(candidate.descriptor.get("suffix_node_count", 0)),
                1,
            )
            prefix_nodes = int(candidate.descriptor.get("prefix_node_count", 0))
            descriptors.append(
                SplitCandidateDescriptor(
                    boundary=str(candidate.boundary),
                    split_id=str(handle.plan.split_id),
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
                    feature_abi_id=str(handle.plan.feature_abi_id),
                    graph_signature=str(handle.plan.graph_signature),
                    framework_backend=self.framework_backend,
                    runtime_backend=self.runtime_backend,
                    valid=True,
                    runtime_contract=dict(handle.plan.runtime_contract),
                    metadata={
                        "candidate_id": candidate.candidate_id,
                        "node_index": candidate.node_index,
                        "boundary_tensor_labels": tuple(candidate.boundary_tensor_labels),
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
        if self.max_candidates is not None and len(descriptors) > self.max_candidates:
            # Deterministic even-spacing retains graph coverage while leaving
            # ``None`` as the operation-level, unpruned default.
            count = self.max_candidates
            indexes = sorted(
                {round(index * (len(descriptors) - 1) / max(count - 1, 1)) for index in range(count)}
            )
            descriptors = [descriptors[index] for index in indexes]
        if not descriptors:
            raise RuntimeError("TorchLens did not expose any valid trainable split candidates")
        catalog = tuple(descriptors)
        self._catalogs[bool(training)] = catalog
        return catalog


__all__ = ["CandidateProvider", "StaticCandidateProvider", "TorchLensCandidateProvider"]
