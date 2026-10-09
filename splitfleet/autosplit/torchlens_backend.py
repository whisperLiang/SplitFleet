"""Backend facade for TorchLens native autosplit runtimes."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
from splitfleet.autosplit.boundary import (
    BoundaryPayload,
    from_torchlens_boundary,
    to_torchlens_boundary,
)
from splitfleet.autosplit.torchlens_candidate import (
    ParameterCountIndex,
    SplitCandidate,
    build_candidate_descriptor,
    candidate_from_plan,
)
from splitfleet.autosplit.torchlens_contract import build_runtime_contract
from splitfleet.autosplit.torchlens_runtime import (
    TORCHLENS_NATIVE_RUNTIME_ADAPTER_VERSION,
    make_split_spec,
    _point,
    normalize_example_inputs,
    normalize_model_call,
    prepare_split_runtime,
    repartition_split_runtime,
    runtime_input_batch_size,
    torchlens_runtime_version,
    trace_signature,
)
from splitfleet.autosplit.types import SplitRuntimePlan
from splitfleet.runtime.torch_suffix_training import train_torch_suffix


@dataclass
class TorchLensRuntimeHandle:
    model: torch.nn.Module
    runtime: Any
    plan: SplitRuntimePlan
    backend: "TorchLensSplitBackend"

    @property
    def feature_abi_id(self) -> str:
        return self.plan.feature_abi_id

    @feature_abi_id.setter
    def feature_abi_id(self, value: str) -> None:
        self.plan.feature_abi_id = value


def _make_plan_id(graph_signature: str, split_id: str, boundary: str, mode: str, *,
                  trainable: bool, training: bool) -> str:
    # Some graphs (e.g. Linear/ReLU) have identical operations in train/eval;
    # their captures must still occupy different runtime registry entries.
    digest = hashlib.sha1(
        "|".join([graph_signature, split_id, boundary, mode, str(trainable), str(training)]).encode("utf-8")
    ).hexdigest()
    return f"torchlens_{digest[:12]}"


def _runtime_device(runtime: Any) -> torch.device | None:
    model = runtime.model
    if isinstance(model, torch.nn.Module):
        for tensor in model.parameters():
            return tensor.device
        for tensor in model.buffers():
            return tensor.device
    return None


def _move_to_device(value: Any, device: torch.device | str | None) -> Any:
    if device is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


class TorchLensSplitBackend:
    """SplitFleet facade over TorchLens native SplitRuntime."""

    def __init__(
        self,
        *,
        device: str | torch.device = "cpu",
        model_name: str | None = None,
        model_family: str | None = None,
    ) -> Any:
        self.device = torch.device(device)
        self.model: torch.nn.Module | None = None
        self.runtime: Any | None = None
        self.split_spec: Any | None = None
        self.current_candidate: SplitCandidate | None = None
        self.candidates: list[SplitCandidate] = []
        self.trace_sample_input: Any = None
        self.trace_sample_kwargs: dict[str, Any] = {}
        self.dynamic_batch: tuple[int, int] | None = None
        self.trace_batch_mode = "batch_gt1"
        self.candidate_report: dict[str, Any] | None = None
        self.model_name = model_name
        self.model_family = model_family
        self.trace_batch_size: int | None = None
        self.framework_backend = "torch"
        self._candidate_analysis: Any | None = None

    def trace(
        self,
        model: torch.nn.Module,
        sample_inputs: Any,
        *,
        split_spec: Any = None,
        sample_kwargs: dict[str, Any] | None = None,
        batch_axes: dict[str, int] | None = None,
        boundary: str = "50%",
        mode: str = "generated_eager",
        trainable: bool = True,
        dynamic_batch: tuple[int, int] | None = None,
        trace_batch_mode: str | None = None,
        model_name: str | None = None,
        model_family: str | None = None,
    ) -> "TorchLensSplitBackend":
        if mode != "generated_eager":
            raise ValueError(f"Unsupported split execution mode {mode!r}; use 'generated_eager'.")
        self.model = model
        self.model_name = model_name or self.model_name or model.__class__.__name__
        self.model_family = model_family or self.model_family or self.model_name
        self.trace_sample_input = normalize_example_inputs(sample_inputs)
        self.trace_sample_kwargs = dict(sample_kwargs or {})
        self.trace_sample_input, self.trace_sample_kwargs, batch_axes = normalize_model_call(
            model, self.trace_sample_input, self.trace_sample_kwargs, batch_axes,
        )
        from splitfleet.backends.utils import detect_torchlens_backend
        complete_inputs = (self.trace_sample_input, self.trace_sample_kwargs)
        self.framework_backend = detect_torchlens_backend(model, complete_inputs)
        if dynamic_batch is not None:
            if len(dynamic_batch) != 2 or not 1 <= dynamic_batch[0] <= dynamic_batch[1]:
                raise ValueError("dynamic_batch must be a positive inclusive (minimum, maximum) range")
        probe_boundary = "50%" if str(boundary) == "auto" else str(boundary)
        self.split_spec = split_spec or make_split_spec(
            probe_boundary,
            trainable=trainable,
            backend=self.framework_backend,
            batch_axes=batch_axes,
        )
        self.runtime = prepare_split_runtime(
            model,
            self.trace_sample_input,
            self.split_spec,
            input_kwargs=self.trace_sample_kwargs,
        )
        self.trace_batch_size = self.runtime.batch_spec.user_batch_size
        self.trace_batch_mode = trace_batch_mode or (
            "batch_gt1" if (self.trace_batch_size or 1) > 1 else "batch_1"
        )
        self.dynamic_batch = dynamic_batch
        if self.dynamic_batch is None and self.runtime.batch_spec.axes:
            self.dynamic_batch = (2, 64) if self.trace_batch_mode == "batch_gt1" else (1, 64)
        self.current_candidate = candidate_from_plan(
            self.runtime,
            self.split_spec,
            self.runtime.plan,
            graph_signature=trace_signature(self.runtime),
        )
        self.current_candidate = self._with_batch_contract(self.current_candidate)
        self.candidates = [self.current_candidate]
        self._candidate_analysis = None
        return self

    def _ensure_runtime(self) -> Any:
        if self.runtime is None:
            raise RuntimeError("TorchLens split runtime has not been traced.")
        return self.runtime

    def _ensure_model(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("TorchLens split backend has no model.")
        return self.model

    def _with_batch_contract(self, candidate: SplitCandidate) -> SplitCandidate:
        return replace(candidate, dynamic_batch=self.dynamic_batch, trace_batch_mode=self.trace_batch_mode)

    def split_points(self, *, diagnose: bool = True):
        """Report every before/after compute boundary, including refusal reasons."""
        report = self._ensure_runtime().split_points(diagnose=diagnose)
        self.candidate_report = report.as_dict()
        return report

    def iter_candidates(
        self,
        *,
        kinds: tuple[str, ...] = ("before", "after"),
        excluded_boundaries: dict[str, str] | None = None,
    ):
        """Inspect each requested cut without building its executables.

        The public undiagnosed site list covers every operation boundary.
        ``analyze`` performs strict capability checks on the shared capture.
        Its plan describes a catalog cut; selected cuts are materialized later
        for training. ``make_plan`` exposes the current catalog metadata.
        Restore the original selection when enumeration finishes.
        """

        from torchlens.split.candidates import SplitCandidateReport
        from torchlens.split.errors import SplitRequestError, SplitUnsupportedError

        if not kinds or any(kind not in ("before", "after") for kind in kinds):
            raise ValueError("Candidate kinds must contain 'before' and/or 'after'.")
        base = self._ensure_runtime()
        original_candidate, original_spec = self.current_candidate, self.split_spec
        original_analysis = self._candidate_analysis
        indexes = {node.canonical_id: index for index, node in enumerate(base.trace_graph.nodes)}
        parameter_index = ParameterCountIndex.from_runtime(base)
        reports = []
        candidates = []
        try:
            for site in base.split_points(diagnose=False).candidates:
                if site.kind not in kinds:
                    continue
                boundary = f"{site.kind}:{site.node_id}"
                reason = (excluded_boundaries or {}).get(boundary)
                if reason is not None:
                    reports.append(replace(
                        site,
                        replay_supported=False,
                        training_supported=False,
                        unsupported_reasons=(f"training_catalog_rejection:{reason}",),
                    ))
                    continue
                try:
                    analysis = base.analyze(site.point)
                except (SplitRequestError, SplitUnsupportedError) as exc:
                    reports.append(replace(site, unsupported_reasons=(str(exc),)))
                    continue
                capability = analysis.capability_report
                reasons = () if capability is None else tuple(capability.unsupported_reasons)
                shape_program = base.trace_graph.shape_program
                diagnosed = replace(
                    site,
                    boundary_value_ids=analysis.plan.boundary_node_ids,
                    boundary_schema=analysis.graph_ir.boundary_schema,
                    replay_supported=not reasons,
                    training_supported=not reasons and (
                        capability is None or capability.training.supported
                    ),
                    unsupported_reasons=reasons,
                    shape_unresolved=() if shape_program is None else tuple(shape_program.unresolved),
                )
                reports.append(diagnosed)
                if not diagnosed.replay_supported:
                    continue
                candidate = self._with_batch_contract(candidate_from_plan(
                    base, analysis.request, analysis.plan,
                    node_index=indexes[site.node_id],
                    graph_signature=analysis.graph_ir.graph_hash,
                    parameter_index=parameter_index,
                ))
                candidate = replace(
                    candidate,
                    is_trainable_tail=candidate.is_trainable_tail and diagnosed.training_supported,
                )
                candidate.descriptor["capabilities"] = diagnosed.as_dict()
                self.split_spec, self.current_candidate = analysis.request, candidate
                self._candidate_analysis = analysis
                candidates.append(candidate)
                yield candidate
        finally:
            self.runtime, self.current_candidate, self.split_spec = base, original_candidate, original_spec
            self._candidate_analysis = original_analysis
            self.candidates = candidates
            self.candidate_report = SplitCandidateReport(tuple(reports)).as_dict()

    def repartition(self, boundary: str) -> TorchLensRuntimeHandle:
        """Select another boundary while reusing capture and native batch semantics."""
        runtime = self._ensure_runtime()
        spec = replace(runtime.request, point=_point(boundary))
        self.runtime = repartition_split_runtime(runtime, spec)
        self._candidate_analysis = None
        self.split_spec = self.runtime.request
        self.current_candidate = self._with_batch_contract(candidate_from_plan(
            self.runtime, self.split_spec, self.runtime.plan,
            graph_signature=trace_signature(self.runtime),
        ))
        return self.make_handle()

    def split(self, candidate: SplitCandidate | None = None) -> SplitCandidate:
        chosen = candidate or self.current_candidate
        if chosen is None:
            raise RuntimeError("No TorchLens split candidate is selected.")
        self.repartition(chosen.boundary)
        self.current_candidate = replace(self.current_candidate, node_index=chosen.node_index, layer_index=chosen.layer_index)
        return self.current_candidate

    def run_prefix(
        self, *inputs: Any, training: bool = False,
        input_kwargs: dict[str, Any] | None = None,
    ) -> BoundaryPayload:
        runtime = self._ensure_runtime()
        args = normalize_example_inputs(inputs)
        kwargs = self.trace_sample_kwargs if input_kwargs is None else input_kwargs
        args, kwargs, _ = normalize_model_call(self._ensure_model(), args, kwargs)
        batch_size = runtime_input_batch_size(runtime, args, kwargs)
        if self.dynamic_batch is not None and batch_size is not None:
            low, high = self.dynamic_batch
            if not low <= batch_size <= high:
                raise ValueError(f"Input batch size {batch_size} is outside configured dynamic_batch={self.dynamic_batch}.")
        if training and self.framework_backend == "tf":
            from splitfleet.runtime.tensorflow_prefix_training import run_tensorflow_training_prefix

            raw = run_tensorflow_training_prefix(runtime, *args, input_kwargs=kwargs)
        else:
            raw = (
                runtime.run_training_prefix(*args, input_kwargs=kwargs)
                if training else runtime.run_prefix(*args, input_kwargs=kwargs)
            )
        return from_torchlens_boundary(raw)

    def run_suffix(self, boundary: BoundaryPayload) -> Any:
        runtime = self._ensure_runtime()
        native = to_torchlens_boundary(boundary)
        device = _runtime_device(runtime)
        if device is not None:
            native = native.to(device)
        return runtime.run_suffix(native)

    def train_suffix(
        self,
        boundary: BoundaryPayload,
        targets: Any,
        *,
        loss_fn=None,
        optimizer=None,
        measurements: dict[str, float] | None = None,
    ):
        """Run the suffix training step, optionally filling ``measurements``.

        Every backend reports ``server_total_ms``. Backends whose suffix step is
        executed phase by phase also report ``server_forward_ms`` and
        ``server_backward_ms``. PyTorch additionally reports ``server_loss_ms``
        and measures its complete suffix step as a wall span. Backends without
        separate phases do not report a fabricated split.
        """

        if loss_fn is None:
            raise ValueError("Split training requires an explicit loss_fn.")
        runtime = self._ensure_runtime()
        native = to_torchlens_boundary(boundary)
        device = _runtime_device(runtime)
        if device is not None:
            native = native.to(device)
            targets = _move_to_device(targets, device)
        if self.framework_backend == "torch":
            return train_torch_suffix(
                runtime,
                native,
                targets,
                loss_fn=loss_fn,
                optimizer=optimizer,
                measurements=measurements,
            )
        if measurements is not None:
            started = time.perf_counter_ns()
            result = runtime.train_suffix(native, targets, loss_fn=loss_fn, optimizer=optimizer)
            measurements["server_total_ms"] = (time.perf_counter_ns() - started) / 1_000_000.0
            return result
        return runtime.train_suffix(native, targets, loss_fn=loss_fn, optimizer=optimizer)

    def backward_prefix(
        self,
        boundary: BoundaryPayload,
        boundary_grads: Any,
        *,
        optimizer=None,
    ) -> Any:
        runtime = self._ensure_runtime()
        native = to_torchlens_boundary(boundary)
        device = _runtime_device(runtime)
        if device is not None:
            native = native.to(device)
            boundary_grads = _move_to_device(boundary_grads, device)
        return runtime.backward_prefix(native, boundary_grads=boundary_grads, optimizer=optimizer)

    def make_plan(self) -> SplitRuntimePlan:
        """Describe the selected runtime or the current catalog analysis."""
        runtime = self._ensure_runtime()
        candidate = self.current_candidate
        if candidate is None:
            raise RuntimeError("No TorchLens split candidate is selected.")
        selected = self._candidate_analysis or runtime
        plan = selected.plan
        split_spec = selected.request
        boundary_schema = dict(candidate.descriptor.get("boundary_schema") or {})
        feature_layout = dict(candidate.descriptor.get("feature_layout") or {})
        contract = build_runtime_contract(
            model_family=str(self.model_family or self.model_name or ""),
            canonical_split_key=candidate.boundary,
            graph_signature=trace_signature(runtime),
            boundary_tensor_labels=list(candidate.boundary_tensor_labels),
            boundary_schema=boundary_schema,
            feature_layout=feature_layout,
            runtime_backend="torchlens_native",
            adapter_version=TORCHLENS_NATIVE_RUNTIME_ADAPTER_VERSION,
            runtime_version=torchlens_runtime_version(),
            trace_batch_mode=self.trace_batch_mode,
            dynamic_batch=self.dynamic_batch,
            trace_batch_size=self.trace_batch_size,
        )
        actual_split_id = str(candidate.boundary)
        from splitfleet.backends.utils import model_training

        return SplitRuntimePlan(
            plan_id=_make_plan_id(
                trace_signature(runtime),
                actual_split_id,
                candidate.boundary,
                "generated_eager",
                trainable=bool(split_spec.features.training),
                training=model_training(self._ensure_model()),
            ),
            split_id=actual_split_id,
            graph_signature=trace_signature(runtime),
            boundary=candidate.boundary,
            mode="generated_eager",
            trainable=split_spec.features.training,
            dynamic_batch=self.dynamic_batch,
            trace_batch_mode=self.trace_batch_mode,
            trace_batch_size=self.trace_batch_size,
            boundary_bytes=int(candidate.estimated_payload_bytes),
            prefix_node_count=len(plan.prefix_node_ids),
            suffix_node_count=len(plan.suffix_node_ids),
            trainable_suffix=bool(candidate.is_trainable_tail),
            candidate_id=candidate.candidate_id,
            split_label=candidate.split_label,
            boundary_tensor_labels=list(candidate.boundary_tensor_labels),
            torchlens_version=torchlens_runtime_version(),
            feature_abi_id=str(contract.get("feature_abi_id", "")),
            runtime_contract=contract,
            metadata={
                "boundary_nodes": tuple(candidate.boundary_tensor_labels),
                "requested_boundary": split_spec.boundary,
                "candidate_descriptor": build_candidate_descriptor(candidate),
                "feature_abi_id": contract.get("feature_abi_id", ""),
                "_example_inputs": self.trace_sample_input,
                "_example_kwargs": self.trace_sample_kwargs,
                "batch_validation": runtime.batch_validation,
                "candidate_report": self.candidate_report,
            },
        )

    def make_handle(self) -> TorchLensRuntimeHandle:
        if self._candidate_analysis is not None:
            raise RuntimeError("Select a catalog candidate with split() before requesting an executable handle.")
        return TorchLensRuntimeHandle(
            model=self._ensure_model(),
            runtime=self._ensure_runtime(),
            plan=self.make_plan(),
            backend=self,
        )


def prepare_torchlens_runtime(
    model: torch.nn.Module,
    sample_inputs: Any,
    *,
    boundary: str = "50%",
    mode: str = "generated_eager",
    trainable: bool = True,
    dynamic_batch: tuple[int, int] | None = None,
    trace_batch_mode: str | None = None,
    model_name: str | None = None,
    model_family: str | None = None,
    candidate: SplitCandidate | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    batch_axes: dict[str, int] | None = None,
) -> TorchLensRuntimeHandle:
    backend = TorchLensSplitBackend(model_name=model_name, model_family=model_family)
    backend.trace(
        model,
        sample_inputs,
        sample_kwargs=sample_kwargs,
        batch_axes=batch_axes,
        boundary=boundary,
        mode=mode,
        trainable=trainable,
        dynamic_batch=dynamic_batch,
        trace_batch_mode=trace_batch_mode,
        model_name=model_name,
        model_family=model_family,
    )
    if candidate is not None:
        backend.split(candidate)
    return backend.make_handle()


def run_prefix(handle: TorchLensRuntimeHandle, *inputs: Any, input_kwargs=None) -> BoundaryPayload:
    return handle.backend.run_prefix(*inputs, input_kwargs=input_kwargs)


def run_training_prefix(handle: TorchLensRuntimeHandle, *inputs: Any, input_kwargs=None) -> BoundaryPayload:
    return handle.backend.run_prefix(*inputs, training=True, input_kwargs=input_kwargs)


def run_suffix(handle: TorchLensRuntimeHandle, boundary: BoundaryPayload) -> Any:
    return handle.backend.run_suffix(boundary)


def train_suffix(
    handle: TorchLensRuntimeHandle,
    boundary: BoundaryPayload,
    targets: Any,
    *,
    loss_fn=None,
    optimizer=None,
    measurements: dict[str, float] | None = None,
):
    return handle.backend.train_suffix(
        boundary,
        targets,
        loss_fn=loss_fn,
        optimizer=optimizer,
        measurements=measurements,
    )


def backward_prefix(
    handle: TorchLensRuntimeHandle,
    boundary: BoundaryPayload,
    boundary_grads: Any,
    *,
    optimizer=None,
) -> Any:
    return handle.backend.backward_prefix(
        boundary,
        boundary_grads,
        optimizer=optimizer,
    )


__all__ = [
    "TorchLensRuntimeHandle",
    "TorchLensSplitBackend",
    "backward_prefix",
    "prepare_torchlens_runtime",
    "run_prefix",
    "run_suffix",
    "run_training_prefix",
    "train_suffix",
]
