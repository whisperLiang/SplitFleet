"""Memory-bounded training capture for the pinned TorchLens 2.34.1 runtime.

Replay recomputes ordinary activations. It needs their operation/shape/gradient
metadata, live parameters, and the values of sources without a producer. This
extension retains those sources in the normalized execution graph, then releases
the temporary diagnostic Trace. Explicit retain_trace=True keeps
the upstream full-value capture. Installed TorchLens globals are never changed.
"""

from __future__ import annotations

import importlib
import types
from typing import Any


def _clone(function, **bindings):
    """Bind dependencies for this preparation without patching a module."""
    local = types.FunctionType(function.__code__, dict(function.__globals__, **bindings),
                               function.__name__, function.__defaults__, function.__closure__)
    local.__kwdefaults__ = function.__kwdefaults__
    return local


def prepare_training_runtime(model: Any, inputs: tuple[Any, ...], request: Any,
                             *, input_kwargs: dict[str, Any] | None = None):
    """Keep training/state/RNG/validation semantics with a compact graph.

    The caller must first require TorchLens 2.34.1. This entry point is only for
    Torch training with default diagnostic retention; all other requests use
    the unmodified public prepare API.
    """
    if not request.trainable or request.features.retain_trace is not None:
        raise ValueError("Selective capture requires training with default trace retention")
    api = importlib.import_module("torchlens.split.api")
    pipeline = importlib.import_module("torchlens.split.pipeline")
    graph_module = importlib.import_module("torchlens.split.graph")
    from torchlens._errors import PayloadUnavailableError
    from torchlens.fastlog.types import CaptureSpec
    from torchlens.user_funcs import trace

    def capture_sources(model, inputs, spec, *, input_kwargs=None,
                        output_observer=None, random_seed=None, shape_witness=False):
        if shape_witness:
            return pipeline.capture_model(model, inputs, spec, input_kwargs=input_kwargs,
                output_observer=output_observer, random_seed=random_seed, shape_witness=True)
        metadata = {}

        def retain(context):
            metadata[context.raw_label] = context.tensor_requires_grad
            return CaptureSpec(
                save_out=context.kind in ("input", "buffer") or not context.parent_labels,
                save_metadata=True, keep_grad=context.tensor_requires_grad is True,
            )

        options = dict(input_kwargs=input_kwargs, save=retain, keep_orphans=True,
            backend="torch", intervention_ready=True, capture_container_structure=True,
            save_arg_values=True, save_rng_states=True, detach_saved_activations=False,
            backward_ready=spec.trainable)
        if output_observer is not None:
            options["output_transform"] = output_observer
        if random_seed is not None:
            options["random_seed"] = random_seed
        with pipeline._torch_capture_state(model):
            captured = trace(model, inputs, **options)
        captured._splitfleet_requires_grad_metadata = metadata
        return captured

    def graph_from_capture(captured):
        metadata = captured._splitfleet_requires_grad_metadata

        def requires_grad(op):
            try:
                return graph_module._requires_grad_for_op(op)
            except PayloadUnavailableError:
                # Upstream derives this boolean from a saved tensor. Use the
                # actual capture-time boolean when that snapshot was omitted.
                label = getattr(op, "_label_raw", None)
                if label in metadata:
                    return metadata[label]
                parents = getattr(op, "parents", ()) or ()
                if getattr(op, "is_output", False) and len(parents) == 1:
                    parent = captured[parents[0]]
                    if hasattr(parent, "ops"):
                        if len(parent.ops) != 1:
                            raise ValueError("Output alias has ambiguous producing passes")
                        parent = parent.ops[0]
                    return requires_grad(parent)
                # Refuse unknown metadata instead of guessing or inventing an
                # activation. Missing replay source values still fail upstream.
                raise

        node = _clone(graph_module._node_from_op, _requires_grad_for_op=requires_grad)
        build = _clone(graph_module.split_graph_from_trace, _node_from_op=node)
        return build(captured)

    canonical = _clone(pipeline.capture_canonical_model, capture_model=capture_sources)
    normalized = _clone(pipeline.normalize_to_split_ir, split_graph_from_trace=graph_from_capture)

    def compact_normalize(captured, spec, **options):
        # The pinned compactor preserves live parameter/buffer identities,
        # argument templates, RNG/autocast and producer-less source payloads.
        # Release the unused capture graph before the independent batch probe.
        options["_release_capture"] = True
        try:
            return normalized(captured, spec, **options)
        finally:
            captured.cleanup()

    prepared = _clone(api.prepare, capture_canonical_model=canonical,
                      normalize_to_split_ir=compact_normalize)
    runtime = prepared(model, inputs, request, input_kwargs=input_kwargs)
    runtime._trace = None  # The temporary Trace was explicitly released above.
    return runtime
