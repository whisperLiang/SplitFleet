"""Torch suffix training compatible with in-place residual operations."""

from __future__ import annotations

import time
from contextlib import nullcontext
from contextvars import ContextVar
import itertools
import threading
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torchlens.split import ReplayBoundary


_RANDOM_FACTORY_OPS = frozenset({
    "aten::rand", "aten::randn", "aten::randint", "aten::randperm",
    "aten::rand_like", "aten::randn_like", "aten::randint_like",
})
_GENERATOR_OVERLOADS = {
    "": "generator", "out": "generator_out",
    "names": "generator_with_names", "names_out": "generator_with_names_out",
    "low": "low_generator", "low_out": "low_generator_out",
    "Tensor": "Tensor_generator", "Tensor_out": "Tensor_generator_out",
    "low_dtype": "low_generator_dtype", "low_dtype_out": "low_generator_dtype_out",
}
_REPLICA_SEED_COUNTER = itertools.count()
_REPLAY_REPLICA_RNG = ContextVar("splitfleet_replay_replica_rng", default=None)
_FUSED_ATTENTION_OPS = frozenset({
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_efficient_attention_backward",
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_flash_attention_backward",
    "aten::_scaled_dot_product_cudnn_attention",
    "aten::_scaled_dot_product_cudnn_attention_backward",
})


class _ReplicaRandomMode(TorchDispatchMode):
    """Route Torch random operators to one replica's private generator.

    PyTorch's Python APIs expose a ``generator`` argument, while many modules
    (notably ``Dropout``) call the generator-free ATen overload internally.
    Dispatch modes are thread-local, so each suffix execution lane can keep a
    private generator without changing process-wide RNG state or serializing
    other lanes.  Operations which already carry an explicit generator pass
    through unchanged.
    """

    def __init__(self, owner: "TorchReplicaRNG") -> None:
        super().__init__()
        self.owner = owner

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # Most suffix, loss and optimizer operations are deterministic. Avoid
        # traversing their tensor arguments and locking a generator per op.
        if torch.Tag.nondeterministic_seeded not in func.tags:
            return func(*args, **(kwargs or {}))
        schema = func._schema
        name = schema.name
        kwargs = dict(kwargs or {})
        if name in _FUSED_ATTENTION_OPS:
            # These kernels carry the seeded tag even with dropout disabled.
            # The zero-dropout forward/backward does not draw random numbers;
            # retain its native kernel rather than rejecting deterministic
            # attention. Nonzero fused dropout still fails closed below.
            index = next(i for i, arg in enumerate(schema.arguments)
                         if arg.name == "dropout_p")
            dropout = (args[index] if len(args) > index else
                       kwargs.get("dropout_p", schema.arguments[index].default_value))
            if dropout == 0:
                return func(*args, **kwargs)
        overload = getattr(schema, "overload_name", "") or ""
        generator_arg = next((index for index, arg in enumerate(schema.arguments)
                              if arg.name == "generator"), None)
        explicit_generator = (args[generator_arg] if generator_arg is not None and len(args) > generator_arg
                              else kwargs.get("generator"))
        if explicit_generator is not None:
            return func(*args, **kwargs)
        default_device = (torch.get_default_device() if any(arg.name == "device" for arg in schema.arguments)
                          else None)
        generator = self.owner.generator_for(args, kwargs, default_device=default_device)
        if generator is None:
            raise RuntimeError(f"Replica RNG isolation does not support the device used by {func}")

        if name == "aten::native_dropout":
            # CUDA's fused dropout has no generator argument. Returning its
            # output/mask pair from a private Bernoulli draw keeps its native
            # autograd formula without consuming the process-wide CUDA RNG.
            value = args[0] if args else kwargs["input"]
            probability = args[1] if len(args) > 1 else kwargs["p"]
            training = args[2] if len(args) > 2 else kwargs.get("train", True)
            if training is False or probability == 0:
                output, mask = value.clone(), torch.ones_like(value, dtype=torch.bool)
            elif probability == 1:
                output, mask = torch.zeros_like(value), torch.zeros_like(value, dtype=torch.bool)
            else:
                mask = torch.empty_like(value).bernoulli_(1 - probability, generator=generator).bool()
                output = value * mask / (1 - probability)
            if overload == "out":
                return kwargs["out0"].copy_(output), kwargs["out1"].copy_(mask)
            return output, mask

        # Includes multinomial, RReLU, normal, Poisson and in-place draws.
        # An explicit None still means the default RNG and must be replaced.
        if generator_arg is not None:
            if len(args) > generator_arg:
                args = (*args[:generator_arg], generator, *args[generator_arg + 1:])
            else:
                kwargs["generator"] = generator
            return func(*args, **kwargs)

        # Generator-free random factory overloads are replaced by their
        # generator variants.  Calling the overload directly avoids re-entering
        # this mode through the Python wrapper.
        if name in _RANDOM_FACTORY_OPS and overload in _GENERATOR_OVERLOADS:
            op = getattr(
                getattr(torch.ops.aten, name.removeprefix("aten::")),
                _GENERATOR_OVERLOADS[overload],
            )
            kwargs["generator"] = generator
            return op(*args, **kwargs)

        # Never silently run a new/fused random op against process-wide state.
        raise RuntimeError(f"Replica RNG isolation does not support random operator {func}")


class TorchReplicaRNG:
    """Per-suffix Torch RNG state with no process-wide RNG mutation."""

    def __init__(
        self,
        *,
        device: torch.device,
        seed: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        if seed is None:
            seed = torch.initial_seed() + next(_REPLICA_SEED_COUNTER)
        self.seed = int(seed or 0) % (2**63 - 1)
        self._generators: dict[str, torch.Generator] = {}
        self._generator_lock = threading.Lock()
        self._generator_for_device(self.device)

    @staticmethod
    def _device_key(device: torch.device | str) -> str:
        normalized = torch.device(device)
        return f"{normalized.type}:{normalized.index or 0}"

    def _generator_for_device(self, device: torch.device) -> torch.Generator:
        device = torch.device(device)
        key = self._device_key(device)
        with self._generator_lock:
            generator = self._generators.get(key)
            if generator is None:
                generator = torch.Generator(device=device)
                # A stable device offset prevents two devices in the same
                # replica from consuming identical streams while retaining
                # deterministic replay.
                offset = int(device.index or 0) * 1_000_003
                generator.manual_seed((self.seed + offset) % (2**63 - 1))
                self._generators[key] = generator
            return generator

    def generator_for(self, args=(), kwargs=None, *, default_device=None) -> torch.Generator | None:
        """Choose the private generator matching an operation's tensor/device."""
        kwargs = kwargs or {}
        found: torch.device | None = None

        def visit(value):
            nonlocal found
            if found is not None:
                return
            if isinstance(value, torch.Tensor):
                found = value.device
            elif isinstance(value, torch.device):
                found = value
            elif isinstance(value, str):
                try:
                    found = torch.device(value)
                except RuntimeError:
                    return
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)

        visit(kwargs.get("device"))
        if found is None:
            visit(kwargs.get("out"))
        if found is None:
            visit(args)
        if found is None:
            found = torch.device(default_device) if default_device is not None else self.device
        # Private generators are only valid for CPU/CUDA random operators.
        if found.type not in {"cpu", "cuda"}:
            return None
        return self._generator_for_device(found)

    def run(self, fn):
        """Run a phase under this replica's thread-local RNG dispatch mode."""
        with _ReplicaRandomMode(self):
            return fn()

    def run_replay(self, fn):
        """Let certified deterministic replay targets bypass RNG dispatch.

        The pinned replay adapter guards unknown/random targets individually;
        loss functions and backward still use the general dispatch scope.
        Context variables keep nested and simultaneous replicas independent.
        """
        token = _REPLAY_REPLICA_RNG.set(self)
        try:
            return fn()
        finally:
            _REPLAY_REPLICA_RNG.reset(token)

    def get_state(self):
        """Return a copy suitable for logical client state persistence."""
        return self._generator_for_device(self.device).get_state().clone()

    def set_state(self, state) -> None:
        """Restore a previously captured replica stream."""
        if state is None:
            return
        self._generator_for_device(self.device).set_state(state)


def _timed_phase(
    fn,
    device: torch.device | None,
    *,
    stream: torch.cuda.Stream | None = None,
    rng: TorchReplicaRNG | None = None,
    replay_rng: bool = False,
) -> tuple[Any, float]:
    def compute():
        if rng is None:
            return fn()
        return rng.run_replay(fn) if replay_rng else rng.run(fn)
    if device is not None and device.type == "cuda":
        # gRPC computation threads need not inherit the model's CUDA device.
        # Record both events on that device's stream and wait on the actual
        # end event; synchronizing another device cannot complete these events.
        with torch.cuda.device(device):
            selected_stream = stream or torch.cuda.current_stream(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if stream is not None:
                # Boundary tensors may have been produced by the caller's
                # stream.  Make this replica's private stream consume that
                # work without forcing every other replica to synchronize.
                selected_stream.wait_stream(torch.cuda.current_stream(device))
            selected_stream.synchronize()
            execution_scope = (
                torch.cuda.stream(selected_stream) if stream is not None else nullcontext()
            )
            with execution_scope:
                start.record(selected_stream)
                value = compute()
                end.record(selected_stream)
            end.synchronize()
            return value, float(start.elapsed_time(end))
    started = time.perf_counter_ns()
    value = compute()
    return value, (time.perf_counter_ns() - started) / 1_000_000.0


def train_torch_suffix(
    runtime,
    boundary: ReplayBoundary,
    targets: Any,
    *,
    loss_fn=None,
    optimizer=None,
    measurements: dict[str, float] | None = None,
    boundary_gradients_only: bool = False,
    stream: torch.cuda.Stream | None = None,
    rng: TorchReplicaRNG | None = None,
):
    """Train suffix while replaying non-leaf views of leaf gradient roots.

    TorchLens 2.34.1 replays its leaf roots directly. Models such as timm ResNet
    use in-place residual adds, which PyTorch correctly rejects on leaf tensors.
    A zero-add view remains connected to the root while being safe for replay.
    """
    if loss_fn is None:
        raise ValueError("Suffix training requires an explicit loss_fn.")
    if boundary_gradients_only and optimizer is not None:
        raise ValueError("Boundary-only gradients cannot update a suffix optimizer")
    service_started_ns = time.perf_counter_ns()
    runtime.validate_boundary(boundary)
    roots: dict[str, torch.Tensor] = {}
    replay: dict[str, Any] = {}
    for key, value in boundary.tensors.items():
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            root = value.detach().clone().requires_grad_(True)
            roots[key] = root
            replay[key] = root + torch.zeros((), dtype=root.dtype, device=root.device)
        else:
            replay[key] = value
    replay_boundary = ReplayBoundary(
        backend=boundary.backend,
        tensors=replay,
        spec=boundary.spec,
        metadata={**boundary.metadata, "suffix_training_roots": tuple(roots)},
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    device = next(
        (value.device for value in replay.values() if isinstance(value, torch.Tensor)),
        None,
    )
    output, forward_ms = _timed_phase(
        lambda: runtime.run_suffix(replay_boundary), device, stream=stream,
        rng=rng, replay_rng=bool(getattr(
            getattr(getattr(runtime, "segments", None), "suffix", None), "_splitfleet_live_rng", False)),
    )
    loss, loss_ms = _timed_phase(
        lambda: loss_fn(output, targets), device, stream=stream,
        rng=rng,
    )
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Split suffix loss is nonfinite; refusing the optimizer update")
    def backward_and_step():
        if boundary_gradients_only:
            # Client calibration updates only its prefix. Compute the same
            # loss VJP without materializing unused suffix parameter gradients.
            values = torch.autograd.grad(loss, tuple(roots.values()), allow_unused=True) if roots else ()
            return {key: value.detach().clone()
                    for key, value in zip(roots, values) if value is not None}
        loss.backward()
        gradients = {
            key: root.grad.detach().clone()
            for key, root in roots.items()
            if root.grad is not None
        }
        if optimizer is not None:
            optimizer.step()
        return gradients

    gradients, backward_ms = _timed_phase(
        backward_and_step, device, stream=stream, rng=rng,
    )
    if measurements is not None:
        # Detection matching can spend substantial time on CPU between GPU
        # kernels. The service wall span includes validation, boundary roots,
        # loss/matching, backward and the optimizer; F+B alone understates it.
        measurements["server_forward_ms"] = forward_ms
        measurements["server_loss_ms"] = loss_ms
        measurements["server_backward_ms"] = backward_ms
        measurements["server_total_ms"] = (time.perf_counter_ns() - service_started_ns) / 1_000_000.0
    return loss, gradients
