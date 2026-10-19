"""CUDA telemetry must follow the model device in a computation thread."""

from contextlib import contextmanager
import time

import pytest
import torch

from splitfleet.runtime.torch_suffix_training import TorchReplicaRNG, _timed_phase


def test_cuda_phase_binds_device_and_waits_for_recorded_end(monkeypatch):
    calls = []
    stream = object()

    @contextmanager
    def guard(device):
        calls.append(("enter", device))
        yield
        calls.append(("exit", device))

    class Stream:
        def synchronize(self):
            calls.append("stream_wait")

    stream = Stream()

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.recorded = False
            self.completed = False

        def record(self, selected_stream):
            assert selected_stream is stream
            self.recorded = True
            calls.append("record")

        def synchronize(self):
            assert self.recorded
            self.completed = True
            calls.append("end_wait")

        def elapsed_time(self, other):
            assert other.completed
            return 3.5

    monkeypatch.setattr(torch.cuda, "device", guard)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    device = torch.device("cuda:1")

    def computation():
        calls.append("compute")
        return 17

    assert _timed_phase(computation, device) == (17, 3.5)
    assert calls == [("enter", device), "stream_wait", "record", "compute",
                     "record", "end_wait", ("exit", device)]


def test_second_gpu_timing_from_default_device_thread():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices required")
    from concurrent.futures import ThreadPoolExecutor

    def invoke():
        torch.cuda.set_device(0)
        inputs = torch.ones((512, 512), device="cuda:1")
        output, elapsed = _timed_phase(lambda: inputs @ inputs, torch.device("cuda:1"))
        assert torch.cuda.current_device() == 0
        assert output.device == torch.device("cuda:1")
        assert output[0, 0].item() == 512
        assert elapsed > 0

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: invoke(), range(6)))


def test_explicit_cuda_stream_is_used_for_suffix_phase():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    observed = []

    def computation():
        observed.append(torch.cuda.current_stream(device).cuda_stream)
        return torch.ones((), device=device)

    value, elapsed = _timed_phase(computation, device, stream=stream)
    assert value.device == device
    assert elapsed >= 0
    assert observed == [stream.cuda_stream]


def test_replica_rng_scope_restores_global_state_and_is_reproducible():
    before = torch.get_rng_state().clone()
    first = TorchReplicaRNG(device=torch.device("cpu"), seed=101)
    second = TorchReplicaRNG(device=torch.device("cpu"), seed=202)
    left = first.run(lambda: torch.rand(8))
    right = second.run(lambda: torch.rand(8))
    assert not torch.equal(left, right)
    assert torch.equal(before, torch.get_rng_state())

    replay = TorchReplicaRNG(device=torch.device("cpu"), seed=101)
    assert torch.equal(left, replay.run(lambda: torch.rand(8)))


@pytest.mark.parametrize("operation", [
    "randint", "randint_low", "randint_low_out", "randperm", "randperm_out",
    "multinomial", "rrelu", "rrelu_inplace", "rrelu_positional_generator",
    "rand_like", "randn_like", "randint_like", "randint_like_low", "normal",
    "poisson", "bernoulli", "uniform_inplace", "random_inplace", "explicit_none",
])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_standard_random_operations_are_private_and_reproducible(operation, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")

    def draw():
        value = torch.full((64,), -1., device=device)
        if operation == "randint":
            return torch.randint(100, (64,), device=device)
        if operation == "randint_low":
            return torch.randint(10, 100, (64,), device=device)
        if operation == "randint_low_out":
            return torch.randint(10, 100, (64,), out=torch.empty(64, dtype=torch.int64, device=device))
        if operation == "randperm":
            return torch.randperm(64, device=device)
        if operation == "randperm_out":
            return torch.randperm(64, out=torch.empty(64, dtype=torch.int64, device=device))
        if operation == "multinomial":
            return torch.multinomial(torch.ones(8, device=device), 64, replacement=True)
        if operation == "rrelu":
            return torch.nn.functional.rrelu(value, training=True)
        if operation == "rrelu_inplace":
            return torch.nn.functional.rrelu(value, training=True, inplace=True)
        if operation == "rrelu_positional_generator":
            return torch.ops.aten.rrelu.default(value, .1, .3, True, None)
        if operation == "rand_like":
            return torch.rand_like(value)
        if operation == "randn_like":
            return torch.randn_like(value)
        if operation == "randint_like":
            return torch.randint_like(value, 100)
        if operation == "randint_like_low":
            return torch.randint_like(value, 10, 100)
        if operation == "normal":
            return torch.normal(torch.zeros_like(value), 1.)
        if operation == "poisson":
            return torch.poisson(torch.ones_like(value))
        if operation == "bernoulli":
            return torch.bernoulli(torch.full_like(value, .5))
        if operation == "uniform_inplace":
            return value.uniform_()
        if operation == "random_inplace":
            return value.random_(10, 100)
        return torch.rand(64, generator=None, device=device)

    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(device).clone() if device.startswith("cuda") else None
    owner = TorchReplicaRNG(device=torch.device(device), seed=818)
    first, next_value = owner.run(draw), owner.run(draw)
    replay = TorchReplicaRNG(device=torch.device(device), seed=818)
    torch.testing.assert_close(first, replay.run(draw), rtol=0, atol=0)
    torch.testing.assert_close(next_value, replay.run(draw), rtol=0, atol=0)
    assert not torch.equal(first, next_value)
    assert torch.equal(before_cpu, torch.get_rng_state())
    if before_cuda is not None:
        assert torch.equal(before_cuda, torch.cuda.get_rng_state(device))


@pytest.mark.parametrize("positional", [False, True])
def test_explicit_generator_keeps_its_own_stream(positional):
    owner = TorchReplicaRNG(device=torch.device("cpu"), seed=11)
    explicit = torch.Generator().manual_seed(12)
    before = owner.get_state()
    if positional:
        value = torch.full((64,), -1.)
        output = owner.run(lambda: torch.ops.aten.rrelu.default(value, .1, .3, True, explicit))
        expected = torch.ops.aten.rrelu.default(value, .1, .3, True, torch.Generator().manual_seed(12))
    else:
        output = owner.run(lambda: torch.randperm(64, generator=explicit))
        expected = torch.randperm(64, generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.equal(owner.get_state(), before)


def test_unhandled_seeded_operator_is_rejected_without_mutating_global_rng():
    library = torch.library.Library("splitfleet_rng_test", "FRAGMENT")
    library.define("unhandled(Tensor value) -> Tensor", tags=(torch.Tag.nondeterministic_seeded,))
    library.impl("unhandled", lambda value: torch.rand_like(value), "CompositeExplicitAutograd")
    owner = TorchReplicaRNG(device=torch.device("cpu"), seed=13)
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="does not support random operator.*unhandled"):
        owner.run(lambda: torch.ops.splitfleet_rng_test.unhandled(torch.ones(64)))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires fused CUDA attention")
def test_zero_dropout_fused_attention_backward_preserves_rng_and_native_gradients():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    generator = torch.Generator(device="cuda").manual_seed(196)
    tensors = [torch.randn(2, 2, 8, 32, device="cuda", generator=generator,
                           requires_grad=True) for _ in range(3)]
    references = [value.detach().clone().requires_grad_() for value in tensors]
    owner = TorchReplicaRNG(device=torch.device("cuda"), seed=197)
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone()
    before_private = owner.get_state()

    def execute(values):
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            output = torch.nn.functional.scaled_dot_product_attention(*values, dropout_p=0.)
            output.square().sum().backward()
        return output

    actual = owner.run(lambda: execute(tensors))
    expected = execute(references)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for value, reference in zip(tensors, references):
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
    assert torch.equal(owner.get_state(), before_private)
    assert torch.equal(torch.get_rng_state(), before_cpu)
    assert torch.equal(torch.cuda.get_rng_state(), before_cuda)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires fused CUDA attention")
def test_fused_attention_with_dropout_still_rejects_global_rng_consumption():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    value = torch.ones(2, 2, 8, 32, device="cuda", requires_grad=True)
    owner = TorchReplicaRNG(device=torch.device("cuda"), seed=198)
    before = torch.cuda.get_rng_state().clone()
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        with pytest.raises(RuntimeError, match="does not support random operator"):
            owner.run(lambda: torch.nn.functional.scaled_dot_product_attention(
                value, value, value, dropout_p=.25))
    assert torch.equal(torch.cuda.get_rng_state(), before)


def test_unseeded_replica_rng_is_local_by_default():
    before = torch.get_rng_state().clone()
    first = TorchReplicaRNG(device=torch.device("cpu"))
    second = TorchReplicaRNG(device=torch.device("cpu"))
    left = first.run(lambda: torch.rand(8))
    right = second.run(lambda: torch.rand(8))
    assert not torch.equal(left, right)
    assert torch.equal(before, torch.get_rng_state())


def test_replica_rng_scopes_support_parallel_dropout_without_global_state_races():
    from concurrent.futures import ThreadPoolExecutor

    model = torch.nn.Dropout(p=0.4).train()
    replicas = [
        TorchReplicaRNG(device=torch.device("cpu"), seed=301),
        TorchReplicaRNG(device=torch.device("cpu"), seed=302),
    ]
    before = torch.get_rng_state().clone()

    def execute(replica):
        return replica.run(lambda: model(torch.ones(512)))

    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = pool.map(execute, replicas)
    assert not torch.equal(left, right)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("probability,training", [(0., True), (.4, True), (1., True), (.4, False)])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fused_dropout_keeps_private_randomness_mask_and_backward(probability, training, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    value = torch.ones(128, device=device, requires_grad=True)
    before = (torch.cuda.get_rng_state(device) if device.startswith("cuda") else torch.get_rng_state()).clone()
    rng = TorchReplicaRNG(device=torch.device(device), seed=702)
    output, mask = rng.run(lambda: torch.ops.aten.native_dropout.default(value, p=probability, train=training))
    scale = 0 if probability == 1 else 1 / (1 - probability) if training else 1
    output.sum().backward()
    torch.testing.assert_close(output, mask.to(value.dtype) * scale)
    torch.testing.assert_close(value.grad, mask.to(value.dtype) * scale)
    after = torch.cuda.get_rng_state(device) if device.startswith("cuda") else torch.get_rng_state()
    assert torch.equal(before, after)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("replica", [False, True])
def test_replay_rng_fast_path_matches_general_isolation_and_optimizer(device, replica):
    """Fast replay must preserve stochastic updates, not just deterministic output."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from copy import deepcopy
    from splitfleet.autosplit import AutoSplitSession

    torch.manual_seed(805)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
        torch.nn.Dropout(.4), torch.nn.Linear(8, 2)).to(device).train()
    sample = torch.ones(3, 4, device=device)
    handles = [AutoSplitSession().prepare_runtime(deepcopy(model), sample,
        boundary="after:linear_1_1:1", trainable=True, dynamic_batch=(1, 3)) for _ in range(2)]
    if replica:
        from splitfleet.autosplit.torchlens_clone import clone_runtime_handle
        handles = [clone_runtime_handle(handles[0], deepcopy(model)) for _ in range(2)]
        assert all(handle is not None and handle.runtime.segments.suffix._splitfleet_live_rng
                   for handle in handles)
    for handle in handles:
        handle.backend.enable_replica_rng_isolation(True, seed=427)
    # The prior path guards every ATen operation in the full forward phase.
    handles[0].runtime.segments.suffix._splitfleet_live_rng = False
    optimizers = [torch.optim.Adam(handle.model.parameters(), lr=1e-3) for handle in handles]
    global_cpu = torch.get_rng_state().clone()
    global_cuda = torch.cuda.get_rng_state(device).clone() if device.startswith("cuda") else None
    for size in (3, 1, 3):
        values = []
        for handle, optimizer in zip(handles, optimizers):
            boundary = handle.backend.run_prefix(sample[:size], training=True)
            values.append(handle.backend.train_suffix(boundary,
                torch.zeros(size, dtype=torch.long, device=device),
                loss_fn=torch.nn.functional.cross_entropy, optimizer=optimizer))
        torch.testing.assert_close(values[0][0], values[1][0], rtol=0, atol=0)
        for key in values[0][1]:
            torch.testing.assert_close(values[0][1][key], values[1][1][key], rtol=0, atol=0)
        for key, value in handles[0].model.state_dict().items():
            torch.testing.assert_close(value, handles[1].model.state_dict()[key], rtol=0, atol=0)
        for left, right in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert torch.equal(global_cpu, torch.get_rng_state())
    if global_cuda is not None:
        assert torch.equal(global_cuda, torch.cuda.get_rng_state(device))


def test_replay_rng_context_restores_on_exception_and_bypasses_pure_dispatch(monkeypatch):
    from splitfleet.autosplit import AutoSplitSession
    from splitfleet.runtime.torch_suffix_training import _REPLAY_REPLICA_RNG, _ReplicaRandomMode

    rng = TorchReplicaRNG(device=torch.device("cpu"), seed=618)
    with pytest.raises(RuntimeError, match="fixture"):
        rng.run_replay(lambda: (_ for _ in ()).throw(RuntimeError("fixture")))
    assert _REPLAY_REPLICA_RNG.get() is None
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    sample = torch.ones(3, 4)
    handle = AutoSplitSession().prepare_runtime(model, sample, boundary="50%", trainable=True)
    boundary = handle.backend.run_prefix(sample, training=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Certified deterministic replay entered the random dispatch scope")

    monkeypatch.setattr(_ReplicaRandomMode, "__torch_dispatch__", forbidden)
    rng.run_replay(lambda: handle.backend.run_suffix(boundary))


def test_each_torch_suffix_replica_gets_a_private_cuda_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend

    device = torch.device("cuda:0")
    left = TorchLensSplitBackend(device=device)
    right = TorchLensSplitBackend(device=device)
    left_stream = left._suffix_execution_stream(device)
    assert left_stream is left._suffix_execution_stream(device)
    right_stream = right._suffix_execution_stream(device)
    assert left_stream.cuda_stream != right_stream.cuda_stream


def test_cpu_loss_work_is_included_in_suffix_service_time():
    from splitfleet.autosplit import AutoSplitSession
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    sample = torch.ones(4, 4)
    handle = AutoSplitSession().prepare_runtime(model, sample, boundary="50%", trainable=True)
    boundary = handle.backend.run_prefix(sample, training=True)
    measurements = {}
    def slow_loss(output, target):
        # Stand in for host-side assignment/matching that has no GPU kernels.
        time.sleep(.025)
        return torch.nn.functional.cross_entropy(output, target)
    loss, gradients = handle.backend.train_suffix(boundary, torch.zeros(4, dtype=torch.long),
        loss_fn=slow_loss, optimizer=torch.optim.SGD(model.parameters(), lr=.01), measurements=measurements)
    assert torch.isfinite(loss) and gradients
    assert measurements["server_loss_ms"] >= 20
    assert measurements["server_total_ms"] >= measurements["server_loss_ms"]
    assert measurements["server_total_ms"] >= sum(measurements[key] for key in (
        "server_forward_ms", "server_loss_ms", "server_backward_ms"))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_suffix_loss_refuses_update(bad_value, monkeypatch):
    from splitfleet.autosplit import AutoSplitSession

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    sample = torch.ones(4, 4)
    handle = AutoSplitSession().prepare_runtime(model, sample, boundary="50%", trainable=True)
    boundary = handle.backend.run_prefix(sample, training=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    original = {key: value.clone() for key, value in model.state_dict().items()}

    def forbidden_step(*args, **kwargs):
        pytest.fail("Nonfinite loss reached the optimizer")

    monkeypatch.setattr(optimizer, "step", forbidden_step)
    with pytest.raises(FloatingPointError, match="refusing the optimizer update"):
        handle.backend.train_suffix(boundary, None,
            loss_fn=lambda output, target: output.sum() * bad_value, optimizer=optimizer)
    assert not optimizer.state
    assert all(parameter.grad is None for parameter in model.parameters())
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])
