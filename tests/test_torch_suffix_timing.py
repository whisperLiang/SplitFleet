"""CUDA telemetry must follow the model device in a computation thread."""

from contextlib import contextmanager
import time

import pytest
import torch

from splitfleet.runtime.torch_suffix_training import _timed_phase


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
