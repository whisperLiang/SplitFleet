"""Share a stable Paddle default device across concurrent native operations."""

from contextlib import contextmanager, nullcontext
from threading import Condition, RLock, get_ident

from .native_capture_scope import NativeCaptureGate


class PaddleDeviceGate:
    """Same-device readers overlap; changing the global place waits for them."""

    def __init__(self):
        self.condition = Condition(RLock())
        self.device = None
        self.readers = {}
        self.guard = None
        self.waiting = 0

    @contextmanager
    def scope(self, device):
        import paddle

        owner = get_ident()
        with self.condition:
            if owner in self.readers and self.device != device:
                if len(self.readers) != 1:
                    raise RuntimeError('Cannot switch Paddle device while other native operations are active')
                # A sole reader can temporarily use another place. Retain the
                # condition lock so no other thread observes that nested place.
                previous = self.device
                self.device = device
                try:
                    with paddle.device.device_guard(device):
                        stream = paddle.device.current_stream(device) if device.startswith('gpu:') else None
                        stream_scope = (
                            paddle.device.stream_guard(stream)
                            if stream is not None else nullcontext()
                        )
                        with stream_scope:
                            yield
                finally:
                    self.device = previous
                return
            self.waiting += 1
            try:
                self.condition.wait_for(lambda: self.device in {None, device})
                if self.device is None:
                    guard = paddle.device.device_guard(device)
                    guard.__enter__()
                    self.guard, self.device = guard, device
                self.readers[owner] = self.readers.get(owner, 0) + 1
            finally:
                self.waiting -= 1
        try:
            stream = paddle.device.current_stream(device) if device.startswith('gpu:') else None
            stream_scope = (
                paddle.device.stream_guard(stream)
                if stream is not None else nullcontext()
            )
            with stream_scope:
                yield
        finally:
            with self.condition:
                self.readers[owner] -= 1
                if not self.readers[owner]:
                    del self.readers[owner]
                if not self.readers:
                    try:
                        self.guard.__exit__(None, None, None)
                    finally:
                        self.guard, self.device = None, None
                self.condition.notify_all()


_DEVICE_GATE = PaddleDeviceGate()
_CAPTURE_GATE = NativeCaptureGate()


def paddle_device_scope(device):
    return _DEVICE_GATE.scope(device)


def paddle_execution_scope(*, capture=False):
    return _CAPTURE_GATE.capture() if capture else _CAPTURE_GATE.execution()
