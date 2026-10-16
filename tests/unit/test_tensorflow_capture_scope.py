"""Threaded exclusion of temporary native capture hooks from execution."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from splitfleet.runtime.tensorflow_capture_scope import TensorFlowCaptureGate


def test_independent_execution_readers_overlap_and_capture_waits():
    gate = TensorFlowCaptureGate()
    started = [Event(), Event()]
    release, captured = Event(), Event()

    def execute(index):
        with gate.execution():
            started[index].set()
            assert release.wait(2)
            # Pending capture cannot deadlock a nested native method.
            with gate.execution():
                assert not captured.is_set()

    def capture():
        with gate.capture(), gate.capture(), gate.execution():
            captured.set()

    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(execute, index) for index in range(2)]
        assert all(event.wait(2) for event in started)
        writer = pool.submit(capture)
        with gate.condition:
            assert gate.condition.wait_for(lambda: gate.waiting_writers == 1, timeout=2)
        assert not captured.is_set()
        release.set()
        for job in jobs:
            job.result(timeout=2)
        writer.result(timeout=2)
    assert captured.is_set()


def test_capture_excludes_new_readers_and_releases_on_error():
    gate = TensorFlowCaptureGate()
    entered, release, executed = Event(), Event(), Event()

    def capture():
        with gate.capture():
            entered.set()
            assert release.wait(2)
            raise ValueError("native preparation failed")

    def execute():
        with gate.execution():
            executed.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(capture)
        assert entered.wait(2)
        reader = pool.submit(execute)
        assert not executed.wait(0.02)
        release.set()
        with pytest.raises(ValueError, match="native preparation failed"):
            writer.result(timeout=2)
        reader.result(timeout=2)
    assert executed.is_set() and not gate.readers and gate.writer is None


def test_execution_to_capture_upgrade_fails_without_stranding_readers():
    gate = TensorFlowCaptureGate()
    with gate.execution(), pytest.raises(RuntimeError, match="active execution"):
        with gate.capture():
            pytest.fail("Capture cannot upgrade an active native operation")
    with gate.capture():
        assert not gate.readers
