"""Coordinate TensorFlow capture hooks with concurrent native execution."""

from .native_capture_scope import NativeCaptureGate

# Preserve the existing public gate name for callers and focused tests.
TensorFlowCaptureGate = NativeCaptureGate
_GATE = NativeCaptureGate()


def tensorflow_capture_scope():
    return _GATE.capture()


def tensorflow_execution_scope():
    return _GATE.execution()
