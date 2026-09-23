"""Optional backend discovery must not load heavyweight native frameworks."""

from __future__ import annotations

import pytest
import numpy as np

from splitfleet.backends import BACKEND_ADAPTERS, BackendAdapterRegistry, JaxBackendAdapter
from splitfleet.backends.array_backends import ArrayBackendAdapter


def test_discovery_reports_missing_dependencies_without_calling_factory(monkeypatch):
    registry = BackendAdapterRegistry()

    def factory():
        raise AssertionError("Discovery must not instantiate a framework")

    registry.register("custom", factory, required_modules=("missing_framework",), install_extra="custom")
    monkeypatch.setattr("splitfleet.backends.registry.find_spec", lambda module: None)

    report, = registry.availability()

    assert registry.names() == ("custom",)
    assert not report.available
    assert report.missing_modules == ("missing_framework",)
    assert report.install_extra == "custom"


def test_builtin_backend_discovery_has_install_hints():
    reports = {report.name: report for report in BACKEND_ADAPTERS.availability()}
    assert reports["torch"].available
    assert reports["tensorflow"].install_extra == reports["tf"].install_extra == "tensorflow"
    assert set(reports) == {"torch", "tf", "tensorflow", "jax", "paddle", "tinygrad"}


def test_factory_errors_are_not_misreported_as_unregistered_backend():
    registry = BackendAdapterRegistry()

    def broken_factory():
        raise KeyError("missing factory configuration")

    registry.register("broken", broken_factory)
    with pytest.raises(KeyError, match="missing factory configuration"):
        registry.create("broken")


@pytest.mark.parametrize("params", [{"weight": object()}, [object()]])
def test_in_place_functional_update_keeps_jax_parameters(params):
    adapter = JaxBackendAdapter()
    adapter.bind_external_params(params)
    original = params.copy()

    # An update callback may mutate the synchronized tree and return that tree.
    adapter.bind_external_params(params)

    assert adapter.external_params is params
    assert params == original


@pytest.mark.parametrize("value", [
    np.array(1.5, dtype=np.float32),
    np.array(7, dtype=np.int64),
    np.arange(12, dtype=np.float32).reshape(3, 4).T,
    np.empty((0, 4), dtype=np.float32),
])
def test_array_wire_codec_preserves_scalar_and_strided_shapes(value):
    class NumpyAdapter(ArrayBackendAdapter):
        def _to_numpy(self, tensor):
            return tensor

        def _from_numpy(self, array, *, device=None):
            return array

    adapter = NumpyAdapter()
    envelope = adapter.encode_tensor("boundary", value)
    restored = adapter.decode_tensor(envelope)

    assert restored.shape == value.shape
    assert restored.dtype == value.dtype
    np.testing.assert_array_equal(restored, value)
