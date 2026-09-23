"""Matching the upstream version alone must not admit the unpatched runtime."""

from types import SimpleNamespace

import pytest

from splitfleet.autosplit import torchlens_runtime


@pytest.mark.parametrize("provenance", [None, "", "unpatched or different local build"])
def test_unpatched_same_version_runtime_is_rejected(monkeypatch, provenance):
    monkeypatch.setattr(torchlens_runtime, "torchlens_runtime_version", lambda: "2.34.1")
    monkeypatch.setattr(
        torchlens_runtime.importlib.metadata, "distribution",
        lambda name: SimpleNamespace(read_text=lambda filename: provenance),
    )

    with pytest.raises(RuntimeError, match="patched TorchLens 2.34.1 build 1"):
        torchlens_runtime.require_torchlens_version()


def test_installed_patched_runtime_is_accepted():
    torchlens_runtime.require_torchlens_version()
