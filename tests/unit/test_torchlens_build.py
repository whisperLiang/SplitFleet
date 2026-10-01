"""Validate the native TorchLens version and the repository wheel source."""

import hashlib
from pathlib import Path
import tomllib

import pytest

from splitfleet.autosplit import torchlens_runtime


@pytest.mark.parametrize("version", ["2.34.0", "2.34.2", "2.35.0"])
def test_different_runtime_version_is_rejected(monkeypatch, version):
    monkeypatch.setattr(torchlens_runtime, "torchlens_runtime_version", lambda: version)
    with pytest.raises(RuntimeError, match="requires torchlens==2.34.1"):
        torchlens_runtime.require_torchlens_version()


def test_installed_native_runtime_is_accepted():
    torchlens_runtime.require_torchlens_version()


def test_repository_wheel_matches_locked_hash():
    root = Path(__file__).resolve().parents[2]
    with (root / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)
    package = next(item for item in lock["package"] if item["name"] == "torchlens")
    wheel = package["wheels"][0]
    expected = "torchlens-2.34.1-py3-none-any.whl"
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    assert project["tool"]["uv"]["sources"]["torchlens"]["path"] == f"./{expected}"
    assert package["source"] == {"path": expected}
    assert wheel["filename"] == expected

    actual = hashlib.sha256((root / wheel["filename"]).read_bytes()).hexdigest()
    assert wheel["hash"] == f"sha256:{actual}"
