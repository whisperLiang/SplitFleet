from __future__ import annotations

from pathlib import Path


def test_production_tree_contains_no_removed_backend_or_legacy_bridge() -> None:
    package_root = Path(__file__).parents[2] / "splitfleet"
    forbidden_backend = "aria" + "dne"
    offenders = []
    for path in package_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        if forbidden_backend in text:
            offenders.append(str(path.relative_to(package_root)))
    assert offenders == []
    assert not (package_root / "split_engine" / "legacy.py").exists()
    assert not (package_root / "autosplit" / "tracer.py").exists()
