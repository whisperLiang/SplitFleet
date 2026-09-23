"""Fetch and verify the AG News CSV files used by the four-task benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path


BASE_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"
EXPECTED_ROWS = {"train.csv": 120_000, "test.csv": 7_600}


def fetch(destination: str | Path, *, timeout: float = 120.0) -> dict[str, object]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"source": BASE_URL, "files": {}}
    for name, expected_count in EXPECTED_ROWS.items():
        target = root / name
        if not target.exists():
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".part", dir=root)
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                with urllib.request.urlopen(BASE_URL + name, timeout=timeout) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                _inspect(temporary, expected_count)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        count, digest = _inspect(target, expected_count)
        manifest["files"][name] = {"rows": count, "sha256": digest, "bytes": target.stat().st_size}
    (root / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _inspect(path: Path, expected_rows: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    with path.open("r", newline="", encoding="utf-8") as stream:
        count = 0
        for row in csv.reader(stream):
            if len(row) != 3 or row[0] not in {"1", "2", "3", "4"}:
                raise ValueError(f"Invalid AG News row {count + 1} in {path}.")
            count += 1
    if count != expected_rows:
        raise ValueError(f"AG News {path.name} has {count} rows; expected {expected_rows}.")
    return count, digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default="data/ag_news_csv")
    args = parser.parse_args(argv)
    print(json.dumps(fetch(args.destination), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
