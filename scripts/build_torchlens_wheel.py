"""Rebuild the pinned TorchLens wheel with the repository's source patches."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path
import subprocess
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "torchlens-2.34.1-py3-none-any.whl"
UPSTREAM_SHA256 = "118fe87e092838664f2daf25c41df6de26b2758e4eeb7445ef636daf9e304993"
OUTPUT = ROOT / "torchlens-2.34.1-1-py3-none-any.whl"
DIST_INFO = "torchlens-2.34.1.dist-info"


def build() -> Path:
    if hashlib.sha256(UPSTREAM.read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("The upstream TorchLens wheel does not match the pinned SHA-256.")
    patches = sorted((ROOT / "patches" / "torchlens").glob("*.patch"))
    if not patches:
        raise ValueError("No TorchLens source patches were found.")

    with tempfile.TemporaryDirectory(prefix="splitfleet-torchlens-") as temp:
        unpacked = Path(temp) / "source"
        with ZipFile(UPSTREAM) as wheel:
            wheel.extractall(unpacked)
        for patch in patches:
            subprocess.run(
                ["patch", "--batch", "--forward", "--fuzz=0", "--no-backup-if-mismatch",
                 "-p1", "-i", str(patch)],
                cwd=unpacked, check=True,
            )
        wheel_metadata = unpacked / DIST_INFO / "WHEEL"
        metadata = wheel_metadata.read_text()
        wheel_metadata.write_text(metadata.rstrip() + "\nBuild: 1\n")
        provenance = [
            "SplitFleet local build 1 of TorchLens 2.34.1.",
            f"Upstream wheel SHA-256: {UPSTREAM_SHA256}",
            "Applied patches:",
            *(f"{patch.name} sha256:{hashlib.sha256(patch.read_bytes()).hexdigest()}" for patch in patches),
        ]
        (unpacked / DIST_INFO / "SPLITFLEET_PATCHES").write_text("\n".join(provenance) + "\n")

        record_name = f"{DIST_INFO}/RECORD"
        payloads = {
            path.relative_to(unpacked).as_posix(): path.read_bytes()
            for path in sorted(unpacked.rglob("*"))
            if path.is_file() and path.relative_to(unpacked).as_posix() != record_name
        }
        record = io.StringIO(newline="")
        writer = csv.writer(record, lineterminator="\n")
        for name, data in sorted(payloads.items()):
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow((name, f"sha256={digest}", len(data)))
        writer.writerow((record_name, "", ""))
        payloads[record_name] = record.getvalue().encode()

        # Fixed metadata and ordering make repeated builds byte-for-byte stable.
        with tempfile.NamedTemporaryFile(dir=ROOT, suffix=".whl", delete=False) as staged:
            staged_path = Path(staged.name)
        try:
            with ZipFile(staged_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as wheel:
                for name, data in sorted(payloads.items()):
                    info = ZipInfo(name, date_time=(2026, 9, 22, 0, 0, 0))
                    info.compress_type = ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    wheel.writestr(info, data, compresslevel=9)
            staged_path.replace(OUTPUT)
        finally:
            staged_path.unlink(missing_ok=True)
    return OUTPUT


if __name__ == "__main__":
    output = build()
    print(f"{output.name} sha256:{hashlib.sha256(output.read_bytes()).hexdigest()}")
