"""Versioned, pickle-free tensor and split-boundary wire envelopes.

The wire representation deliberately contains no runtime/autograd objects.  A
backend adapter turns its tensors into canonical little-endian byte strings and
keeps graph-connected state in a process-local context store.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import lz4.frame
import torch

PROTOCOL_VERSION = "splitfleet-wire.v1"
_MAGIC = b"SFW1"
_NONE = 0
_LZ4 = 1


@dataclass(frozen=True)
class EnvelopeLimits:
    max_header_bytes: int = 1024 * 1024
    max_tensors: int = 1024
    max_tensor_bytes: int = 64 * 1024 * 1024
    max_payload_bytes: int = 256 * 1024 * 1024
    max_wire_bytes: int = 256 * 1024 * 1024


DEFAULT_LIMITS = EnvelopeLimits()


@dataclass(frozen=True)
class TensorEnvelope:
    tensor_id: str
    shape: tuple[int, ...]
    dtype: str
    payload: bytes
    layout: str = "contiguous"
    encoding: str = "raw-le"
    requires_grad: bool = False


@dataclass(frozen=True)
class BoundaryEnvelope:
    tensors: tuple[TensorEnvelope, ...]
    protocol_version: str = PROTOCOL_VERSION
    engine: str = "torchlens"
    backend: str = "torch"
    round_id: int = 0
    client_id: str = ""
    step_id: str = ""
    plan_id: str = ""
    split_id: str = ""
    canonical_graph_hash: str = ""
    boundary_schema_hash: str = ""
    model_version: int = 0
    batch_size: int = 0
    compression: str = "none"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    checksum: str = ""


@dataclass(frozen=True)
class GradientEnvelope:
    tensors: tuple[TensorEnvelope, ...]
    protocol_version: str = PROTOCOL_VERSION
    backend: str = "torch"
    round_id: int = 0
    client_id: str = ""
    step_id: str = ""
    plan_id: str = ""
    split_id: str = ""
    model_version: int = 0
    compression: str = "none"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    checksum: str = ""


_TORCH_DTYPES = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def encode_tensor(tensor_id: str, tensor: torch.Tensor) -> TensorEnvelope:
    value = tensor.detach().to("cpu").contiguous()
    # Viewing as bytes also supports bfloat16, which NumPy cannot represent on
    # all supported versions.
    # PyTorch does not allow a zero-dimensional tensor to be reinterpreted as
    # a dtype with a different element size. Flattening first preserves the raw
    # storage while allowing scalar boundary metadata to use the same codec.
    payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return TensorEnvelope(
        tensor_id=str(tensor_id),
        shape=tuple(int(dim) for dim in value.shape),
        dtype=_dtype_name(value.dtype),
        payload=payload,
        requires_grad=bool(tensor.requires_grad),
    )


def decode_tensor(envelope: TensorEnvelope, device: str | torch.device = "cpu") -> torch.Tensor:
    if envelope.layout != "contiguous" or envelope.encoding != "raw-le":
        raise ValueError("Unsupported tensor layout or encoding")
    try:
        dtype = _TORCH_DTYPES[envelope.dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor dtype {envelope.dtype!r}") from exc
    if any(int(dim) < 0 for dim in envelope.shape):
        raise ValueError("Tensor envelope contains negative shape dimension")
    expected = int(torch.empty((), dtype=dtype).element_size())
    for dim in envelope.shape:
        expected *= int(dim)
    if expected != len(envelope.payload):
        raise ValueError(
            f"Tensor payload length mismatch: expected {expected}, got {len(envelope.payload)}"
        )
    raw = torch.frombuffer(bytearray(envelope.payload), dtype=torch.uint8)
    value = raw.view(dtype).reshape(envelope.shape).clone().to(device)
    if envelope.requires_grad and (value.is_floating_point() or value.is_complex()):
        value.requires_grad_(True)
    return value


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _pack(kind: str, envelope: BoundaryEnvelope | GradientEnvelope, compression: str) -> bytes:
    if compression not in {"none", "lz4"}:
        raise ValueError(f"Unsupported compression {compression!r}")
    tensors = list(envelope.tensors)
    header = asdict(envelope)
    header.pop("tensors", None)
    header["checksum"] = ""
    header["compression"] = compression
    header["kind"] = kind
    header["metadata"] = _json_value(header.get("metadata", {}))
    header["tensors"] = [
        {key: _json_value(value) for key, value in asdict(tensor).items() if key != "payload"}
        | {"nbytes": len(tensor.payload)}
        for tensor in tensors
    ]
    data = b"".join(tensor.payload for tensor in tensors)
    checksum = hashlib.sha256(data).hexdigest()
    header["checksum"] = checksum
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body = struct.pack("!I", len(header_bytes)) + header_bytes + data
    flag = _LZ4 if compression == "lz4" else _NONE
    if flag == _LZ4:
        body = lz4.frame.compress(body)
    return _MAGIC + bytes([flag]) + body


def _decompress_lz4_bounded(payload: bytes, max_bytes: int) -> bytes:
    decoder = lz4.frame.LZ4FrameDecompressor()
    chunks: list[bytes] = []
    total = 0
    source = payload
    while True:
        chunk = decoder.decompress(source, max_length=max_bytes - total + 1)
        source = b""
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("SplitFleet envelope exceeds decompressed size limit")
        if decoder.eof:
            break
        if decoder.needs_input:
            raise ValueError("Truncated SplitFleet lz4 envelope")
    return b"".join(chunks)


def _unpack(payload: bytes, expected_kind: str, limits: EnvelopeLimits = DEFAULT_LIMITS) -> tuple[dict[str, Any], tuple[TensorEnvelope, ...]]:
    if len(payload) > limits.max_wire_bytes:
        raise ValueError("SplitFleet envelope exceeds wire size limit")
    if len(payload) < 9 or payload[:4] != _MAGIC:
        raise ValueError("Not a SplitFleet wire envelope")
    flag, body = payload[4], payload[5:]
    if flag == _LZ4:
        body = _decompress_lz4_bounded(
            body, limits.max_header_bytes + limits.max_payload_bytes + 4
        )
    elif flag != _NONE:
        raise ValueError(f"Unknown SplitFleet compression flag {flag}")
    if len(body) > limits.max_header_bytes + limits.max_payload_bytes + 4:
        raise ValueError("SplitFleet envelope exceeds decompressed size limit")
    if len(body) < 4:
        raise ValueError("Truncated SplitFleet envelope header")
    header_size = struct.unpack("!I", body[:4])[0]
    if header_size > limits.max_header_bytes:
        raise ValueError("SplitFleet envelope header exceeds size limit")
    if header_size > len(body) - 4:
        raise ValueError("Truncated SplitFleet envelope metadata")
    header = json.loads(body[4 : 4 + header_size].decode("utf-8"))
    expected_compression = "lz4" if flag == _LZ4 else "none"
    if header.get("compression") != expected_compression:
        raise ValueError("SplitFleet envelope compression metadata mismatch")
    if header.get("kind") != expected_kind:
        raise ValueError(f"Expected {expected_kind} envelope, got {header.get('kind')!r}")
    if header.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version {header.get('protocol_version')!r}")
    data = body[4 + header_size :]
    if len(data) > limits.max_payload_bytes:
        raise ValueError("SplitFleet envelope payload exceeds size limit")
    if hashlib.sha256(data).hexdigest() != header.get("checksum"):
        raise ValueError("SplitFleet envelope checksum mismatch")
    offset = 0
    tensors = []
    tensor_headers = header.pop("tensors")
    if not isinstance(tensor_headers, list) or len(tensor_headers) > limits.max_tensors:
        raise ValueError("SplitFleet envelope tensor count exceeds limit")
    tensor_ids: set[str] = set()
    for item in tensor_headers:
        size = int(item.pop("nbytes"))
        if size > limits.max_tensor_bytes:
            raise ValueError("SplitFleet tensor exceeds size limit")
        if size < 0 or offset + size > len(data):
            raise ValueError("Truncated SplitFleet tensor payload")
        tensor_id = str(item.get("tensor_id", ""))
        if not tensor_id or tensor_id in tensor_ids:
            raise ValueError("SplitFleet envelope contains invalid or duplicate tensor id")
        tensor_ids.add(tensor_id)
        tensors.append(TensorEnvelope(payload=data[offset : offset + size], shape=tuple(item.pop("shape")), **item))
        offset += size
    if offset != len(data):
        raise ValueError("SplitFleet envelope contains trailing tensor bytes")
    header.pop("kind", None)
    return header, tuple(tensors)


def encode_boundary(envelope: BoundaryEnvelope, *, compression: str | None = None) -> bytes:
    return _pack("boundary", envelope, compression or envelope.compression)


def decode_boundary(payload: bytes, *, limits: EnvelopeLimits = DEFAULT_LIMITS) -> BoundaryEnvelope:
    header, tensors = _unpack(payload, "boundary", limits)
    return BoundaryEnvelope(tensors=tensors, **header)


def encode_gradients(envelope: GradientEnvelope, *, compression: str | None = None) -> bytes:
    return _pack("gradients", envelope, compression or envelope.compression)


def decode_gradients(payload: bytes, *, limits: EnvelopeLimits = DEFAULT_LIMITS) -> GradientEnvelope:
    header, tensors = _unpack(payload, "gradients", limits)
    return GradientEnvelope(tensors=tensors, **header)
