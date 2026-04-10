"""Optimized serialization utilities for gRPC communication.

Performance optimizations:
- Dictionary comprehension for faster iteration
- Pre-computed control code mappings
- Reduced branching in hot paths
"""

from typing import Dict, List, Union

from splitfleet.proto import server_model_pb2
from splitfleet.proto.server_model_pb2 import ControlCode as GrpcControlCode
from splitfleet.common.typing import ControlCode


# Pre-computed control code mappings for O(1) lookup
_CONTROL_CODE_TO_PROTO: Dict[ControlCode, GrpcControlCode] = {
    ControlCode.OK: GrpcControlCode.OK,
    ControlCode.DO_CLOSE_STREAM: GrpcControlCode.DO_CLOSE_STREAM,
    ControlCode.ERROR_PROCESSING_STREAM: GrpcControlCode.ERROR_PROCESSING_STREAM,
    ControlCode.STREAM_CLOSED_OK: GrpcControlCode.STREAM_CLOSED_OK,
    ControlCode.INIT_STREAM: GrpcControlCode.INIT_STREAM,
}

_CONTROL_CODE_FROM_PROTO: Dict[GrpcControlCode, ControlCode] = {
    GrpcControlCode.OK: ControlCode.OK,
    GrpcControlCode.DO_CLOSE_STREAM: ControlCode.DO_CLOSE_STREAM,
    GrpcControlCode.ERROR_PROCESSING_STREAM: ControlCode.ERROR_PROCESSING_STREAM,
    GrpcControlCode.STREAM_CLOSED_OK: ControlCode.STREAM_CLOSED_OK,
    GrpcControlCode.INIT_STREAM: ControlCode.INIT_STREAM,
}


def control_code_to_proto(in_code: ControlCode) -> GrpcControlCode:
    """Serialize `Status` to ProtoBuf using O(1) lookup."""
    return _CONTROL_CODE_TO_PROTO.get(in_code, GrpcControlCode.OK)


def control_code_from_proto(in_code: GrpcControlCode) -> ControlCode:
    """Deserialize `Status` from ProtoBuf using O(1) lookup."""
    return _CONTROL_CODE_FROM_PROTO.get(in_code, ControlCode.OK)


def from_grpc_format(
    data: Dict[str, server_model_pb2.ByteTensor]
) -> Dict[str, Union[bytes, List[bytes]]]:
    """Convert a batch of data in protobuf format to a native python dictionary.

    Optimized with dictionary comprehension for faster iteration.

    Parameters
    ----------
    data : Dict[str, server_model_pb2.ByteTensor]
        Data to be deserialized

    Returns
    -------
    Dict[str, Union[bytes, List[bytes]]]
        Deserialized data
    """
    return {
        key: list(value.tensors.tensors) if value.WhichOneof("data") == "tensors" else value.single_tensor
        for key, value in data.items()
    }


def to_grpc_format(
    data: Dict[str, Union[bytes, List[bytes]]]
) -> Dict[str, server_model_pb2.ByteTensor]:
    """Serialize a batch of data to protobuf format.

    Optimized with dictionary comprehension for faster iteration.

    Parameters
    ----------
    data : Dict[str, Union[bytes, List[bytes]]]
        Data to be serialized

    Returns
    -------
    Dict[str, server_model_pb2.ByteTensor]
        Serialized data
    """
    return {
        key: (
            server_model_pb2.ByteTensor(single_tensor=value)
            if isinstance(value, bytes)
            else server_model_pb2.ByteTensor(tensors=server_model_pb2.TensorList(tensors=value))
        )
        for key, value in data.items()
    }
