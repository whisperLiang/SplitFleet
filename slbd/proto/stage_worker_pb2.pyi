from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RegisterPlanRequest(_message.Message):
    __slots__ = ("worker_id", "plan_id", "plan_descriptor", "model_state")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    PLAN_ID_FIELD_NUMBER: _ClassVar[int]
    PLAN_DESCRIPTOR_FIELD_NUMBER: _ClassVar[int]
    MODEL_STATE_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    plan_id: str
    plan_descriptor: bytes
    model_state: bytes
    def __init__(self, worker_id: _Optional[str] = ..., plan_id: _Optional[str] = ..., plan_descriptor: _Optional[bytes] = ..., model_state: _Optional[bytes] = ...) -> None: ...

class RegisterPlanResponse(_message.Message):
    __slots__ = ("worker_id", "plan_id", "accepted", "graph_signature", "message")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    PLAN_ID_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    GRAPH_SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    plan_id: str
    accepted: bool
    graph_signature: str
    message: str
    def __init__(self, worker_id: _Optional[str] = ..., plan_id: _Optional[str] = ..., accepted: bool = ..., graph_signature: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class StageExecutionRequest(_message.Message):
    __slots__ = ("plan_id", "stage_id", "execution_id", "client_id", "round_id", "differentiable", "detach_boundary", "seeded_values", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PLAN_ID_FIELD_NUMBER: _ClassVar[int]
    STAGE_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    ROUND_ID_FIELD_NUMBER: _ClassVar[int]
    DIFFERENTIABLE_FIELD_NUMBER: _ClassVar[int]
    DETACH_BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    SEEDED_VALUES_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    plan_id: str
    stage_id: str
    execution_id: str
    client_id: str
    round_id: int
    differentiable: bool
    detach_boundary: bool
    seeded_values: bytes
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, plan_id: _Optional[str] = ..., stage_id: _Optional[str] = ..., execution_id: _Optional[str] = ..., client_id: _Optional[str] = ..., round_id: _Optional[int] = ..., differentiable: bool = ..., detach_boundary: bool = ..., seeded_values: _Optional[bytes] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class StageExecutionResponse(_message.Message):
    __slots__ = ("available_updates", "stored_updates", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    AVAILABLE_UPDATES_FIELD_NUMBER: _ClassVar[int]
    STORED_UPDATES_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    available_updates: bytes
    stored_updates: bytes
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, available_updates: _Optional[bytes] = ..., stored_updates: _Optional[bytes] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class StageBackwardRequest(_message.Message):
    __slots__ = ("plan_id", "stage_id", "execution_id", "upstream_grads")
    PLAN_ID_FIELD_NUMBER: _ClassVar[int]
    STAGE_ID_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    UPSTREAM_GRADS_FIELD_NUMBER: _ClassVar[int]
    plan_id: str
    stage_id: str
    execution_id: str
    upstream_grads: bytes
    def __init__(self, plan_id: _Optional[str] = ..., stage_id: _Optional[str] = ..., execution_id: _Optional[str] = ..., upstream_grads: _Optional[bytes] = ...) -> None: ...

class StageBackwardResponse(_message.Message):
    __slots__ = ("input_grads", "parameter_grads")
    INPUT_GRADS_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_GRADS_FIELD_NUMBER: _ClassVar[int]
    input_grads: bytes
    parameter_grads: bytes
    def __init__(self, input_grads: _Optional[bytes] = ..., parameter_grads: _Optional[bytes] = ...) -> None: ...

class ClearExecutionRequest(_message.Message):
    __slots__ = ("execution_id",)
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    execution_id: str
    def __init__(self, execution_id: _Optional[str] = ...) -> None: ...

class ClearExecutionResponse(_message.Message):
    __slots__ = ("cleared",)
    CLEARED_FIELD_NUMBER: _ClassVar[int]
    cleared: bool
    def __init__(self, cleared: bool = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("worker_id",)
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    def __init__(self, worker_id: _Optional[str] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ("worker_id", "online")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ONLINE_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    online: bool
    def __init__(self, worker_id: _Optional[str] = ..., online: bool = ...) -> None: ...
