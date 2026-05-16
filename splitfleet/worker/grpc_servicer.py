"""gRPC servicer for remote stage workers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import grpc

from splitfleet.autosplit.serde import (
    deserialize_plan_descriptor,
    dumps_torch_object,
    loads_torch_object,
)
from splitfleet.autosplit.types import WorkerSpec
from splitfleet.proto import stage_worker_pb2, stage_worker_pb2_grpc
from splitfleet.worker.runtime import StageWorkerRuntime


class StageWorkerServicer(stage_worker_pb2_grpc.StageWorkerServicer):
    """Expose a StageWorkerRuntime over synchronous gRPC."""

    def __init__(self, runtime: StageWorkerRuntime) -> None:
        super().__init__()
        self.runtime = runtime

    def RegisterPlan(self, request, context):
        _ = context
        descriptor = deserialize_plan_descriptor(request.plan_descriptor)
        placement = self.runtime.register_plan(
            descriptor,
            model_state=request.model_state,
        )
        return stage_worker_pb2.RegisterPlanResponse(
            worker_id=self.runtime.worker_spec.worker_id,
            plan_id=placement.plan_id,
            accepted=True,
            graph_signature=placement.graph_signature,
            message="ok",
        )

    def ExecuteStage(self, request, context):
        _ = context
        seeded_values = loads_torch_object(
            request.seeded_values,
            map_location=self.runtime.worker_spec.device,
        )
        result = self.runtime.execute_stage(
            plan_id=request.plan_id,
            stage_id=request.stage_id,
            execution_id=request.execution_id,
            seeded_values=seeded_values,
            differentiable=request.differentiable,
            detach_boundary=request.detach_boundary,
        )
        return stage_worker_pb2.StageExecutionResponse(
            available_updates=dumps_torch_object(result.available_updates),
            stored_updates=dumps_torch_object(result.stored_updates),
            metadata={
                "worker_id": self.runtime.worker_spec.worker_id,
                "client_id": request.client_id,
                "round_id": str(request.round_id),
                **dict(request.metadata),
            },
        )

    def BackwardStage(self, request, context):
        _ = context
        upstream_grads = loads_torch_object(
            request.upstream_grads,
            map_location=self.runtime.worker_spec.device,
        )
        result = self.runtime.backward_stage(
            plan_id=request.plan_id,
            stage_id=request.stage_id,
            execution_id=request.execution_id,
            upstream_grads=upstream_grads,
        )
        return stage_worker_pb2.StageBackwardResponse(
            input_grads=dumps_torch_object(result.input_grads),
            parameter_grads=dumps_torch_object(result.parameter_grads),
        )

    def ClearExecution(self, request, context):
        _ = context
        return stage_worker_pb2.ClearExecutionResponse(
            cleared=self.runtime.clear_execution(request.execution_id)
        )

    def Heartbeat(self, request, context):
        _ = context
        return stage_worker_pb2.HeartbeatResponse(
            worker_id=request.worker_id or self.runtime.worker_spec.worker_id,
            online=True,
        )


@dataclass
class WorkerServiceHandle:
    """Handle for a running worker gRPC server."""

    worker_spec: WorkerSpec
    server: grpc.Server
    runtime: StageWorkerRuntime

    @property
    def address(self) -> str:
        return self.worker_spec.address or ""

    def stop(self, grace: float = 0.0) -> None:
        self.server.stop(grace)

    def wait(self, timeout: Optional[float] = None) -> None:
        self.server.wait_for_termination(timeout=timeout)


def start_stage_worker_server(
    *,
    runtime: StageWorkerRuntime,
    server_address: str,
    max_workers: int = 8,
) -> WorkerServiceHandle:
    """Start a synchronous gRPC server for a stage worker."""

    server = grpc.server(
        ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )
    stage_worker_pb2_grpc.add_StageWorkerServicer_to_server(
        StageWorkerServicer(runtime),
        server,
    )
    bound_port = server.add_insecure_port(server_address)
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind worker server to {server_address}")
    host, _, raw_port = server_address.rpartition(":")
    if raw_port == "0":
        address = f"{host}:{bound_port}"
    else:
        address = server_address
    runtime.worker_spec = WorkerSpec(
        worker_id=runtime.worker_spec.worker_id,
        address=address,
        device=runtime.worker_spec.device,
        bandwidth_mbps=runtime.worker_spec.bandwidth_mbps,
        memory_bytes=runtime.worker_spec.memory_bytes,
        tags=runtime.worker_spec.tags,
        online=runtime.worker_spec.online,
    )
    server.start()
    return WorkerServiceHandle(
        worker_spec=runtime.worker_spec,
        server=server,
        runtime=runtime,
    )
