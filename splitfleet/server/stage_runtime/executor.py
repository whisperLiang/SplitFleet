"""Executor stubs retained for worker registry compatibility."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import grpc

from splitfleet.autosplit.serde import serialize_plan_descriptor
from splitfleet.autosplit.types import PlacementPlan, WorkerSpec
from splitfleet.proto import stage_worker_pb2, stage_worker_pb2_grpc


REMOTE_STAGE_ERROR = (
    "TorchLens autosplit backend supports coordinator-local suffix execution only; "
    "node-level remote stage execution is not active."
)


class StageExecutor(ABC):
    """Abstract worker-side execution surface."""

    def __init__(self, worker_spec: WorkerSpec) -> None:
        self.worker_spec = worker_spec

    @abstractmethod
    def run_eval(self, placement_plan: PlacementPlan, inputs: Any) -> Any:
        """Run evaluation for a placement plan."""

    @abstractmethod
    def run_train(self, placement_plan: PlacementPlan, inputs: Any, **kwargs) -> dict:
        """Run training for a placement plan."""

    def load_placement_plan(
        self,
        placement_plan: PlacementPlan,
        *,
        model_state: bytes | None = None,
    ) -> None:
        _ = (placement_plan, model_state)

    def run_stage_forward(self, *args, **kwargs):
        _ = (args, kwargs)
        raise NotImplementedError(REMOTE_STAGE_ERROR)

    def run_stage_backward(self, *args, **kwargs):
        _ = (args, kwargs)
        raise NotImplementedError(REMOTE_STAGE_ERROR)

    def clear_execution(self, execution_id: str) -> None:
        _ = execution_id

    @property
    def is_remote(self) -> bool:
        return False


class LocalStageExecutor(StageExecutor):
    """In-process executor used by the registry for coordinator-local execution."""

    def __init__(self, worker_spec: WorkerSpec, session=None) -> None:
        super().__init__(worker_spec)
        self.session = session

    def run_eval(self, placement_plan: PlacementPlan, inputs: Any) -> Any:
        if self.session is None:
            raise RuntimeError("LocalStageExecutor has no AutoSplitSession.")
        return self.session.run_eval(placement_plan, inputs)

    def run_train(self, placement_plan: PlacementPlan, inputs: Any, **kwargs) -> dict:
        if self.session is None:
            raise RuntimeError("LocalStageExecutor has no AutoSplitSession.")
        return self.session.run_train(placement_plan, inputs, **kwargs)


class RemoteStageExecutor(StageExecutor):
    """gRPC-backed compatibility shell for removed node-level stage execution."""

    def __init__(
        self,
        worker_spec: WorkerSpec,
        *,
        channel: Optional[grpc.Channel] = None,
    ) -> None:
        super().__init__(worker_spec)
        if not worker_spec.address:
            raise ValueError("RemoteStageExecutor requires a worker address.")
        self.channel = channel or grpc.insecure_channel(
            worker_spec.address,
            options=[
                ("grpc.max_send_message_length", 256 * 1024 * 1024),
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ],
        )
        self.stub = stage_worker_pb2_grpc.StageWorkerStub(self.channel)

    @property
    def is_remote(self) -> bool:
        return True

    def run_eval(self, placement_plan: PlacementPlan, inputs: Any) -> Any:
        _ = (placement_plan, inputs)
        raise NotImplementedError(REMOTE_STAGE_ERROR)

    def run_train(self, placement_plan: PlacementPlan, inputs: Any, **kwargs) -> dict:
        _ = (placement_plan, inputs, kwargs)
        raise NotImplementedError(REMOTE_STAGE_ERROR)

    def load_placement_plan(
        self,
        placement_plan: PlacementPlan,
        *,
        model_state: bytes | None = None,
    ) -> None:
        self.stub.RegisterPlan(
            stage_worker_pb2.RegisterPlanRequest(
                worker_id=self.worker_spec.worker_id,
                plan_id=placement_plan.plan_id,
                plan_descriptor=serialize_plan_descriptor(placement_plan),
                model_state=model_state or b"",
            )
        )

    def run_stage_forward(self, *args, **kwargs):
        _ = (args, kwargs)
        raise NotImplementedError(REMOTE_STAGE_ERROR)

    def run_stage_backward(self, *args, **kwargs):
        _ = (args, kwargs)
        raise NotImplementedError(REMOTE_STAGE_ERROR)

    def clear_execution(self, execution_id: str) -> None:
        self.stub.ClearExecution(stage_worker_pb2.ClearExecutionRequest(execution_id=execution_id))
