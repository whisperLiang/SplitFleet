"""Executors that host autosplit stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import grpc

from slbd.autosplit.runtime import AutoSplitSession
from slbd.autosplit.serde import (
    dump_model_state,
    dumps_torch_object,
    loads_torch_object,
    serialize_plan_descriptor,
)
from slbd.autosplit.types import PlacementPlan, WorkerSpec
from slbd.proto import stage_worker_pb2, stage_worker_pb2_grpc


class StageExecutor(ABC):
    """Abstract worker-side execution surface."""

    def __init__(self, worker_spec: WorkerSpec) -> None:
        self.worker_spec = worker_spec

    @abstractmethod
    def run_eval(
        self,
        placement_plan: PlacementPlan,
        inputs: Any,
        *,
        input_kwargs: Optional[dict] = None,
    ) -> Any:
        """Run evaluation for a placement plan."""

    @abstractmethod
    def run_train(
        self,
        placement_plan: PlacementPlan,
        inputs: Any,
        *,
        input_kwargs: Optional[dict] = None,
        targets: Any = None,
        loss_fn=None,
        optimizer=None,
        zero_grad: bool = True,
        step_optimizer: bool = True,
    ) -> dict:
        """Run training for a placement plan."""

    def load_placement_plan(
        self,
        placement_plan: PlacementPlan,
        *,
        model_state: bytes | None = None,
    ) -> None:
        """Synchronize a placement plan onto the executor."""

    def run_stage_forward(
        self,
        placement_plan: PlacementPlan,
        stage_id: str,
        execution_id: str,
        seeded_values: Dict[int, Any],
        *,
        client_id: str = "",
        round_id: int = 0,
        differentiable: bool,
        detach_boundary: bool,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """Run forward for a single stage."""

    def run_stage_backward(
        self,
        placement_plan: PlacementPlan,
        stage_id: str,
        execution_id: str,
        upstream_grads: Dict[int, Any],
    ):
        """Run backward for a single stage."""

    def clear_execution(self, execution_id: str) -> None:
        """Release any retained execution state."""

    @property
    def is_remote(self) -> bool:
        return False


class LocalStageExecutor(StageExecutor):
    """In-process executor used by the coordinator and local workers."""

    def __init__(self, worker_spec: WorkerSpec, session: Optional[AutoSplitSession] = None) -> None:
        super().__init__(worker_spec)
        self.session = session or AutoSplitSession(device=worker_spec.device)
        self._stage_traces: Dict[str, Dict[str, Any]] = {}

    def run_eval(
        self,
        placement_plan: PlacementPlan,
        inputs: Any,
        *,
        input_kwargs: Optional[dict] = None,
    ) -> Any:
        return self.session.run_eval(
            placement_plan,
            inputs,
            input_kwargs=input_kwargs,
        )

    def run_train(
        self,
        placement_plan: PlacementPlan,
        inputs: Any,
        *,
        input_kwargs: Optional[dict] = None,
        targets: Any = None,
        loss_fn=None,
        optimizer=None,
        zero_grad: bool = True,
        step_optimizer: bool = True,
    ) -> dict:
        return self.session.run_train(
            placement_plan,
            inputs,
            input_kwargs=input_kwargs,
            targets=targets,
            loss_fn=loss_fn,
            optimizer=optimizer,
            zero_grad=zero_grad,
            step_optimizer=step_optimizer,
        )

    def run_stage_forward(
        self,
        placement_plan: PlacementPlan,
        stage_id: str,
        execution_id: str,
        seeded_values: Dict[int, Any],
        *,
        client_id: str = "",
        round_id: int = 0,
        differentiable: bool,
        detach_boundary: bool,
        metadata: Optional[Dict[str, str]] = None,
    ):
        _ = (client_id, round_id, metadata)
        stage = next(
            stage
            for stage in placement_plan.partition_plan.stages
            if stage.stage_id == stage_id
        )
        stage_trace, result = self.session.run_stage_forward(
            placement_plan,
            stage,
            seeded_values,
            differentiable=differentiable,
            detach_boundary=detach_boundary,
            output_indices=self.session.get_output_indices(placement_plan),
        )
        self._stage_traces.setdefault(execution_id, {})[stage_id] = stage_trace
        return result

    def run_stage_backward(
        self,
        placement_plan: PlacementPlan,
        stage_id: str,
        execution_id: str,
        upstream_grads: Dict[int, Any],
    ):
        stage_trace = self._stage_traces.get(execution_id, {}).pop(stage_id)
        result = self.session.run_stage_backward(
            placement_plan,
            stage_trace,
            upstream_grads,
        )
        if execution_id in self._stage_traces and not self._stage_traces[execution_id]:
            self._stage_traces.pop(execution_id, None)
        return result

    def clear_execution(self, execution_id: str) -> None:
        self._stage_traces.pop(execution_id, None)


class RemoteStageExecutor(StageExecutor):
    """Executor backed by a remote worker runtime over gRPC."""

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

    def run_eval(
        self,
        placement_plan: PlacementPlan,
        inputs: Any,
        *,
        input_kwargs: Optional[dict] = None,
    ) -> Any:
        raise NotImplementedError("Remote stage executors are orchestrated stage-by-stage.")

    def run_train(
        self,
        placement_plan: PlacementPlan,
        inputs: Any,
        *,
        input_kwargs: Optional[dict] = None,
        targets: Any = None,
        loss_fn=None,
        optimizer=None,
        zero_grad: bool = True,
        step_optimizer: bool = True,
    ) -> dict:
        raise NotImplementedError("Remote stage executors are orchestrated stage-by-stage.")

    def load_placement_plan(
        self,
        placement_plan: PlacementPlan,
        *,
        model_state: bytes | None = None,
    ) -> None:
        model = placement_plan.partition_plan.execution_plan.model
        state = model_state if model_state is not None else (
            dump_model_state(model) if model is not None else b""
        )
        self.stub.RegisterPlan(
            stage_worker_pb2.RegisterPlanRequest(
                worker_id=self.worker_spec.worker_id,
                plan_id=placement_plan.plan_id,
                plan_descriptor=serialize_plan_descriptor(placement_plan),
                model_state=state,
            )
        )

    def run_stage_forward(
        self,
        placement_plan: PlacementPlan,
        stage_id: str,
        execution_id: str,
        seeded_values: Dict[int, Any],
        *,
        client_id: str = "",
        round_id: int = 0,
        differentiable: bool,
        detach_boundary: bool,
        metadata: Optional[Dict[str, str]] = None,
    ):
        response = self.stub.ExecuteStage(
            stage_worker_pb2.StageExecutionRequest(
                plan_id=placement_plan.plan_id,
                stage_id=stage_id,
                execution_id=execution_id,
                client_id=client_id,
                round_id=round_id,
                differentiable=differentiable,
                detach_boundary=detach_boundary,
                seeded_values=dumps_torch_object(seeded_values),
                metadata=metadata or {},
            )
        )
        return type(
            "RemoteStageForwardResult",
            (),
            {
                "available_updates": loads_torch_object(
                    response.available_updates,
                    map_location="cpu",
                ),
                "stored_updates": loads_torch_object(
                    response.stored_updates,
                    map_location="cpu",
                ),
            },
        )()

    def run_stage_backward(
        self,
        placement_plan: PlacementPlan,
        stage_id: str,
        execution_id: str,
        upstream_grads: Dict[int, Any],
    ):
        response = self.stub.BackwardStage(
            stage_worker_pb2.StageBackwardRequest(
                plan_id=placement_plan.plan_id,
                stage_id=stage_id,
                execution_id=execution_id,
                upstream_grads=dumps_torch_object(upstream_grads),
            )
        )
        return type(
            "RemoteStageBackwardResult",
            (),
            {
                "input_grads": loads_torch_object(response.input_grads, map_location="cpu"),
                "parameter_grads": loads_torch_object(
                    response.parameter_grads,
                    map_location="cpu",
                ),
            },
        )()

    def clear_execution(self, execution_id: str) -> None:
        self.stub.ClearExecution(
            stage_worker_pb2.ClearExecutionRequest(execution_id=execution_id)
        )
