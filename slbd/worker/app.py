"""Local worker entrypoint for autosplit execution."""

from __future__ import annotations

import copy
from typing import Optional

from slbd.autosplit.runtime import AutoSplitSession
from slbd.autosplit.types import WorkerSpec
from slbd.server.stage_runtime.executor import LocalStageExecutor
from slbd.server.stage_runtime.registry import GLOBAL_WORKER_REGISTRY, WorkerRegistry
from slbd.worker.grpc_servicer import WorkerServiceHandle, start_stage_worker_server
from slbd.worker.runtime import StageWorkerRuntime


def start_worker(
    *,
    worker_id: str,
    model=None,
    sample_inputs=None,
    sample_kwargs: Optional[dict] = None,
    server_address: Optional[str] = None,
    device: str = "cpu",
    bandwidth_mbps: float = 1000.0,
    memory_bytes: Optional[int] = None,
    tags: tuple[str, ...] = (),
    worker_registry: Optional[WorkerRegistry] = None,
    register_with_registry: bool = True,
) -> WorkerSpec | WorkerServiceHandle:
    """Register a local worker executor and return its worker spec."""

    registry = worker_registry or GLOBAL_WORKER_REGISTRY
    worker_spec = WorkerSpec(
        worker_id=worker_id,
        address=server_address,
        device=device,
        bandwidth_mbps=bandwidth_mbps,
        memory_bytes=memory_bytes,
        tags=tags,
    )
    runtime = None
    if model is not None:
        runtime = StageWorkerRuntime(
            worker_spec=worker_spec,
            model=copy.deepcopy(model),
            sample_inputs=sample_inputs,
            sample_kwargs=sample_kwargs,
            autosplit_session=AutoSplitSession(device=device),
        )

    if server_address is not None:
        if runtime is None:
            raise ValueError("Remote workers require `model` and `sample_inputs`.")
        handle = start_stage_worker_server(
            runtime=runtime,
            server_address=server_address,
        )
        if register_with_registry:
            registry.register(handle.worker_spec)
        return handle

    registry.register(
        worker_spec,
        executor=LocalStageExecutor(
            worker_spec,
            session=runtime.autosplit_session if runtime is not None else AutoSplitSession(device=device),
        ),
    )
    return worker_spec
