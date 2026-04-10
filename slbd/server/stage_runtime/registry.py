"""Worker registration for autosplit stage execution."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from slbd.autosplit.runtime import AutoSplitSession
from slbd.autosplit.types import WorkerSpec
from slbd.server.stage_runtime.executor import (
    LocalStageExecutor,
    RemoteStageExecutor,
    StageExecutor,
)


class WorkerRegistry:
    """Registry of available stage workers."""

    def __init__(self) -> None:
        self._workers: Dict[str, WorkerSpec] = {}
        self._executors: Dict[str, StageExecutor] = {}

    def register(
        self,
        worker_spec: WorkerSpec,
        executor: Optional[StageExecutor] = None,
    ) -> WorkerSpec:
        self._workers[worker_spec.worker_id] = worker_spec
        if executor is None:
            if worker_spec.address:
                executor = RemoteStageExecutor(worker_spec)
            else:
                executor = LocalStageExecutor(
                    worker_spec,
                    session=AutoSplitSession(device=worker_spec.device),
                )
        self._executors[worker_spec.worker_id] = executor
        return worker_spec

    def unregister(self, worker_id: str) -> None:
        self._workers.pop(worker_id, None)
        self._executors.pop(worker_id, None)

    def get_worker(self, worker_id: str) -> WorkerSpec:
        return self._workers[worker_id]

    def get_executor(self, worker_id: str) -> StageExecutor:
        return self._executors[worker_id]

    def list_workers(self, *, online_only: bool = True) -> list[WorkerSpec]:
        workers = list(self._workers.values())
        if online_only:
            workers = [worker for worker in workers if worker.online]
        return sorted(workers, key=lambda worker: worker.worker_id)

    def update_status(self, worker_id: str, *, online: bool) -> None:
        worker = self._workers[worker_id]
        self._workers[worker_id] = WorkerSpec(
            worker_id=worker.worker_id,
            address=worker.address,
            device=worker.device,
            bandwidth_mbps=worker.bandwidth_mbps,
            memory_bytes=worker.memory_bytes,
            tags=worker.tags,
            online=online,
        )

    def __len__(self) -> int:
        return len(self._workers)

    def __iter__(self) -> Iterable[WorkerSpec]:
        return iter(self.list_workers(online_only=False))


GLOBAL_WORKER_REGISTRY = WorkerRegistry()
