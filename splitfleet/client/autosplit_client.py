"""Generic Flower client that feeds local data into the autosplit server runtime."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import numpy as np

from splitfleet.client.numpy_client import NumPyClient
from splitfleet.tasks import TaskAdapter, prepare_task_batch
from splitfleet.tasks.transport import encode_task_batch



class AutoSplitNumPyClient(NumPyClient):
    """Data-owning Flower client that delegates model execution to autosplit runtimes."""

    def __init__(
        self,
        *,
        train_data: Iterable[Any],
        evaluate_data: Optional[Iterable[Any]] = None,
        batch_adapter: Optional[Callable[[Any], tuple[Any, Any]]] = None,
        task: TaskAdapter | None = None,
        backend: str = "torch",
    ) -> None:
        self.train_data = train_data
        self.evaluate_data = evaluate_data if evaluate_data is not None else train_data
        self.batch_adapter = batch_adapter
        self.task = task
        self.backend = backend

    def get_parameters(self, config):
        _ = config
        return []

    def fit(self, parameters, config):
        _ = (parameters, config)
        proxy = self._require_server_model_proxy()
        proxy.numpy()
        num_examples = 0
        weighted_loss = 0.0
        for batch in self.train_data:
            task_batch = prepare_task_batch(batch, task=self.task, batch_adapter=self.batch_adapter, training=True)
            response = proxy.train_task_batch(
                payload=encode_task_batch(task_batch, backend=self.backend), _streams_=False,
            )
            batch_examples = task_batch.num_examples
            batch_loss = float(np.asarray(response["loss"]).reshape(-1)[0])
            num_examples += batch_examples
            weighted_loss += batch_loss * batch_examples
        metrics = {}
        if num_examples > 0:
            metrics["loss"] = weighted_loss / num_examples
        return [], num_examples, metrics

    def evaluate(self, parameters, config):
        _ = (parameters, config)
        proxy = self._require_server_model_proxy()
        proxy.numpy()
        num_examples = 0
        weighted_loss = 0.0
        for batch in self.evaluate_data:
            task_batch = prepare_task_batch(batch, task=self.task, batch_adapter=self.batch_adapter, training=False)
            response = proxy.evaluate_task_batch(
                payload=encode_task_batch(task_batch, backend=self.backend), _streams_=False,
            )
            batch_examples = task_batch.num_examples
            batch_loss = float(np.asarray(response["loss"]).reshape(-1)[0])
            num_examples += batch_examples
            weighted_loss += batch_loss * batch_examples
        average_loss = weighted_loss / max(num_examples, 1)
        return float(average_loss), num_examples, {"loss": average_loss}

    def _require_server_model_proxy(self):
        proxy = getattr(self, "server_model_proxy", None)
        if proxy is None:
            raise RuntimeError("AutoSplitNumPyClient requires a server_model_proxy.")
        return proxy
