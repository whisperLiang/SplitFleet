"""Generic Flower client that feeds local data into the autosplit server runtime."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch

from splitfleet.client.numpy_client import NumPyClient


def _to_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_numpy(item) for item in value)
    return value


def _batch_size(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.shape[0]) if value.ndim > 0 else 1
    if isinstance(value, torch.Tensor):
        return int(value.shape[0]) if value.ndim > 0 else 1
    if isinstance(value, dict):
        for item in value.values():
            size = _batch_size(item)
            if size > 0:
                return size
    if isinstance(value, (list, tuple)) and value:
        return _batch_size(value[0])
    return 1


class AutoSplitNumPyClient(NumPyClient):
    """Data-owning Flower client that delegates model execution to autosplit runtimes."""

    def __init__(
        self,
        *,
        train_data: Iterable[Any],
        evaluate_data: Optional[Iterable[Any]] = None,
        batch_adapter: Optional[Callable[[Any], tuple[Any, Any]]] = None,
    ) -> None:
        self.train_data = train_data
        self.evaluate_data = evaluate_data if evaluate_data is not None else train_data
        self.batch_adapter = batch_adapter or self._default_batch_adapter

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
            inputs, targets = self.batch_adapter(batch)
            response = proxy.train_batch(
                inputs=_to_numpy(inputs),
                targets=_to_numpy(targets),
                _streams_=False,
            )
            batch_examples = _batch_size(inputs)
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
            inputs, targets = self.batch_adapter(batch)
            response = proxy.evaluate_batch(
                inputs=_to_numpy(inputs),
                targets=_to_numpy(targets),
                _streams_=False,
            )
            batch_examples = _batch_size(inputs)
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

    @staticmethod
    def _default_batch_adapter(batch: Any) -> tuple[Any, Any]:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        raise ValueError(
            "Expected each batch to be a `(inputs, targets)` pair. "
            "Provide `batch_adapter` to customize parsing."
        )
