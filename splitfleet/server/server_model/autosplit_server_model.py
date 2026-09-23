"""Server-model bridge that drives autosplit execution inside Flower rounds."""

from __future__ import annotations

import threading
import uuid
from typing import Any

import numpy as np

from splitfleet.server.server_model.numpy_server_model import NumPyServerModel
from splitfleet.backends.utils import adapter_for, bind_model_inputs
from splitfleet.tasks import ModelInputs, TaskAdapter
from splitfleet.tasks.transport import decode_task_batch
from splitfleet.common.constants import AUTOSPLIT_PLAN_ID_CONFIG_KEY

class AutoSplitServerModel(NumPyServerModel):
    """Expose autosplit training/eval as server-model methods consumable by clients."""

    def __init__(
        self,
        *,
        runtime_manager,
        model: Any,
        optimizer_fn=None,
        loss_fn=None,
        task: TaskAdapter | None = None,
        device: str = "cpu",
    ) -> None:
        self.runtime_manager = runtime_manager
        self.optimizer_fn = optimizer_fn
        self.task = task
        self.loss_fn = loss_fn if loss_fn is not None else (task.loss if task is not None else None)
        self.device = device
        sample_inputs = runtime_manager._require_runtime_handle().plan.metadata["_example_inputs"]
        self.backend_adapter = adapter_for(model, sample_inputs)
        self.model = self.backend_adapter.move_model(self.backend_adapter.clone_model(model), device)
        self.optimizer = None
        self.runtime_handle = None
        self.sid = ""
        self.num_examples = 0
        self.loss_total = 0.0
        self._lock = threading.Lock()

    def get_parameters(self):
        return self.backend_adapter.export_ndarrays(self.model)

    def configure_fit(self, parameters, config) -> None:
        self._configure(parameters=parameters, sid=config.get("sid", ""), training=True,
                        plan_id=config.get(AUTOSPLIT_PLAN_ID_CONFIG_KEY))
        if self.optimizer_fn is not None:
            self.optimizer = self.optimizer_fn(self.model)
        else:
            self.optimizer = self.backend_adapter.build_optimizer(self.model, {"name": "sgd", "lr": 0.01})
        if self.optimizer is not None and self.backend_adapter.backend_name == "tf":
            # Both split segments share this optimizer. Keras otherwise builds
            # its variable registry on the suffix's first partial update and
            # rejects the prefix variables when their gradients arrive.
            self.optimizer.build(self.model.trainable_variables)

    def configure_evaluate(self, parameters, config) -> None:
        self._configure(parameters=parameters, sid=config.get("sid", ""), training=False,
                        plan_id=config.get(AUTOSPLIT_PLAN_ID_CONFIG_KEY))
        self.optimizer = None

    def get_fit_result(self):
        average_loss = self.loss_total / max(self.num_examples, 1)
        return self.get_parameters(), {
            "num_examples": self.num_examples,
            "avg_loss": average_loss,
        }

    def train_task_batch(self, payload):
        """Train a nested task batch after the ordinary NumPy RPC wire round trip."""
        with self._lock:
            responses = []
            for encoded in payload:
                batch = decode_task_batch(encoded, device=self.device, backend=self.backend_adapter.backend_name)
                call = ModelInputs(bind_model_inputs(batch.inputs.args, self.backend_adapter), batch.inputs.kwargs)
                result = self.runtime_manager.autosplit_session.run_train(
                    self.runtime_handle, call, targets=batch.targets, loss_fn=self.loss_fn,
                    prefix_optimizer=self.optimizer, suffix_optimizer=self.optimizer,
                )
                loss_value = self.backend_adapter.scalar_value(result["loss"])
                self.num_examples += batch.num_examples
                self.loss_total += loss_value * batch.num_examples
                responses.append({"loss": np.asarray([loss_value], dtype=np.float32),
                                  "num_examples": np.asarray([batch.num_examples], dtype=np.int64)})
            return responses

    def evaluate_task_batch(self, payload):
        with self._lock:
            responses = []
            for encoded in payload:
                batch = decode_task_batch(encoded, device=self.device, backend=self.backend_adapter.backend_name)
                call = ModelInputs(bind_model_inputs(batch.inputs.args, self.backend_adapter), batch.inputs.kwargs)
                outputs = self.runtime_manager.autosplit_session.run_eval(self.runtime_handle, call)
                loss = self.runtime_manager.autosplit_session.compute_loss(outputs, batch.targets, self.loss_fn)
                responses.append({"loss": np.asarray([self.backend_adapter.scalar_value(loss)], dtype=np.float32),
                                  "num_examples": np.asarray([batch.num_examples], dtype=np.int64)})
            return responses

    def _configure(self, *, parameters, sid: str, training: bool, plan_id: str | None = None) -> None:
        self.sid = sid
        self.num_examples = 0
        self.loss_total = 0.0
        self.backend_adapter.load_ndarrays(self.model, parameters)
        self.backend_adapter.set_training(self.model, training)
        plan_suffix = f"{sid or 'shared'}_{uuid.uuid4().hex[:8]}"
        self.runtime_handle = self.runtime_manager.clone_runtime_for_model(
            self.model,
            suffix=plan_suffix,
            plan_id=plan_id,
        )
