"""Isolated full-model native CPU regression, not scientific gate admission."""

import numpy as np

from tests.integration.resnet50_reference.array_helpers import canonical_array
from experiments.resnet50_quality.gate_torch import compare_tensor
from experiments.resnet50_quality.model_contract import NAMES, canonical_state
from tests.integration.resnet50_reference.native import (
    build_model,
    configure_backend,
    export_state,
    native_batch,
    optimizer,
    train_step,
)
from tests.integration.resnet50_reference.staged_reference import train_staged
from splitfleet.backends.utils import tensorflow_tensor_device
from splitfleet.virtual.backend_ops import FrameworkOps


def main():
    configure_backend("tensorflow")
    initial = canonical_state(233)
    model = build_model("tensorflow", initial, "cpu")
    rng = np.random.default_rng(233)
    images, labels = native_batch("tensorflow", rng.normal(size=(2, 3, 64, 64)).astype(np.float32),
                                 np.asarray([2, 3], np.int64), "cpu")
    opt = optimizer("tensorflow", model, .05)
    adapter_opt = FrameworkOps("tf").optimizer(model, "sgd", .05)
    for native_optimizer in (opt, adapter_opt):
        assert {tensorflow_tensor_device(variable) for variable in native_optimizer.variables} == {"cpu"}
    state = initial
    for step, stage in enumerate((1, 3), start=1):
        expected = train_step("tensorflow", model, images, labels,
                              native_optimizer=opt, learning_rate=.05, capture=True)
        actual = train_staged("tensorflow", state, images, labels, stage=stage,
                              client_device="cpu", server_device="cpu", learning_rate=.05)
        checks = [compare_tensor(actual["gradients"][name], canonical_array("tensorflow", name, value), name=name)
                  for name, value in expected["gradients"].items()]
        expected_state = export_state("tensorflow", model)
        checks.extend(compare_tensor(actual["state"][name], expected_state[name], name=name) for name in NAMES)
        assert len(checks) == 481
        assert all(row["passed"] for row in checks), [row["name"] for row in checks if not row["passed"]]
        assert all(int(value) == step for name, value in actual["state"].items()
                   if name.endswith("num_batches_tracked"))
        state = actual["state"]


if __name__ == "__main__":
    main()
