"""Multi-batch Adam with two independent replicas and a partial final batch."""

import copy

import pytest
import torch
from torch import nn

from experiments.physical_multitask import _resolve_fixed_cuts
from splitfleet.autosplit import AutoSplitSession
from splitfleet.autosplit.state_ownership import state_ownership
from splitfleet.backends.utils import adapter_for


class ResidualClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv2d(3, 4, 3, padding=1)
        self.branch = nn.Conv2d(4, 4, 3, padding=1)
        self.output = nn.Linear(4, 3)

    def forward(self, inputs):
        hidden = torch.relu(self.input(inputs))
        residual = hidden + torch.relu(self.branch(hidden))
        return self.output(residual.mean(dim=(-2, -1)))


@pytest.mark.parametrize("fraction", ["25%", "50%", "75%"])
def test_separate_prefix_suffix_adam_matches_native_multiple_batches(fraction):
    torch.manual_seed(724)
    model = ResidualClassifier().train()
    initial = copy.deepcopy(model.state_dict())
    sample = torch.randn(4, 3, 8, 8)
    resolved = _resolve_fixed_cuts(
        model=model, sample_inputs=sample,
        bundle={"task": "image_classification", "batch_size": 4, "model_id": "numerical_fixture"},
    )[fraction]
    native, prefix, suffix = (copy.deepcopy(model) for _ in range(3))
    prefix_handle = AutoSplitSession().prepare_runtime(
        prefix, sample, boundary=resolved, trainable=True, dynamic_batch=(1, 4))
    suffix_handle = AutoSplitSession().prepare_runtime(
        suffix, sample, boundary=resolved, trainable=True, dynamic_batch=(1, 4))
    schema = adapter_for(prefix, sample).state_manifest(prefix).schema_hash
    ownership = state_ownership(prefix_handle, schema)
    assert set(ownership["owners"]) >= {"prefix", "suffix"}
    for replica in (native, prefix, suffix):
        replica.load_state_dict(initial)
        replica.train()
    native_optimizer, prefix_optimizer, suffix_optimizer = (
        torch.optim.Adam(replica.parameters(), lr=1e-3) for replica in (native, prefix, suffix))
    loss_fn = nn.CrossEntropyLoss()

    # Optimizers persist through several batches as in a physical local epoch.
    for batch_size in (4, 4, 1):
        inputs = torch.randn(batch_size, 3, 8, 8)
        labels = torch.randint(0, 3, (batch_size,))
        native_optimizer.zero_grad(set_to_none=True)
        native_loss = loss_fn(native(inputs), labels)
        native_loss.backward()
        native_optimizer.step()
        boundary = prefix_handle.backend.run_prefix(inputs, training=True)
        split_loss, gradients = suffix_handle.backend.train_suffix(
            boundary, labels, loss_fn=loss_fn, optimizer=suffix_optimizer)
        prefix_handle.backend.backward_prefix(boundary, boundary_grads=gradients,
                                               optimizer=prefix_optimizer)
        torch.testing.assert_close(split_loss, native_loss, rtol=2e-5, atol=2e-6)
        states = {"prefix": prefix.state_dict(), "suffix": suffix.state_dict(), "initial": initial}
        for name, owner in zip(ownership["names"], ownership["owners"]):
            torch.testing.assert_close(states[owner][name], native.state_dict()[name],
                                       rtol=2e-5, atol=2e-6, msg=lambda msg: f"{name}: {msg}")
