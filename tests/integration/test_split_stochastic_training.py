"""Fresh dropout, wire exchange, independent replicas, and persistent Adam."""

import copy

import pytest
import torch
from torch import nn

from experiments.physical_multitask import _resolve_fixed_cuts
from splitfleet.autosplit import AutoSplitSession
from splitfleet.autosplit.state_ownership import state_ownership
from splitfleet.backends.utils import adapter_for
from splitfleet.split_engine import graph_contract_for_runtime_handle
from splitfleet.transport import encode_boundary, decode_boundary, encode_gradients, decode_gradients
from splitfleet.transport.split_wire import (
    boundary_to_envelope, envelope_to_boundary, gradients_to_envelope, envelope_to_gradients,
)


class StochasticClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(8, 16), nn.Dropout(.3), nn.ReLU(),
                                    nn.Linear(16, 16), nn.Dropout(.4), nn.ReLU(),
                                    nn.Linear(16, 3))

    def forward(self, inputs):
        return self.layers(inputs)


def test_capture_preserves_rng_and_repartition_uses_fresh_dropout():
    torch.manual_seed(982)
    model = StochasticClassifier().train()
    sample = torch.ones(4, 8)
    rng = torch.get_rng_state().clone()
    runtime = AutoSplitSession().prepare_runtime(model, sample, boundary="50%",
                                                 trainable=True, dynamic_batch=(1, 4))
    assert torch.equal(rng, torch.get_rng_state())
    for cut in ("50%", "25%", "75%"):
        runtime = runtime.backend.repartition(cut)
        rng = torch.get_rng_state().clone()
        outputs = [runtime.backend.run_suffix(runtime.backend.run_prefix(sample, training=True))
                   for _ in range(4)]
        assert not torch.equal(rng, torch.get_rng_state())
        assert any(not torch.equal(outputs[0], output) for output in outputs[1:])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("fraction", ["25%", "50%", "75%"])
def test_stochastic_split_matches_native_gradients_and_adam_across_wire(device, fraction):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(983)
    model = StochasticClassifier().to(device).train()
    initial = copy.deepcopy(model.state_dict())
    sample = torch.ones(4, 8, device=device)
    cut = _resolve_fixed_cuts(model=model, sample_inputs=sample,
        bundle={"task": "image_classification", "batch_size": 4, "model_id": "numerical_fixture"})[fraction]
    native, prefix, suffix = (copy.deepcopy(model) for _ in range(3))
    first = AutoSplitSession().prepare_runtime(prefix, sample, boundary=cut,
                                               trainable=True, dynamic_batch=(1, 4))
    last = AutoSplitSession().prepare_runtime(suffix, sample, boundary=cut,
                                              trainable=True, dynamic_batch=(1, 4))
    owners = state_ownership(first, adapter_for(prefix, sample).state_manifest(prefix).schema_hash)
    for replica in (native, prefix, suffix):
        replica.load_state_dict(initial)
    optimizers = [torch.optim.Adam(replica.parameters(), lr=1e-3)
                  for replica in (native, prefix, suffix)]
    contract = graph_contract_for_runtime_handle(first)
    for index, size in enumerate((4, 4, 1)):
        inputs = sample[:size] + index / 10
        targets = torch.arange(size, device=device) % 3
        rng = torch.cuda.get_rng_state(device) if device.startswith("cuda") else torch.get_rng_state()
        optimizers[0].zero_grad(set_to_none=True)
        reference_loss = nn.functional.cross_entropy(native(inputs), targets)
        reference_loss.backward()
        optimizers[0].step()
        advanced = torch.cuda.get_rng_state(device) if device.startswith("cuda") else torch.get_rng_state()
        if device.startswith("cuda"):
            torch.cuda.set_rng_state(rng, device)
        else:
            torch.set_rng_state(rng)
        boundary = first.backend.run_prefix(inputs, training=True)
        envelope = boundary_to_envelope(boundary, round_id=1, client_id="stochastic",
            step_id=str(index), plan_id=first.plan.plan_id, split_id=contract.split_id,
            canonical_graph_hash=contract.canonical_graph_hash,
            boundary_schema_hash=contract.boundary_schema_hash, model_version=1)
        remote = envelope_to_boundary(decode_boundary(encode_boundary(envelope)), last.runtime, device)
        loss, gradients = last.backend.train_suffix(remote, targets,
            loss_fn=nn.functional.cross_entropy, optimizer=optimizers[2])
        gradient_envelope = gradients_to_envelope(envelope, gradients)
        returned = envelope_to_gradients(decode_gradients(encode_gradients(gradient_envelope)), device)
        first.backend.backward_prefix(boundary, boundary_grads=returned, optimizer=optimizers[1])
        current_rng = torch.cuda.get_rng_state(device) if device.startswith("cuda") else torch.get_rng_state()
        assert torch.equal(advanced, current_rng)
        torch.testing.assert_close(loss, reference_loss, rtol=2e-5, atol=2e-6)
        replicas = {"prefix": (prefix, optimizers[1]), "suffix": (suffix, optimizers[2])}
        native_params = dict(native.named_parameters())
        for name, owner in zip(owners["names"], owners["owners"]):
            if owner == "initial":
                continue
            replica, optimizer = replicas[owner]
            parameter = dict(replica.named_parameters())[name]
            reference = native_params[name]
            torch.testing.assert_close(parameter, reference, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=2e-5, atol=2e-6)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[parameter][key],
                    optimizers[0].state[reference][key], rtol=2e-5, atol=2e-6)
