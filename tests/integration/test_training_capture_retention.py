"""Selective training snapshots preserve the executable graph contract."""

import copy
from dataclasses import replace
import importlib

import pytest
import torch
from torch import nn

from splitfleet.autosplit.torch_training_capture import prepare_training_runtime
from splitfleet.autosplit.torchlens_runtime import make_split_spec, prepare_split_runtime


class BufferedResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(8, 8)
        self.dropout = nn.Dropout(.4)
        self.head = nn.Linear(8, 3)
        self.register_buffer("scale", torch.arange(1, 9).float() / 8)

    def forward(self, inputs):
        features = self.dropout(torch.relu(self.first(inputs))) + inputs
        return {"logits": self.head(features * self.scale), "features": features}


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_selective_capture_preserves_graph_grad_metadata_and_source_values(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(1064)
    model = BufferedResidual().to(device).train()
    sample = torch.randn(2, 8, device=device)
    request = make_split_spec("50%", trainable=True)
    full_request = replace(request, features=replace(request.features, retain_trace=True))
    modules = [importlib.import_module(name) for name in
               ("torchlens.split.api", "torchlens.split.pipeline", "torchlens.split.graph")]
    identities = (modules[0].prepare, modules[1].capture_canonical_model,
                  modules[1].normalize_to_split_ir, modules[2].split_graph_from_trace)
    full = prepare_split_runtime(copy.deepcopy(model), sample, full_request)
    selective = prepare_training_runtime(copy.deepcopy(model), (sample,), request)
    assert selective.graph_ir.graph_hash == full.graph_ir.graph_hash
    assert [(node.canonical_id, node.requires_grad) for node in selective.trace_graph.nodes] == [
        (node.canonical_id, node.requires_grad) for node in full.trace_graph.nodes]
    assert not selective.retains_trace and full.retains_trace
    retained = [node for node in selective.trace_graph.nodes if node.op.out is not None]
    assert len(retained) < len(selective.trace_graph.nodes) / 2
    assert any(node.is_buffer for node in retained)
    assert selective.batch_validation["status"] == full.batch_validation["status"]
    # A prepared training replay must accept both the probe batch and a tail.
    for size in (2, 1):
        output = selective.run_suffix(selective.run_prefix(sample[:size]))
        assert output["logits"].shape == (size, 3)
        assert output["features"].shape == (size, 8)
    assert identities == (modules[0].prepare, modules[1].capture_canonical_model,
                          modules[1].normalize_to_split_ir, modules[2].split_graph_from_trace)
