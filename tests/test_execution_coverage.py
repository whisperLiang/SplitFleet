"""Evidence checks must catch state errors and exercise independent split replicas."""
import json

import pytest
import torch
from torch import nn

from experiments.analysis.execution_coverage import compare, tail_batch_check, validate
from splitfleet.tasks import ModelInputs


def test_comparison_retains_schema_and_missing_gradient_failures():
    assert not compare({"p": None}, {"p": torch.zeros(2)}, rtol=1e-4, atol=1e-6)["passed"]
    assert not compare(torch.tensor([1]), torch.tensor([1.]), rtol=1, atol=1)["passed"]
    assert not compare(torch.tensor([1]), torch.tensor([2]), rtol=1, atol=1)["passed"]
    result = compare({"p": torch.tensor([1., 2.])}, {"p": torch.tensor([1., 3.])}, rtol=1e-4, atol=1e-6)
    assert not result["passed"] and result["max_abs"] == 1 and result["reason"] == "value.p"
    assert not compare(torch.tensor([float("nan")]), torch.tensor([float("nan")]), rtol=1, atol=1)["passed"]


class ResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first, self.branch, self.head = nn.Linear(4, 8), nn.Linear(8, 8), nn.Linear(8, 3)
        self.offset = nn.Parameter(torch.randn(1, 4))
        self.register_buffer("scale", torch.tensor(0.5), persistent=False)

    def forward(self, inputs):
        hidden = self.first(inputs + self.offset.expand_as(inputs)).relu()
        return self.head(hidden + self.branch(hidden).relu() * self.scale)


class MutableBufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first, self.head = nn.Linear(4, 8), nn.Linear(8, 3)
        self.register_buffer("calls", torch.zeros((), dtype=torch.int64), persistent=False)

    def forward(self, inputs):
        self.calls.add_(1)
        return self.head(self.first(inputs).relu())


def test_mutable_nonpersistent_buffers_reset_between_cuts_and_rpc(tmp_path):
    torch.manual_seed(791)
    model = MutableBufferModel()
    batches = [(ModelInputs((torch.randn(1, 4),)), torch.tensor([i % 3])) for i in range(3)]
    result = validate(model, batches, nn.functional.cross_entropy, device="cpu", steps=3,
                      output=tmp_path / "mutable_evidence.json")
    assert all(c["passed"] for c in result["native_repeat"].values())
    assert result["counts"].get("passed", 0) > 1
    assert result["counts"].get("failed", 0) == 0
    assert result["rpc"]["status"] == "passed"
    assert result["trajectories"]
    assert all(len(row["steps"]) == 3 and row["status"] == "passed"
               for row in result["trajectories"])


@pytest.mark.parametrize("model_type", [ResidualModel, MutableBufferModel])
def test_tail_batch_capture_restores_nonpersistent_buffers(model_type, tmp_path):
    torch.manual_seed(791)
    model = model_type()
    batches = [(ModelInputs((torch.randn(1, 4),)), torch.tensor([i % 3])) for i in range(3)]
    catalog = validate(model, batches, nn.functional.cross_entropy, device="cpu", steps=3,
                       output=tmp_path / "catalog.json", numeric=False)
    boundary = catalog["pressure_boundaries"][0]
    result = tail_batch_check(model, batches, nn.functional.cross_entropy,
                             boundary, "cpu")
    assert result["status"] == "passed"
    assert not result["recaptured"]
    assert [step["batch_size"] for step in result["steps"]] == [2, 1]


def test_all_cuts_multistep_nonpersistent_buffer_and_production_rpc(tmp_path):
    torch.manual_seed(791)
    model = ResidualModel()
    batches = [(ModelInputs((torch.randn(1, 4),)), torch.tensor([i % 3])) for i in range(3)]
    result = validate(model, batches, nn.functional.cross_entropy, device="cpu", steps=3,
                      output=tmp_path / "evidence.json")
    assert result["counts"].get("failed", 0) == 0
    assert result["counts"].get("admitted_unchecked", 0) == 0
    assert result["counts"]["rejected"] > 0
    assert max(row["frontier_count"] or 0 for row in result["cuts"]) > 1
    assert all(len(row["steps"]) == 3 and row["status"] == "passed" for row in result["trajectories"])
    assert result["rpc"]["status"] == "passed"
    assert all(c["passed"] for c in result["native_repeat"].values())
    boundary = result["pressure_boundaries"][len(result["pressure_boundaries"]) // 2]
    tail = tail_batch_check(model, batches, nn.functional.cross_entropy, boundary, "cpu")
    assert tail["status"] == "passed" and not tail["recaptured"]
    assert [step["batch_size"] for step in tail["steps"]] == [2, 1]
    assert json.loads((tmp_path / "evidence.json").read_text())["counts"] == result["counts"]
