"""Prediction failures must not be hidden by argmax, sigmoid or box clipping."""
from types import SimpleNamespace
import pytest
import torch
from torch.utils.data import TensorDataset

from experiments.common.workload_training import _evaluate
from experiments.rfdetr_nano_physical import evaluate as evaluate_detection
from splitfleet.tasks import ImageClassificationTask


@pytest.mark.parametrize("task_name", ["image_classification", "text_classification", "semantic_segmentation"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_predictions_fail_before_native_metric_conversion(task_name, bad_value):
    calls = []
    task = SimpleNamespace(name=task_name, make_adapter=ImageClassificationTask,
                           evaluate=lambda *args: calls.append(args) or {"accuracy": 1.})
    workload = SimpleNamespace(task=task, test_dataset=TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long)),
                               collate_fn=None)
    class BadModel(torch.nn.Module):
        def forward(self, inputs):
            logits = torch.full((len(inputs), 2), bad_value)
            return {"out": logits} if task_name == "semantic_segmentation" else logits
    with pytest.raises(FloatingPointError, match="predictions"):
        _evaluate(workload, BadModel(), batch_size=2, device=torch.device("cpu"))
    assert not calls


@pytest.mark.parametrize("field", ["pred_logits", "pred_boxes"])
def test_nonfinite_detector_output_fails_before_sigmoid_and_clipping(field):
    calls = []
    workload = SimpleNamespace(task=SimpleNamespace(evaluate=lambda *a: calls.append(a) or {"map50": 1.}),
        test_dataset=[(torch.ones(3, 2, 2), {"boxes": torch.zeros(1, 4)})],
        collate_fn=lambda items: (torch.stack([x[0] for x in items]), [x[1] for x in items]))
    class BadDetector(torch.nn.Module):
        def forward(self, images):
            result = {"pred_logits": torch.zeros(1, 1, 91), "pred_boxes": torch.full((1, 1, 4), .5)}
            result[field].fill_(float("inf"))
            return result
    with pytest.raises(FloatingPointError, match="predictions"):
        evaluate_detection(workload, BadDetector(), device=torch.device("cpu"))
    assert not calls
