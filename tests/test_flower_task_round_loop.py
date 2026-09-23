import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from splitfleet.client.autosplit_client import AutoSplitNumPyClient
from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.common import ServerModelEvaluateIns, ServerModelFitIns
from splitfleet.common.constants import AUTOSPLIT_PLAN_ID_CONFIG_KEY
from splitfleet.server.server_model.autosplit_server_model import AutoSplitServerModel
from splitfleet.server.stage_runtime.manager import StageRuntimeManager
from splitfleet.server.strategy import AutoSplitStrategy
from splitfleet.tasks import DetectionTask, SemanticSegmentationTask, TextClassificationTask
from tests.test_flower_autosplit_round_loop import InProcessServerModelProxy


class KeywordClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 6)
        self.classifier = nn.Linear(6, 3)

    def forward(self, *, input_ids, attention_mask):
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        return {"logits": self.classifier(hidden.sum(1) / attention_mask.sum(1, keepdim=True))}


class PixelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 5, 1)
        self.classifier = nn.Conv2d(5, 3, 1)

    def forward(self, images):
        return {"out": self.classifier(torch.relu(self.encoder(images)))}


class TimeMajorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 5)
        self.classifier = nn.Linear(5, 3)

    def forward(self, tokens):
        return self.classifier(torch.relu(self.hidden(tokens)).mean(0))


class NativeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 6)
        self.classifier = nn.Linear(6, 3)
        self.box_head = nn.Linear(6, 4)

    def forward(self, images, targets):
        features = torch.relu(self.encoder(torch.stack([image.mean((1, 2)) for image in images])))
        logits, boxes = self.classifier(features), self.box_head(features)
        if self.training:
            labels = torch.cat([target["labels"] for target in targets])
            expected_boxes = torch.cat([target["boxes"] for target in targets])
            return {"loss_classifier": F.cross_entropy(logits, labels),
                    "loss_box_reg": F.smooth_l1_loss(boxes, expected_boxes)}
        return {"logits": logits, "boxes": boxes}


def detection_eval_loss(outputs, targets):
    return F.cross_entropy(outputs["logits"], torch.cat([target["labels"] for target in targets])) + F.smooth_l1_loss(
        outputs["boxes"], torch.cat([target["boxes"] for target in targets]))


def task_case(name):
    if name == "time_major_text":
        return TimeMajorClassifier(), TextClassificationTask(), (
            torch.randn(7, 2, 4), torch.tensor([0, 2])), "after:hidden", {"/args/0": 1}
    if name == "text":
        return KeywordClassifier(), TextClassificationTask(), {
            "input_ids": torch.tensor([[1, 2, 0], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]), "labels": torch.tensor([0, 2]),
        }, "after:embedding", None
    if name == "pixels":
        return PixelClassifier(), SemanticSegmentationTask(), (
            torch.randn(2, 3, 4, 4), torch.randint(0, 3, (2, 4, 4))), "after:encoder", None
    targets = [{"boxes": torch.tensor([[0., 0., 1., 1.]]), "labels": torch.tensor([index])}
               for index in (0, 2)]
    return NativeDetector(), DetectionTask(evaluation_loss_fn=detection_eval_loss), (
        [torch.randn(3, 4, 4), torch.randn(3, 5, 5)], targets), "after:encoder", {}


@pytest.mark.parametrize("name", ["text", "time_major_text", "pixels", "detection"])
def test_task_split_flower_train_and_eval_match_real_objectives(name):
    torch.manual_seed(19)
    model, task, raw_batch, boundary, batch_axes = task_case(name)
    batch = task.prepare_batch(raw_batch)
    initial = [value.detach().numpy().copy() for value in model.state_dict().values()]
    expected = task.loss(model(*batch.inputs.args, **batch.inputs.kwargs), batch.targets).item()
    strategy = AutoSplitStrategy(model=model, sample_inputs=batch, task=task,
                                 boundary=boundary, batch_axes=batch_axes)
    manager = StageRuntimeManager(autosplit_session=strategy.autosplit_session)
    strategy.bind_stage_runtime_manager(manager)
    fit_config = strategy._autosplit_config(1, training=True)
    server = strategy._make_server_model()
    server.configure_fit(ServerModelFitIns(parameters=initial, config=fit_config, sid=""))
    client = AutoSplitSplitLearningClient(model=model, sample_inputs=batch,
                                         task=task, train_data=[raw_batch], batch_axes=batch_axes)
    client.server_model_proxy = InProcessServerModelProxy(server_model=server)
    updated, count, metrics = client.fit(initial, fit_config)
    assert count == 2
    assert server.get_fit_result().config["num_examples"] == 2
    assert metrics["loss"] == pytest.approx(expected, rel=1e-5, abs=1e-6)
    assert any(not np.array_equal(a, b) for a, b in zip(initial, updated))

    eval_config = strategy._autosplit_config(1, training=False)
    server.configure_evaluate(ServerModelEvaluateIns(parameters=initial, config=eval_config, sid=""))
    loss, count, _ = client.evaluate(initial, eval_config)
    eval_batch = task.prepare_batch(raw_batch, training=False)
    reference = copy.deepcopy(model).eval()
    expected_eval = task.loss(reference(*eval_batch.inputs.args, **eval_batch.inputs.kwargs), eval_batch.targets).item()
    assert count == 2
    assert loss == pytest.approx(expected_eval, rel=1e-5, abs=1e-6)
    if name == "detection":
        assert fit_config[AUTOSPLIT_PLAN_ID_CONFIG_KEY] != eval_config[AUTOSPLIT_PLAN_ID_CONFIG_KEY]


def test_numpy_task_client_runs_keyword_batch_through_flower_serialization():
    torch.manual_seed(23)
    model, task, raw_batch, boundary, _ = task_case("text")
    batch = task.prepare_batch(raw_batch)
    expected = task.loss(model(**batch.inputs.kwargs), batch.targets).item()
    strategy = AutoSplitStrategy(model=model, sample_inputs=(), sample_kwargs=batch.inputs.kwargs,
                                 task=task, boundary=boundary)
    manager = StageRuntimeManager(autosplit_session=strategy.autosplit_session)
    strategy.bind_stage_runtime_manager(manager)
    config = strategy._autosplit_config()
    server = AutoSplitServerModel(runtime_manager=manager, model=model, task=task)
    initial = server.get_parameters()
    server.configure_fit(initial, config)
    client = AutoSplitNumPyClient(train_data=[raw_batch], task=task)
    client.server_model_proxy = InProcessServerModelProxy(server_model=server.to_server_model())
    _, count, metrics = client.fit([], config)
    assert count == 2
    assert metrics["loss"] == pytest.approx(expected, rel=1e-5, abs=1e-6)
    eval_config = strategy._autosplit_config(training=False)
    server.configure_evaluate(initial, eval_config)
    loss, count, _ = client.evaluate([], eval_config)
    assert count == 2
    assert loss == pytest.approx(expected, rel=1e-5, abs=1e-6)
