import numpy as np
import pytest
import torch
from torch.nn import functional as F

from splitfleet.tasks import (
    DetectionTask, ImageClassificationTask, InstanceSegmentationTask, ModelInputs,
    SemanticSegmentationTask, TaskBatch, TextClassificationTask, prepare_task_batch,
)
from splitfleet.tasks.transport import decode_task_batch, encode_task_batch


def test_classification_uses_cross_entropy_and_preserves_gradients():
    logits = torch.tensor([[2., -1., .5], [-2., 3., 1.]], requires_grad=True)
    labels = torch.tensor([0, 2])
    loss = ImageClassificationTask().loss({"logits": logits}, labels)
    torch.testing.assert_close(loss, F.cross_entropy(logits, labels))
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


@pytest.mark.parametrize("task,target_key,target_shape", [
    (ImageClassificationTask(), "labels", (2,)),
    (SemanticSegmentationTask(), "masks", (2, 4, 5)),
])
@pytest.mark.parametrize("as_mapping", [False, True])
def test_supervised_tasks_count_targets_for_functional_model_inputs(task, target_key, target_shape, as_mapping):
    parameters = {"weight": np.zeros((4, 3), dtype=np.float32)}
    inputs = (parameters, np.zeros((2, 4), dtype=np.float32))
    targets = np.zeros(target_shape, dtype=np.int64)
    raw_batch = {"images": inputs, target_key: targets} if as_mapping else (inputs, targets)

    batch = task.prepare_batch(raw_batch)
    restored = decode_task_batch(encode_task_batch(batch))

    assert batch.num_examples == restored.num_examples == 2
    np.testing.assert_array_equal(restored.inputs.args[0]["weight"], parameters["weight"])


@pytest.mark.parametrize("layout", ["mapping", "tuple", "keyword_tuple"])
def test_text_counts_labels_for_time_major_inputs(layout):
    tokens = torch.zeros(7, 2, dtype=torch.int64)
    labels = torch.tensor([0, 1])
    raw_batch = {
        "mapping": {"input_ids": tokens, "labels": labels},
        "tuple": (tokens, labels),
        "keyword_tuple": ({"input_ids": tokens}, labels),
    }[layout]

    batch = TextClassificationTask().prepare_batch(raw_batch)

    assert batch.num_examples == 2


@pytest.mark.parametrize("task", [ImageClassificationTask(), TextClassificationTask(), SemanticSegmentationTask()])
def test_supervised_tasks_preserve_explicit_batch_count(task):
    batch = TaskBatch(ModelInputs((torch.zeros(7, 2, 4),)), torch.zeros(2), num_examples=3)
    assert task.prepare_batch(batch) is batch


def test_text_keywords_survive_task_transport_and_count_examples():
    batch = TextClassificationTask().prepare_batch({
        "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        "labels": torch.tensor([0, 1]),
    })
    restored = decode_task_batch(encode_task_batch(batch))
    assert restored.num_examples == 2
    assert restored.inputs.args == ()
    assert set(restored.inputs.kwargs) == {"input_ids", "attention_mask"}
    torch.testing.assert_close(restored.inputs.kwargs["input_ids"], batch.inputs.kwargs["input_ids"])
    torch.testing.assert_close(restored.targets, batch.targets)


def test_semantic_segmentation_uses_pixel_ce_ignore_index_and_auxiliary_loss():
    logits = torch.randn(2, 3, 4, 5, requires_grad=True)
    auxiliary = torch.randn_like(logits, requires_grad=True)
    masks = torch.randint(0, 3, (2, 4, 5))
    masks[:, 0, :] = 255
    task = SemanticSegmentationTask(aux_weight=.25)
    batch = task.prepare_batch({"images": torch.randn(2, 3, 4, 5), "masks": masks})
    loss = task.loss({"out": logits, "aux": auxiliary}, batch.targets)
    expected = F.cross_entropy(logits, masks, ignore_index=255) + .25 * F.cross_entropy(auxiliary, masks, ignore_index=255)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :, 0, :]) == 0
    assert auxiliary.grad.abs().sum() > 0


def test_semantic_segmentation_all_ignored_keeps_finite_loss_and_zero_gradients():
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    auxiliary = torch.randn(1, 3, 2, 2, requires_grad=True)
    masks = torch.full((1, 2, 2), 255)

    loss = SemanticSegmentationTask().loss({"out": logits, "aux": auxiliary}, masks)

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
    torch.testing.assert_close(auxiliary.grad, torch.zeros_like(auxiliary))


def test_detection_image_list_counts_images_and_transports_empty_targets():
    images = [torch.randn(3, 4, 5), torch.randn(3, 6, 7)]
    targets = [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.int64)},
               {"boxes": torch.ones(1, 4), "labels": torch.ones(1, dtype=torch.int64)}]
    batch = DetectionTask().prepare_batch((images, targets))
    restored = decode_task_batch(encode_task_batch(batch))
    assert restored.num_examples == 2
    assert len(restored.inputs.args) == 2
    assert isinstance(restored.inputs.args[0], list)
    assert restored.targets[0]["boxes"].shape == (0, 4)
    torch.testing.assert_close(restored.inputs.args[0][1], images[1])


def test_detector_native_losses_are_summed_with_explicit_weights():
    classification = torch.tensor(2., requires_grad=True)
    regression = torch.tensor(.5, requires_grad=True)
    task = DetectionTask(loss_weights={"loss_box_reg": 2.})
    loss = task.loss({"loss_classifier": classification, "loss_box_reg": regression})
    torch.testing.assert_close(loss, torch.tensor(3.))
    loss.backward()
    assert classification.grad.item() == 1.
    assert regression.grad.item() == 2.
    with pytest.raises(ValueError, match="evaluation_loss_fn"):
        task.loss([{"boxes": torch.zeros(1, 4), "scores": torch.ones(1)}])


def test_instance_segmentation_requires_matching_masks():
    task = InstanceSegmentationTask()
    target = {"boxes": torch.zeros(1, 4), "labels": torch.ones(1, dtype=torch.long)}
    with pytest.raises(ValueError, match="masks"):
        task.prepare_batch(([torch.zeros(3, 4, 4)], [target]))
    target["masks"] = torch.zeros(1, 4, 4)
    assert task.prepare_batch(([torch.zeros(3, 4, 4)], [target])).num_examples == 1


def test_explicit_batch_count_is_required_for_ambiguous_calls():
    with pytest.raises(ValueError, match="example count"):
        TaskBatch(ModelInputs(kwargs={"temperature": .5}))
    with pytest.raises(ValueError, match="positive integer"):
        TaskBatch(ModelInputs((torch.zeros(2, 3),)), num_examples=0)
    prepared = TaskBatch(ModelInputs((torch.zeros(2, 3),)), num_examples=2)
    assert prepare_task_batch(prepared) is prepared
    with pytest.raises(ValueError, match="uint8"):
        decode_task_batch(np.zeros(2, dtype=np.float32))
