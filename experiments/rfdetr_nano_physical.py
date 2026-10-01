"""Small, real RF-DETR Nano VOC07 workload for the physical six-device runner."""

from __future__ import annotations

import copy
import torch
from torch import nn
from torch.utils.data import DataLoader

from splitfleet.tasks import DetectionTask
from splitfleet.server.placement.cosplit_ucb.candidate_provider import TorchLensCandidateProvider


RESOLUTION = 384
NUM_CLASSES = 90
NUM_QUERIES = 50
GROUP_DETR = 1
# Pascal VOC 2007 class order mapped to the COCO category ids retained by
# RF-DETR's pretrained 91-logit detection head (90 classes plus no-object).
VOC_TO_COCO = (5, 2, 16, 9, 44, 6, 3, 17, 62, 21,
               67, 18, 19, 4, 1, 64, 20, 63, 7, 72)


def _config(device: str = "cpu"):
    from rfdetr.config import RFDETRNanoConfig

    return RFDETRNanoConfig(
        pretrain_weights=None, resolution=RESOLUTION, num_classes=NUM_CLASSES,
        group_detr=GROUP_DETR, num_queries=NUM_QUERIES, num_select=NUM_QUERIES,
        device=device,
    )


class RFDETRNanoDetector(nn.Module):
    """Expose RF-DETR's original detector as a tensor-input split model."""

    def __init__(self, *, pretrain_weights: str | None = None) -> None:
        super().__init__()
        from rfdetr import RFDETRNano

        options = _config().model_dump()
        options["pretrain_weights"] = pretrain_weights
        self.model = RFDETRNano(**options).model.model

    def forward(self, images: torch.Tensor):
        from rfdetr.utilities.tensors import NestedTensor

        mask = torch.zeros(images.shape[:1] + images.shape[-2:],
                           dtype=torch.bool, device=images.device)
        return self.model(NestedTensor(images, mask))


class RFDETRDetectionTask(DetectionTask):
    """Use RF-DETR's Hungarian matching and weighted detection losses."""

    def __init__(self) -> None:
        super().__init__(model_loss=False)
        self._criterion = None
        self._criterion_device = None

    def prepare_batch(self, batch, *, training: bool = True):
        prepared = super().prepare_batch(batch, training=training)
        targets = []
        for target in prepared.targets:
            xyxy = target["boxes"]
            cxcywh = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2,
                                xyxy[:, 2:] - xyxy[:, :2]), dim=-1)
            mapping = torch.as_tensor(VOC_TO_COCO, dtype=torch.long,
                                      device=target["labels"].device)
            targets.append({"boxes": cxcywh,
                            "labels": mapping[target["labels"]]})
        return type(prepared)(prepared.inputs, targets, prepared.num_examples)

    def loss(self, outputs, targets=None):
        from rfdetr._namespace import build_namespace
        from rfdetr.config import TrainConfig
        from rfdetr.models.lwdetr import build_criterion_and_postprocessors

        device = outputs["pred_logits"].device
        if self._criterion is None or self._criterion_device != device:
            config = _config(str(device))
            train = TrainConfig(dataset_dir=".", output_dir=".", batch_size=1)
            self._criterion, _ = build_criterion_and_postprocessors(
                build_namespace(config, train)
            )
            self._criterion_device = device
        losses = self._criterion(outputs, targets)
        return sum(value * self._criterion.weight_dict[key]
                   for key, value in losses.items() if key in self._criterion.weight_dict)


def decode_predictions(outputs) -> list[dict[str, torch.Tensor]]:
    """Convert RF-DETR query outputs to the existing VOC mAP50 record format."""

    probabilities = outputs["pred_logits"][..., list(VOC_TO_COCO)].sigmoid()
    scores, labels = probabilities.max(dim=-1)
    cxcywh = outputs["pred_boxes"]
    xyxy = torch.cat((cxcywh[..., :2] - cxcywh[..., 2:] / 2,
                      cxcywh[..., :2] + cxcywh[..., 2:] / 2), dim=-1).clamp(0, 1)
    return [dict(boxes=xyxy[index].detach().cpu(),
                 labels=labels[index].detach().cpu(),
                 scores=scores[index].detach().cpu())
            for index in range(xyxy.shape[0])]


def evaluate(workload, model: nn.Module, *, device: torch.device) -> dict[str, float]:
    model.eval()
    loader = DataLoader(workload.test_dataset, batch_size=1, shuffle=False,
                        collate_fn=workload.collate_fn)
    targets, predictions = [], []
    with torch.inference_mode():
        for images, records in loader:
            outputs = model(images.to(device))
            targets.extend(records)
            predictions.extend(decode_predictions(outputs))
    return workload.task.evaluate(targets, predictions)


class RFDETRCandidateProvider(TorchLensCandidateProvider):
    """Expose every valid RF-DETR cut from an independent training capture."""

    def __init__(self, *, model: nn.Module, sample_inputs,
                 batch_axes: dict[str, int] | None = None,
                 dynamic_batch: tuple[int, int] = (1, 1)) -> None:
        # Global evaluation must not change the modes of the captured model.
        super().__init__(
            model=copy.deepcopy(model),
            sample_inputs=sample_inputs,
            batch_axes={"/args/0": 0} if batch_axes is None else dict(batch_axes),
            dynamic_batch=dynamic_batch,
            kinds=("before", "after"),
            require_trainable_prefix=True,
        )

__all__ = ["RFDETRNanoDetector", "RFDETRDetectionTask",
           "RFDETRCandidateProvider",
           "decode_predictions", "evaluate"]
