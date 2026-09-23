"""Small, traceable reference architectures for four supervised task families."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images).mean(dim=(2, 3)))


class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.hidden = nn.Linear(32, 48)
        self.classifier = nn.Linear(48, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = attention_mask.unsqueeze(-1).to(dtype=self.embedding.weight.dtype)
        embedded = self.embedding(input_ids) * mask
        pooled = embedded.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return {"logits": self.classifier(F.relu(self.hidden(pooled)))}


class GridDetector(nn.Module):
    """A lightweight, multi-object grid detector with 21 VOC classes.

    Each spatial cell predicts one object.  Ground-truth objects sharing a cell
    are resolved deterministically by the criterion; the collision count should
    be reported for a real benchmark and this architecture is not a VOC SOTA
    claim.
    """

    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.head = nn.Conv2d(32, 5 + self.num_classes, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = self.head(self.backbone(images))
        return {
            "objectness": prediction[:, 0],
            "boxes": prediction[:, 1:5],
            "classes": prediction[:, 5:],
        }


def detection_loss(outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    objectness = outputs["objectness"]
    box_raw = outputs["boxes"]
    class_logits = outputs["classes"]
    batch_size, height, width = objectness.shape
    desired_objectness = torch.zeros_like(objectness)
    positive_batch: list[int] = []
    positive_y: list[int] = []
    positive_x: list[int] = []
    positive_boxes: list[torch.Tensor] = []
    positive_labels: list[torch.Tensor] = []
    for batch_index, target in enumerate(targets):
        occupied: set[tuple[int, int]] = set()
        for box, label in zip(target["boxes"], target["labels"], strict=True):
            center_x = float((box[0] + box[2]) / 2)
            center_y = float((box[1] + box[3]) / 2)
            x = min(width - 1, max(0, int(center_x * width)))
            y = min(height - 1, max(0, int(center_y * height)))
            if (x, y) in occupied:
                continue
            occupied.add((x, y))
            desired_objectness[batch_index, y, x] = 1
            positive_batch.append(batch_index)
            positive_y.append(y)
            positive_x.append(x)
            positive_boxes.append(torch.stack((
                (box[0] + box[2]) * width / 2 - x,
                (box[1] + box[3]) * height / 2 - y,
                box[2] - box[0],
                box[3] - box[1],
            )))
            positive_labels.append(label)
    loss = F.binary_cross_entropy_with_logits(objectness, desired_objectness)
    if not positive_batch:
        return loss
    indices = (positive_batch, positive_y, positive_x)
    predicted_box = box_raw.permute(0, 2, 3, 1)[indices].sigmoid()
    box_loss = F.smooth_l1_loss(predicted_box, torch.stack(positive_boxes))
    predicted_classes = class_logits.permute(0, 2, 3, 1)[indices]
    class_loss = F.cross_entropy(predicted_classes, torch.stack(positive_labels).long())
    return loss + 5.0 * box_loss + class_loss


def decode_detections(
    outputs: dict[str, torch.Tensor], *, score_threshold: float = 0.05, top_k: int = 100
) -> list[dict[str, torch.Tensor]]:
    objectness = outputs["objectness"].sigmoid()
    raw_boxes = outputs["boxes"].sigmoid()
    class_scores = outputs["classes"].softmax(dim=1)
    batch_size, height, width = objectness.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=objectness.device),
        torch.arange(width, device=objectness.device),
        indexing="ij",
    )
    center_x = (grid_x.unsqueeze(0) + raw_boxes[:, 0]) / width
    center_y = (grid_y.unsqueeze(0) + raw_boxes[:, 1]) / height
    box_width = raw_boxes[:, 2]
    box_height = raw_boxes[:, 3]
    boxes = torch.stack(
        (
            (center_x - box_width / 2).clamp(0, 1),
            (center_y - box_height / 2).clamp(0, 1),
            (center_x + box_width / 2).clamp(0, 1),
            (center_y + box_height / 2).clamp(0, 1),
        ),
        dim=-1,
    )
    scores, labels = (class_scores * objectness.unsqueeze(1)).max(dim=1)
    records = []
    for index in range(batch_size):
        flat_scores = scores[index].reshape(-1)
        chosen = torch.argsort(flat_scores, descending=True)[:top_k]
        chosen = chosen[flat_scores[chosen] >= score_threshold]
        records.append(
            {
                "boxes": boxes[index].reshape(-1, 4)[chosen].detach().cpu(),
                "labels": labels[index].reshape(-1)[chosen].detach().cpu(),
                "scores": flat_scores[chosen].detach().cpu(),
            }
        )
    return records


class Segmenter(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(),
            nn.Conv2d(24, num_classes, 1),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"out": self.decoder(self.encoder(images))}


__all__ = [
    "ImageClassifier", "TextClassifier", "GridDetector", "Segmenter",
    "detection_loss", "decode_detections",
]
