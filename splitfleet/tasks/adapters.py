"""Input and objective adapters for common vision and language tasks."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from splitfleet.tasks.base import ModelInputs, TaskBatch, infer_num_examples
from splitfleet.tasks.losses import extract_logits, sparse_cross_entropy, tensor_backend


class ImageClassificationTask:
    """Supervised batches count examples from the leading target dimension."""

    def __init__(self, *, loss_fn: Callable | None = None, input_key: str = "images",
                 target_key: str = "labels", output_key: str = "logits") -> None:
        self.loss_fn, self.input_key, self.target_key, self.output_key = loss_fn, input_key, target_key, output_key

    def prepare_batch(self, batch: Any, *, training: bool = True) -> TaskBatch:
        if isinstance(batch, TaskBatch):
            return batch
        if isinstance(batch, Mapping):
            inputs, targets = batch[self.input_key], batch[self.target_key]
        else:
            inputs, targets = batch
        return TaskBatch(ModelInputs.from_value(inputs), targets, num_examples=infer_num_examples(targets))

    def __call__(self, batch: Any) -> TaskBatch:
        return self.prepare_batch(batch)

    def loss(self, outputs: Any, targets: Any = None) -> Any:
        logits = extract_logits(outputs, self.output_key)
        return self.loss_fn(logits, targets) if self.loss_fn is not None else sparse_cross_entropy(logits, targets)


class TextClassificationTask(ImageClassificationTask):
    """Keep tokenizer keyword arguments and remove labels from the model call."""

    def prepare_batch(self, batch: Any, *, training: bool = True) -> TaskBatch:
        if isinstance(batch, TaskBatch):
            return batch
        if isinstance(batch, Mapping):
            targets = batch[self.target_key]
            kwargs = {key: value for key, value in batch.items() if key != self.target_key}
            return TaskBatch(ModelInputs(kwargs=kwargs), targets, num_examples=infer_num_examples(targets))
        inputs, targets = batch
        call = ModelInputs(kwargs=inputs) if isinstance(inputs, Mapping) else ModelInputs.from_value(inputs)
        return TaskBatch(call, targets, num_examples=infer_num_examples(targets))


class SemanticSegmentationTask(ImageClassificationTask):
    """Pixelwise sparse CE; torchvision's auxiliary output is weighted explicitly."""

    def __init__(self, *, loss_fn: Callable | None = None, input_key: str = "images",
                 target_key: str = "masks", output_key: str = "out",
                 ignore_index: int = 255, aux_weight: float = 0.4) -> None:
        super().__init__(loss_fn=loss_fn, input_key=input_key, target_key=target_key, output_key=output_key)
        self.ignore_index, self.aux_weight = ignore_index, aux_weight

    def loss(self, outputs: Any, targets: Any = None) -> Any:
        logits = extract_logits(outputs, self.output_key)
        if self.loss_fn is not None:
            return self.loss_fn(logits, targets)
        loss = sparse_cross_entropy(logits, targets, ignore_index=self.ignore_index)
        if isinstance(outputs, Mapping) and outputs.get("aux") is not None and self.aux_weight:
            loss = loss + self.aux_weight * sparse_cross_entropy(outputs["aux"], targets, ignore_index=self.ignore_index)
        return loss


class DetectionTask:
    """Use detector-native training losses or a caller-supplied detection criterion.

    Native detectors usually emit loss dictionaries in training and predictions
    in evaluation. Evaluating predictions requires ``evaluation_loss_fn`` (or
    ``loss_fn`` for a detector with an external criterion). No surrogate loss is
    inferred from boxes or scores. Train and eval graphs must be traced separately.
    """

    def __init__(self, *, loss_fn: Callable | None = None,
                 evaluation_loss_fn: Callable | None = None, model_loss: bool = True,
                 loss_weights: Mapping[str, float] | None = None,
                 input_key: str = "images", target_key: str = "targets") -> None:
        self.loss_fn, self.evaluation_loss_fn, self.model_loss = loss_fn, evaluation_loss_fn, model_loss
        self.loss_weights = dict(loss_weights or {})
        self.input_key, self.target_key = input_key, target_key

    def prepare_batch(self, batch: Any, *, training: bool = True) -> TaskBatch:
        if isinstance(batch, TaskBatch):
            return batch
        if isinstance(batch, Mapping):
            images, targets = batch[self.input_key], batch[self.target_key]
        else:
            images, targets = batch
        if not isinstance(targets, (list, tuple)) or not all(isinstance(target, Mapping) for target in targets):
            raise ValueError("Detection targets must be a list/tuple of per-image dictionaries.")
        targets = list(targets)
        count = len(images) if isinstance(images, (list, tuple)) else infer_num_examples(images)
        if len(targets) != count:
            raise ValueError("Detection needs exactly one target dictionary per image.")
        for target in targets:
            self._validate_target(target)
        # Keep the image list as a single positional argument. Targets may also
        # be required by eval traces of native detector APIs (e.g. torchvision).
        if isinstance(images, tuple):
            images = list(images)
        args = (images, targets) if self.model_loss else (images,)
        return TaskBatch(ModelInputs(args=args), targets, num_examples=count)

    def _validate_target(self, target: Mapping[str, Any]) -> None:
        if "boxes" not in target or "labels" not in target:
            raise ValueError("Each detection target requires boxes [N, 4] and labels [N].")
        boxes, labels = target["boxes"], target["labels"]
        if len(boxes.shape) != 2 or int(boxes.shape[-1]) != 4 or len(labels.shape) != 1:
            raise ValueError("Detection boxes must have shape [N, 4] and labels shape [N].")
        if int(boxes.shape[0]) != int(labels.shape[0]):
            raise ValueError("Detection box and label counts differ.")

    def __call__(self, batch: Any) -> TaskBatch:
        return self.prepare_batch(batch)

    def loss(self, outputs: Any, targets: Any = None) -> Any:
        if self.loss_fn is not None:
            return self.loss_fn(outputs, targets)
        if isinstance(outputs, Mapping) and outputs and all(str(key).startswith("loss") for key in outputs):
            losses = []
            unknown = set(self.loss_weights) - set(outputs)
            if unknown:
                raise ValueError(f"Unknown detector loss weights: {sorted(unknown)}")
            for key, value in outputs.items():
                backend = tensor_backend(value)
                if tuple(value.shape) not in ((), (1,)):
                    raise ValueError(f"Native detector loss {key!r} must be scalar.")
                if backend == "tf":
                    from tensorflow import reshape
                    value = reshape(value, ())
                else:
                    value = value.reshape(())
                losses.append(value * self.loss_weights.get(key, 1.0))
            total = losses[0]
            for item in losses[1:]:
                total = total + item
            return total
        if self.evaluation_loss_fn is not None:
            return self.evaluation_loss_fn(outputs, targets)
        raise ValueError("Detection predictions require an explicit evaluation_loss_fn; native loss dictionaries are only available in training mode.")


class InstanceSegmentationTask(DetectionTask):
    """Native detector losses including the model's mask loss."""

    def _validate_target(self, target: Mapping[str, Any]) -> None:
        super()._validate_target(target)
        if "masks" not in target:
            raise ValueError("Instance segmentation targets also require masks [N, H, W].")
        masks = target["masks"]
        if len(masks.shape) != 3 or int(masks.shape[0]) != int(target["boxes"].shape[0]):
            raise ValueError("Instance masks must have shape [N, H, W] with one mask per box.")
