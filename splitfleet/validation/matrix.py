"""Compare split execution against native training on small, actual task losses."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np


TASKS = ("image_classification", "text_classification", "detection", "segmentation", "instance_segmentation")
TASK_BACKENDS = ("torch", "jax")


@dataclass
class ValidationCase:
    backend: str
    task_name: str
    model: Any
    inputs: tuple[Any, ...]
    targets: Any
    task: Any
    batch_size: int = 2


def build_case(task_name: str, *, backend: str = "torch", seed: int = 2026) -> ValidationCase:
    """Build deterministic synthetic fixtures without fetching datasets or weights."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task {task_name!r}; choose from {TASKS}")
    if backend == "torch":
        return _torch_case(task_name, seed)
    if backend == "jax":
        return _jax_case(task_name, seed)
    raise NotImplementedError(
        f"The task-equivalence fixtures currently support torch and jax, not {backend!r}. "
        "Other backend replay/training checks are in tests/integration/test_torchlens_optional_frameworks.py."
    )


def _torch_case(task_name: str, seed: int) -> ValidationCase:
    import torch
    from torch import nn
    from torch.nn import functional as F

    from splitfleet.tasks import (
        DetectionTask, ImageClassificationTask, InstanceSegmentationTask, SemanticSegmentationTask, TextClassificationTask,
    )

    torch.manual_seed(seed)

    class ImageClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.classifier = nn.Linear(4, 3)

        def forward(self, images):
            features = F.relu(self.conv(images))
            return self.classifier(features.mean(dim=(2, 3)))

    class TextClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 4)
            self.hidden = nn.Linear(4, 4)
            self.classifier = nn.Linear(4, 3)

        def forward(self, input_ids, attention_mask):
            mask = attention_mask.unsqueeze(-1).float()
            embedded = self.embedding(input_ids) * mask
            pooled = embedded.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return {"logits": self.classifier(F.relu(self.hidden(pooled)) + pooled)}

    class Detector(nn.Module):
        """One supervised object per image, with separate class and box heads."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.classes = nn.Linear(4, 3)
            self.boxes = nn.Linear(4, 4)
            if task_name == "instance_segmentation":
                self.masks = nn.Conv2d(4, 1, 1)

        def forward(self, images):
            features = F.relu(self.conv(images))
            pooled = features.mean(dim=(2, 3))
            output = {"class_logits": self.classes(pooled), "boxes": self.boxes(pooled).sigmoid()}
            if task_name == "instance_segmentation":
                output["mask_logits"] = self.masks(features)
            return output

    class Segmenter(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.classifier = nn.Conv2d(4, 3, 1)

        def forward(self, images):
            features = F.relu(self.conv(images))
            return {"out": self.classifier(features)}

    images = torch.randn(2, 3, 8, 8)
    labels = torch.tensor([0, 2])
    if task_name == "image_classification":
        model, inputs, targets, task = ImageClassifier(), (images,), labels, ImageClassificationTask()
    elif task_name == "text_classification":
        model = TextClassifier()
        inputs = (torch.randint(0, 16, (2, 5)), torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]))
        targets, task = labels, TextClassificationTask()
    elif task_name in ("detection", "instance_segmentation"):
        def criterion(output, target):
            loss = F.cross_entropy(output["class_logits"], torch.cat([item["labels"] for item in target]))
            loss = loss + F.smooth_l1_loss(output["boxes"], torch.cat([item["boxes"] for item in target]))
            if task_name == "instance_segmentation":
                loss = loss + F.binary_cross_entropy_with_logits(
                    output["mask_logits"], torch.stack([item["masks"] for item in target]),
                )
            return loss

        model, inputs = Detector(), (images,)
        boxes = torch.tensor([[0.2, 0.3, 0.5, 0.6], [0.1, 0.2, 0.7, 0.8]])
        targets = [{"labels": labels[i:i+1], "boxes": boxes[i:i+1]} for i in range(2)]
        if task_name == "instance_segmentation":
            for target in targets:
                target["masks"] = torch.randint(0, 2, (1, 8, 8)).float()
        task_type = InstanceSegmentationTask if task_name == "instance_segmentation" else DetectionTask
        task = task_type(loss_fn=criterion, model_loss=False)
        task.prepare_batch((images, targets))
    else:
        model, inputs = Segmenter(), (images,)
        targets, task = torch.randint(0, 3, (2, 8, 8)), SemanticSegmentationTask()
    return ValidationCase("torch", task_name, model, inputs, targets, task)


def _jax_case(task_name: str, seed: int) -> ValidationCase:
    import jax
    import jax.numpy as jnp

    from splitfleet.tasks import (
        DetectionTask, ImageClassificationTask, InstanceSegmentationTask, SemanticSegmentationTask, TextClassificationTask,
    )

    rng = np.random.default_rng(seed)
    params = {
        "stem": jnp.asarray(rng.normal(0, 0.2, (3, 4)), dtype=jnp.float32),
        "classifier": jnp.asarray(rng.normal(0, 0.2, (4, 3)), dtype=jnp.float32),
    }
    images = jnp.asarray(rng.normal(size=(2, 8, 8, 3)), dtype=jnp.float32)
    labels = jnp.asarray([0, 2], dtype=jnp.int32)
    if task_name == "text_classification":
        params["embedding"] = jnp.asarray(rng.normal(size=(16, 3)), dtype=jnp.float32)

        def model(parameters, tokens, mask):
            embedded = parameters["embedding"][tokens]
            pooled = (embedded * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
            return {"logits": jax.nn.relu(pooled @ parameters["stem"]) @ parameters["classifier"]}

        inputs = (params, jnp.asarray(rng.integers(0, 16, (2, 5))), jnp.ones((2, 5), dtype=jnp.float32))
        task, targets = TextClassificationTask(), labels
    elif task_name in ("detection", "instance_segmentation"):
        if task_name == "instance_segmentation":
            params["masks"] = jnp.asarray(rng.normal(0, 0.2, (4, 1)), dtype=jnp.float32)
        params["boxes"] = jnp.asarray(rng.normal(0, 0.2, (4, 4)), dtype=jnp.float32)

        def model(parameters, image):
            features = jax.nn.relu(image @ parameters["stem"])
            pooled = features.mean(axis=(1, 2))
            output = {"class_logits": pooled @ parameters["classifier"], "boxes": jax.nn.sigmoid(pooled @ parameters["boxes"])}
            if task_name == "instance_segmentation":
                output["mask_logits"] = jnp.transpose(features @ parameters["masks"], (0, 3, 1, 2))
            return output

        def criterion(output, target):
            log_probs = jax.nn.log_softmax(output["class_logits"], axis=-1)
            labels = jnp.concatenate([item["labels"] for item in target])
            classification = -jnp.take_along_axis(log_probs, labels[:, None], axis=-1).mean()
            delta = jnp.abs(output["boxes"] - jnp.concatenate([item["boxes"] for item in target]))
            loss = classification + jnp.where(delta < 1, 0.5 * delta ** 2, delta - 0.5).mean()
            if task_name == "instance_segmentation":
                logits = output["mask_logits"]
                masks = jnp.stack([item["masks"] for item in target])
                loss = loss + (jax.nn.softplus(logits) - logits * masks).mean()
            return loss

        inputs = (params, images)
        boxes = jnp.asarray([[0.2, 0.3, 0.5, 0.6], [0.1, 0.2, 0.7, 0.8]])
        targets = [{"labels": labels[i:i+1], "boxes": boxes[i:i+1]} for i in range(2)]
        if task_name == "instance_segmentation":
            for target in targets:
                target["masks"] = jnp.asarray(rng.integers(0, 2, (1, 8, 8)), dtype=jnp.float32)
        task_type = InstanceSegmentationTask if task_name == "instance_segmentation" else DetectionTask
        task = task_type(loss_fn=criterion, model_loss=False)
        task.prepare_batch((images, targets))
    elif task_name == "segmentation":
        def model(parameters, image):
            logits = jax.nn.relu(image @ parameters["stem"]) @ parameters["classifier"]
            return {"out": jnp.transpose(logits, (0, 3, 1, 2))}

        inputs = (params, images)
        targets = jnp.asarray(rng.integers(0, 3, (2, 8, 8)), dtype=jnp.int32)
        task = SemanticSegmentationTask()
    else:
        def model(parameters, image):
            return jax.nn.relu(image.mean(axis=(1, 2)) @ parameters["stem"]) @ parameters["classifier"]

        inputs, targets, task = (params, images), labels, ImageClassificationTask()
    return ValidationCase("jax", task_name, model, inputs, targets, task)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _compare(expected: Any, actual: Any, *, rtol: float, atol: float) -> float:
    """Check complete nested structure and values; return maximum absolute error."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError("Output dictionary keys differ")
        return max((_compare(expected[key], actual[key], rtol=rtol, atol=atol) for key in expected), default=0.0)
    if isinstance(expected, (tuple, list)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise AssertionError("Output sequence structure differs")
        return max((_compare(a, b, rtol=rtol, atol=atol) for a, b in zip(expected, actual, strict=True)), default=0.0)
    if expected is None or actual is None:
        if expected is not actual:
            raise AssertionError("One execution omitted a parameter gradient")
        return 0.0
    left, right = _numpy(expected), _numpy(actual)
    if left.shape != right.shape:
        raise AssertionError(f"Shape mismatch: {left.shape} != {right.shape}")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise AssertionError("Native or split execution produced non-finite values")
    np.testing.assert_allclose(left, right, rtol=rtol, atol=atol)
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _boundaries(backend, all_nodes: bool, max_nodes: int):
    report = backend.split_points()
    sites = list(report.candidates)
    if not all_nodes and len(sites) > max_nodes:
        sites = [sites[index] for index in np.linspace(0, len(sites) - 1, max_nodes, dtype=int)]
    return report, sites


def _any_changed(expected, actual) -> bool:
    if isinstance(expected, dict):
        return any(_any_changed(expected[key], actual[key]) for key in expected)
    if isinstance(expected, (tuple, list)):
        return any(_any_changed(left, right) for left, right in zip(expected, actual, strict=True))
    return not np.array_equal(_numpy(expected), _numpy(actual))


def validate_case(
    case: ValidationCase, *, all_nodes: bool = False, max_nodes: int = 4,
    learning_rate: float = 0.01, rtol: float = 2e-4, atol: float = 2e-6,
) -> dict[str, Any]:
    """Check wire replay, loss, all parameter gradients and one SGD update.

    Every requested node receives its own outcome. Backend capability refusals
    never count as passed training checks. Other exceptions are failures.
    """
    from splitfleet.autosplit import prepare_torchlens_runtime

    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    report: dict[str, Any] = {"backend": case.backend, "task": case.task_name, "scope": "synthetic correctness", "nodes": []}
    try:
        native = _NativeTraining(case, learning_rate)
        initial = native.snapshot()
        handle = prepare_torchlens_runtime(
            case.model, case.inputs, boundary="50%", trainable=True,
            dynamic_batch=(case.batch_size, case.batch_size), model_name=f"validation-{case.task_name}",
            batch_axes={"/args/1": 0, **({"/args/2": 0} if case.task_name == "text_classification" else {})} if case.backend == "jax" else None,
        )
        native.restore(initial)
        expected_output, expected_loss, expected_gradients, expected_state = native.full_step()
        boundary_report, sites = _boundaries(handle.backend, all_nodes, max_nodes)
        report["available_boundaries"] = boundary_report.total
        report["selection"] = "all" if all_nodes else "sampled"
        for site in sites:
            boundary_name = site.point.as_boundary()
            node: dict[str, Any] = {"boundary": boundary_name, "capabilities": site.as_dict()}
            report["nodes"].append(node)
            try:
                if not site.replay_supported:
                    node.update(status="unsupported", reason=site.unsupported_reason or "backend refused replay")
                    continue
                native.restore(initial)
                runtime = handle.backend.repartition(boundary_name)
                payload = runtime.backend.run_prefix(*case.inputs)
                remote, _ = _wire_boundary(runtime, payload)
                actual_output = runtime.backend.run_suffix(remote)
                node["output_max_abs_error"] = _compare(expected_output, actual_output, rtol=rtol, atol=atol)
                if not site.training_supported:
                    node.update(status="unsupported", reason=site.unsupported_reason or "backend refused training", output_verified=True)
                    continue
                native.restore(initial)
                loss, gradients, state = native.split_step(runtime)
                node["loss_max_abs_error"] = _compare(expected_loss, loss, rtol=rtol, atol=atol)
                node["gradient_max_abs_error"] = _compare(expected_gradients, gradients, rtol=rtol, atol=atol)
                node["parameter_max_abs_error"] = _compare(expected_state, state, rtol=rtol, atol=atol)
                if not _any_changed(initial, state):
                    raise AssertionError("Training changed no parameter")
                node["status"] = "passed"
            except AssertionError as exc:
                # A changed state is checked separately; comparisons above must
                # never be downgraded to unsupported capabilities.
                node.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            except Exception as exc:
                node.update(status=_exception_status(exc), reason=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        report.update(status=_exception_status(exc), reason=f"{type(exc).__name__}: {exc}")
    finally:
        if "initial" in locals():
            native.restore(initial)
    counts = {status: sum(node["status"] == status for node in report["nodes"]) for status in ("passed", "failed", "unsupported")}
    report["counts"] = counts
    if "status" not in report:
        report["status"] = "failed" if counts["failed"] else "partial" if counts["unsupported"] else "passed"
        if not counts["passed"]:
            report["status"] = "unsupported" if not counts["failed"] else "failed"
    return report


def _exception_status(exc: Exception) -> str:
    from torchlens.split.errors import SplitUnsupportedError

    return "unsupported" if isinstance(exc, SplitUnsupportedError) else "failed"


def _wire_boundary(runtime, payload):
    from splitfleet.split_engine import graph_contract_for_runtime_handle
    from splitfleet.transport import encode_boundary, decode_boundary
    from splitfleet.transport.split_wire import boundary_to_envelope, envelope_to_boundary

    contract = graph_contract_for_runtime_handle(runtime)
    wire = boundary_to_envelope(
        payload, round_id=1, client_id="validation", step_id="1", plan_id=runtime.plan.plan_id,
        split_id=contract.split_id, canonical_graph_hash=contract.canonical_graph_hash,
        boundary_schema_hash=contract.boundary_schema_hash, model_version=1,
    )
    remote = envelope_to_boundary(decode_boundary(encode_boundary(wire)), runtime.runtime, None)
    return remote, wire


class _NativeTraining:
    def __init__(self, case: ValidationCase, learning_rate: float):
        self.case = case
        self.lr = learning_rate
        if case.backend == "torch":
            self.case.model.train()
        elif case.backend != "jax":
            raise NotImplementedError(f"Native gradient comparator for {case.backend!r} is unavailable")

    def snapshot(self):
        if self.case.backend == "torch":
            return {name: tensor.detach().clone() for name, tensor in self.case.model.state_dict().items()}
        return copy.deepcopy(self.case.inputs[0])

    def restore(self, state):
        if self.case.backend == "torch":
            self.case.model.load_state_dict(state)
            self.case.model.zero_grad(set_to_none=True)
        else:
            self.case.inputs = (copy.deepcopy(state), *self.case.inputs[1:])

    def full_step(self):
        case = self.case
        if case.backend == "torch":
            import torch

            output = case.model(*case.inputs)
            loss = case.task.loss(output, case.targets)
            loss.backward()
            gradients = {name: None if param.grad is None else param.grad.detach().clone() for name, param in case.model.named_parameters()}
            torch.optim.SGD(case.model.parameters(), lr=self.lr).step()
            return output, loss.detach().clone(), gradients, self.snapshot()
        import jax

        output = case.model(*case.inputs)
        loss, gradients = jax.value_and_grad(lambda p: case.task.loss(case.model(p, *case.inputs[1:]), case.targets))(case.inputs[0])
        self.case.inputs = (jax.tree_util.tree_map(lambda p, g: p - self.lr * g, case.inputs[0], gradients), *case.inputs[1:])
        return output, loss, gradients, self.snapshot()

    def split_step(self, runtime):
        from splitfleet.transport import encode_gradients, decode_gradients
        from splitfleet.transport.split_wire import (
            gradients_to_envelope, envelope_to_gradients,
        )

        case = self.case
        local = runtime.backend.run_prefix(*case.inputs, training=True)
        remote, wire = _wire_boundary(runtime, local)
        loss, boundary_grads = runtime.backend.train_suffix(remote, case.targets, loss_fn=case.task.loss)
        gradient_wire = encode_gradients(gradients_to_envelope(wire, boundary_grads))
        received_grads = envelope_to_gradients(decode_gradients(gradient_wire), None)
        prefix_result = runtime.backend.backward_prefix(local, received_grads)
        if case.backend == "torch":
            import torch

            gradients = {name: None if param.grad is None else param.grad.detach().clone() for name, param in case.model.named_parameters()}
            torch.optim.SGD(case.model.parameters(), lr=self.lr).step()
        else:
            import jax

            if not prefix_result or "inputs" not in prefix_result:
                raise AssertionError("The trainable boundary returned no input parameter gradients")
            gradients = prefix_result["inputs"][0]
            case.inputs = (jax.tree_util.tree_map(lambda p, g: p - self.lr * g, case.inputs[0], gradients), *case.inputs[1:])
        return loss, gradients, self.snapshot()
