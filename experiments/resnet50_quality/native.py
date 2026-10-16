"""PyTorch full-model SGD used by the paired convergence experiment."""

from __future__ import annotations

import numpy as np

from . import torch_model

CLIENTS = 50
BACKENDS = {"pytorch": CLIENTS}


def require_pytorch(backend):
    if backend != "pytorch":
        raise ValueError(f"Unknown native backend {backend!r}; this experiment requires pytorch")


def configure_backend(backend="pytorch"):
    require_pytorch(backend)
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def build_model(backend, state, device):
    require_pytorch(backend)
    return torch_model.build_model(state, device)


def export_state(backend, model):
    require_pytorch(backend)
    return torch_model.export_state(model)


def numpy_value(value):
    return np.asarray(value.detach().cpu().numpy()).copy()


def training_mode(backend, model, training=True):
    require_pytorch(backend)
    model.train(training)


def native_batch(backend, images, labels, device):
    require_pytorch(backend)
    import torch
    return torch.as_tensor(images, device=device), torch.as_tensor(labels, device=device)


def call_inputs(backend, model, images):
    require_pytorch(backend)
    return (images,)


def native_parameters(backend, model):
    require_pytorch(backend)
    return dict(model.named_parameters())


def optimizer(backend, model, learning_rate):
    require_pytorch(backend)
    import torch
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0, weight_decay=0)


def train_step(backend, model, images, labels, *, learning_rate=.05, native_optimizer=None, capture=False):
    """One real CE/backpropagation/SGD step; snapshots serve numerical controls."""
    require_pytorch(backend)
    import torch
    model.train()
    opt = native_optimizer if native_optimizer is not None else optimizer(backend, model, learning_rate)
    opt.zero_grad()
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    parameters = dict(model.named_parameters())
    if any(value.grad is None for value in parameters.values()):
        raise RuntimeError("Full ResNet50 training is missing parameter gradients")
    result = {"loss": float(loss.detach()),
              "logits": numpy_value(logits) if capture else None,
              "gradients": {name: numpy_value(value.grad) for name, value in parameters.items()} if capture else None}
    opt.step()
    return result
