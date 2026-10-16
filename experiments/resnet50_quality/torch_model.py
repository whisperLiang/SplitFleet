"""Full native Torch ResNet50 with the portable initial state."""

import numpy as np
import torch

from .model_contract import NAMES, validate_state


def build_model(state, device="cpu"):
    from torchvision.models import resnet50

    validate_state(state)
    model = resnet50(weights=None, num_classes=10)
    model.load_state_dict({name: torch.from_numpy(np.asarray(value).copy()) for name, value in state.items()})
    return model.to(device)


def export_state(model):
    state = {name: model.state_dict()[name].detach().cpu().numpy().copy() for name in NAMES}
    validate_state(state)
    return state
