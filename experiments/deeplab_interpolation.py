# Torchvision forward methods adapted under the BSD 3-Clause license:
# Copyright (c) Soumith Chintala 2016, All rights reserved.
# See dependency_patches/TORCHVISION_BSD_LICENSE.txt.
"""Device-independent bilinear resizing for the frozen DeepLab study.

CUDA's fused bilinear backward is nondeterministic, and PyTorch's strict-mode
decomposition differs from its CPU graph. Explicit index selection gives every
method and device the same align_corners=False math and a replayable graph.
"""

from __future__ import annotations

from collections import OrderedDict

import torch


def bilinear_resize(input: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize NCHW tensors with half-pixel coordinates and border replication."""
    if input.ndim != 4 or not input.is_floating_point():
        raise ValueError("DeepLab interpolation requires a floating NCHW tensor")
    if len(size) != 2 or min(size) <= 0:
        raise ValueError("DeepLab interpolation requires two positive output sizes")

    result = input
    # Width first, then height, matching the separable native bilinear formula.
    for axis, output_size in ((3, size[1]), (2, size[0])):
        input_size = result.shape[axis]
        coordinates = torch.arange(output_size, device=input.device).to(input.dtype)
        coordinates = ((coordinates + 0.5) * (input_size / output_size) - 0.5).clamp(min=0)
        lower = coordinates.to(torch.int64)
        upper = (lower + 1).clamp(max=input_size - 1)
        shape = [1, 1, 1, 1]
        shape[axis] = output_size
        weight = (coordinates - lower.to(input.dtype)).clamp(0, 1).reshape(shape)
        first = result.index_select(axis, lower)
        second = result.index_select(axis, upper)
        result = first + (second - first) * weight
    return result


def _pooling_forward(self, x: torch.Tensor) -> torch.Tensor:
    size = x.shape[-2:]
    for module in self:
        x = module(x)
    return bilinear_resize(x, size)


def _segmentation_forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    input_shape = x.shape[-2:]
    features = self.backbone(x)
    result = OrderedDict()
    result["out"] = bilinear_resize(self.classifier(features["out"]), input_shape)
    if self.aux_classifier is not None:
        result["aux"] = bilinear_resize(self.aux_classifier(features["aux"]), input_shape)
    return result


def install_deeplab_interpolation() -> None:
    """Retain all torchvision modules and weights, replacing only resize math."""
    from torchvision.models.segmentation.deeplabv3 import ASPPPooling, DeepLabV3

    for module, forward in ((ASPPPooling, _pooling_forward), (DeepLabV3, _segmentation_forward)):
        if not hasattr(module, "_native_forward_for_math_audit"):
            module._native_forward_for_math_audit = module.forward
        module.forward = forward
