# Adapted from RF-DETR (Copyright 2025 Roboflow), LW-DETR (Copyright 2024
# Baidu), Conditional DETR (Copyright 2021 Microsoft), and DETR (Copyright
# Facebook, Inc. and its affiliates). Apache License, Version 2.0; see
# dependency_patches/RFDETR_APACHE_LICENSE.txt and RFDETR_SAMPLER_NOTICE.md.
"""RF-DETR arithmetic expressed with deterministic PyTorch operations.

The CUDA grid_sample backward is nondeterministic. This implements the same
zero-padded, align_corners=False interpolation used by deformable attention;
PyTorch's strict deterministic mode supplies deterministic gather backward.
Inputs, outputs, and gradients stay on their original device. The arithmetic
follows RF-DETR's gather-based MPS sampler (Apache-2.0), without device dispatch.
"""

from __future__ import annotations

import torch


def bilinear_grid_sample(
    input: torch.Tensor,
    grid: torch.Tensor,
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """Sample NCHW feature maps at NHW2 coordinates, with zero padding."""
    if padding_mode != "zeros" or align_corners:
        raise ValueError("RF-DETR sampling requires zeros padding and align_corners=False")
    if input.ndim != 4 or grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError("RF-DETR sampling requires NCHW input and NHW2 grid")
    if input.shape[0] != grid.shape[0] or input.device != grid.device:
        raise ValueError("RF-DETR input and grid must share batch size and device")

    batch, channels, height, width = input.shape
    grid_height, grid_width = grid.shape[1:3]
    x = (grid[..., 0] + 1) * width / 2 - 0.5
    y = (grid[..., 1] + 1) * height / 2 - 0.5
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = x0 + 1, y0 + 1
    wx = (x - x0.to(x.dtype)).to(input.dtype).unsqueeze(1)
    wy = (y - y0.to(y.dtype)).to(input.dtype).unsqueeze(1)
    flat = input.flatten(2)

    def gather(iy: torch.Tensor, ix: torch.Tensor) -> torch.Tensor:
        valid = ((iy >= 0) & (iy < height) & (ix >= 0) & (ix < width))
        indices = (iy.clamp(0, height - 1) * width + ix.clamp(0, width - 1))
        indices = indices.flatten(1).unsqueeze(1).expand(batch, channels, -1)
        values = flat.gather(2, indices).view(batch, channels, grid_height, grid_width)
        return values * valid.unsqueeze(1)

    v00, v10 = gather(y0, x0), gather(y0, x1)
    v01, v11 = gather(y1, x0), gather(y1, x1)
    return (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10 \
        + (1 - wx) * wy * v01 + wx * wy * v11


def integer_position_embedding(self, tensor_list, align_dim_orders=True):
    """Retain the sine encoding, counting the boolean mask with integer sums.

    Float CUDA cumsum is refused by strict mode in NVIDIA PyTorch 2.5. Mask
    counts within the RF-DETR image dimensions are exactly representable in
    FP32, so integer accumulation followed by conversion gives the same values.
    """
    features, mask = tensor_list.tensors, tensor_list.mask
    if mask is None or mask.dtype != torch.bool:
        raise ValueError("RF-DETR position encoding requires a boolean mask")
    if max(mask.shape[1:]) >= 2**24:
        raise ValueError("Position mask counts must be exactly representable in FP32")
    valid = ~mask
    y = valid.cumsum(1, dtype=torch.int64).to(torch.float32)
    x = valid.cumsum(2, dtype=torch.int64).to(torch.float32)
    if self.normalize:
        y = y / (y[:, -1:, :] + 1e-6) * self.scale
        x = x / (x[:, :, -1:] + 1e-6) * self.scale
    frequencies = torch.arange(self.num_pos_feats, dtype=torch.float32, device=features.device)
    frequencies = self.temperature ** (2 * (frequencies // 2) / self.num_pos_feats)
    px, py = x[:, :, :, None] / frequencies, y[:, :, :, None] / frequencies
    px = torch.stack((px[:, :, :, 0::2].sin(), px[:, :, :, 1::2].cos()), dim=4).flatten(3)
    py = torch.stack((py[:, :, :, 0::2].sin(), py[:, :, :, 1::2].cos()), dim=4).flatten(3)
    positions = torch.cat((py, px), dim=3)
    return positions.permute(1, 2, 0, 3) if align_dim_orders else positions.permute(0, 3, 1, 2)


def install_rfdetr_sampler() -> None:
    """Use identical deterministic arithmetic on the coordinator and workers."""
    import rfdetr.models.ops.functions.ms_deform_attn_func as deform
    from rfdetr.models.position_encoding import PositionEmbeddingSine

    deform._bilinear_grid_sample = bilinear_grid_sample
    # Keep the original method available for independent numerical audits. The
    # module class, attributes, state schema and positional formula are retained.
    if not hasattr(PositionEmbeddingSine, "_native_forward_for_math_audit"):
        PositionEmbeddingSine._native_forward_for_math_audit = PositionEmbeddingSine.forward
    PositionEmbeddingSine.forward = integer_position_embedding
