# RF-DETR sampler attribution

`experiments/rfdetr_grid_sampling.py` adapts the gather-based interpolation in
RF-DETR 1.6.5.post2 `rfdetr/utilities/tensors.py::_bilinear_grid_sample`.

The source file carries these notices:

- RF-DETR: Copyright (c) 2025 Roboflow. All Rights Reserved.
- LW-DETR: Copyright (c) 2024 Baidu. All Rights Reserved.
- Conditional DETR: Copyright (c) 2021 Microsoft. All Rights Reserved.
- DETR: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

RF-DETR and LW-DETR identify Apache License, Version 2.0 in that file.
The installed distribution's license is retained in `RFDETR_APACHE_LICENSE.txt`.

The adaptation uses its zero-padded, align_corners=False interpolation on all
requested devices, uses dtype-preserving coordinate arithmetic, limits the
supported argument domain, and binds that implementation to RF-DETR deformable
attention. Strict deterministic PyTorch gather backward replaces the
nondeterministic fused CUDA grid_sample backward in the physical study.

The sine-position adaptation preserves all attributes and normalization, counts
the boolean mask using int64 cumsum and converts those exact counts to FP32.
It follows `rfdetr/models/position_encoding.py::PositionEmbeddingSine.forward`.
