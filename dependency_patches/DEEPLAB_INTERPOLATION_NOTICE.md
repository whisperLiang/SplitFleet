# DeepLab interpolation used by the physical study

The frozen pretrained DeepLabV3 ResNet50 retains all torchvision modules,
parameter names, buffers, pretrained weights and auxiliary classifier. Only
bilinear resizing is expressed with separable index selection, using half-pixel
coordinates, border replication and `align_corners=False`. All CPU/GPU workers
and all six methods use this implementation under strict deterministic mode.
The primary training batches, samples, seeds and rounds remain unchanged.

The forwarding methods in `experiments/deeplab_interpolation.py` are adapted
from the installed torchvision `_SimpleSegmentationModel.forward` and
`ASPPPooling.forward`. Copyright (c) Soumith Chintala 2016, All rights reserved.
The full BSD 3-Clause license is retained in `TORCHVISION_BSD_LICENSE.txt`.
Installed torchvision files are not modified.

`tests/test_deeplab_interpolation.py` compares the independent native CPU
bilinear kernel's output and input gradients for up/down sampling, degenerate
axes and unchanged sizes in FP32/FP64. The real-device admission additionally
compares native and split full/partial batches, Adam updates, gradients, RNG,
wire payloads and graph contracts before primary training.

Spatial sparse cross entropy is reduced over equivalent flattened pixel rows
to avoid the nondeterministic CUDA spatial NLL kernel. Independent native loss
and gradient comparisons are in `tests/test_sparse_ce_reduction.py`.
