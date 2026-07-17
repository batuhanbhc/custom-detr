import torch
import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    """Project-wide LayerNorm backed by PyTorch's AMP-safe implementation."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__(d_model, eps=eps)


class BatchNorm2d(nn.BatchNorm2d):
    """Project-wide BatchNorm2d pinned to fp32 under autocast.

    Autocast doesn't force batch_norm to fp32 the way it does layer_norm, and
    while cuDNN's kernel is safe with fp16 input in practice, this makes the
    running-stats reduction explicitly fp32 regardless of backend/version."""

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float()).type_as(x)
