import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    """Project-wide LayerNorm backed by PyTorch's AMP-safe implementation."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__(d_model, eps=eps)
