import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        # Assuming input is [B, seq, d_model]
        batch_size, seq, d_model = x.size()

        var, mean = torch.var_mean(x, dim=-1, keepdim=True, correction=0)

        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.scale + self.bias