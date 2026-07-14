import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
import torch.nn.functional as F
import copy

class ConvBatchAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        bias=False,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

    def fuse(self):
        self.eval()

        self.conv = fuse_conv_bn_eval(
            self.conv,
            self.bn,
        )

        self.bn = nn.Identity()
        return self

class RepVGG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_3x3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn_3x3 = nn.BatchNorm2d(out_channels)

        self.conv_1x1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn_1x1 = nn.BatchNorm2d(out_channels)

        self.act = nn.SiLU()
        self.reparam_conv = None

    def forward(self, x):
        if self.reparam_conv is not None:
            return self.act(self.reparam_conv(x))

        x_3x3 = self.bn_3x3(self.conv_3x3(x))
        x_1x1 = self.bn_1x1(self.conv_1x1(x))

        return self.act(x_3x3 + x_1x1)

    def fuse(self):
        self.eval()

        fused_3x3 = fuse_conv_bn_eval(
            self.conv_3x3,
            self.bn_3x3,
        )
        fused_1x1 = fuse_conv_bn_eval(
            self.conv_1x1,
            self.bn_1x1,
        )

        # [out_channels, in_channels, 1, 1]
        # becomes [out_channels, in_channels, 3, 3].
        kernel_1x1_padded = F.pad(
            fused_1x1.weight,
            [1, 1, 1, 1],
        )

        equivalent_kernel = (
            fused_3x3.weight + kernel_1x1_padded
        )
        equivalent_bias = (
            fused_3x3.bias + fused_1x1.bias
        )

        self.reparam_conv = nn.Conv2d(
            in_channels=self.conv_3x3.in_channels,
            out_channels=self.conv_3x3.out_channels,
            kernel_size=3,
            stride=self.conv_3x3.stride,
            padding=self.conv_3x3.padding,
            dilation=self.conv_3x3.dilation,
            groups=self.conv_3x3.groups,
            bias=True,
        ).to(
            device=equivalent_kernel.device,
            dtype=equivalent_kernel.dtype,
        )

        with torch.no_grad():
            self.reparam_conv.weight.copy_(equivalent_kernel)
            self.reparam_conv.bias.copy_(equivalent_bias)

        # Remove training-only branches.
        del self.conv_3x3
        del self.bn_3x3
        del self.conv_1x1
        del self.bn_1x1

        return self


class SimpleELAN(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, hidden_channels=None):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = in_channels // 2

        self.in_proj = ConvBatchAct(in_channels, 2*hidden_channels, kernel_size=1)
        self.num_blocks = num_blocks

        repvgg = RepVGG(hidden_channels, hidden_channels)
        self.repvgg_blocks = nn.ModuleList(
            [copy.deepcopy(repvgg) for _ in range(num_blocks)]
        )

        self.out_proj = ConvBatchAct(hidden_channels*(num_blocks+2), out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.in_proj(x)

        up, down = torch.chunk(x, 2, dim=1)
        out = [up, down]
        for block in self.repvgg_blocks:
            out.append(block(out[-1]))
        
        x = torch.cat(out, dim=1)
        return self.out_proj(x)



