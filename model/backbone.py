from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FrozenBatchNorm2d
import torch.nn as nn


def freeze_batchnorm(module):
    """Replace every BatchNorm2d with a frozen copy (fixed running stats/affine)."""
    if isinstance(module, nn.BatchNorm2d):
        frozen = FrozenBatchNorm2d(module.num_features, eps=module.eps)
        frozen.weight.data.copy_(module.weight.data)
        frozen.bias.data.copy_(module.bias.data)
        frozen.running_mean.data.copy_(module.running_mean.data)
        frozen.running_var.data.copy_(module.running_var.data)
        return frozen
    for name, child in module.named_children():
        setattr(module, name, freeze_batchnorm(child))
    return module


class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        freeze_batchnorm(self)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        s3 = self.layer2(x)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)

        return s3, s4, s5