from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

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

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        s3 = self.layer2(x)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)

        return s3, s4, s5