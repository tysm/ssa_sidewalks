import torch.nn as nn
from torchvision import models

from networks.utils import DoubleConv2dBnReluConv1x1, scale_as


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, pretrained=True):
        super().__init__()

        # Backbone
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(
            nn.Sequential(
                resnet.conv1 if in_channels == 3 else nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            ),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        backbone_out_channels = 512

        # Semantic Head
        self.head = DoubleConv2dBnReluConv1x1(backbone_out_channels, 256, out_channels)

    def forward(self, x):
        return self.head(scale_as(self.backbone(x), x))
