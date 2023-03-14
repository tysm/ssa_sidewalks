import torch.nn as nn
from torchvision import models

from networks.utils import DoubleConv2dBnReluConv1x1, resize, scale_as


class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()


class HMSANet(nn.Module):
    def __init__(self, in_channels, out_channels, pretrained=True):
        super().__init__()

        self._training_low_scale_factor = 0.5
        self._evaluation_scale_factors = [0.25, 0.5, 1.0, 2.0]

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

        # Attention Head
        self.attention_head = nn.Sequential(
            DoubleConv2dBnReluConv1x1(backbone_out_channels, 256, 1),
            nn.Sigmoid()
        )

        # Semantic Head
        self.head = DoubleConv2dBnReluConv1x1(backbone_out_channels, 256, out_channels)

    def _forward(self, x):
        features = self.backbone(x)
        attention = self.attention_head(features)

        features = scale_as(features, x)
        attention = scale_as(attention, x)
        return self.head(features), attention

    def _training_forward(self, x):
        x_lo, x_hi = resize(x, self._training_low_scale_factor), x

        logits_lo, attention = self._forward(x_lo)
        logits_hi, _ = self._forward(x_hi)

        logits_lo = attention * logits_lo

        logits_lo = scale_as(logits_lo, logits_hi)
        attention = scale_as(attention, logits_hi)
        return logits_lo + (1 - attention) * logits_hi

    def _evaluation_forward(self, x):
        assert 1.0 in self._evaluation_scale_factors
        scale_factors = sorted(self._evaluation_scale_factors, reverse=True)

        logits = None
        for scale_factor in scale_factors:
            scaled_logits, scaled_attention = self._forward(resize(x, scale_factor))

            if logits is None:
                logits = scaled_logits
            elif scale_factor >= 1.0:
                scaled_logits = scaled_attention * scaled_logits

                logits = scale_as(logits, scaled_logits)
                logits = scaled_logits + (1 - scaled_attention) * logits
            else:
                scaled_logits = scaled_attention * scaled_logits

                scaled_logits = scale_as(scaled_logits, logits)
                scaled_attention = scale_as(scaled_attention, logits)
                logits = scaled_logits + (1 - scaled_attention) * logits
        return logits

    def forward(self, x):
        return self._training_forward(x) if self.training else self._evaluation_forward(x)
