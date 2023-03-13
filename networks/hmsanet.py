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

        resnet = models.resnet50(pretrained=pretrained)
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
        backbone_out_channels = 2048

        self.attention_head = nn.Sequential(
            DoubleConv2dBnReluConv1x1(backbone_out_channels, 256, 1),
            nn.Sigmoid()
        )

        self.head = DoubleConv2dBnReluConv1x1(backbone_out_channels, 256, out_channels)

    def _forward(self, x):
        logits = self.backbone(x)
        attention_logits = self.attention_head(logits)

        logits = scale_as(logits, x)
        attention_logits = scale_as(attention_logits, x)
        return logits, attention_logits

    def _training_forward(self, x):
        x_lo, x_hi = resize(x, self._training_low_scale_factor), x

        logits_lo, attention_logits = self._forward(x_lo)
        logits_hi, _ = self._forward(x_hi)

        logits_lo = attention_logits * logits_lo

        logits_lo = scale_as(logits_lo, logits_hi)
        attention_logits = scale_as(attention_logits, logits_hi)
        return logits_lo + (1 - attention_logits) * logits_hi

    def _evaluation_forward(self, x):
        assert 1.0 in self._evaluation_scale_factors
        scale_factors = sorted(self._evaluation_scale_factors, reverse=True)

        logits = None
        for scale_factor in scale_factors:
            scaled_logits, scaled_attention_logits = self._forward(resize(x, scale_factor))

            if logits is None:
                logits = scaled_logits
            elif scale_factor >= 1.0:
                scaled_logits = scaled_attention_logits * scaled_logits

                logits = scale_as(logits, scaled_logits)
                logits = scaled_logits + (1 - scaled_attention_logits) * logits
            else:
                scaled_logits = scaled_attention_logits * scaled_logits

                scaled_logits = scale_as(scaled_logits, logits)
                scaled_attention_logits = scale_as(scaled_attention_logits, logits)
                logits = scaled_logits + (1 - scaled_attention_logits) * logits
        return logits

    def forward(self, x):
        return self._training_forward(x) if self.training else self._evaluation_forward(x)
