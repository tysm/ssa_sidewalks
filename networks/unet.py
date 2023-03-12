import torch
import torch.nn as nn
from networks.utils import Conv2dBnRelu


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        in_features = in_channels
        encoder_intermediate_features = [64, 128, 256, 512]

        # Encoder
        self.encoder = nn.ModuleList()
        for out_features in encoder_intermediate_features:
            self.encoder.append(nn.Sequential(
                Conv2dBnRelu(in_features, out_features),
                Conv2dBnRelu(out_features, out_features)
            ))
            in_features = out_features
        self.pool = nn.MaxPool2d(2, stride=2)

        # Decoder
        self.decoder = nn.ModuleList()
        for out_features in reversed(encoder_intermediate_features):
            self.decoder.append(nn.Sequential(
                Conv2dBnRelu(in_features, 2*out_features),
                Conv2dBnRelu(2*out_features, 2*out_features),
                nn.ConvTranspose2d(2*out_features, out_features, 2, stride=2)
            ))
            in_features = 2*out_features

        # Head
        self.head = nn.Sequential(
            Conv2dBnRelu(in_features, in_features//2),
            Conv2dBnRelu(in_features//2, in_features//2),
            nn.Conv2d(in_features//2, out_channels, 1)
        )

    def forward(self, x):
        encoder_activations = []
        for block in self.encoder:
            x = block(x)
            encoder_activations.append(x)
            x = self.pool(x)

        for i, block in enumerate(self.decoder):
            x = block(x)

            encoder_activation = encoder_activations[-1-i]
            if x.shape != encoder_activation.shape:
                x = x.resize_(*encoder_activation.shape[2:])  # FIXME can't resize
            x = torch.cat((encoder_activation, x), dim=1)

        return self.head(x)
