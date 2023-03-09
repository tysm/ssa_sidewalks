import torch
import torch.nn as nn


class Conv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, eps=1e-05, inplace=True):
        super().__init__()
        self._forward = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels, eps=eps),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x):
        return self._forward(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._intermediate_features = [64, 128, 256, 512]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # UNet Encoder
        in_features = in_channels
        for out_features in self._intermediate_features:
            self.encoder.append(nn.Sequential(
                Conv2dBnRelu(in_features, out_features),
                Conv2dBnRelu(out_features, out_features)
            ))
            self.encoder.append(nn.MaxPool2d(2, stride=2))
            in_features = out_features

        # UNet Decoder
        for out_features in reversed(self._intermediate_features):
            self.decoder.append(nn.Sequential(
                Conv2dBnRelu(in_features, 2*out_features),
                Conv2dBnRelu(2*out_features, 2*out_features),
                nn.ConvTranspose2d(2*out_features, out_features, 2, stride=2)
            ))
            in_features = 2*out_features

        # Semantic Head
        self.semantic_head = nn.Sequential(
            Conv2dBnRelu(in_features, in_features//2),
            Conv2dBnRelu(in_features//2, in_features//2),
            nn.Conv2d(in_features//2, out_channels, 1)
        )

    def forward(self, x):
        encoder_activations = []
        for i in range(0, len(self.encoder), 2):
            x = self.encoder[i](x)
            encoder_activations.append(x)
            x = self.encoder[i+1](x)

        for i, block in enumerate(self.decoder):
            x = block(x)

            encoder_activation = encoder_activations[-1-i]
            if x.shape != encoder_activation.shape:
                x = x.resize_(*encoder_activation.shape[2:])
            x = torch.cat((encoder_activation, x), dim=1)

        return self.semantic_head(x)
