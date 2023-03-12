import torch
import torch.nn as nn
from torchvision import models
from networks.utils import Conv2dBnRelu


class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels, pretrained=True):
        super().__init__()
        vgg_bn = models.vgg16_bn(pretrained=pretrained)
        vgg_bn_encoder = list(vgg_bn.features.children())

        if in_channels != 3:
            vgg_bn_encoder[0] = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False),

        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential(*vgg_bn_encoder[:6]))
        self.encoder.append(nn.Sequential(*vgg_bn_encoder[7:13]))
        self.encoder.append(nn.Sequential(*vgg_bn_encoder[14:23]))
        self.encoder.append(nn.Sequential(*vgg_bn_encoder[24:33]))
        self.encoder.append(nn.Sequential(*vgg_bn_encoder[34:-1]))
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

        # Decoder
        in_features = 512
        decoder_conv_structure = reversed([(2, 64), (2, 64), (3, 128), (3, 256), (3, 512)])
        self.decoder = nn.ModuleList()
        for num_convs, out_features in decoder_conv_structure:
            conv_block = nn.Sequential(Conv2dBnRelu(in_features, out_features))
            for _ in range(1, num_convs):
                conv_block.append(Conv2dBnRelu(out_features, out_features))
            self.decoder.append(conv_block)
            in_features = out_features
        self.unpool = nn.MaxUnpool2d(2, stride=2)

        # Head
        self.head = nn.Conv2d(in_features, out_channels, 1)

    def forward(self, x):
        encoder_activation_sizes = []
        encoder_pooling_indices = []

        for block in self.encoder:
            x = block(x)
            encoder_activation_sizes.append(x.size())
            x, indices = self.pool(x)
            encoder_pooling_indices.append(indices)

        for i, block in enumerate(self.decoder):
            x = self.unpool(x, indices=encoder_pooling_indices[-1-i], output_size=encoder_activation_sizes[-1-i])
            x = block(x)

        return self.head(x)
