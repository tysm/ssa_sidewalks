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


class DoubleConv2dBnReluConv1x1(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self._forward = nn.Sequential(
            Conv2dBnRelu(in_channels, mid_channels),
            Conv2dBnRelu(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        )

    def forward(self, x):
        return self._forward(x)


def resize(x, scale_factor):
    return nn.functional.interpolate(x, scale_factor=scale_factor, mode='bilinear')


def scale_as(x, y):
    y_size = y.size(2), y.size(3)
    return nn.functional.interpolate(x, size=y_size, mode='bilinear')
