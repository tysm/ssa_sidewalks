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
