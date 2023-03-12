from networks.segnet import SegNet
from networks.unet import UNet


__all__ = [SegNet, UNet]


def get_model(architecture, in_channels, out_channels):
    if architecture == "UNet":
        return UNet(in_channels, out_channels)
    elif architecture == "SegNet":
        return SegNet(in_channels, out_channels)
    raise ValueError(f"unknown network architecture \"{architecture}\"")
