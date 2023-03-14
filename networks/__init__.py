from networks.hmsanet import HMSANet
from networks.resnet import ResNet
from networks.segnet import SegNet
from networks.unet import UNet


__all__ = [HMSANet, SegNet, UNet]


def get_model(architecture, in_channels, out_channels, pretrained):
    if architecture == "HMSANet":
        return HMSANet(in_channels, out_channels, pretrained=pretrained)
    elif architecture == "ResNet":
        return ResNet(in_channels, out_channels, pretrained=pretrained)
    elif architecture == "SegNet":
        return SegNet(in_channels, out_channels, pretrained=pretrained)
    elif architecture == "UNet":
        return UNet(in_channels, out_channels)
    raise ValueError(f"unknown network architecture \"{architecture}\"")
