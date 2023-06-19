import torch
import torch.nn as nn
from torch import Tensor
from .initializers import he_normalizer


__all__ = [
    "Conv2dWithHeNorm",
    "CustomConv2d"
]

class Conv2dWithHeNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding="same", padding_mode="zeros", bias=False, use_bn=False, use_activation=True):
        super(Conv2dWithHeNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding, padding_mode=padding_mode)
        self.conv.apply(he_normalizer)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.99, affine=False) if use_bn else None
        self.act_func = nn.ReLU() if use_activation else None

    def forward(self, input: Tensor) -> Tensor:
        output = self.conv(input)
        if self.bn:
            output = self.bn(output)
        
        if self.act_func:
            output = self.act_func(output)

        return output


class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, maxpool_kernel_size, maxpool_stride, padding="same", padding_mode="zeros", bias: bool = False, dropout: float = 0.0):
        super(CustomConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding="same", bias=bias),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.99),
            nn.ReLU(),
            # kernel equal with stride in size is the padding 'same'
            nn.MaxPool2d(maxpool_kernel_size, maxpool_stride),
        )
        self.conv[0] = self.conv[0].apply(he_normalizer)

        if dropout != 0.0:
            self.conv.add_module("dropout", nn.Dropout(0.5))

    def forward(self, input: Tensor) -> Tensor:
        return self.con(input)


class SpatialAttentionLeft(nn.Module):
    pass

class SpatialAttentionRight(nn.Module):
    pass