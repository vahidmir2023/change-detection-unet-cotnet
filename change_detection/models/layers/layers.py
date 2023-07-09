import torch
import torch.nn as nn
from torch import Tensor
from .initializers import he_normalizer
from models.cotnet import CotLayer
from cupy_layers.aggregation_zeropad import LocalConvolution
from models.layers import get_act_layer


__all__ = [
    "Conv2dWithHeNorm",
    "CustomConv2d",
    "ConvBlock",
    "UpsamplingBlock",
    "BackBone",
]


class ConvBlock(nn.Module):
  def __init__(self, in_channels, cotlayer_channels, cotlayer_kernel_size):
    super(ConvBlock, self).__init__()

    self.net = nn.Sequential(
            nn.Conv2d(in_channels, cotlayer_channels, 1, bias=False),
            CotLayer(cotlayer_channels, cotlayer_kernel_size),
            nn.BatchNorm2d(cotlayer_channels),
            nn.ReLU(inplace=True),
    )

    # self.net[0].weight = self.net[0].apply(he_normalizer)

    self.maxpool = nn.MaxPool2d(2, 2)

  def forward(self, x):
    """
    accepts a 2D-Tensor and returns two 2D-Tensors
    """
    out = self.net(x)
    return out, self.maxpool(out)


class Conv2dWithHeNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding="same", padding_mode="zeros", bias=False, use_bn=False, use_activation=True):
        super(Conv2dWithHeNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding, padding_mode=padding_mode)
        self.conv.apply(he_normalizer)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.99, affine=False) if use_bn else None
        self.act_func = nn.ReLU(inplace=True) if use_activation else None

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
            nn.ReLU(inplace=True),
            # kernel equal with stride in size is the padding 'same'
            nn.MaxPool2d(maxpool_kernel_size, maxpool_stride),
        )
        self.conv[0] = self.conv[0].apply(he_normalizer)

        if dropout != 0.0:
            self.conv.add_module("dropout", nn.Dropout(0.5))

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input)


class BackBone(nn.Module):
    def __init__(self, in_channels, dropout_val = 0.5):
        super(BackBone, self).__init__()

        cotlayer_kernel_size = 5

        self.nets = nn.ModuleDict({
            "net64": ConvBlock(in_channels, 64, cotlayer_kernel_size),
            "net128": ConvBlock(64, 128, cotlayer_kernel_size),
            "net256": ConvBlock(128, 256, cotlayer_kernel_size),
            "net512": ConvBlock(256, 512, cotlayer_kernel_size),
        })

        self.dropout = nn.Dropout(dropout_val)

    def forward(self, x):
        out64, maxpool64 = self.nets["net64"](x)
        out128, maxpool128 = self.nets["net128"](maxpool64)
        out256, maxpool256 = self.nets["net256"](maxpool128)
        _, maxpool512 = self.nets["net512"](maxpool256)
        maxpool512 = self.dropout(maxpool512)
        return [out64, out128, out256, maxpool512]


class UpsamplingBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        translation_in_channels, 
        translation_out_channels, 
        merge_upsample=True
    ):
        super(UpsamplingBlock, self).__init__()

        self.merge_upsample = merge_upsample

        self.conv_bn2 = nn.Sequential(
            nn.Upsample((2,2)),
            nn.Conv2d(in_channels, out_channels, 2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if merge_upsample:
            self.conv_bn3 = nn.Sequential(
                nn.Conv2d(translation_in_channels*2, translation_out_channels, 3, padding=1),
                nn.BatchNorm2d(translation_out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_bn3 = nn.Sequential(
                nn.Conv2d(translation_in_channels, translation_out_channels, 3, padding=1),
                nn.BatchNorm2d(translation_out_channels),
                nn.ReLU(inplace=True)
            )

        self.conv_bn4 = nn.Sequential(
            nn.Conv2d(translation_in_channels, translation_out_channels, 3, padding=1),
            nn.BatchNorm2d(translation_out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, left_cot_x = None, right_cot_x = None):
        """
        receives a merged or not 4D-Tensor
        """
        out = self.conv_bn2(x)
        print("how ever the entry shape is --> ", x.shape)
        print("out shape from upsampling block --> ", out.shape)
        if self.merge_upsample:
            out = torch.cat([
                out,
                torch.cat([left_cot_x, right_cot_x], axis=1)], 
                1,
            )
        
        return self.conv_bn4(self.conv_bn3(out))
