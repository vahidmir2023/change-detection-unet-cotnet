import torch
import torch.nn as nn
from .layers.initializers import he_normalizer
from .layers.layers import CustomConv2d


class E2EConvBlock(nn.Module):
    def __init__(self):
        super(E2EConvBlock, self).__init__()
        self.seq = nn.Sequential(
            CustomConv2d(3, 64, 5, (1,1), 2, 2),
            CustomConv2d(64, 128, 5, (1,1), 2, 2),
            CustomConv2d(128, 256, 5, (1,1), 2, 2),
            CustomConv2d(256, 512, 5, (1,1), 2, 2, dropout=0.5),
        )

    def forward(self, input):
        return self.seq(input)


class SpatialAttentionBlock(nn.Module):
    def __init__(self, conv_output):
        super(SpatialAttentionBlock, self).__init__()
        self.c1 = conv_output.size(1) // 8
        self.c = conv_output.size(3)
        self.d = conv_output.size(1)
        self.conv1 = self.conv1x1(d, c1)
        self.conv2 = self.conv1x1(d, c1)
        self.conv3 = self.conv1x1(d, d)

    @staticmethod
    def conv1x1(in_channels, out_channels):
        net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding="same"),
            nn.ReLU()
        )
        net[0].weight = net[0].apply(he_normalizer)

        return net

    def forward(self, input):
        #TODO: finish this one
        pass



class LeftBlock(nn.Module):
    def __init__(self):
        super(LeftBlock, self).__init__()
        self.conv = E2EConvBlock()

    def forward(self, input):
        output = self.conv(input)
        return output
        

class RightBlock(nn.Module):
    def __init__(self):
        super(RightBlock, self).__init__()
        self.conv = E2EConvBlock()

    def forward(self, input):
        output = self.conv(input)
        return output
