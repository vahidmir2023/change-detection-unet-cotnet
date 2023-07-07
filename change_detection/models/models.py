import torch
import torch.nn as nn
from .layers.initializers import he_normalizer
from .layers import layers


class SpatialAttentionBlock(nn.Module):
    def __init__(self, conv_output, reduction_factor=8):
        super(SpatialAttentionBlock, self).__init__()
        self.c1 = conv_output.size(1) // reduction_factor
        self.c = conv_output.size(3)
        self.d = conv_output.size(1)
        self.key_embed = self.conv1x1(d, c1)
        self.query_embed = self.conv1x1(d, c1)
        self.val_embed = self.conv1x1(d, d)

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
        fa = self.key_embed(input)
        fb = self.query_embed(input)
        fc = self.val_embed(input)
        fa1 = fa.view()
        pass


class ChannelAttentionBlock(nn.Module):
    def __init__(self, reduction_factor=8):
        self.reduction_factor = reduction_factor

    def forward(self, input):
        c1 = conv_output.size(1) // self.reduction_factor
        c = conv_output.size(3)
        d = conv_output.size(1)
        pass


###### New Network ######
class UnetCotnetNetwork(nn.Module):
    def __init__(self, in_channels, classes):
        super(UnetCotnetNetwork, self).__init__()

        # left side blocks
        self.left_backbone = BackBone(in_channels)

        # right side blocks
        self.right_backbone = BackBone(in_channels)

        # upsampling block
        self.upsamples = nn.ModuleDict({
            "512_512": UpsamplingBlock(512, 512, 512, 512), 
            "256_256": UpsamplingBlock(512, 256, 256, 256), 
            "128_128": UpsamplingBlock(256, 128, 128, 128), 
            "128_64": UpsamplingBlock(128, 128, 64, 64, merge_upsample=False), 
        })

        self.conv_bn = nn.Sequential(
            nn.Conv2d(64, classes, 1),
            nn.Batchnorm2d(classes),
            nn.Sigmoid(),
        )

    def forward(self, left_x, right_x):
        # Left part
        left_bb_outputs = self.left_backbone(left_x)

        # Right part
        right_bb_outputs = self.right_backbone(right_x)

        # apply spatial and channel attentions
        # from both sides 

        # Upsampling part
        return left_bb_outputs, right_bb_outputs
