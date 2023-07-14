import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.initializers import he_normalizer
from .layers import layers


# class SpatialAttentionBlock(nn.Module):
#     def __init__(self, conv_output_dims, reduction_factor=8):
#         super(SpatialAttentionBlock, self).__init__()
#         self.c1 = conv_output_dims[1] // reduction_factor
#         self.c = conv_output_dims[3]
#         self.d = conv_output_dims[1]
#         self.key_embed = self.conv1x1(d, c1)
#         self.query_embed = self.conv1x1(d, c1)
#         self.val_embed = self.conv1x1(d, d)

#     @staticmethod
#     def conv1x1(in_channels, out_channels):
#         net = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1),
#             nn.ReLU()
#         )
#         net[0].weight = net[0].apply(he_normalizer)

#         return net

#     def forward(self, x):
#         """
#         Accepts a 4-D tensor as input (B, C, H, W)
#         """
#         key_embed = self.key_embed(x)
#         key_embed = key_emmbed.view(-1, self.c1, self.c**2)
#         query_embed = self.query_embed(x)
#         query_embed = query_embed.view(-1, self.c1, self.c**2)
#         val_embed = self.val_embed(x)
#         pass


class ChannelAttentionBlock(nn.Module):
    def forward(self, x):
        c = x.size(3)
        d = x.size(1)

        fch = x.view(-1, c*c, d)
        out = torch.linalg.matmul(fch, fch.transpose(1,2))
        out = torch.nn.functional.softmax(out, dim=-1)
        out = torch.linalg.matmul(fch.transpose(1,2), out)
        out = out.view(-1, d, c, c)
        return out


###### New Network ######
class UnetCotnetNetwork(nn.Module):
    def __init__(self, in_channels, classes):
        super(UnetCotnetNetwork, self).__init__()

        # self.batch_size = batch_size

        self.channel_att = ChannelAttentionBlock()

        # left side blocks
        self.left_backbone = layers.BackBone(in_channels)
        # self.left_spa_att = SpatialAttentionBlock((batch_size, 512, 8, 8))
        # self.left_chn_att = ChannelAttentionBlock()

        # right side blocks
        self.right_backbone = layers.BackBone(in_channels)
        # self.right_spa_att = SpatialAttentionBlock((batch_size, 512, 8, 8))
        # self.right_chn_att = ChannelAttentionBlock()


        # upsampling block
        self.upsamples = nn.ModuleDict({
            "512_512": layers.UpsamplingBlock(512, 512, (512, 512), (512, 512)), 
            "256_256": layers.UpsamplingBlock(512, 256, (256, 256), (256, 256)), 
            "128_128": layers.UpsamplingBlock(256, 128, (128, 128), (128, 128)), 
            "128_64": layers.UpsamplingBlock(128, 128, (128, 64), (64, 64), merge_upsample=False), 
        })

        self.conv_bn = nn.Sequential(
            nn.Conv2d(64, classes, 1),
            nn.BatchNorm2d(classes),
            # nn.Sigmoid(), # since we are using BCEWithLogitsLoss
        )

    def forward(self, left_x, right_x):
        # Left part
        left_bb_outputs = self.left_backbone(left_x)

        # Right part
        right_bb_outputs = self.right_backbone(right_x)

        # Channel part
        dist = (left_bb_outputs[-1] - right_bb_outputs[-1]).pow(2)
        dist = self.channel_att(dist)

        # for i in range(len(left_bb_outputs)):
        #     print(f"idx: {i-4}\tconv{2**(i+6)} --> {left_bb_outputs[i].shape}")

        # for i in range(len(right_bb_outputs)):
        #     print(f"idx: {i-4}\tconv{2**(i+6)} --> {right_bb_outputs[i].shape}")

        # print(f"dist.shape --> {dist.shape}\n====================\n")

        # Upsampling part
        up6_out = self.upsamples["512_512"](dist, left_bb_outputs[-2], right_bb_outputs[-2])
        # print("up6_out shape --> ", up6_out.shape)
        up7_out = self.upsamples["256_256"](up6_out, left_bb_outputs[-3], right_bb_outputs[-3])
        # print("up7_out shape --> ", up7_out.shape)
        up8_out = self.upsamples["128_128"](up7_out, left_bb_outputs[-4], right_bb_outputs[-4])
        # print("up8_out shape --> ", up8_out.shape)
        up9_out = self.upsamples["128_64"](up8_out)

        out = self.conv_bn(up9_out)
        return out
