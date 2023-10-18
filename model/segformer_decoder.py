# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import List
import warnings
#from mmseg.utils import get_root_logger
#from mmcv.runner import load_checkpoint
import math
from ops import resize

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class ConvModule(nn.Module):

    def __init__(self, channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        #nn.ReLU(), # why relu? Who knows
        self.bn = nn.BatchNorm2d(channels) # why batchnorm and not layer norm? Idk
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(F.relu(x))
        return x

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self,
                 feature_strides,
                 in_channels,
                 channels,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 ignore_index=255,
                 align_corners=False):
        super().__init__()
        self.in_channels = in_channels
        self.in_index = in_index
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        #self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        self.conv_seg = nn.Conv2d(128, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(channels, kernel_size=1)

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        inputs = [inputs[i] for i in self.in_index]


        return inputs

    def forward(self, inputs):
        #x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c1.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        fused_feats = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = self.linear_fuse(fused_feats)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        x = resize(x, size=(h*4, w*4), mode='bilinear', align_corners=False )

        return x
    
if __name__ == "__main__":
    from segformer_encoder import mit_b0, mit_b5
    encoder = mit_b5()
    test_dummy = MLP()
    decoder = SegFormerHead(
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=None,
        align_corners=False,
    )
    
    N, C, H, W = 1, 3, 640, 640
    x = torch.rand([N, C, H, W])
    y = encoder(x)
    y = decoder(y)
    for i, out in enumerate(y):
        print("Output {} shape: {}".format(i, out.shape))
    
    