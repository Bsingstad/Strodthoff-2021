import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fastai.layers import *
from fastai.core import *

from models.basic_conv1d import AdaptiveConcatPool1d,create_head1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fastai.layers import *
from fastai.core import *

from models.basic_conv1d import AdaptiveConcatPool1d, create_head1d

# ConvNeXt-style depthwise convolution
def convnext_block(in_planes, out_planes, kernel_size=7, stride=1):
    return nn.Sequential(
        # Depthwise convolution
        nn.Conv1d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=in_planes, bias=False),
        # Pointwise convolution
        nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False),
        nn.LayerNorm(out_planes),
        nn.GELU()
    )

class ConvNeXtBlock1d(nn.Module):
    def __init__(self, ni, nf, kernel_size=7, stride=1):
        super().__init__()
        self.convnext_block = convnext_block(ni, nf, kernel_size, stride)
        self.downsample = nn.Conv1d(ni, nf, 1) if ni != nf else nn.Identity()
        self.ln = nn.LayerNorm(nf)

    def forward(self, x):
        identity = self.downsample(x)
        x = self.convnext_block(x)
        x += identity
        return x

class InceptionBackboneWithConvNeXt(nn.Module):
    def __init__(self, input_channels, kss, depth, bottleneck_size, nb_filters, use_residual):
        super().__init__()

        self.depth = depth
        assert((depth % 3) == 0)
        self.use_residual = use_residual

        n_ks = len(kss) + 1
        self.im = nn.ModuleList(
            [InceptionBlock1d(input_channels if d == 0 else n_ks * nb_filters,
                              nb_filters=nb_filters, kss=kss, bottleneck_size=bottleneck_size)
             for d in range(depth)]
        )

        self.cnx = nn.ModuleList(
            [ConvNeXtBlock1d(input_channels if d == 0 else n_ks * nb_filters, nb_filters, kernel_size=7)
             for d in range(depth)]
        )

        self.sk = nn.ModuleList([Shortcut1d(input_channels if d == 0 else n_ks * nb_filters, n_ks * nb_filters)
                                 for d in range(depth // 3)])

    def forward(self, x):
        input_res = x
        for d in range(self.depth):
            x = self.im[d](x)
            # Add ConvNeXt Block here for additional feature extraction
            x = self.cnx[d](x)
            if self.use_residual and d % 3 == 2:
                x = self.sk[d // 3](input_res, x)
                input_res = x.clone()
        return x

class Inception1dWithConvNeXt(nn.Module):
    '''inception time architecture with ConvNeXt-style layers'''
    def __init__(self, num_classes=2, input_channels=8, kernel_size=40, depth=6, bottleneck_size=32, nb_filters=32, use_residual=True, lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
        super().__init__()
        assert(kernel_size >= 40)
        kernel_size = [k-1 if k % 2 == 0 else k for k in [kernel_size, kernel_size // 2, kernel_size // 4]]  # was 39,19,9

        layers = [InceptionBackboneWithConvNeXt(input_channels=input_channels, kss=kernel_size, depth=depth, bottleneck_size=bottleneck_size, nb_filters=nb_filters, use_residual=use_residual)]

        n_ks = len(kernel_size) + 1
        # head
        head = create_head1d(n_ks * nb_filters, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        layers.append(head)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def get_layer_groups(self):
        depth = self.layers[0].depth
        if depth > 3:
            return (self.layers[0].im[3:], self.layers[0].sk[1:]), self.layers[-1]
        else:
            return self.layers[-1]

    def get_output_layer(self):
        return self.layers[-1][-1]

    def set_output_layer(self, x):
        self.layers[-1][-1] = x


def inception1d_with_convnext(**kwargs):
    """Constructs an Inception model with ConvNeXt-style layers"""
    return Inception1dWithConvNeXt(**kwargs)

