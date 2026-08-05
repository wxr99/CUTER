import copy
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .tresnet_l_ica import Tresnet_L_ICA
from torch.nn import Module as Module
from collections import OrderedDict
from src.models.tresnet.layers.anti_aliasing import AntiAliasDownsampleLayer
from .layers.avg_pool import FastAvgPool2d
from .layers.general_layers import SEModule, SpaceToDepthModule
# from inplace_abn import InPlaceABN, ABN
from .tresnet import bottleneck_head, conv2d, conv2d_ABN, BasicBlock, Bottleneck, TResNet
from continual.convit import Block, ClassAttention
from continual.classifier import ContinualClassifier
import continual.utils as cutils


class Tresnet_M_ICA(Tresnet_L_ICA):
    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0,
                 do_bottleneck_head=False, bottleneck_features=512, first_two_layers='basicblock',
                 embed_dim=2432, depth=6,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer='layer', use_head_div=False, use_pos_embed=False
                 ):

        # This is for Tresnet_L_v2
        if first_two_layers == 'bottleneck':
            self.first_two_layers = Bottleneck
        else:
            self.first_two_layers = BasicBlock

        super(Tresnet_M_ICA, self).__init__(layers=layers, in_chans=in_chans, num_classes=num_classes, width_factor=width_factor,
                 do_bottleneck_head=do_bottleneck_head, bottleneck_features=bottleneck_features, first_two_layers=first_two_layers,
                 embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=norm_layer, use_head_div=use_head_div, use_pos_embed=use_pos_embed)
