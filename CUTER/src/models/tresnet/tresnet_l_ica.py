import copy
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
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

class InPlaceABN(nn.Module):
    def __init__(self, num_features, activation, activation_param):
        super(InPlaceABN, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.activation_param = activation_param

    def forward(self,x):
        x = self.bn(x)
        return F.leaky_relu(x, negative_slope=self.activation_param)

class Tresnet_L_ICA(TResNet):
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

        super(Tresnet_L_ICA, self).__init__(layers, in_chans, num_classes, width_factor,
                                           do_bottleneck_head, bottleneck_features)

        self.embed_dim = embed_dim
        self.use_pos_embed = use_pos_embed
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, 49, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        # Last projection
        self.last_proj = nn.Conv2d(2048, embed_dim, 1, 1)

        # TAB
        task_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(task_token, std=.02)
        self.task_tokens = nn.ParameterList([task_token])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.tab = Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[5], norm_layer=nn.LayerNorm,
                    attention_type=ClassAttention
                )

        self.head = nn.ModuleList([
            ContinualClassifier(embed_dim, num_classes).cuda()
        ])

        self.use_head_div = use_head_div
        self.head_div = None
        
        # Put at Last to operate kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            # elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, InPlaceABN):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].bn.weight = nn.Parameter(torch.zeros_like(m.conv2[1].bn.weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].bn.weight = nn.Parameter(torch.zeros_like(m.conv3[1].bn.weight))  # BN to zero
            if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)
        
        global_pool_layer = FastAvgPool2d(flatten=True)
        self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
        

    def freeze(self, names):
        """Choose what to freeze depending on the name of the module."""
        requires_grad = False
        cutils.freeze_parameters(self, requires_grad=not requires_grad)
        self.train()

        for name in names:
            if name == 'all':
                self.eval()
                return cutils.freeze_parameters(self)
            elif name == 'old_task_tokens':
                cutils.freeze_parameters(self.task_tokens[:-1], requires_grad=requires_grad)
            elif name == 'task_tokens':
                cutils.freeze_parameters(self.task_tokens, requires_grad=requires_grad)
            elif name == 'tab':
                self.tabs.eval()
                cutils.freeze_parameters(self.tabs, requires_grad=requires_grad)
            elif name == 'old_heads':
                self.head[:-1].eval()
                cutils.freeze_parameters(self.head[:-1], requires_grad=requires_grad)
            elif name == 'heads':
                self.head.eval()
                cutils.freeze_parameters(self.head, requires_grad=requires_grad)
            elif name == 'head_div':
                self.head_div.eval()
                cutils.freeze_parameters(self.head_div, requires_grad=requires_grad)
            else:
                raise NotImplementedError(f'Unknown name={name}.')

    def add_model(self, nb_new_classes):
        """
        Expand model as per the KRT framework given `nb_new_classes`.
        param nb_new_classes: Number of new classes brought by the new task.
        """

        # Class tokens
        new_task_token = copy.deepcopy(self.task_tokens[-1])
        trunc_normal_(new_task_token, std=.02)
        self.task_tokens.append(new_task_token)

        # Diversity head
        if self.use_head_div:
            self.head_div = ContinualClassifier(self.embed_dim, nb_new_classes + 1).cuda()

        # Classifier
        self.head.append(ContinualClassifier(self.embed_dim, nb_new_classes).cuda())

    def forward(self, x):
        B = x.shape[0]

        # feature extractor: Tresnet_L
        x = self.space_to_depth(x)
        x = self.conv1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)  # B x 2432 x 7 x 7

        pool_embeddings = self.global_pool(x4)

        proj = self.last_proj(x4)

        # Token Attention Block
        tab_input = proj.flatten(2).transpose(1, 2)  # B x 49 x 2432

        if self.use_pos_embed:
            tab_input = tab_input + self.pos_embed

        tokens = []
        for task_token in self.task_tokens:
            task_token = task_token.expand(B, -1, -1)
            cls_token = self.cls_token.expand(B, -1, -1)
            task_token, attn, v = self.tab(torch.cat((cls_token, task_token, tab_input), dim=1))  # (b, 1 2048)?
            tokens.append(task_token[:, 0])  # e1, e2, ..., ei

        # Classifier
        logits = []
        for i, head in enumerate(self.head):
            logits.append(head(tokens[i]))
        logits = torch.cat(logits, dim=1)

        # Divergence Classifier: Not suitable for Multilabel
        logits_div = None
        if self.use_head_div:
            logits_div = self.head_div(tokens[-1])  # only last token

        return {
            'logits': logits,
            'logits_div': logits_div,
            'tokens': tokens,
           # 'attentions': [x1, x2, x3, x4],
            'attentions': [x1, x2, x3],
            'embeddings': torch.cat(tokens, dim=1),  # embedding is the output of global_pool(x4)
            'pool_embeddings': pool_embeddings, 
        }
