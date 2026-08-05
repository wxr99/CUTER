import logging

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL, Tresnet_M_ICA, Tresnet_L_ICA
from ..resnet import Resnet
from ..vit import vit_small_patch16_224, vit_base_patch16_224

def my_create_model(args, num_classes):
    """Create a model, with model_name and num_classes
    """
    model_params = {'args': args, 'num_classes': num_classes}
    model_name = args.model_name.lower()

    if model_name=='tresnet_m':
        model = TResnetM(model_params)
    elif model_name=='tresnet_l':
        model = TResnetL(model_params)
    elif model_name=='tresnet_xl':
        model = TResnetXL(model_params)

    elif model_name == 'tresnet_m_ica':
        model = Tresnet_M_ICA(layers=[3, 4, 11, 3], in_chans=3, num_classes=num_classes,  # Tresnet_M params
                                width_factor=1.0, do_bottleneck_head=args.do_bottleneck_head, first_two_layers='basicblock',
                                embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=4.,  # dytox params
                                qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                                drop_path_rate=0., norm_layer='layer', use_head_div=args.use_head_div,
                                use_pos_embed=args.use_pos_embed)
        
    elif model_name == 'tresnet_l_ica':
        model = Tresnet_L_ICA(layers=[4, 5, 18, 3], in_chans=3, num_classes=num_classes,  # Tresnet_L params
                                width_factor=1.2, do_bottleneck_head=args.do_bottleneck_head, first_two_layers='basicblock',
                                embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=4.,  # dytox params
                                qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                                drop_path_rate=0., norm_layer='layer', use_head_div=args.use_head_div,
                                use_pos_embed=args.use_pos_embed)
    elif model_name == 'resnet':
        model = Resnet(num_class=num_classes)

    elif model_name == 'vitb':
        model = vit_base_patch16_224(pretrained=True, num_classes=num_classes, pretrained_path=args.pretrained_path)

    elif model_name == 'vits':
        model = vit_small_patch16_224(pretrained=True, num_classes=num_classes, pretrained_path=args.pretrained_path)

    else:
        print("model: {} not found !!".format(model_name))
        exit(-1)

    return model