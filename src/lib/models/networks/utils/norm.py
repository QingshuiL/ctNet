import torch.nn as nn
from models.networks.utils.CBN import CBatchNorm2d

import sys 
print(sys.path)

norm_cfg = {
    # format: layer_type: (abbreviation, module)
    'BN': ('bn', nn.BatchNorm2d),
    'GN': ('gn', nn.GroupNorm),
    'CBN': ('bn', CBatchNorm2d),
    # and potentially 'IN' 'LN' 'BRN'
}


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            frozen (bool): [optional] whether stop gradient updates
                of norm layer, it is helpful to set frozen mode
                in backbone's norms.
        num_features (int): number of channels from input
        postfix (int, str): appended into norm abbreation to
            create named layer.

    Returns:
        name (str): abbreation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    frozen = cfg_.pop('frozen', False)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN-pytorch':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    if frozen:
        for param in layer.parameters():
            param.requires_grad = False

    return name, layer



class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(normalize, out_dim, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, out_dim, postfix=2)

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.add_module(self.norm1_name, norm1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.add_module(self.norm2_name, norm2)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.norm1(conv1, self.conv1.weight)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.norm2(conv2, self.conv2.weight)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)
