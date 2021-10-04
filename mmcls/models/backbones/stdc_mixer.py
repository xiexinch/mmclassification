import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.runner import BaseModule
from mmcv.cnn import Linear, build_activation_layer

from mmcls.models import BACKBONES, BaseBackbone
from mmcls.models.utils import  PatchEmbed, PatchMerging


class Mlp(BaseModule):

    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_cfg=dict(type='GELU'), drop_rate=0., init_cfg=None):
        super(Mlp, self).__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = Linear(in_channels, hidden_channels)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerBlock(BaseModule):

    def __init__(self, channels, seq_len, mlp_ratio=(0.5, 4.0), norm_cfg=None, act_cfg=None, drop_rate=0., init_cfg=None):
        super(MixerBlock, self).__init__(init_cfg)





class MLPBlock(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass


class OverlapPatchEmbed(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass


@BACKBONES.register_module()
class STDCMlpMixer(BaseBackbone):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
