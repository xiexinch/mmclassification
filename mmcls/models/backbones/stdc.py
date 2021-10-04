import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class STDCModule(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 norm_cfg=None,
                 act_cfg=None,
                 num_convs=4,
                 fusion_type='add',
                 init_cfg=None):
        super(STDCModule, self).__init__(init_cfg)
        assert num_convs > 1
        assert fusion_type in ['add', 'cat']
        self.stride = stride
        self.with_avg_pool = True if stride == 2 else False
        self.fusion_type = fusion_type

        self.layers = ModuleList()
        conv_0 = ConvModule(in_channels, out_channels // 2, kernel_size=1, norm_cfg=norm_cfg)

        if self.with_avg_pool:
            self.avg_pool = ConvModule(
                out_channels // 2,
                out_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=out_channels // 2,
                norm_cfg=norm_cfg,
                act_cfg=None)
            if self.fusion_type == 'add':
                self.layers.append(Sequential(conv_0, self.avg_pool))
                self.skip = Sequential(
                    ConvModule(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=in_channels,
                        norm_cfg=norm_cfg,
                        act_cfg=None),
                    ConvModule(
                        in_channels,
                        out_channels,
                        1,
                        norm_cfg=norm_cfg,
                        act_cfg=None))
            else:
                self.layers.append(conv_0)
                self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.layers.append(conv_0)

        for i in range(1, num_convs):
            out_factor = 2 ** (i + 1) if i != num_convs - 1 else 2 ** i
            self.layers.append(
                ConvModule(
                    out_channels // 2 ** i,
                    out_channels // out_factor,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        if self.fusion_type == 'add':
            out = self.forward_add(inputs)
        else:
            out = self.forward_cat(inputs)
        return out

    def forward_add(self, inputs):
        layer_outputs = []
        x = inputs.clone()
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        if self.with_avg_pool:
            inputs = self.skip(inputs)
        return torch.cat(layer_outputs, dim=1) + inputs

    def forward_cat(self, inputs):
        x0 = self.layers[0](inputs)
        layer_outputs = [x0]
        for i, layer in enumerate(self.layers[1:]):
            if i == 0:
                if self.with_avg_pool:
                    x = layer(self.avg_pool(x0))
                else:
                    x = layer(x0)
            else:
                x = layer(x)
            layer_outputs.append(x)
        if self.with_avg_pool:
            layer_outputs[0] = self.skip(x0)
        return torch.cat(layer_outputs, dim=1)


@BACKBONES.register_module()
class STDCNet(BaseBackbone):
    arch_settings = {
        'STDCNet813': [(2, 1), (2, 1), (2, 1)],
        'STDCNet1446': [(2, 1, 1, 1), (2, 1, 1, 1, 1), (2, 1, 1)]
    }

    def __init__(self,
                 stdc_type,
                 in_channels,
                 channels,
                 bottleneck_type,
                 norm_cfg,
                 act_cfg,
                 stdc_num_convs=4,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super(STDCNet, self).__init__(init_cfg)
        if stdc_type not in self.arch_settings:
            raise KeyError(f'invalid depth {stdc_type} for STDCNet')

        assert bottleneck_type in ['add', 'cat'], \
            f'bottleneck_type must be `add` or `cat`, got {bottleneck_type}'

        assert len(channels) == 5

        self.in_channels = in_channels
        self.channels = channels
        self.stage_strides = self.arch_settings[stdc_type]
        self.pretrained = pretrained
        self.stdc_num_convs = stdc_num_convs
        self.with_cp = with_cp

        self.stages = ModuleList([
            ConvModule(
                self.in_channels,
                self.channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                self.channels[0],
                self.channels[1],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        ])

        for strides in self.stage_strides:
            idx = len(self.stages) - 1
            self.stages.append(
                self._make_stages(self.channels[idx], self.channels[idx + 1],
                                  strides, norm_cfg, act_cfg, bottleneck_type))

        self.final_conv = ConvModule(
            self.channels[-1],
            max(1024, self.channels[-1]),
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # self.global_pool = nn.AvgPool2d(7)

    def _make_stages(self, in_channels, out_channels, strides, norm_cfg,
                     act_cfg, bottleneck_type):
        layers = []
        for i, stride in enumerate(strides):
            layers.append(
                STDCModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride,
                    norm_cfg,
                    act_cfg,
                    num_convs=self.stdc_num_convs,
                    fusion_type=bottleneck_type))
        return Sequential(*layers)

    def forward(self, x):
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        # outs.append(self.global_pool(self.final_conv(x)))
        outs.append(self.final_conv(x))
        return tuple(outs)
