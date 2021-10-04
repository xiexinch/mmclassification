import pytest
import torch
from mmcls.models import STDCNet

norm_cfg = dict(type='BN', requires_grad=True)


def test_stdcnet():
    with pytest.raises(KeyError):
        backbone = STDCNet('STDCNet',
                           3,
                           channels=(32, 64, 256, 512, 1024),
                           bottleneck_type='cat',
                           stdc_num_convs=4,
                           norm_cfg=norm_cfg,
                           act_cfg=dict(type='ReLU'), )

    img = torch.randn(1, 3, 224, 224)
    backbone = STDCNet('STDCNet813',
                       3,
                       channels=(32, 64, 256, 512, 1024),
                       bottleneck_type='cat',
                       stdc_num_convs=4,
                       norm_cfg=norm_cfg,
                       act_cfg=dict(type='ReLU'), )

    outs = backbone(img)
    assert len(outs) == 6
    assert outs[0].shape == torch.Size((1, 32, 112, 112))
    assert outs[1].shape == torch.Size((1, 64, 56, 56))
    assert outs[2].shape == torch.Size((1, 256, 28, 28))
    assert outs[3].shape == torch.Size((1, 512, 14, 14))
    assert outs[4].shape == torch.Size((1, 1024, 7, 7))
    assert outs[5].shape == torch.Size((1, 1024, 1, 1))
