norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='ImageClassifier',
    backbone=dict(type='STDCNet',
                  stdc_type='STDCNet813',
                  in_channels=3,
                  channels=(32, 64, 256, 512, 1024),
                  bottleneck_type='cat',
                  stdc_num_convs=4,
                  norm_cfg=norm_cfg,
                  act_cfg=dict(type='ReLU')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=1000,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False))
train_cfg = dict(mixup=dict(alpha=0.2, num_classes=1000))
