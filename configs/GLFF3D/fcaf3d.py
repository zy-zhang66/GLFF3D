_base_ = ['./sunrgbd-3d-10class.py']

voxel_size = 0.01

model = dict(
    type='SingleStageSparse3DDetector',
    voxel_size=voxel_size,
    backbone=dict(
        type='MEResNet3D',
        in_channels=3,
        depth=34),
    neck_with_head=dict(
        type='Fcaf3DNeckWithHead',
        in_channels=(64, 128, 256, 512),
        out_channels=128,
        pts_threshold=100000,
        n_classes=10,
        n_reg_outs=8,
        voxel_size=voxel_size,
        assigner=dict(
            type='Fcaf3DAssigner',
            limit=27,
            topk=18,
            n_scales=4),
        loss_bbox=dict(type='IoU3DLoss', loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(
        nms_pre=1000,
        iou_thr=.5,
        score_thr=.01,
        ensemble_stages=[0,1]))

find_unused_parameters = True
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# lr_config = dict(policy='step', warmup=None, step=[8, 11])
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=1e-5,  # 你可以调整这个最小学习率
    warmup='linear',  # 可选：如果需要 warmup，则启用
    warmup_iters=500,  # 需要的 warmup iterations 数
    warmup_ratio=0.001,  # 起始学习率
)
runner = dict(type='EpochBasedRunner', max_epochs=16)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]