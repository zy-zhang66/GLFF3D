# Copyright (c) OpenMMLab. All rights reserved.
# Follow https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/resnet.py # noqa
# and mmcv.cnn.ResNet
try:
    import MinkowskiEngine as ME
    from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')
    # blocks are used in the static part of MinkResNet
    BasicBlock, Bottleneck = None, None

import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet3d.models.builder import BACKBONES


@BACKBONES.register_module()
class MinkResNet(BaseModule):
    r"""Minkowski ResNet backbone. See `4D Spatio-Temporal ConvNets
    <https://arxiv.org/abs/1904.08755>`_ for more details.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (ont): Number of input channels, 3 for RGB.
        num_stages (int, optional): Resnet stages. Default: 4.
        pool (bool, optional): Add max pooling after first conv if True.
            Default: True.
    """
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels,
                 max_channels=None,
                 num_stages=4,
                 pool=True,
                 norm='instance'):
        super(MinkResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        assert 4 >= num_stages >= 1
        block, stage_blocks = self.arch_settings[depth]
        stage_blocks = stage_blocks[:num_stages]
        self.max_channels = max_channels
        self.num_stages = num_stages
        self.pool = pool

        self.inplanes = 64
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=3, stride=2, dimension=3)
        norm1 = ME.MinkowskiInstanceNorm if norm == 'instance' \
            else ME.MinkowskiBatchNorm
        self.norm1 = norm1(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        if self.pool:
            self.maxpool = ME.MinkowskiMaxPooling(
                kernel_size=2, stride=2, dimension=3)

        for i, _ in enumerate(stage_blocks):
            n_channels = 64 * 2**i
            if self.max_channels is not None:
                n_channels = min(n_channels, self.max_channels)
            setattr(
                self, f'layer{i + 1}',
                self._make_layer(block, n_channels, stage_blocks[i], stride=2))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=3),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                dimension=3))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dimension=3))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of ResNet.

        Args:
            x (ME.SparseTensor): Input sparse tensor.

        Returns:
            list[ME.SparseTensor]: Output sparse tensors.
        """

        x = self.conv1(x)

        x = self.norm1(x)

        x = self.relu(x)

        if self.pool:

            x = self.maxpool(x)

        outs = []
        for i in range(self.num_stages):
            x = getattr(self, f'layer{i + 1}')(x)
            outs.append(x)
        return outs


@BACKBONES.register_module()
class MinkFFResNet(MinkResNet):
    def forward(self, x, f):
        """Forward pass of ResNet.

        Args:
            x (ME.SparseTensor): Input sparse tensor.

        Returns:
            list[ME.SparseTensor]: Output sparse tensors.
        """
        # 输入x的维度是(543993,3)
        x = self.conv1(x)
        # 输出x的维度是(318872,64)
        x = self.norm1(x)
        # 输出x的维度是(318872,64)
        x = self.relu(x)
        # 输出x的维度是(318872,64)
        if self.pool:
            # 输入x的维度是(126442,64)
            x = self.maxpool(x)
            # 输出x的维度是(126442,64)

        # 这里会跳转到tr3d_ff.py的_f函数
        # 输入的x和输出的x维度是相同的
        x = f(x)
        outs = []
        for i in range(self.num_stages):
            x = getattr(self, f'layer{i + 1}')(x)
            outs.append(x)

        # 返回后的outs是一个长度为4的列表，每个张量的维度是(41002,64),(12607,128),(3997,128),(1258,128)
        # 128的原因是max_channels=128
        # 返回值之后就到neck了
        return outs
