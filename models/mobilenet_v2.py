#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

# https://github.com/tonylins/pytorch-mobilenet-v2

import math

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MobileNetV2"]


def conv_bn(inp, oup, stride):
    """
    用于创建一个包含卷积层、批归一化层和激活函数的序列模块。

    Args:
        inp: 输入通道数
        oup: 输出通道数
        stride: 步长
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    """
    用于创建一个包含1x1卷积层、批归一化层和激活函数的序列模块。

    Args:
        inp: 输入通道数
        oup: 输出通道数
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    """
    实现了MobileNetV2中的倒置残差块，用于构建网络的核心模块。
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        """
        Args:
            inp: 输入通道数
            oup: 输出通道数
            stride: 步长
            expand_ratio: 扩展比例，用于增加或减少中间层的通道数
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        # 计算中间层的通道数
        hidden_dim = round(inp * expand_ratio)
        # 判断是否使用残差连接
        self.use_res_connect = self.stride == 1 and inp == oup

        # 构建卷积序列
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), # Depthwise卷积
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),                                # Pointwise卷积
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),                                # Pointwise卷积
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), # Depthwise卷积
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),                                # Pointwise卷积
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        """
        前向传播方法。
        """
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    实现了MobileNetV2魔性的整体结构，用于图像分类任务。
    """
    def __init__(self, n_class=1000, input_size=224, width_mult=1.0, dropout=0.0):
        """
        Args:
            n_class: 类别数
            input_size: 输入尺寸
            width_mult: 宽度倍数，用于调整网络宽度
            dropout: Dropout概率
        """
        super().__init__()
        # 模块定义
        block = InvertedResidual        # 基本块
        input_channel = 32              # 初始输入通道数
        last_channel = 1280             # 最后一层的通道数
        inverted_residual_setting = [   # 每一层倒置残差块的配置
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        # 构建第一层
        assert input_size % 32 == 0
        # 调整初始输入通道数
        input_channel = int(input_channel * width_mult)
        # 调整最后一层的通道数
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        # 添加第一层卷积模块
        features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        # 构建倒置残差块
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        # 构建最后几层
        features.append(conv_1x1_bn(input_channel, self.last_channel))  # 添加1x1卷积模块
        features.append(nn.AvgPool2d(input_size // 32))                 # 添加全局平均池化层
        # make it nn.Sequential
        self.features = nn.Sequential(*features)                        # 封装特征层

        # building classifier
        # 构建分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channel, n_class),
        )

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        """
        前向传播方法。
        """
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()  # type: ignore[arg-type] # accepted slang
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        权重初始化方法。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))    # 符合He初始化方法
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
