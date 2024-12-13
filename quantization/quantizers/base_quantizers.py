#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

from torch import nn


class QuantizerBase(nn.Module):
    """
    一个基类，用于定义量化器的基本接口和通用属性，为具体的量化器实现提供了统一的框架。
    主要负责初始化量化参数、定义必需的方法接口以及提供一些通用的功能，如重置量化器状态和额外的表示信息。
    """
    def __init__(self, n_bits, per_channel=False, *args, **kwargs):
        """
        Args:
            n_bits: 量化的位数，即每个量化级别所使用的比特数。
            per_channel: 是否进行逐通道量化，默认为False，表示全局量化。
        """
        super().__init__(*args, **kwargs)
        self.n_bits = n_bits
        self.per_channel = per_channel
        self.state = None                               # 初始化量化器的状态
        self.x_min_fp32 = self.x_max_fp32 = None        # 初始化FP32精度下的最小和最大值

    @property
    def is_initialized(self):
        """
        检查量化器是否已经初始化。
        """
        raise NotImplementedError()

    @property
    def x_max(self):
        """
        获取量化范围的最大值。
        """
        raise NotImplementedError()

    @property
    def symmetric(self):
        """
        指示量化是否是对称的。
        """
        raise NotImplementedError()

    @property
    def x_min(self):
        """
        获取量化范围的最小值。
        """
        raise NotImplementedError()

    def forward(self, x_float):
        """
        前向传播方法，用于将输入浮点数张量x_float进行量化处理。
        """
        raise NotImplementedError()

    def _adjust_params_per_channel(self, x):
        """
        根据输入张量x的每个通道调整量化参数。
        """
        raise NotImplementedError()

    def set_quant_range(self, x_min, x_max):
        """
        设置量化的最小值x_min和最大值x_max。
        """
        raise NotImplementedError()

    def extra_repr(self):
        """
        返回类的额外字符串表示信息。
        """
        return "n_bits={}, per_channel={}, is_initalized={}".format(
            self.n_bits, self.per_channel, self.is_initialized
        )

    def reset(self):
        """
        重置量化器的内部状态。
        """
        self._delta = None
