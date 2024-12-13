#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd

from quantization.hijacker import QuantizationHijacker


class BNFusedHijacker(QuantizationHijacker):
    """
    `BNFusedHijacker` 类是 `QuantizationHijacker` 的扩展，
    负责将批量归一化（Batch Normalization, BN）层与前置的权重层（如卷积层或线性层）融合为一个联合模块。
    在融合过程中，BN 层的参数和统计数据保持为全精度。这种融合有助于优化模型的推理性能，同时保持量化过程的准确性。
    Extension to the QuantizationHijacker that fuses batch normalization (BN) after a weight
    layer into a joined module. The parameters and the statistics of the BN layer remain in
    full-precision.
    """

    def __init__(self, *args, **kwargs):
        # 移除bias参数
        kwargs.pop("bias", None)  # Bias will be learned by BN params
        super().__init__(*args, **kwargs, bias=False)
        # 获取BN层的维度
        bn_dim = self.get_bn_dim()
        # 注册缓冲区
        self.register_buffer("running_mean", torch.zeros(bn_dim))
        self.register_buffer("running_var", torch.ones(bn_dim))
        # 设置动量参数
        self.momentum = kwargs.pop("momentum", 0.1)
        # 定义BN层可学习参数
        self.gamma = nn.Parameter(torch.ones(bn_dim))
        self.beta = nn.Parameter(torch.zeros(bn_dim))
        self.epsilon = kwargs.get("eps", 1e-5)
        self.bias = None

    def forward(self, x):
        """
        前向传播方法。
        """
        # Quantize input
        # 量化输入
        if self.quantize_input and self._quant_a:
            x = self.activation_quantizer(x)

        # Get quantized weight
        # 获取量化后的权重
        weight, bias = self.get_params()
        # 执行前向传播
        res = self.run_forward(x, weight, bias)

        # 应用批量归一化
        res = F.batch_norm(
            res,
            self.running_mean,
            self.running_var,
            self.gamma,
            self.beta,
            self.training,
            self.momentum,
            self.epsilon,
        )
        # Apply fused activation function
        # 应用融合的激活函数
        if self.activation_function is not None:
            res = self.activation_function(res)

        # Quantize output
        # 量化输出
        if not self.quantize_input and self._quant_a:
            res = self.activation_quantizer(res)
        return res

    def get_bn_dim(self):
        """
        确定BN层的维度。
        """
        # 检查是否为线性层
        if isinstance(self, nn.Linear):
            return self.out_features
        # 检查是否为卷积层
        elif isinstance(self, _ConvNd):
            return self.out_channels
        else:
            msg = (
                f"Unsupported type used: {self}. Must be a linear or (transpose)-convolutional "
                f"nn.Module"
            )
            raise NotImplementedError(msg)
