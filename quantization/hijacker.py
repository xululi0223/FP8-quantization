#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy

from timm.models.layers.activations import Swish, HardSwish, HardSigmoid
from timm.models.layers.activations_me import SwishMe, HardSwishMe, HardSigmoidMe
from torch import nn

from quantization.base_quantized_classes import QuantizedModule
from quantization.quantization_manager import QuantizationManager
from quantization.range_estimators import RangeEstimators

activations_set = [
    nn.ReLU,
    nn.ReLU6,
    nn.Hardtanh,
    nn.Sigmoid,
    nn.Tanh,
    nn.GELU,
    nn.PReLU,
    Swish,
    SwishMe,
    HardSwish,
    HardSwishMe,
    HardSigmoid,
    HardSigmoidMe,
]


class QuantizationHijacker(QuantizedModule):
    """
    继承自 `QuantizedModule` 的混入类（Mixin），其主要功能是在模块的前向传播过程中“劫持”操作，以执行权重和输出的量化与反量化操作。
    通过这种方式，`QuantizationHijacker` 可以在不修改原有模块逻辑的基础上，添加量化功能，从而支持量化感知训练（Quantization-Aware Training, QAT）。

    Mixin class that 'hijacks' the forward pass in a module to perform quantization and
    dequantization on the weights and output distributions.

    Usage:
    To make a quantized nn.Linear layer:
    class HijackedLinear(QuantizationHijacker, nn.Linear):
        pass
    """

    def __init__(self, *args, activation: nn.Module = None, **kwargs):
        """
        Args:
            activation: 激活函数，用于在量化模块的前向传播过程中应用激活函数。
        """

        super().__init__(*args, **kwargs)
        
        # 激活函数验证
        if activation:
            assert isinstance(activation, tuple(activations_set)), str(activation)

        # 激活函数深拷贝
        self.activation_function = copy.deepcopy(activation) if activation else None

        # 实例化激活量化器
        self.activation_quantizer = QuantizationManager(
            qmethod=self.act_method,
            init=self.act_range_method,
            qparams=self.act_qparams,
            range_estim_params=self.act_range_options,
        )

        # 权重量化范围初始化参数
        if self.weight_range_method == RangeEstimators.current_minmax:
            weight_init_params = dict(percentile=self.percentile)
        else:
            weight_init_params = self.weight_range_options

        # 实例化权重量化器
        self.weight_quantizer = QuantizationManager(
            qmethod=self.method,
            init=self.weight_range_method,
            per_channel=self.per_channel_weights,
            qparams=self.weight_qparams,
            range_estim_params=weight_init_params,
        )

    def forward(self, x, offsets=None):
        """
        前向传播方法。
        
        Args:
            x: 输入张量。
            offsets: 用于特定量化操作的偏移量参数。
        """
        # Quantize input
        # 输入量化
        if self.quantize_input and self._quant_a:
            x = self.activation_quantizer(x)

        # Get quantized weight
        # 获取量化权重
        weight, bias = self.get_params()
        res = self.run_forward(x, weight, bias, offsets=offsets)    # 执行实际的前向传播逻辑，接收量化后的输入、权重、偏置和偏移量参数

        # Apply fused activation function
        # 应用激活函数
        if self.activation_function is not None:
            res = self.activation_function(res)

        # Quantize output
        # 输出量化
        if not self.quantize_input and self._quant_a:
            res = self.activation_quantizer(res)
        return res

    def get_params(self):
        """
        获取量化后的模型权重和偏置。
        """
        # 获取权重和偏置
        weight, bias = self.get_weight_bias()

        # 对权重进行量化
        if self._quant_w:
            weight = self.quantize_weights(weight)

        return weight, bias

    def quantize_weights(self, weights):
        """
        权重量化方法。
        """
        return self.weight_quantizer(weights)

    def get_weight_bias(self):
        """
        获取权重和偏置方法。
        """
        bias = None
        if hasattr(self, "bias"):
            bias = self.bias
        return self.weight, bias

    def run_forward(self, x, weight, bias, offsets=None):
        """
        权重量化执行方法。
        用于执行层的实际前向操作。
        
        Args:
            x: 输入张量。
            weight: 权重张量。
            bias: 偏置张量。
            offsets: 用于特定量化操作的偏移量参数。
        """
        # Performs the actual linear operation of the layer
        raise NotImplementedError()

    def extra_repr(self):
        """
        额外的字符串表示方法。
        """
        activation = "input" if self.quantize_input else "output"
        return f"{super().extra_repr()}-{activation}"
