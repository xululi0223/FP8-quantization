#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
from torch import nn

from quantization.quantization_manager import QuantizationManager
from quantization.quantizers import QuantizerBase, AsymmetricUniformQuantizer
from quantization.range_estimators import (
    RangeEstimatorBase,
    CurrentMinMaxEstimator,
    RunningMinMaxEstimator,
)


def _set_layer_learn_ranges(layer):
    """
    辅助函数，用于在给定的层（layer）是 `QuantizationManager` 的实例且其量化器已初始化的情况下，调用该层的 `learn_ranges` 方法。
    该方法使量化器的范围参数变为可学习的，以便在训练过程中通过反向传播进行优化。
    """
    if isinstance(layer, QuantizationManager):
        if layer.quantizer.is_initialized:
            layer.learn_ranges()


def _set_layer_fix_ranges(layer):
    """
    辅助函数，用于在给定的层（layer）是 `QuantizationManager` 的实例且其量化器已初始化的情况下，调用该层的 `fix_ranges` 方法。
    该方法固定量化器的范围参数，使其不再变化。
    """
    if isinstance(layer, QuantizationManager):
        if layer.quantizer.is_initialized:
            layer.fix_ranges()


def _set_layer_estimate_ranges(layer):
    """
    辅助函数，用于在给定的层（layer）是 `QuantizationManager` 的实例的情况下，调用该层的 `estimate_ranges` 方法。
    该方法将量化器的状态设置为估计范围模式，表示当前正在估计量化范围。
    """
    if isinstance(layer, QuantizationManager):
        layer.estimate_ranges()


def _set_layer_estimate_ranges_train(layer):
    """
    辅助函数，用于在给定的层（layer）是 `QuantizationManager` 的实例的情况下，调用该层的 `estimate_ranges_train` 方法。
    该方法将量化器的状态设置为在训练期间估计范围模式，同时在评估期间保持固定范围。
    """
    if isinstance(layer, QuantizationManager):
        if layer.quantizer.is_initialized:
            layer.estimate_ranges_train()


class QuantizedModule(nn.Module):
    """
    用于封装和管理量化模块的基本功能。
    它提供了在量化和全精度模式之间切换的能力，定义了缓存参数，并妥善处理缓存的重置。
    该类还包含了管理量化范围估计的方法，如学习范围、固定范围和估计范围。
    Parent class for a quantized module. It adds the basic functionality of switching the module
    between quantized and full precision mode. It also defines the cached parameters and handles
    the reset of the cache properly.
    """

    def __init__(
        self,
        *args,
        method: QuantizerBase = AsymmetricUniformQuantizer,
        act_method=None,
        weight_range_method: RangeEstimatorBase = CurrentMinMaxEstimator,
        act_range_method: RangeEstimatorBase = RunningMinMaxEstimator,
        n_bits=8,
        n_bits_act=None,
        per_channel_weights=False,
        percentile=None,
        weight_range_options=None,
        act_range_options=None,
        scale_domain="linear",
        act_quant_kwargs={},
        weight_quant_kwargs={},
        quantize_input=False,
        fp8_kwargs=None,
        **kwargs
    ):
        """
        Args:
            method: 量化方法，默认为非对称均匀量化器。
            act_method: 激活量化方法，默认为 method。
            weight_range_method: 权重范围估计方法，默认为当前最小最大值估计器。
            act_range_method: 激活范围估计方法，默认为运行最小最大值估计器。
            n_bits: 量化位数，默认为 8。
            n_bits_act: 激活量化位数，默认为 n_bits。
            per_channel_weights: 是否按通道进行权重量化，默认为 False。
            percentile: 百分位数，用于估计范围。
            weight_range_options: 权重范围估计器参数。
            act_range_options: 激活范围估计器参数。
            scale_domain: 量化域，默认为线性。
            act_quant_kwargs: 激活量化器的额外参数。
            weight_quant_kwargs: 权重量化器的额外参数。
            quantize_input: 是否量化输入，默认为 False。
            fp8_kwargs: FP8量化器的额外参数。
        """
        # 移除多余参数
        kwargs.pop("act_quant_dict", None)

        super().__init__(*args, **kwargs)

        self.method = method
        self.act_method = act_method or method
        self.n_bits = n_bits
        self.n_bits_act = n_bits_act or n_bits
        self.per_channel_weights = per_channel_weights
        self.percentile = percentile
        self.weight_range_method = weight_range_method
        self.weight_range_options = weight_range_options if weight_range_options else {}
        self.act_range_method = act_range_method
        self.act_range_options = act_range_options if act_range_options else {}
        self.scale_domain = scale_domain
        self.quantize_input = quantize_input
        self.fp8_kwargs = fp8_kwargs or {}

        # 定义量化参数缓存
        self.quant_params = None
        self.register_buffer("_quant_w", torch.BoolTensor([False]))     # 表示权重默认为全精度
        self.register_buffer("_quant_a", torch.BoolTensor([False]))     # 表示激活默认为全精度

        # 定义量化器参数
        self.act_qparams = dict(
            n_bits=self.n_bits_act,
            scale_domain=self.scale_domain,
            **act_quant_kwargs,
            **self.fp8_kwargs
        )
        self.weight_qparams = dict(
            n_bits=self.n_bits,
            scale_domain=self.scale_domain,
            **weight_quant_kwargs,
            **self.fp8_kwargs
        )

    def quantized_weights(self):
        """
        设置权重为量化状态。
        """
        self._quant_w = torch.BoolTensor([True])

    def full_precision_weights(self):
        """
        设置权重为全精度状态。
        """
        self._quant_w = torch.BoolTensor([False])

    def quantized_acts(self):
        """
        设置激活为量化状态。
        """
        self._quant_a = torch.BoolTensor([True])

    def full_precision_acts(self):
        """
        设置激活为全精度状态。
        """
        self._quant_a = torch.BoolTensor([False])

    def quantized(self):
        """
        设置权重和激活为量化状态。
        """
        self.quantized_weights()
        self.quantized_acts()

    def full_precision(self):
        """
        设置权重和激活为全精度状态。
        """
        self.full_precision_weights()
        self.full_precision_acts()

    def get_quantizer_status(self):
        """
        获取量化器状态。
        """
        return dict(quant_a=self._quant_a.item(), quant_w=self._quant_w.item())

    def set_quantizer_status(self, quantizer_status):
        """
        设置量化器状态。
        
        Args:
            quantizer_status: 量化器状态字典。
        """
        if quantizer_status["quant_a"]:
            self.quantized_acts()
        else:
            self.full_precision_acts()

        if quantizer_status["quant_w"]:
            self.quantized_weights()
        else:
            self.full_precision_weights()

    def learn_ranges(self):
        """
        应用_set_layer_learn_ranges辅助函数，使量化器的范围参数变为可学习的。
        """
        self.apply(_set_layer_learn_ranges)

    def fix_ranges(self):
        """
        应用_set_layer_fix_ranges辅助函数，固定量化器的范围参数。
        """
        self.apply(_set_layer_fix_ranges)

    def estimate_ranges(self):
        """
        应用_set_layer_estimate_ranges辅助函数，估计量化器的范围。
        """
        self.apply(_set_layer_estimate_ranges)

    def estimate_ranges_train(self):
        """
        应用_set_layer_estimate_ranges_train辅助函数，训练时估计量化器的范围。
        """
        self.apply(_set_layer_estimate_ranges_train)

    def extra_repr(self):
        """
        额外的字符串表示方法。
        """
        quant_state = "weight_quant={}, act_quant={}".format(
            self._quant_w.item(), self._quant_a.item()
        )
        parent_repr = super().extra_repr()
        return "{},\n{}".format(parent_repr, quant_state) if parent_repr else quant_state


class QuantizedActivation(QuantizedModule):
    """
    继承自QuantizedModule，专门用于管理和量化激活值。
    它初始化并配置了激活量化器 (QuantizationManager)，并在前向传播过程中应用激活量化操作。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 实例化激活量化器
        self.activation_quantizer = QuantizationManager(
            qmethod=self.act_method,
            qparams=self.act_qparams,
            init=self.act_range_method,
            range_estim_params=self.act_range_options,
        )

    def quantize_activations(self, x):
        """
        量化激活。
        """
        # 如果激活量化器为量化状态，则对输入进行量化
        if self._quant_a:
            return self.activation_quantizer(x)
        else:
            return x

    def forward(self, x):
        """
        前向传播方法。
        """
        return self.quantize_activations(x)


class FP32Acts(nn.Module):
    """
    专门用于处理激活值而不进行量化。
    它实现了一个直接传递输入的前向方法，并提供了一个空的 reset_ranges 方法，用于与量化模块接口的一致性。
    """
    def forward(self, x):
        return x

    def reset_ranges(self):
        pass
