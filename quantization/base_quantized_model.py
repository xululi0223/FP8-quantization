#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
from typing import Union, Dict

import torch
from torch import nn, Tensor

from quantization.base_quantized_classes import (
    QuantizedModule,
    _set_layer_estimate_ranges,
    _set_layer_estimate_ranges_train,
    _set_layer_learn_ranges,
    _set_layer_fix_ranges,
)
from quantization.quantizers import QuantizerBase


class QuantizedModel(nn.Module):
    """
    作为一个量化模型的父类，提供了便捷的方法来将整个模型切换到量化模式或全精度模式。
    它还覆盖了 `load_state_dict` 方法，以确保量化参数的正确加载。
    此外，该类包含了管理量化范围估计和学习的方法，支持量化感知训练（QAT）。

    Parent class for a quantized model. This allows you to have convenience functions to put the
    whole model into quantization or full precision.
    """

    def __init__(self, input_size=(1, 3, 224, 224)):
        """
        Parameters
        ----------
        input_size:     Tuple with the input dimension for the model (including batch dimension)
        """
        super().__init__()
        self.input_size = input_size

    def load_state_dict(
        self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool = True
    ):
        """
        重写 nn.Module 的 load_state_dict 方法，以确保在加载量化模型时正确加载量化参数。
        This function overwrites the load_state_dict of nn.Module to ensure that quantization
        parameters are loaded correctly for quantized model.

        """
        # 筛选量化参数
        quant_state_dict = {
            k: v for k, v in state_dict.items() if k.endswith("_quant_a") or k.endswith("_quant_w")
        }

        # 加载量化参数
        if quant_state_dict:
            super().load_state_dict(quant_state_dict, strict=False)
        else:
            raise ValueError(
                "The quantization states of activations or weights should be "
                "included in the state dict "
            )
        # Pass dummy data through quantized model to ensure all quantization parameters are
        # initialized with the correct dimensions (None tensors will lead to issues in state dict
        # loading)
        # 初始化量化参数
        device = next(self.parameters()).device
        dummy_input = torch.rand(*self.input_size, device=device)
        with torch.no_grad():
            self.forward(dummy_input)               # 通过前向传播传递dummy_input，以确保所有量化参数都以正确的维度初始化

        # Load state dict
        # 加载状态字典
        super().load_state_dict(state_dict, strict)

    def quantized_weights(self):
        """
        将所有权重设置为量化状态。
        """
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized_weights()

        self.apply(_fn)

    def full_precision_weights(self):
        """
        将所有权重设置为全精度状态。
        """
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision_weights()

        self.apply(_fn)

    def quantized_acts(self):
        """
        将所有激活设置为量化状态。
        """
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized_acts()

        self.apply(_fn)

    def full_precision_acts(self):
        """
        将所有激活设置为全精度状态。
        """
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision_acts()

        self.apply(_fn)

    def quantized(self):
        """
        将所有权重和激活设置为量化状态。
        """
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized()

        self.apply(_fn)

    def full_precision(self):
        """
        将所有权重和激活设置为全精度状态。
        """
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision()

        self.apply(_fn)

    def estimate_ranges(self):
        """
        将所有相关层设置为估计量化范围模式。
        """
        self.apply(_set_layer_estimate_ranges)

    def estimate_ranges_train(self):
        """
        将所有相关层设置为训练时估计量化范围模式。
        """
        self.apply(_set_layer_estimate_ranges_train)

    def set_quant_state(self, weight_quant, act_quant):
        """
        根据参数设置模型中权重和激活的量化状态。
        
        Args:
            weight_quant: 权重量化状态
            act_quant: 激活量化状态
        """
        if act_quant:
            self.quantized_acts()
        else:
            self.full_precision_acts()

        if weight_quant:
            self.quantized_weights()
        else:
            self.full_precision_weights()

    def grad_scaling(self, grad_scaling=True):
        """
        控制模型中所有量化器的梯度缩放属性。
        
        Args:
            grad_scaling: 是否启用梯度缩放
        """
        def _fn(module):
            if isinstance(module, QuantizerBase):
                module.grad_scaling = grad_scaling

        self.apply(_fn)
        # Methods for switching quantizer quantization states

    def learn_ranges(self):
        """
        使所有量化器的范围参数变为可学习的。
        """
        self.apply(_set_layer_learn_ranges)

    def fix_ranges(self):
        """
        固定所有量化器的范围参数。
        """
        self.apply(_set_layer_fix_ranges)
