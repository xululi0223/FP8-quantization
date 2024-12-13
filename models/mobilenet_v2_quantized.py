#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import re
import torch
from collections import OrderedDict
from models.mobilenet_v2 import MobileNetV2, InvertedResidual
from quantization.autoquant_utils import quantize_sequential, Flattener, quantize_model, BNQConv
from quantization.base_quantized_classes import QuantizedActivation, FP32Acts
from quantization.base_quantized_model import QuantizedModel


class QuantizedInvertedResidual(QuantizedActivation):
    """
    继承自`QuantizedActivation`，用于量化MobileNet中的倒置残差块。
    """
    def __init__(self, inv_res_orig, **quant_params):
        """
        Args:
            inv_res_orig: 原始的倒置残差块
        """
        super().__init__(**quant_params)
        # 决定是否使用残差连接
        self.use_res_connect = inv_res_orig.use_res_connect
        # 量化卷积层
        self.conv = quantize_sequential(inv_res_orig.conv, **quant_params)

    def forward(self, x):
        """
        前向传播方法。
        """
        if self.use_res_connect:
            x = x + self.conv(x)
            return self.quantize_activations(x)     # 对相加后的结果进行激活量化
        else:
            return self.conv(x)


class QuantizedMobileNetV2(QuantizedModel):
    """
    继承自`QuantizedModel`，用于量化MobileNetV2模型。
    """
    def __init__(self, model_fp, input_size=(1, 3, 224, 224), quant_setup=None, **quant_params):
        """
        Args:
            model_fp: 原始的MobileNetV2模型
            input_size: 输入尺寸
            quant_setup: 量化设置的类型，决定具体的量化策略
        """
        super().__init__(input_size)
        # 定义特殊的量化层
        specials = {InvertedResidual: QuantizedInvertedResidual}
        # quantize and copy parts from original model
        # 量化模型的特征提取部分
        quantize_input = quant_setup and quant_setup == "LSQ_paper"     # 判断是否使用LSQ_paper量化策略
        self.features = quantize_sequential(
            model_fp.features,
            tie_activation_quantizers=not quantize_input,
            specials=specials,
            **quant_params,
        )

        # 量化模型的分类器部分
        self.flattener = Flattener()
        self.classifier = quantize_model(model_fp.classifier, **quant_params)

        # 量化设置`FP_logits`表示不量化最后的全连接层
        if quant_setup == "FP_logits":
            print("Do not quantize output of FC layer")
            self.classifier[1].activation_quantizer = FP32Acts()
            # self.classifier.activation_quantizer = FP32Acts()  # no activation quantization of logits
        # 量化设置`fc4`表示全连接层使用4位量化，第一个卷积层使用8位量化
        elif quant_setup == "fc4":
            self.features[0][0].weight_quantizer.quantizer.n_bits = 8
            self.classifier[1].weight_quantizer.quantizer.n_bits = 4
        # 量化设置`fc4_dw8`表示全连接层使用4位量化，深度可分离卷积层使用8位量化，第一个卷积层使用8位量化
        elif quant_setup == "fc4_dw8":
            print("\n\n### fc4_dw8 setup ###\n\n")
            # FC layer in 4 bits, depth-wise separable once in 8 bit
            self.features[0][0].weight_quantizer.quantizer.n_bits = 8
            self.classifier[1].weight_quantizer.quantizer.n_bits = 4
            for name, module in self.named_modules():
                if isinstance(module, BNQConv) and module.groups == module.in_channels:
                    module.weight_quantizer.quantizer.n_bits = 8
                    print(f"Set layer {name} to 8 bits")
        # 量化设置`LSQ`表示第一个和最后一个卷积层使用8位量化，全连接层权重使用8位量化，分类器激活值不量化
        elif quant_setup == "LSQ":
            print("Set quantization to LSQ (first+last layer in 8 bits)")
            # Weights of the first layer
            self.features[0][0].weight_quantizer.quantizer.n_bits = 8
            # The quantizer of the last conv_layer layer (input to avgpool with tied quantizers)
            self.features[-2][0].activation_quantizer.quantizer.n_bits = 8
            # Weights of the last layer
            self.classifier[1].weight_quantizer.quantizer.n_bits = 8
            # no activation quantization of logits
            self.classifier[1].activation_quantizer = FP32Acts()
        # 量化设置`LSQ_paper`表示第一层卷积层使用8位量化，全连接层权重使用8位量化，不对特征提取层中的激活值进行量化
        elif quant_setup == "LSQ_paper":
            # Weights of the first layer
            self.features[0][0].activation_quantizer = FP32Acts()
            self.features[0][0].weight_quantizer.quantizer.n_bits = 8
            # Weights of the last layer
            self.classifier[1].weight_quantizer.quantizer.n_bits = 8
            self.classifier[1].activation_quantizer.quantizer.n_bits = 8
            # Set all QuantizedActivations to FP32
            for layer in self.features.modules():
                if isinstance(layer, QuantizedActivation):
                    layer.activation_quantizer = FP32Acts()
        # 其他量化设置
        elif quant_setup is not None and quant_setup != "all":
            raise ValueError(
                "Quantization setup '{}' not supported for MobilenetV2".format(quant_setup)
            )

    def forward(self, x):
        """
        前向传播方法。
        """
        x = self.features(x)
        x = self.flattener(x)
        x = self.classifier(x)

        return x


def mobilenetv2_quantized(pretrained=True, model_dir=None, load_type="fp32", **qparams):
    """
    用于创建和加载量化后的MobileNetV2模型。

    Args:
        pretrained: 是否使用预训练的模型，默认为True
        model_dir: 模型的路径
        load_type: 加载模型的类型，可以是"fp32"或"quantized"
    """
    # 初始化浮点模型实例
    fp_model = MobileNetV2()
    # 加载预训练的FP32模型
    if pretrained and load_type == "fp32":
        # Load model from pretrained FP32 weights
        assert os.path.exists(model_dir)
        print(f"Loading pretrained weights from {model_dir}")
        state_dict = torch.load(model_dir)
        fp_model.load_state_dict(state_dict)
        # 量化模型
        quant_model = QuantizedMobileNetV2(fp_model, **qparams)
    # 加载量化模型
    elif load_type == "quantized":
        # Load pretrained QuantizedModel
        print(f"Loading pretrained quantized model from {model_dir}")
        state_dict = torch.load(model_dir)
        quant_model = QuantizedMobileNetV2(fp_model, **qparams)
        quant_model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError("wrong load_type specified")

    return quant_model
