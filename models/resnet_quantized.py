#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models import resnet18, resnet50

from quantization.autoquant_utils import quantize_model, Flattener, QuantizedActivationWrapper
from quantization.base_quantized_classes import QuantizedActivation, FP32Acts
from quantization.base_quantized_model import QuantizedModel


class QuantizedBlock(QuantizedActivation):
    """
    对ResNet模型中的Basic块和Bottleneck块进行量化处理。
    """
    def __init__(self, block, **quant_params):
        """
        Args:
            block: 原始的Basic块或Bottleneck
        """
        super().__init__(**quant_params)

        # 特征提取层的定义
        if isinstance(block, Bottleneck):
            features = nn.Sequential(
                block.conv1,
                block.bn1,
                block.relu,
                block.conv2,
                block.bn2,
                block.relu,
                block.conv3,
                block.bn3,
            )
        elif isinstance(block, BasicBlock):
            features = nn.Sequential(block.conv1, block.bn1, block.relu, block.conv2, block.bn2)

        # 量化特征提取层
        self.features = quantize_model(features, **quant_params)
        # 量化下采样层
        self.downsample = (
            quantize_model(block.downsample, **quant_params) if block.downsample else None
        )

        self.relu = block.relu

    def forward(self, x):
        """
        前向传播方法。
        """
        residual = x if self.downsample is None else self.downsample(x)
        out = self.features(x)

        out += residual
        out = self.relu(out)

        return self.quantize_activations(out)   # 对激活输出进行量化


class QuantizedResNet(QuantizedModel):
    """
    对整个ResNet模型进行量化处理。
    """
    def __init__(self, resnet, input_size=(1, 3, 224, 224), quant_setup=None, **quant_params):
        """
        Args:
            resnet: 原始的ResNet模型
            input_size: 输入数据的大小
            quant_setup: 量化设置
            quant_params: 量化参数
        """
        super().__init__(input_size)
        # 定义特殊的量化块
        specials = {BasicBlock: QuantizedBlock, Bottleneck: QuantizedBlock}

        # 特征提取层的定义
        if hasattr(resnet, "maxpool"):
            # ImageNet ResNet case
            features = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )
        else:
            # Tiny ImageNet ResNet case
            features = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )

        # 量化特征提取层
        self.features = quantize_model(features, specials=specials, **quant_params)

        # 池化层的定义
        if quant_setup and quant_setup == "LSQ_paper":
            # Keep avgpool intact as we quantize the input the last layer
            self.avgpool = resnet.avgpool
        else:
            # 包装avgpool层，以实现激活量化
            self.avgpool = QuantizedActivationWrapper(
                resnet.avgpool,
                tie_activation_quantizers=True,
                input_quantizer=self.features[-1][-1].activation_quantizer,
                **quant_params,
            )
        # 扁平化层的定义
        self.flattener = Flattener()
        # 量化全连接层
        self.fc = quantize_model(resnet.fc, **quant_params)

        # Adapt to specific quantization setup
        # 量化设置`LSQ`：第一层卷积层权重量化为8位，最后一个卷积层的激活量化为8位，全连接层权重量化为8位，全连接层激活不量化
        if quant_setup == "LSQ":
            print("Set quantization to LSQ (first+last layer in 8 bits)")
            # Weights of the first layer
            self.features[0].weight_quantizer.quantizer.n_bits = 8
            # The quantizer of the residual (input to last layer)
            self.features[-1][-1].activation_quantizer.quantizer.n_bits = 8
            # Output of the last conv (input to last layer)
            self.features[-1][-1].features[-1].activation_quantizer.quantizer.n_bits = 8
            # Weights of the last layer
            self.fc.weight_quantizer.quantizer.n_bits = 8
            # no activation quantization of logits
            self.fc.activation_quantizer = FP32Acts()
        # 量化设置`LSQ_paper`：第一层卷积层权重量化为8位，第一个卷积层的激活不量化，全连接层权重量化为8位，全连接层激活量化为8位，特征提取层的激活不量化
        elif quant_setup == "LSQ_paper":
            # Weights of the first layer
            self.features[0].activation_quantizer = FP32Acts()
            self.features[0].weight_quantizer.quantizer.n_bits = 8
            # Weights of the last layer
            self.fc.activation_quantizer.quantizer.n_bits = 8
            self.fc.weight_quantizer.quantizer.n_bits = 8
            # Set all QuantizedActivations to FP32
            for layer in self.features.modules():
                if isinstance(layer, QuantizedActivation):
                    layer.activation_quantizer = FP32Acts()
        # 量化设置`FP_logits`：全连接层的激活不量化
        elif quant_setup == "FP_logits":
            print("Do not quantize output of FC layer")
            self.fc.activation_quantizer = FP32Acts()  # no activation quantization of logits
        # 量化设置`fc4`：第一个卷积层权重量化为8位，全连接层权重量化为4位
        elif quant_setup == "fc4":
            self.features[0].weight_quantizer.quantizer.n_bits = 8
            self.fc.weight_quantizer.quantizer.n_bits = 4
        # 其他量化设置
        elif quant_setup is not None and quant_setup != "all":
            raise ValueError("Quantization setup '{}' not supported for Resnet".format(quant_setup))

    def forward(self, x):
        """
        前向传播方法。
        """
        x = self.features(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.flattener(x)
        x = self.fc(x)

        return x


def resnet18_quantized(pretrained=True, model_dir=None, load_type="fp32", **qparams):
    """
    创建和加载量化后的ResNet-18模型。

    Args:
        pretrained: 是否加载预训练模型
        model_dir: 模型路径
        load_type: 加载类型
        qparams: 量化参数
    """
    # 加载FP32模型并量化
    if load_type == "fp32":
        # Load model from pretrained FP32 weights
        fp_model = resnet18(pretrained=pretrained)
        quant_model = QuantizedResNet(fp_model, **qparams)
    # 加载预训练的量化模型
    elif load_type == "quantized":
        # Load pretrained QuantizedModel
        print(f"Loading pretrained quantized model from {model_dir}")
        state_dict = torch.load(model_dir)
        fp_model = resnet18()
        quant_model = QuantizedResNet(fp_model, **qparams)
        quant_model.load_state_dict(state_dict)
    else:
        raise ValueError("wrong load_type specified")
    return quant_model


def resnet50_quantized(pretrained=True, model_dir=None, load_type="fp32", **qparams):
    """
    创建和加载量化后的ResNet-50模型。

    Args:
        pretrained: 是否加载预训练模型
        model_dir: 模型路径
        load_type: 加载类型
        qparams: 量化参数
    """
    # 加载FP32模型并量化
    if load_type == "fp32":
        # Load model from pretrained FP32 weights
        fp_model = resnet50(pretrained=pretrained)
        quant_model = QuantizedResNet(fp_model, **qparams)
    # 加载预训练的量化模型
    elif load_type == "quantized":
        # Load pretrained QuantizedModel
        print(f"Loading pretrained quantized model from {model_dir}")
        state_dict = torch.load(model_dir)
        fp_model = resnet50()
        quant_model = QuantizedResNet(fp_model, **qparams)
        quant_model.load_state_dict(state_dict)
    else:
        raise ValueError("wrong load_type specified")
    return quant_model
