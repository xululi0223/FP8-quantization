#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
import warnings

from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.pooling import _AdaptiveAvgPoolNd, _AvgPoolNd


from quantization.base_quantized_classes import QuantizedActivation, QuantizedModule
from quantization.hijacker import QuantizationHijacker, activations_set
from quantization.quantization_manager import QuantizationManager
from quantization.quantized_folded_bn import BNFusedHijacker


class QuantConv1d(QuantizationHijacker, nn.Conv1d):
    """
    `QuantConv1d` 类结合了 `QuantizationHijacker` 和 `nn.Conv1d`，用于一维卷积层的量化。
    该类重写了前向传播方法，以支持权重量化。
    """
    def run_forward(self, x, weight, bias, offsets=None):
        """
        重写的前向传播方法。
        """
        return F.conv1d(
            x.contiguous(),
            weight.contiguous(),
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class QuantConv(QuantizationHijacker, nn.Conv2d):
    """
    `QuantConv` 类结合了 `QuantizationHijacker` 和 `nn.Conv2d`，用于二维卷积层的量化。
    该类重写了前向传播方法，以支持权重量化。
    """
    def run_forward(self, x, weight, bias, offsets=None):
        """
        重写的前向传播方法。
        """
        return F.conv2d(
            x.contiguous(),
            weight.contiguous(),
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class QuantConvTransposeBase(QuantizationHijacker):
    """
    QuantConvTransposeBase 类是转置卷积层的基类，结合了 QuantizationHijacker。
    它提供了权重量化的通用方法，支持按通道量化和全局量化。
    """
    def quantize_weights(self, weights):
        """
        定义权重量化方法。
        """
        # 按通道量化，权重维度转置
        if self.per_channel_weights:
            # NOTE: ND tranpose conv weights are stored as (in_channels, out_channels, *)
            # instead of (out_channels, in_channels, *) for convs
            # and per-channel quantization should be applied to out channels
            # transposing before passing to quantizer is trick to avoid
            # changing logic in range estimators and quantizers
            weights = weights.transpose(1, 0).contiguous()
        # 对权重进行量化
        weights = self.weight_quantizer(weights)
        # 按通道量化，恢复原始权重维度
        if self.per_channel_weights:
            weights = weights.transpose(1, 0).contiguous()
        return weights


class QuantConvTranspose1d(QuantConvTransposeBase, nn.ConvTranspose1d):
    """
    QuantConvTranspose1d 类结合了 QuantConvTransposeBase 和 nn.ConvTranspose1d，用于一维转置卷积层的量化。
    它实现了前向传播方法以支持一维转置卷积的量化。
    """
    def run_forward(self, x, weight, bias, offsets=None):
        """
        重写的前向传播方法。
        """
        return F.conv_transpose1d(
            x.contiguous(),
            weight.contiguous(),
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class QuantConvTranspose(QuantConvTransposeBase, nn.ConvTranspose2d):
    """
    QuantConvTranspose 类结合了 QuantConvTransposeBase 和 nn.ConvTranspose2d，用于二维转置卷积层的量化。
    它实现了前向传播方法以支持二维转置卷积的量化。
    """
    def run_forward(self, x, weight, bias, offsets=None):
        """
        重写的前向传播方法。
        """
        return F.conv_transpose2d(
            x.contiguous(),
            weight.contiguous(),
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class QuantLinear(QuantizationHijacker, nn.Linear):
    """
    `QuantLinear` 类结合了 `QuantizationHijacker` 和 `nn.Linear`，用于全连接层的量化。
    该类重写了前向传播方法，以支持权重量化。
    """
    def run_forward(self, x, weight, bias, offsets=None):
        """
        重写的前向传播方法。
        """
        return F.linear(x.contiguous(), weight.contiguous(), bias=bias)


class BNQConv1d(BNFusedHijacker, nn.Conv1d):
    """
    BNQConv1d 类结合了 BNFusedHijacker 和 nn.Conv1d，用于一维卷积层与批量归一化层的融合量化。
    它重写了前向传播方法，以支持一维卷积的融合BN层量化。
    """
    def run_forward(self, x, weight, bias, offsets=None):
        """
        重写的前向传播方法。
        """
        return F.conv1d(
            x.contiguous(),
            weight.contiguous(),
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class BNQConv(BNFusedHijacker, nn.Conv2d):
    """
    BNQConv 类结合了 BNFusedHijacker 和 nn.Conv2d，用于二维卷积层与批量归一化层的融合量化。
    它重写了前向传播方法，以支持二维卷积的融合BN层量化。
    """
    def run_forward(self, x, weight, bias, offsets=None):
        """
        重写的前向传播方法。
        """
        return F.conv2d(
            x.contiguous(),
            weight.contiguous(),
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class BNQLinear(BNFusedHijacker, nn.Linear):
    """
    BNQLinear 类结合了 BNFusedHijacker 和 nn.Linear，用于全连接层与批量归一化层的融合量化。
    """
    def run_forward(self, x, weight, bias, offsets=None):
        """
        重写的前向传播方法。
        """
        return F.linear(x.contiguous(), weight.contiguous(), bias=bias)


class QuantizedActivationWrapper(QuantizedActivation):
    """
    继承自 QuantizedActivation，用于包装一个层并对其激活进行量化。
    它支持将输入和输出量化器绑定在一起，适用于如平均池化等层。
    Wraps over a layer and quantized the activation.
    It also allow for tying the input and output quantizer which is helpful
    for layers such Average Pooling
    """

    def __init__(
        self,
        layer,
        tie_activation_quantizers=False,
        input_quantizer: QuantizationManager = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            layer: 要包装的层。
            tie_activation_quantizers: 是否绑定输入和输出量化器，默认为 False。
            input_quantizer: 输入量化器，默认为 None。
        """
        super().__init__(*args, **kwargs)
        self.tie_activation_quantizers = tie_activation_quantizers
        # 设置输入量化器
        if input_quantizer:
            assert isinstance(input_quantizer, QuantizationManager)
            self.activation_quantizer = input_quantizer
        self.layer = layer

    def quantize_activations_no_range_update(self, x):
        """
        不更新量化范围的激活量化方法。
        """
        if self._quant_a:
            return self.activation_quantizer.quantizer(x)
        else:
            return x

    def forward(self, x):
        """
        前向传播方法。
        """
        # 执行包装层的前向传播
        x = self.layer(x)
        
        # 如果绑定输入和输出量化器，则执行量化激活
        if self.tie_activation_quantizers:
            # The input activation quantizer is used to quantize the activation
            # but without updating the quantization range
            return self.quantize_activations_no_range_update(x)
        else:
            return self.quantize_activations(x)

    def extra_repr(self):
        """
        额外的字符串表示方法。
        """
        return f"tie_activation_quantizers={self.tie_activation_quantizers}"


class QuantLayerNorm(QuantizationHijacker, nn.LayerNorm):
    """
    `QuantLayerNorm` 类结合了 `QuantizationHijacker` 和 `nn.LayerNorm`，用于 LayerNorm 层的量化。
    该类重写了前向传播方法，以支持 LayerNorm 层的量化。
    """
    def run_forward(self, x, weight, bias, offsets=None):
        """
        重写的前向传播方法。
        """
        return F.layer_norm(
            input=x.contiguous(),
            normalized_shape=self.normalized_shape,
            weight=weight.contiguous(),
            bias=bias.contiguous(),
            eps=self.eps,
        )


class Flattener(nn.Module):
    """
    将输入张量展平成二维张量，通常用于连接卷积层和全连接层。
    """
    def forward(self, x):
        return x.view(x.shape[0], -1)


# Non BN Quant Modules Map
non_bn_module_map = {
    nn.Conv1d: QuantConv1d,
    nn.Conv2d: QuantConv,
    nn.ConvTranspose1d: QuantConvTranspose1d,
    nn.ConvTranspose2d: QuantConvTranspose,
    nn.Linear: QuantLinear,
    nn.LayerNorm: QuantLayerNorm,
}

non_param_modules = (_AdaptiveAvgPoolNd, _AvgPoolNd)
# BN Quant Modules Map
bn_module_map = {nn.Conv1d: BNQConv1d, nn.Conv2d: BNQConv, nn.Linear: BNQLinear}

quant_conv_modules = (QuantConv1d, QuantConv, BNQConv1d, BNQConv)


def next_bn(module, i):
    """
    用于检查在给定模块列表中，索引 `i` 后是否存在批量归一化（Batch Normalization, BN）层。
    它用于在量化过程中判断是否需要将卷积层与BN层进行融合。

    Args:
        module: 模块列表
        i: 当前模块的索引
    """
    return len(module) > i + 1 and isinstance(module[i + 1], (nn.BatchNorm2d, nn.BatchNorm1d))


def get_act(module, i):
    """
    用于在模块列表中，从当前索引 i 开始，查找与卷积层（或其他支持的层）关联的激活函数。
    它返回找到的激活函数及其索引位置。
    如果未找到，则返回 (None, None)。
    
    Args:
        module: 模块列表
        i: 当前模块的索引
    """
    # Case 1: conv + act
    if len(module) - i > 1 and isinstance(module[i + 1], tuple(activations_set)):
        return module[i + 1], i + 1

    # Case 2: conv + bn + act
    if (
        len(module) - i > 2
        and next_bn(module, i)
        and isinstance(module[i + 2], tuple(activations_set))
    ):
        return module[i + 2], i + 2

    # Case 3: conv + bn + X -> return false
    # Case 4: conv + X -> return false
    return None, None


def get_conv_args(module):
    """
    用于从卷积层模块中提取必要的参数，并以字典形式返回。这些参数用于创建量化后的卷积层实例。
    """
    # 提取卷积层参数
    args = dict(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
    )
    # 处理转置卷积层的特殊参数
    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d)):
        args["output_padding"] = module.output_padding
    return args


def get_linear_args(module):
    """
    用于从全连接层模块中提取必要的参数，并以字典形式返回。这些参数用于创建量化后的全连接层实例。
    """
    args = dict(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
    )
    return args


def get_layernorm_args(module):
    """
    用于从 LayerNorm 层模块中提取必要的参数，并以字典形式返回。这些参数用于创建量化后的 LayerNorm 层实例。
    """
    args = dict(normalized_shape=module.normalized_shape, eps=module.eps)
    return args


def get_module_args(mod, act):
    """
    根据模块类型（卷积层、线性层、层归一化层）调用相应的参数提取函数，并添加激活函数参数。
    它返回包含所有必要参数的字典，用于创建量化后的模块实例。
    
    Args:
        mod: 待处理的模块
        act: 关联的激活函数
    """
    if isinstance(mod, _ConvNd):
        kwargs = get_conv_args(mod)
    elif isinstance(mod, nn.Linear):
        kwargs = get_linear_args(mod)
    elif isinstance(mod, nn.LayerNorm):
        kwargs = get_layernorm_args(mod)
    else:
        raise ValueError

    kwargs["activation"] = act

    return kwargs


def fold_bn(module, i, **quant_params):
    """
    将卷积层（或线性层）与其后紧跟的BN层融合为一个量化后的联合模块。
    它结合了权重量化和BN层的参数，优化模型的推理性能。
    
    Args:
        module: 模块列表
        i: 当前模块的索引
    """
    # 检查是否存在BN层
    bn = next_bn(module, i)
    # 获取激活函数及其索引
    act, act_idx = get_act(module, i)
    # 根据是否存在BN层，选择对应的模块映射表
    modmap = bn_module_map if bn else non_bn_module_map
    # 确定新模块的类型
    modtype = modmap[type(module[i])]

    # 获取新模块的参数
    kwargs = get_module_args(module[i], act)
    # 创建新模块实例
    new_module = modtype(**kwargs, **quant_params)
    # 复制权重数据
    new_module.weight.data = module[i].weight.data.clone()

    # 如果后续存在BN层，则复制BN层参数
    if bn:
        new_module.gamma.data = module[i + 1].weight.data.clone()
        new_module.beta.data = module[i + 1].bias.data.clone()
        new_module.running_mean.data = module[i + 1].running_mean.data.clone()
        new_module.running_var.data = module[i + 1].running_var.data.clone()
        # 如果卷积层存在偏置，则调整BN层的running_mean
        if module[i].bias is not None:
            new_module.running_mean.data -= module[i].bias.data
            print("Warning: bias in conv/linear before batch normalization.")
        new_module.epsilon = module[i + 1].eps

    # 处理无BN层，但存在偏置项的情况
    elif module[i].bias is not None:
        new_module.bias.data = module[i].bias.data.clone()

    # 返回新模块及其索引
    return new_module, i + int(bool(act)) + int(bn) + 1


def quantize_sequential(model, specials=None, tie_activation_quantizers=False, **quant_params):
    """
    量化一个 nn.Sequential 模型。
    它遍历模型中的各个子模块，应用相应的量化逻辑，将支持的模块替换为量化后的模块，并处理特殊情况（如激活函数绑定、BN层融合等）。
    
    Args:
        model: 待量化的模型
        specials: 特殊模块映射表，默认为 None
        tie_activation_quantizers: 是否绑定输入和输出量化器，默认为 False
        quant_params: 量化参数
    """
    # 初始化特殊模块映射表
    specials = specials or dict()

    # 初始化索引和量化模块列表
    i = 0
    quant_modules = []
    
    # 遍历模型中的各个子模块
    while i < len(model):
        # 处理已量化的模块
        if isinstance(model[i], QuantizedModule):
            quant_modules.append(model[i])
        # 处理支持BN层融合的模块
        elif type(model[i]) in non_bn_module_map:
            new_module, new_i = fold_bn(model, i, **quant_params)
            quant_modules.append(new_module)
            i = new_i
            continue

        # 处理特殊模块
        elif type(model[i]) in specials:
            quant_modules.append(specials[type(model[i])](model[i], **quant_params))

        # 处理非参数模块（如池化层）
        elif isinstance(model[i], non_param_modules):
            # Check for last quantizer
            input_quantizer = None
            # 检查前一个模块是否为量化模块，获取输入量化器
            if quant_modules and isinstance(quant_modules[-1], QuantizedModule):
                last_layer = quant_modules[-1]
                input_quantizer = quant_modules[-1].activation_quantizer
            elif (
                quant_modules
                and isinstance(quant_modules[-1], nn.Sequential)
                and isinstance(quant_modules[-1][-1], QuantizedModule)
            ):
                last_layer = quant_modules[-1][-1]
                input_quantizer = quant_modules[-1][-1].activation_quantizer

            # 根据tie_activation_quantizers参数，绑定输入和输出量化器
            if input_quantizer and tie_activation_quantizers:
                # If input quantizer is found the tie input/output act quantizers
                print(
                    f"Tying input quantizer {i-1}^th layer of type {type(last_layer)} to the "
                    f"quantized {type(model[i])} following it"
                )
                quant_modules.append(
                    QuantizedActivationWrapper(
                        model[i],
                        tie_activation_quantizers=tie_activation_quantizers,
                        input_quantizer=input_quantizer,
                        **quant_params,
                    )
                )
            else:
                # Input quantizer not found
                quant_modules.append(QuantizedActivationWrapper(model[i], **quant_params))
                if tie_activation_quantizers:
                    warnings.warn("Input quantizer not found, so we do not tie quantizers")
        # 处理其他模块
        else:
            quant_modules.append(quantize_model(model[i], specials=specials, **quant_params))   # 递归调用quantize_model函数
        i += 1
    return nn.Sequential(*quant_modules)


def quantize_model(model, specials=None, tie_activation_quantizers=False, **quant_params):
    """
    量化一个PyTorch模型。
    它根据模型的类型（如 nn.Sequential、支持的模块、特殊模块等）选择相应的量化策略，并返回量化后的模型。
    若模块类型不受支持，则递归地量化其子模块。
    
    Args:
        model: 待量化的模型
        specials: 特殊模块映射表，默认为 None
        tie_activation_quantizers: 是否绑定输入和输出量化器，默认为 False
        quant_params: 量化参数
    """
    # 初始化特殊模块映射表
    specials = specials or dict()

    # 处理 nn.Sequential 模型
    if isinstance(model, nn.Sequential):
        quant_model = quantize_sequential(
            model, specials, tie_activation_quantizers, **quant_params
        )

    # 处理特殊模块
    elif type(model) in specials:
        quant_model = specials[type(model)](model, **quant_params)

    # 处理非参数模块（如池化层）
    elif isinstance(model, non_param_modules):
        quant_model = QuantizedActivationWrapper(model, **quant_params)

    # 处理支持BN层融合的模块
    elif type(model) in non_bn_module_map:
        # If we do isinstance() then we might run into issues with modules that inherit from
        # one of these classes, for whatever reason
        modtype = non_bn_module_map[type(model)]
        kwargs = get_module_args(model, None)
        quant_model = modtype(**kwargs, **quant_params)

        quant_model.weight.data = model.weight.data
        if getattr(model, "bias", None) is not None:
            quant_model.bias.data = model.bias.data

    # 处理未知类型的模块，递归量化其子模块
    else:
        # Unknown type, try to quantize all child modules
        quant_model = copy.deepcopy(model)
        for name, module in quant_model._modules.items():
            new_model = quantize_model(module, specials=specials, **quant_params)
            if new_model is not None:
                setattr(quant_model, name, new_model)

    return quant_model
