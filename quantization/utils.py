#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.


import torch
import torch.serialization

from quantization.quantizers import QuantizerBase
from quantization.quantizers.rounding_utils import ParametrizedGradEstimatorBase
from quantization.range_estimators import RangeEstimators
from utils import StopForwardException, get_layer_by_name


def separate_quantized_model_params(quant_model):
    """
    将量化模型的参数分离成四类：
    1. 量化参数（Quantization Parameters）：例如，delta 和 zero_float 等量化相关参数。
    2. 模型参数（Model Parameters）：不包含任何量化操作的基础模型参数。
    3. 梯度估计器参数（Gradient Estimator Parameters）：存在于梯度估计器（`ParametrizedGradEstimatorBase`）中的参数。
    通过这种分离，用户可以对不同类别的参数进行独立的优化或冻结，从而在量化感知训练（QAT）过程中更好地控制模型行为。

    This method separates the parameters of the quantized model to 4 categories.
    Parameters
    ----------
    quant_model:      (QuantizedModel)

    Returns
    -------
    quant_params:       (list)
        Quantization parameters, e.g. delta and zero_float
    model_params:    (list)
        The model parameters of the base model without any quantization operations
    grad_params:        (list)
        Parameters found in the gradient estimators (ParametrizedGradEstimatorBase)
    -------

    """
    # 初始化参数列表
    quant_params, grad_params = [], []
    quant_params_names, grad_params_names = [], []
    
    for mod_name, module in quant_model.named_modules():    # 遍历所有模块
        if isinstance(module, QuantizerBase):               # 如果模块是量化器
            for name, param in module.named_parameters(recurse=False):  # 遍历模块的参数，添加到量化参数列表中
                quant_params.append(param)
                quant_params_names.append(".".join((mod_name, name)))
        if isinstance(module, ParametrizedGradEstimatorBase):   # 如果模块是梯度估计器
            # gradient estimator params
            for name, param in module.named_parameters(recurse=False):  # 遍历模块的参数，添加到梯度估计器参数列表中
                grad_params.append(param)
                grad_params_names.append(".".join((mod_name, name)))

    def tensor_in_list(tensor, lst):
        """
        辅助函数，检查一个张量是否存在于给定的列表中。
        
        Args:
            tensor: 待检查的张量
            lst: 给定的列表
        """
        return any([e is tensor for e in lst])

    # 组合已找到的参数
    found_params = quant_params + grad_params

    # 筛选基础模型参数
    # 遍历模型的所有参数，将不在量化参数和梯度估计器参数中的参数添加到基础模型参数列表中
    model_params = [p for p in quant_model.parameters() if not tensor_in_list(p, found_params)] 
    model_param_names = [
        n for n, p in quant_model.named_parameters() if not tensor_in_list(p, found_params)
    ]

    print("Quantization parameters ({}):".format(len(quant_params_names)))
    print(quant_params_names)

    print("Gradient estimator parameters ({}):".format(len(grad_params_names)))
    print(grad_params_names)

    print("Other model parameters ({}):".format(len(model_param_names)))
    print(model_param_names)

    assert len(model_params + quant_params + grad_params) == len(
        list(quant_model.parameters())
    ), "{}; {}; {} -- {}".format(
        len(model_params), len(quant_params), len(grad_params), len(list(quant_model.parameters()))
    )

    return quant_params, model_params, grad_params


def pass_data_for_range_estimation(
    loader, model, act_quant, weight_quant, max_num_batches=20, cross_entropy_layer=None, inp_idx=0
):
    """
    通过传递训练数据来估计量化范围（最小值和最大值）。
    它在模型处于评估模式时运行，以确保批标准化（BN）等层的均值和方差不会被更新。
    该函数还支持指定特定层的交叉熵估计器，以增强范围估计的准确性。
    
    Args:
        loader: 训练数据加载器
        model: 量化模型
        act_quant: 激活量化器
        weight_quant: 权重量化器
        max_num_batches: 最大批次数
        cross_entropy_layer: 指定的交叉熵层名称
        inp_idx: 输入索引
    """
    print("\nEstimate quantization ranges on training data")
    # 设置量化状态
    model.set_quant_state(weight_quant, act_quant)
    # Put model in eval such that BN EMA does not get updated
    model.eval()

    # 设置交叉熵估计器
    if cross_entropy_layer is not None:
        layer_xent = get_layer_by_name(model, cross_entropy_layer)      # 获取指定的交叉熵层
        if layer_xent:
            print('Set cross entropy estimator for layer "{}"'.format(cross_entropy_layer))
            act_quant_mgr = layer_xent.activation_quantizer             # 获取激活量化器
            act_quant_mgr.range_estimator = RangeEstimators.cross_entropy.cls(  # 将该量化管理器的范围估计器设置为交叉熵范围估计器
                per_channel=act_quant_mgr.per_channel,
                quantizer=act_quant_mgr.quantizer,
                **act_quant_mgr.range_estim_params,
            )
        else:
            raise ValueError("Cross-entropy layer not found")

    # 初始化处理过的批次数据
    batches = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for i, data in enumerate(loader):                   # 遍历数据加载器
            try:
                if isinstance(data, (tuple, list)):
                    x = data[inp_idx].to(device=device)     # 提取指定索引的输入数据
                    batches.append(x.data.cpu().numpy())    # 添加到处理过的批次数据中
                    model(x)                                # 前向传播，以更新量化范围估计器的状态
                    print(f"proccesed step={i}")
                else:
                    x = {k: v.to(device=device) for k, v in data.items()}
                    model(**x)                              # 前向传播，以更新量化范围估计器的状态
                    print(f"proccesed step={i}")

                if i >= max_num_batches - 1 or not act_quant:
                    break
            except StopForwardException:
                pass
        return batches


def set_range_estimators(config, model):
    """
    配置量化模型中的范围估计器和梯度缩放设置。
    它确保量化器可以学习其范围参数，并根据配置启用或禁用梯度缩放，以优化量化感知训练过程。
    
    Args:
        config: 配置对象，包含量化感知训练（QAT）的配置参数
        model: 量化模型
    """
    # 使量化器的范围参数可学习
    print("Make quantizers learnable")
    model.learn_ranges()

    # 启用或禁用梯度缩放
    if config.qat.grad_scaling:
        print("Activate gradient scaling")
        model.grad_scaling(True)

    # Ensure we have the desired quant state
    # 设置量化状态
    model.set_quant_state(config.quant.weight_quant, config.quant.act_quant)
