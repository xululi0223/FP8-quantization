#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
import torch

from quantization.quantized_folded_bn import BNFusedHijacker
from utils.imagenet_dataloaders import ImageNetDataLoaders


def get_dataloaders_and_model(config, load_type="fp32", **qparams):
    """
    用于根据给定的配置加载数据集和模型。
    这包括创建数据加载器（dataloaders）以便于训练和验证，以及初始化指定的模型架构。
    函数支持加载预训练模型，并将模型移动到CUDA设备（如果配置中指定）。
    
    Args:
        config: 包含模型和数据加载配置的对象。
        load_type: 模型加载类型，默认值为"fp32"，表示使用32位浮点数。
    """
    # 创建数据加载器
    dataloaders = ImageNetDataLoaders(
        config.base.images_dir,
        224,
        config.base.batch_size,
        config.base.num_workers,
        config.base.interpolation,
    )

    # 初始化模型
    model = config.base.architecture(
        pretrained=config.base.pretrained,
        load_type=load_type,
        model_dir=config.base.model_dir,
        **qparams,
    )
    # 将模型移动到CUDA设备
    if config.base.cuda:
        model = model.cuda()

    return dataloaders, model


class ReestimateBNStats:
    """
    用于在训练过程中重新估计模型中批量归一化（Batch Normalization, BN）层的统计量（如均值和方差）。
    这是量化感知训练（Quantization-Aware Training, QAT）中的一个关键步骤，确保BN层的统计量能够适应新的量化配置，从而提高模型的量化性能。
    """
    def __init__(self, model, data_loader, num_batches=50):
        """
        Args:
            model: 要重新估计BN统计量的模型。
            data_loader: 用于重新估计BN统计量的数据加载器。
            num_batches: 用于重新估计BN统计量的批次数。
        """
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.num_batches = num_batches

    def __call__(self, engine):
        """
        调用方法，用于重新估计模型中的BN统计量。
        """
        print("-- Reestimate current BN statistics --")
        reestimate_BN_stats(self.model, self.data_loader, self.num_batches)


def reestimate_BN_stats(model, data_loader, num_batches=50, store_ema_stats=False):
    """
    用于实际执行BN层统计量的重新估计算法。
    它通过遍历模型中的BN层，调整其动量参数，并利用指定数量的数据批次来计算新的运行均值和方差。
    这一过程确保BN层的统计量能够精准反映当前模型在量化配置下的表现。
    
    Args:
        model: 要重新估计BN统计量的模型。
        data_loader: 用于重新估计BN统计量的数据加载器。
        num_batches: 用于重新估计BN统计量的批次数。
        store_ema_stats: 是否存储指数移动平均（Exponential Moving Average, EMA）统计量，默认值为False。
    """
    # We set BN momentum to 1 an use train mode
    # -> the running mean/var have the current batch statistics
    model.eval()
    # 初始化原始动量字典
    org_momentum = {}
    # 遍历模型中的所有模块
    for name, module in model.named_modules():
        # 判断是否为BNFusedHijacker模块
        if isinstance(module, BNFusedHijacker):
            org_momentum[name] = module.momentum    # 保存原始动量值
            module.momentum = 1.0                   # 设置动量值为1
            # 初始化运行均值和方差的累加器
            module.running_mean_sum = torch.zeros_like(module.running_mean)
            module.running_var_sum = torch.zeros_like(module.running_var)
            # Set all BNFusedHijacker modules to train mode for but not its children
            module.training = True

            # 存储指数移动平均（EMA）统计量
            if store_ema_stats:
                # Save the original EMA, make sure they are in buffers so they end in the state dict
                if not hasattr(module, "running_mean_ema"):
                    module.register_buffer("running_mean_ema", copy.deepcopy(module.running_mean))
                    module.register_buffer("running_var_ema", copy.deepcopy(module.running_var))
                else:
                    module.running_mean_ema = copy.deepcopy(module.running_mean)
                    module.running_var_ema = copy.deepcopy(module.running_var)

    # Run data for estimation
    device = next(model.parameters()).device
    # 初始化批次计数器
    batch_count = 0
    # 重新估计BN统计量
    with torch.no_grad():
        # 遍历数据加载器
        for x, y in data_loader:
            model(x.to(device))     # 前向传播
            # We save the running mean/var to a buffer
            # 累加BN层的统计量
            for name, module in model.named_modules():
                if isinstance(module, BNFusedHijacker):
                    module.running_mean_sum += module.running_mean
                    module.running_var_sum += module.running_var

            batch_count += 1
            if batch_count == num_batches:
                break
    # At the end we normalize the buffer and write it into the running mean/var
    # 计算平均统计量并恢复BN层的原始动量
    for name, module in model.named_modules():
        if isinstance(module, BNFusedHijacker):
            module.running_mean = module.running_mean_sum / batch_count
            module.running_var = module.running_var_sum / batch_count
            # We reset the momentum in case it would be used anywhere else
            module.momentum = org_momentum[name]
    model.eval()
