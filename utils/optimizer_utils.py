#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch


def get_lr_scheduler(optimizer, lr_schedule, epochs):
    """
    根据提供的学习率调度策略字符串，创建并返回相应的学习率调度器实例。
    支持的调度策略有：`multistep`和`cosine`。
    
    Args:
        optimizer: 优化器实例
        lr_schedule: 学习率调度策略字符串
        epochs: 总训练周期数，仅用于余弦退火调度
    """
    scheduler = None
    if lr_schedule:                                                                 # 如果提供了学习率调度策略
        if lr_schedule.startswith("multistep"):                                     # 如果调度策略是多步调度
            epochs = [int(s) for s in lr_schedule.split(":")[1:]]                   # 解析调度策略字符串，提取多步调度的周期数
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epochs)
        elif lr_schedule.startswith("cosine"):                                      # 如果调度策略是余弦退火调度
            eta_min = float(lr_schedule.split(":")[1])                              # 解析调度策略字符串，提取余弦退火的最小学习率
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, epochs, eta_min=eta_min
            )
    return scheduler


def optimizer_lr_factory(config_optim, params, epochs):
    """
    根据提供的优化器配置参数，创建并返回相应的优化器实例及其学习率调度器。
    支持的优化器包括：`SGD`和`Adam`。
    
    Args:
        config_optim: 优化器配置参数，包含优化器类型和相关参数
        params: 待优化的模型参数
        epochs: 总训练周期数
    """
    if config_optim.optimizer.lower() == "sgd":                                     # 如果优化器是SGD
        optimizer = torch.optim.SGD(                                                # 创建SGD优化器实例
            params,
            lr=config_optim.learning_rate,
            momentum=config_optim.momentum,
            weight_decay=config_optim.weight_decay,
        )
    elif config_optim.optimizer.lower() == "adam":                                  # 如果优化器是Adam
        optimizer = torch.optim.Adam(                                               # 创建Adam优化器实例
            params, lr=config_optim.learning_rate, weight_decay=config_optim.weight_decay
        )
    else:
        raise ValueError()

    # 创建并返回学习率调度器实例
    lr_scheduler = get_lr_scheduler(optimizer, config_optim.learning_rate_schedule, epochs)

    return optimizer, lr_scheduler
