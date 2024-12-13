# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
import logging
import os

import click
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, TopKCategoricalAccuracy, Loss
from torch.nn import CrossEntropyLoss

from quantization.utils import pass_data_for_range_estimation
from utils import DotDict
from utils.click_options import (
    qat_options,
    quantization_options,
    fp8_options,
    quant_params_dict,
    base_options,
)
from utils.qat_utils import get_dataloaders_and_model, ReestimateBNStats


class Config(DotDict):
    """
    继承自 `DotDict`，用于存储和管理配置参数。
    通过继承 `DotDict`，`Config` 实例可以像字典一样访问其属性，同时支持通过点操作符访问（例如 `config.quant`）。
    """
    pass


@click.group()
def fp8_cmd_group():
    """
    Click的命令组，用于组织多个子命令。
    它主要负责初始化日志配置，并作为顶级命令组的容器。
    """
    # 配置全局日志记录的基本设置
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# 用于将配置参数传递给子命令的装饰器，确保每个命令函数都能访问到共享的配置对象
pass_config = click.make_pass_decorator(Config, ensure=True)

# 注册为Click的子命令
@fp8_cmd_group.command()
# 传递配置对象
@pass_config
# 统一管理和配置与基础设置、FP8量化、量化选项以及QAT相关的参数
@base_options
@fp8_options
@quantization_options
@qat_options
# 指定加载的模型类型
@click.option(
    "--load-type",
    type=click.Choice(["fp32", "quantized"]),
    default="quantized",
    help='Either "fp32", or "quantized". Specify weather to load a quantized or a FP ' "model.",
)

def validate_quantized(config, load_type):
    """
    Click 的子命令，负责对预训练的量化模型进行验证。
    它加载模型和数据加载器，执行范围估计（如果加载类型为 fp32），
    固定量化范围，创建评估器，进行 BatchNorm 统计的重新估计（如果需要），并最终在验证集上运行评估，输出最终的度量指标。
    function for running validation on pre-trained quantized models
    
    Args:
        config: 由pass_config装饰器传递的Config实例，包含所有配置参数。
        load_type: 通过命令行选项--load-type传递的参数，指定加载的模型类型。
    """
    print("Setting up network and data loaders")
    # 获取量化参数字典
    qparams = quant_params_dict(config)

    # 获取数据加载器和模型
    dataloaders, model = get_dataloaders_and_model(config=config, load_type=load_type, **qparams)

    # 根据加载类型执行相关操作
    if load_type == "fp32":
        # 执行范围估计和量化状态设置
        # Estimate ranges using training data
        pass_data_for_range_estimation(
            loader=dataloaders.train_loader,
            model=model,
            act_quant=config.quant.act_quant,
            weight_quant=config.quant.weight_quant,
            max_num_batches=config.quant.num_est_batches,
        )
        # Ensure we have the desired quant state
        model.set_quant_state(config.quant.weight_quant, config.quant.act_quant)

    # Fix ranges
    # 固定量化范围
    model.fix_ranges()

    # Create evaluator
    # 定义损失函数
    loss_func = CrossEntropyLoss()
    # 定义评估指标
    metrics = {
        "top_1_accuracy": Accuracy(),
        "top_5_accuracy": TopKCategoricalAccuracy(),
        "loss": Loss(loss_func),
    }

    # 初始化进度条和评估器
    pbar = ProgressBar()
    evaluator = create_supervised_evaluator(
        model=model, metrics=metrics, device="cuda" if config.base.cuda else "cpu"
    )
    # 附加进度条到评估器
    pbar.attach(evaluator)
    print("Model with the ranges estimated:\n{}".format(model))

    # BN Re-estimation
    # BN统计的重新估计
    if config.qat.reestimate_bn_stats:
        ReestimateBNStats(
            model, dataloaders.train_loader, num_batches=int(0.02 * len(dataloaders.train_loader))
        )(None)

    # 启动验证过程
    print("Start quantized validation")
    evaluator.run(dataloaders.val_loader)
    final_metrics = evaluator.state.metrics
    print(final_metrics)


if __name__ == "__main__":
    fp8_cmd_group()
