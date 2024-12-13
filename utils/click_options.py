#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import click

from models import QuantArchitectures
from functools import wraps, partial
from quantization.quantization_manager import QMethods
from quantization.range_estimators import RangeEstimators, OptMethod
from utils import split_dict, DotDict, ClickEnumOption, seed_all
from utils.imagenet_dataloaders import ImageInterpolation

click.option = partial(click.option, show_default=True)

_HELP_MSG = (
    "Enforce determinism also on the GPU by disabling CUDNN and setting "
    "`torch.set_deterministic(True)`. In many cases this comes at the cost of efficiency "
    "and performance."
)


def base_options(func):
    """
    一个装饰器函数，用于为命令行界面（CLI）添加一组基础选项。
    这些选项涵盖了数据路径、训练参数、模型架构等基本配置。
    通过使用这个装饰器，开发者可以简化命令的参数定义，并确保所有命令共享相同的一组基础选项。
    """
    @click.option(
        "--images-dir", type=click.Path(exists=True), help="Root directory of images", required=True
    )
    @click.option("--max-epochs", default=90, type=int, help="Maximum number of training epochs.")
    @click.option(
        "--interpolation",
        type=ClickEnumOption(ImageInterpolation),
        default=ImageInterpolation.bilinear.name,
        help="Desired interpolation to use for resizing.",
    )
    @click.option(
        "--save-checkpoint-dir",
        type=click.Path(exists=False),
        default=None,
        help="Directory where to save checkpoints (model, optimizer, lr_scheduler).",
    )
    @click.option(
        "--tb-logging-dir", default=None, type=str, help="The logging directory " "for tensorboard"
    )
    @click.option("--cuda/--no-cuda", is_flag=True, default=True, help="Use GPU")
    @click.option("--batch-size", default=128, type=int, help="Mini-batch size")
    @click.option("--num-workers", default=16, type=int, help="Number of workers for data loading")
    @click.option("--seed", default=None, type=int, help="Random number generator seed to set")
    @click.option("--deterministic/--nondeterministic", default=False, help=_HELP_MSG)
    # Architecture related options
    @click.option(
        "--architecture",
        type=ClickEnumOption(QuantArchitectures),
        required=True,
        help="Quantized architecture",
    )
    @click.option(
        "--model-dir",
        type=click.Path(exists=True),
        default=None,
        help="Path for model directory. If the model does not exist it will downloaded "
        "from a URL",
    )
    @click.option(
        "--pretrained/--no-pretrained",
        is_flag=True,
        default=True,
        help="Use pretrained model weights",
    )
    @click.option(
        "--progress-bar/--no-progress-bar", is_flag=True, default=False, help="Show progress bar"
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        # 分割配置和其他参数
        config.base, remaining_kwargs = split_dict(
            kwargs,
            [
                "images_dir",
                "max_epochs",
                "interpolation",
                "save_checkpoint_dir",
                "tb_logging_dir",
                "cuda",
                "batch_size",
                "num_workers",
                "seed",
                "model_dir",
                "architecture",
                "pretrained",
                "deterministic",
                "progress_bar",
            ],
        )

        # 处理种子和确定性选项
        seed, deterministic = config.base.seed, config.base.deterministic

        if seed is None:
            if deterministic is True:
                raise ValueError("Enforcing determinism without providing a seed is not supported")
        else:
            seed_all(seed=seed, deterministic=deterministic)

        return func(config, *args, **remaining_kwargs)

    return func_wrapper


class multi_optimizer_options:
    """
    一个装饰器类，用于为命令行接口添加多个优化器相关的选项。
    通过传递不同的前缀，用户可以为不同的优化器（例如主优化器和量化优化器）定义独立的命令行选项。
    这种设计提高了灵活性，使得不同优化器的参数可以单独配置。
    
    An instance of this class is a callable object to serve as a decorator;
    hence the lower case class name.

    Among the CLI options defined in the decorator, `--{prefix-}optimizer-type`
    requires special attention. Default value for that variable for
    {prefix-}optimizer is the value in use by the main optimizer.

    Examples:
        @multi_optimizer_options('quant')
        @pass_config
        def command(config):
            ...
    """

    def __init__(self, prefix: str = ""):
        """
        Args:
            prefix: 字符串类型，用于区分不同的优化器选项。
        """
        self.optimizer_name = prefix + "_optimizer" if prefix else "optimizer"
        self.prefix_option = prefix + "-" if prefix else ""
        self.prefix_attribute = prefix + "_" if prefix else ""

    def __call__(self, func):
        """
        调用方法，作为装饰器应用于函数。
        
        Args:
            func: 要装饰的函数。
        """
        prefix_option = self.prefix_option
        prefix_attribute = self.prefix_attribute

        # 优化器类型
        @click.option(
            f"--{prefix_option}optimizer",
            default="SGD",
            type=click.Choice(["SGD", "Adam"], case_sensitive=False),
            help=f"Class name of torch Optimizer to be used.",
        )
        # 初始学习率
        @click.option(
            f"--{prefix_option}learning-rate",
            default=None,
            type=float,
            help="Initial learning rate.",
        )
        # 优化器动量
        @click.option(
            f"--{prefix_option}momentum", default=0.9, type=float, help=f"Optimizer momentum."
        )
        # 权重衰减
        @click.option(
            f"--{prefix_option}weight-decay",
            default=None,
            type=float,
            help="Weight decay for the network.",
        )
        # 学习率调度器
        @click.option(
            f"--{prefix_option}learning-rate-schedule",
            default=None,
            type=str,
            help="Learning rate scheduler, 'MultiStepLR:10:20:40' or "
            "'cosine:1e-4' for cosine decay",
        )
        @wraps(func)
        def func_wrapper(config, *args, **kwargs):
            """
            包装器函数，用于处理命令行参数并将其存储到配置对象中。
            """
            # 定义要处理的基础参数名称列表
            base_arg_names = [
                "optimizer",
                "learning_rate",
                "momentum",
                "weight_decay",
                "learning_rate_schedule",
            ]

            # 存储优化器选项
            optimizer_opt = DotDict()

            # Collect basic arguments
            # 遍历基础参数名称列表，提取对应的参数值
            for arg in base_arg_names:
                option_name = prefix_attribute + arg
                optimizer_opt[arg] = kwargs.pop(option_name)

            # config.{prefix_attribute}optimizer = optimizer_opt
            # 将优化器选项存储到配置对象中
            setattr(config, prefix_attribute + "optimizer", optimizer_opt)

            return func(config, *args, **kwargs)

        return func_wrapper


def qat_options(func):
    """
    装饰器函数，用于为命令行接口添加量化感知训练（Quantization-Aware Training, QAT）相关的选项。
    这些选项包括是否重新估计 BN 统计量、梯度缩放、是否使用独立的量化优化器等。
    通过使用这个装饰器，开发者可以为 QAT 配置提供一组统一的命令行参数。
    """
    # 控制是否在每次评估前重新估计BN统计量
    @click.option(
        "--reestimate-bn-stats/--no-reestimate-bn-stats",
        is_flag=True,
        default=True,
        help="Reestimates the BN stats before every evaluation.",
    )
    # 选择是否进行梯度缩放
    @click.option(
        "--grad-scaling/--no-grad-scaling",
        is_flag=True,
        default=False,
        help="Do gradient scaling as in LSQ paper.",
    )
    # 选择是否为量化器使用单独的优化器
    @click.option(
        "--sep-quant-optimizer/--no-sep-quant-optimizer",
        is_flag=True,
        default=False,
        help="Use a separate optimizer for the quantizers.",
    )
    # 应用`multi_optimizer_options`装饰器，为量化优化器添加一系列命令行选项
    @multi_optimizer_options("quant")
    # 应用`oscillations_dampen_options`装饰器，为命令行接口添加振荡抑制相关的选项
    @oscillations_dampen_options
    # 应用`oscillations_freeze_options`装饰器，为命令行接口添加振荡冻结相关的选项
    @oscillations_freeze_options
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        """
        包装器函数，用于处理命令行参数并将其存储到配置对象中。
        """
        # 将kwargs中的QAT选项提取出来，并存储到config.qat中，剩余的参数存储到remainder_kwargs中
        config.qat, remainder_kwargs = split_dict(
            kwargs, ["reestimate_bn_stats", "grad_scaling", "sep_quant_optimizer"]
        )
        return func(config, *args, **remainder_kwargs)

    return func_wrapper


def oscillations_dampen_options(func):
    """
    装饰器函数，用于为命令行接口添加与振荡抑制（oscillations dampening）相关的选项。
    这些选项允许用户在训练过程中添加振荡抑制到损失函数中，以减少训练过程中的振荡现象，从而提高模型的稳定性和性能。
    """
    # 设定振荡抑制的权重
    @click.option(
        "--oscillations-dampen-weight",
        default=None,
        type=float,
        help="If given, adds a oscillations dampening to the loss with given  " "weighting.",
    )
    # 选择振荡抑制在损失函数中的聚合方式
    @click.option(
        "--oscillations-dampen-aggregation",
        type=click.Choice(["sum", "mean", "kernel_mean"]),
        default="kernel_mean",
        help="Aggregation type for bin regularization loss.",
    )
    # 设定振荡抑制正则化在退火计划中的最终值
    @click.option(
        "--oscillations-dampen-weight-final",
        type=float,
        default=None,
        help="Dampening regularization final value for annealing schedule.",
    )
    # 设定振荡抑制正则化的退火开始位置
    @click.option(
        "--oscillations-dampen-anneal-start",
        default=0.25,
        type=float,
        help="Start of annealing (relative to total number of iterations).",
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        """
        包装器函数，用于处理命令行参数并将其存储到配置对象中。
        """
        # 分割配置和其他参数
        config.osc_damp, remainder_kwargs = split_dict(
            kwargs,
            [
                "oscillations_dampen_weight",
                "oscillations_dampen_aggregation",
                "oscillations_dampen_weight_final",
                "oscillations_dampen_anneal_start",
            ],
            "oscillations_dampen",
        )

        return func(config, *args, **remainder_kwargs)

    return func_wrapper


def oscillations_freeze_options(func):
    """
    装饰器函数，用于为命令行接口添加与振荡冻结（oscillations freezing）相关的选项。
    这些选项允许用户在训练过程中冻结频率高于特定阈值的振荡，以进一步提高模型的稳定性。
    """
    # 设定震荡冻结的频率阈值
    @click.option(
        "--oscillations-freeze-threshold",
        default=0.0,
        type=float,
        help="If greater than 0, we will freeze oscillations which frequency (EMA) is "
        "higher  than the given threshold. Frequency is defined as 1/period length.",
    )
    # 设定计算振荡频率EMA的动量
    @click.option(
        "--oscillations-freeze-ema-momentum",
        default=0.001,
        type=float,
        help="The momentum to calculate the EMA frequency of the oscillation. In case"
        "freezing is used, this should be at least 2-3 times lower than the "
        "freeze threshold.",
    )
    # 选择是否使用EMA来确定冻结的内部值
    @click.option(
        "--oscillations-freeze-use-ema/--no-oscillation-freeze-use-ema",
        is_flag=True,
        default=True,
        help="Uses an EMA of past x_int to find the correct freezing int value.",
    )
    # 设定振荡跟踪和冻结的最大位宽
    @click.option(
        "--oscillations-freeze-max-bits",
        default=4,
        type=int,
        help="Max bit-width for oscillation tracking and freezing. If layers weight is in"
        "higher bits we do not track or freeze oscillations.",
    )
    # 设定震荡冻结在退火计划中的最终阈值
    @click.option(
        "--oscillations-freeze-threshold-final",
        type=float,
        default=None,
        help="Oscillation freezing final value for annealing schedule.",
    )
    # 设定震荡冻结的退火开始位置
    @click.option(
        "--oscillations-freeze-anneal-start",
        default=0.25,
        type=float,
        help="Start of annealing (relative to total number of iterations).",
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        """
        包装器函数，用于处理命令行参数并将其存储到配置对象中。
        """
        # 分割配置和其他参数
        config.osc_freeze, remainder_kwargs = split_dict(
            kwargs,
            [
                "oscillations_freeze_threshold",
                "oscillations_freeze_ema_momentum",
                "oscillations_freeze_use_ema",
                "oscillations_freeze_max_bits",
                "oscillations_freeze_threshold_final",
                "oscillations_freeze_anneal_start",
            ],
            "oscillations_freeze",
        )

        return func(config, *args, **remainder_kwargs)

    return func_wrapper


def quantization_options(func):
    """
    装饰器函数，用于为命令行接口添加与量化相关的选项。
    这些选项涵盖了权重量化、激活量化的方法和参数，以及其他与量化设置相关的配置。
    通过使用这个装饰器，开发者可以为模型量化配置提供一组统一且灵活的命令行参数。
    """
    # 选择是否进行权重量化
    # Weight quantization options
    @click.option(
        "--weight-quant/--no-weight-quant",
        is_flag=True,
        default=True,
        help="Run evaluation weight quantization or use FP32 weights",
    )
    # 选择量化方案
    @click.option(
        "--qmethod",
        type=ClickEnumOption(QMethods),
        default=QMethods.symmetric_uniform.name,
        help="Quantization scheme to use.",
    )
    # 选择确定权重量化裁剪阈值的方法
    @click.option(
        "--weight-quant-method",
        default=RangeEstimators.current_minmax.name,
        type=ClickEnumOption(RangeEstimators),
        help="Method to determine weight quantization clipping thresholds.",
    )
    # 选择优化激活量化裁剪阈值的方法
    @click.option(
        "--weight-opt-method",
        default=OptMethod.grid.name,
        type=ClickEnumOption(OptMethod),
        help="Optimization procedure for activation quantization clipping thresholds",
    )
    # 设定在MSE范围方法中的网格搜索点数
    @click.option(
        "--num-candidates",
        type=int,
        default=None,
        help="Number of grid points for grid search in MSE range method.",
    )
    # 设定默认的量化位数
    @click.option("--n-bits", default=8, type=int, help="Default number of quantization bits.")
    # 选择是否对每个通道单独量化
    @click.option(
        "--per-channel/--no-per-channel",
        is_flag=True,
        default=False,
        help="If given, quantize each channel separately.",
    )
    # 选择是否进行激活量化
    # Activation quantization options
    @click.option(
        "--act-quant/--no-act-quant",
        is_flag=True,
        default=True,
        help="Run evaluation with activation quantization or use FP32 activations",
    )
    # 选择激活量化的量化方案
    @click.option(
        "--qmethod-act",
        type=ClickEnumOption(QMethods),
        default=None,
        help="Quantization scheme for activation to use. If not specified `--qmethod` " "is used.",
    )
    # 设定激活量化的位数
    @click.option(
        "--n-bits-act", default=None, type=int, help="Number of quantization bits for activations."
    )
    # 选择确定激活量化裁剪阈值的方法
    @click.option(
        "--act-quant-method",
        default=RangeEstimators.running_minmax.name,
        type=ClickEnumOption(RangeEstimators),
        help="Method to determine activation quantization clipping thresholds",
    )
    # 选择优化激活量化裁剪阈值的方法
    @click.option(
        "--act-opt-method",
        default=OptMethod.grid.name,
        type=ClickEnumOption(OptMethod),
        help="Optimization procedure for activation quantization clipping thresholds",
    )
    # 设定在MSE/SQNR/Cross-entropy中的网格搜索点数
    @click.option(
        "--act-num-candidates",
        type=int,
        default=None,
        help="Number of grid points for grid search in MSE/SQNR/Cross-entropy",
    )
    # 设定running_minmax的指数平均因子
    @click.option(
        "--act-momentum",
        type=float,
        default=None,
        help="Exponential averaging factor for running_minmax",
    )
    # 设定用于激活范围估计的训练批次数
    @click.option(
        "--num-est-batches",
        type=int,
        default=1,
        help="Number of training batches to be used for activation range estimation",
    )
    # 选择量化网络的方法
    # Other options
    @click.option(
        "--quant-setup",
        default="all",
        type=click.Choice(["all", "LSQ", "FP_logits", "fc4", "fc4_dw8", "LSQ_paper"]),
        help="Method to quantize the network.",
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        """
        包装器函数，用于处理命令行参数并将其存储到配置对象中。
        """
        # 分割配置和其他参数
        config.quant, remainder_kwargs = split_dict(
            kwargs,
            [
                "qmethod",
                "qmethod_act",
                "weight_quant_method",
                "weight_opt_method",
                "num_candidates",
                "n_bits",
                "n_bits_act",
                "per_channel",
                "act_quant",
                "weight_quant",
                "quant_setup",
                "num_est_batches",
                "act_momentum",
                "act_num_candidates",
                "act_opt_method",
                "act_quant_method",
            ],
        )

        # 如果没有指定激活量化的量化方案，则使用权重量化的量化方案
        config.quant.qmethod_act = config.quant.qmethod_act or config.quant.qmethod

        return func(config, *args, **remainder_kwargs)

    return func_wrapper


def fp8_options(func):
    """
    装饰器函数，用于为命令行接口添加与 FP8 量化相关的选项。
    FP8 量化是一种低精度浮点数格式，用于加速深度学习模型的训练和推理。
    通过使用这个装饰器，开发者可以为 FP8 量化配置提供一组统一且灵活的命令行参数。
    """
    # Weight quantization options
    # 设定FP8量化的最大值
    @click.option("--fp8-maxval", type=float, default=None)
    # 设定FP8量化的小数位数（尾数位数）
    @click.option("--fp8-mantissa-bits", type=int, default=4)
    # 选择是否设置FP8量化的最大值
    @click.option("--fp8-set-maxval/--no-fp8-set-maxval", is_flag=True, default=False)
    # 选择是否学习FP8量化的最大值
    @click.option("--fp8-learn-maxval/--no-fp8-learn-maxval", is_flag=True, default=False)
    # 选择是否学习FP8量化的小数位数（尾数位数）
    @click.option(
        "--fp8-learn-mantissa-bits/--no-fp8-learn-mantissa-bits", is_flag=True, default=False
    )
    # 选择是否包括FP8尾数位数在MSE计算中
    @click.option(
        "--fp8-mse-include-mantissa-bits/--no-fp8-mse-include-mantissa-bits",
        is_flag=True,
        default=True,
    )
    # 选择是否允许FP8量化无符号表示
    @click.option("--fp8-allow-unsigned/--no-fp8-allow-unsigned", is_flag=True, default=False)
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        """
        包装器函数，用于处理命令行参数并将其存储到配置对象中。
        """
        # 分割配置和其他参数
        config.fp8, remainder_kwargs = split_dict(
            kwargs,
            [
                "fp8_maxval",
                "fp8_mantissa_bits",
                "fp8_set_maxval",
                "fp8_learn_maxval",
                "fp8_learn_mantissa_bits",
                "fp8_mse_include_mantissa_bits",
                "fp8_allow_unsigned",
            ],
        )
        return func(config, *args, **remainder_kwargs)

    return func_wrapper


def quant_params_dict(config):
    """
    辅助函数，用于根据配置对象 config 构建一个量化参数的字典。
    该字典包含了模型量化所需的各种参数，如量化方法、位数、聚合方式、范围估计方法等。
    这个函数主要用于在模型训练或推理过程中，将命令行参数转化为量化配置，以便于后续的量化操作。
    """
    # 初始化存储权重量化范围估计的选项
    weight_range_options = {}
    # 设置权重量化范围选项
    if config.quant.weight_quant_method == RangeEstimators.MSE:
        weight_range_options = dict(opt_method=config.quant.weight_opt_method)
    if config.quant.num_candidates is not None:
        weight_range_options["num_candidates"] = config.quant.num_candidates

    # 初始化存储激活量化范围估计的选项
    act_range_options = {}
    # 设置激活量化范围选项
    if config.quant.act_quant_method == RangeEstimators.MSE:
        act_range_options = dict(opt_method=config.quant.act_opt_method)
    if config.quant.act_num_candidates is not None:
        act_range_options["num_candidates"] = config.quant.num_candidates

    # 构建量化参数字典
    qparams = {
        "method": config.quant.qmethod.cls,
        "n_bits": config.quant.n_bits,
        "n_bits_act": config.quant.n_bits_act,
        "act_method": config.quant.qmethod_act.cls,
        "per_channel_weights": config.quant.per_channel,
        "quant_setup": config.quant.quant_setup,
        "weight_range_method": config.quant.weight_quant_method.cls,
        "weight_range_options": weight_range_options,
        "act_range_method": config.quant.act_quant_method.cls,
        "act_range_options": act_range_options,
        "quantize_input": True if config.quant.quant_setup == "LSQ_paper" else False,
    }

    # 检查并添加FP8量化参数
    if config.quant.qmethod.name.startswith("fp_quantizer"):
        fp8_kwargs = {
            k.replace("fp8_", ""): v for k, v in config.fp8.items() if k.startswith("fp8")
        }
    qparams["fp8_kwargs"] = fp8_kwargs

    return qparams
