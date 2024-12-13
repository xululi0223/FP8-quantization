#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
import numpy as np
from utils.grid import integrate_pdf_grid_func_analyt
from quantization.quantizers.fp8_quantizer import FPQuantizer, generate_all_float_values_scaled


def generate_integr_grid_piecewise(integr_discontin, num_intervals_smallest_bin):
    """
    用于生成分段积分网格（integral grid），该网格基于给定的不连续点（`integr_discontin`）和最小子区间数量。
    这个网格在量化误差估计中用于积分计算，确保在每个区间内有足够的细分，以提高估计的精度。

    Args:
        integr_discontin: 不连续点的列表，用于定义不同的区间。
        num_intervals_smallest_bin: 最小子区间数量，用于确定每个区间内的步长。
    """
    # 计算区间宽度
    bin_widths = np.diff(integr_discontin)
    # 找出最小的非零区间宽度
    min_bin_width = np.min(bin_widths[bin_widths > 0.0])
    # 计算每个子区间的最小步长
    integr_min_step = min_bin_width / num_intervals_smallest_bin
    # 初始化网格列表
    grid_list = []
    # 遍历每个区间并生成网格点
    for i in range(len(integr_discontin) - 1):
        # 获取当前区间的最小值和最大值
        curr_interv_min = integr_discontin[i]
        curr_interv_max = integr_discontin[i + 1]
        # 计算当前区间的宽度
        curr_interv_width = curr_interv_max - curr_interv_min

        # 跳过零宽度区间
        if curr_interv_width == 0.0:
            continue
        assert curr_interv_min < curr_interv_max
        # 确定子区间数量
        curr_interv_n_subintervals = np.ceil(curr_interv_width / integr_min_step).astype("int")
        # 确定网格点数量
        curr_interv_n_pts = curr_interv_n_subintervals + 1
        # 生成线性空间网格点
        lspace = torch.linspace(curr_interv_min, curr_interv_max, curr_interv_n_pts)
        # 添加到网格列表
        grid_list.append(lspace)

    # 合并所有区间的网格点
    grid_all = torch.cat(grid_list)
    # 去除重复的网格点
    grid_all_no_dup = torch.unique(grid_all)

    return grid_all_no_dup


def estimate_rounding_error_analyt(distr, grid):
    """
    分析性地估计量化过程中的舍入误差。
    它基于给定的分布（distr）和网格（grid），通过积分计算得到舍入误差的平方和。
    
    Args:
        distr: 概率分布对象，用于描述量化前的数值分布。
        grid: 积分网格，用于定义积分区间和点。
    """
    err = integrate_pdf_grid_func_analyt(distr, grid, "integr_interv_p_sqr_r")
    return err


def estimate_dot_prod_error_analyt(distr_x, grid_x, distr_y, grid_y):
    """
    分析性地估计两个量化分布在点积运算中的误差。
    它基于两个输入分布（distr_x 和 distr_y）及其对应的网格，计算点积误差的平方和。
    
    Args:
        distr_x: 第一个输入分布对象。
        grid_x: 第一个输入分布的积分网格。
        distr_y: 第二个输入分布对象。
        grid_y: 第二个输入分布的积分网格。
    """
    # 计算单个量化误差
    rounding_err_x = integrate_pdf_grid_func_analyt(distr_x, grid_x, "integr_interv_p_sqr_r")
    rounding_err_y = integrate_pdf_grid_func_analyt(distr_y, grid_y, "integr_interv_p_sqr_r")
    # 计算二阶矩
    second_moment_x = distr_x.eval_non_central_second_moment()
    second_moment_y = distr_y.eval_non_central_second_moment()
    # 计算交叉项积分
    y_p_y_R_y_signed = integrate_pdf_grid_func_analyt(distr_y, grid_y, "integr_interv_x_p_signed_r")
    x_p_x_R_x_signed = integrate_pdf_grid_func_analyt(distr_x, grid_x, "integr_interv_x_p_signed_r")

    # 计算各项误差的贡献
    term_rounding_err_x_moment_y = rounding_err_x * second_moment_y
    term_rounding_err_y_moment_x = rounding_err_y * second_moment_x
    term_mixed_rounding_err = rounding_err_x * rounding_err_y
    term_mixed_R_signed = 2.0 * y_p_y_R_y_signed * x_p_x_R_x_signed
    term_rounding_err_x_R_y_signed = 2.0 * rounding_err_x * y_p_y_R_y_signed
    term_rounding_err_y_R_x_signed = 2.0 * rounding_err_y * x_p_x_R_x_signed

    # 汇总所有误差项
    total_sum = (
        term_rounding_err_x_moment_y
        + term_rounding_err_y_moment_x
        + term_mixed_R_signed
        + term_mixed_rounding_err
        + term_rounding_err_x_R_y_signed
        + term_rounding_err_y_R_x_signed
    )

    return total_sum


def estimate_rounding_error_empirical(W, quantizer, range_min, range_max):
    """
    经验性地估计量化过程中的舍入误差。
    通过对量化后的权重与原始权重之间的平方差进行平均，得到舍入误差的均值。
    
    Args:
        W: 原始权重张量。
        quantizer: 量化器对象。
        range_min: 量化范围的最小值。
        range_max: 量化范围的最大值。
    """
    # 设置量化范围
    quantizer.set_quant_range(range_min, range_max)
    # 对权重进行量化
    W_int_quant = quantizer.forward(W)

    # 计算平方舍入误差
    round_err_sqr_int_quant_emp = (W_int_quant - W) ** 2
    # 计算平均舍入误差
    res = torch.mean(round_err_sqr_int_quant_emp.flatten()).item()
    return res


def estimate_dot_prod_error_empirical(
    x, y, quantizer_x, quantizer_y, x_range_min, x_range_max, y_range_min, y_range_max
):
    """
    经验性地估计两个量化向量在进行点积运算时的误差。
    通过实际量化两个向量，并计算量化后的点积与原始点积之间的平方差，得到点积误差。
    
    Args:
        x: 第一个输入向量。
        y: 第二个输入向量。
        quantizer_x: 第一个输入向量的量化器。
        quantizer_y: 第二个输入向量的量化器。
        x_range_min: 第一个输入向量的量化范围最小值。
        x_range_max: 第一个输入向量的量化范围最大值。
        y_range_min: 第二个输入向量的量化范围最小值。
        y_range_max: 第二个输入向量的量化范围最大值。
    """
    # 设置量化范围
    quantizer_x.set_quant_range(x_range_min, x_range_max)
    quantizer_y.set_quant_range(y_range_min, y_range_max)
    # 对输入向量进行量化
    x_quant = quantizer_x.forward(x)
    y_quant = quantizer_y.forward(y)

    # 计算量化点积误差的平方
    scalar_prod_err_emp = (torch.mul(x, y) - torch.mul(x_quant, y_quant)) ** 2
    # 计算平均点积误差
    res = torch.mean(scalar_prod_err_emp.flatten()).item()
    return res


def compute_expected_dot_prod_mse(
    distr_x,
    distr_y,
    quant_x,
    quant_y,
    quant_x_range_min,
    quant_x_range_max,
    quant_y_range_min,
    quant_y_range_max,
    num_samples=2000000,
):
    """
    计算期望的点积均方误差（MSE）。
    它结合分析性和经验性的方法，通过对两个分布进行样本采样和量化，估计点积运算中的误差。
    
    Args:
        distr_x: 第一个输入分布。
        distr_y: 第二个输入分布。
        quant_x: 第一个输入量化器。
        quant_y: 第二个输入量化器。
        quant_x_range_min: 第一个输入量化范围的最小值。
        quant_x_range_max: 第一个输入量化范围的最大值。
        quant_y_range_min: 第二个输入量化范围的最小值。
        quant_y_range_max: 第二个输入量化范围的最大值。
        num_samples: 用于经验性估计的样本数量。
    """
    # 设置量化范围
    quant_x.set_quant_range(quant_x_range_min, quant_x_range_max)
    quant_y.set_quant_range(quant_y_range_min, quant_y_range_max)
    
    # 生成x的网格点
    if isinstance(quant_x, FPQuantizer):
        grid_x = generate_all_float_values_scaled(
            quant_x.n_bits, quant_x.ebits, quant_x.default_bias, quant_x_range_max
        )
    else:
        grid_x = quant_x.generate_grid().numpy()

    # 生成y的网格点
    if isinstance(quant_y, FPQuantizer):
        grid_y = generate_all_float_values_scaled(
            quant_y.n_bits, quant_y.ebits, quant_y.default_bias, quant_y_range_max
        )
    else:
        grid_y = quant_x.generate_grid().numpy()

    # 计算分析性误差
    err_analyt = estimate_dot_prod_error_analyt(distr_x, grid_x, distr_y, grid_y)
    
    # 采样并构建样本，用于经验性误差估计
    distr_x_sample = torch.tensor(distr_x.sample((num_samples,)))
    distr_y_sample = torch.tensor(distr_x.sample((num_samples,)))
    # 计算经验性误差
    err_emp = estimate_dot_prod_error_empirical(
        distr_x_sample,
        distr_y_sample,
        quant_x,
        quant_y,
        quant_x_range_min,
        quant_x_range_max,
        quant_y_range_min,
        quant_y_range_max,
    )

    # 计算分析性误差与经验性误差之间的相对误差
    rel_err = np.abs((err_emp - err_analyt) / err_analyt)
    return err_analyt


def compute_expected_quant_mse(distr, quant, quant_range_min, quant_range_max, num_samples):
    """
    用于计算量化过程中的期望均方误差（MSE）。它
    结合分析性和经验性的方法，通过对一个分布进行样本采样和量化，估计量化操作中的舍入误差。
    
    Args:
        distr: 输入的概率分布。
        quant: 量化器对象。
        quant_range_min: 量化范围的最小值。
        quant_range_max: 量化范围的最大值。
        num_samples: 用于经验性估计的样本数量。
    """
    # 设置量化范围
    quant.set_quant_range(quant_range_min, quant_range_max)

    # 生成积分网格
    if isinstance(quant, FPQuantizer):
        grid = generate_all_float_values_scaled(
            quant.n_bits, quant.ebits, quant.default_bias, quant_range_max
        )
    else:
        grid = quant.generate_grid().numpy()

    # 计算分析性误差
    err_analyt = estimate_rounding_error_analyt(distr, grid)
    
    # 采样并构建样本，用于经验性误差估计
    distr_sample = torch.tensor(
        distr.sample(
            num_samples,
        )
    )
    # 计算经验性误差
    err_emp = estimate_rounding_error_empirical(
        distr_sample, quant, quant_range_min, quant_range_max
    )

    # 计算分析性误差与经验性误差之间的相对误差
    rel_err = np.abs((err_emp - err_analyt) / err_analyt)
    if rel_err > 0.1:
        print(
            "Warning: the relative difference between the analytical and empirical error estimate is too high,\n"
            "please consider increasing the number of samples for the quantization range estimator."
        )

    return err_analyt
