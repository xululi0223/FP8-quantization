#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
import numpy as np
from utils.distributions import ClippedGaussDistr, ClippedStudentTDistr


def rounding_error_abs_nearest(x_float, grid):
    """
    计算浮点数x_float与网格grid之间的绝对舍入误差，返回每个浮点数到其最近网格点的最小距离。
    
    Args:
        x_float: 需要计算舍入误差的浮点数。
        grid: 网格点。
    """
    n_grid = grid.size          # 获取网格大小
    grid_row = np.array(grid).reshape(1, n_grid)    # 重塑网格为行向量
    m_vals = x_float.numel()                        # 获取浮点数的数量
    x_float = x_float.cpu().detach().numpy().reshape(m_vals, 1)     # 重塑浮点数为列向量

    dist = np.abs(x_float - grid_row)               # 计算浮点数与网格之间的绝对距离矩阵
    min_dist = np.min(dist, axis=1)                 # 获取每个浮点数到其最近网格点的最小距离

    return min_dist


def quant_scalar_nearest(x_float, grid):
    """
    将标量x_float量化到最近的网格点，返回量化后的值。
    
    Args:
        x_float: 需要量化的标量。
        grid: 网格点。
    """
    dist = np.abs(x_float - grid)                   # 计算浮点数与网格之间的绝对距离
    idx = np.argmin(dist)                           # 获取最小距离的索引
    q_x = grid[idx]                                 # 获取梁化后的值
    return q_x


def clip_grid_exclude_bounds(grid, min_val, max_val):
    """
    从网格grid中排除边界值，返回大于min_val且小于max_val的网格点。
    
    Args:
        grid: 网格点。
        min_val: 最小值。
        max_val: 最大值。
    """
    idx_subset = torch.logical_and(grid > min_val, grid < max_val)  # 创建逻辑掩码，获取大于min_val且小于max_val的网格点
    return grid[idx_subset]                                         # 返回大于min_val且小于max_val的网格点


def clip_grid_include_bounds(grid, min_val, max_val):
    """
    从网格grid中包含边界值，返回大于等于min_val且小于等于max_val的网格点。
    
    Args:
        grid: 网格点。
        min_val: 最小值。
        max_val: 最大值。
    """
    idx_subset = torch.logical_and(grid >= min_val, grid <= max_val)    # 创建逻辑掩码，获取大于等于min_val且小于等于max_val的网格点
    return grid[idx_subset]                                             # 返回大于等于min_val且小于等于max_val的网格点


def clip_grid_add_bounds(grid, min_val, max_val):
    """
    从网格grid中排除边界值，添加新的边界值min_val和max_val，并对结果进行排序，返回新的网格点。
    
    Args:
        grid: 网格点。
        min_val: 新的最小值。
        max_val: 新的最大值。
    """
    grid_clipped = clip_grid_exclude_bounds(grid, min_val, max_val)     # 从网格grid中排除边界值
    bounds_np = np.array([min_val, max_val])                            # 创建新的边界值
    clipped_with_bounds = np.sort(np.concatenate((grid_clipped, bounds_np)))    # 添加新的边界值，并对结果进行排序
    return clipped_with_bounds


def integrate_pdf_grid_func_analyt(distr, grid, distr_attr_func_name):
    """
    在给定的网格grid上，对分布distr的某个属性函数（如积分函数）执行分段积分，并累加结果。
    
    Args:
        distr: 分布对象。
        grid: 网格点。
        distr_attr_func_name: 分布对象中需要调用的属性函数名称。
    """
    grid = np.sort(grid)            # 对网格进行排序
    interval_integr_func = getattr(distr, distr_attr_func_name)     # 获取分布对象中的属性函数
    res = 0.0

    if distr.range_min < grid[0]:                                       # 处理分布范围下界与网格起点之间的积分
        res += interval_integr_func(distr.range_min, grid[0], grid[0])

    # 遍历网格区间，执行分段积分
    for i_interval in range(0, len(grid) - 1):                      # 遍历grid每一个相邻区间
        grid_mid = 0.5 * (grid[i_interval] + grid[i_interval + 1])  # 计算网格区间的中点

        # 第一子区间
        first_half_a = max(grid[i_interval], distr.range_min)
        first_half_b = min(grid_mid, distr.range_max)

        # 第二子区间
        second_half_a = max(grid_mid, distr.range_min)
        second_half_b = min(grid[i_interval + 1], distr.range_max)

        # 对第一子区间和第二子区间执行分段积分
        if first_half_a < first_half_b:
            res += interval_integr_func(first_half_a, first_half_b, grid[i_interval])
        if second_half_a < second_half_b:
            res += interval_integr_func(second_half_a, second_half_b, grid[i_interval + 1])

    # 处理分布范围上界与网格终点之间的积分
    if distr.range_max > grid[-1]:
        res += interval_integr_func(grid[-1], distr.range_max, grid[-1])

    # 处理特定分布类型的点质量贡献
    if (
        isinstance(distr, ClippedGaussDistr) or isinstance(distr, ClippedStudentTDistr)     # 如果分布是ClippedGaussDistr或ClippedStudentTDistr
    ) and distr_attr_func_name == "integr_interv_x_p_signed_r":                             # 并且属性函数是integr_interv_x_p_signed_r
        # 计算量化的边界点
        q_range_min = quant_scalar_nearest(torch.Tensor([distr.range_min]), grid)
        q_range_max = quant_scalar_nearest(torch.Tensor([distr.range_max]), grid)

        # 计算点质量项
        term_point_mass = (
            distr.range_min * (q_range_min - distr.range_min) * distr.point_mass_range_min
            + distr.range_max * (q_range_max - distr.range_max) * distr.point_mass_range_max
        )
        res += term_point_mass
    elif (
        isinstance(distr, ClippedGaussDistr) or isinstance(distr, ClippedStudentTDistr)     # 如果分布是ClippedGaussDistr或ClippedStudentTDistr
    ) and distr_attr_func_name == "integr_interv_p_sqr_r":                                  # 并且属性函数是integr_interv_p_sqr_r
        # 计算量化的边界点
        q_range_min = quant_scalar_nearest(torch.Tensor([distr.range_min]), grid)
        q_range_max = quant_scalar_nearest(torch.Tensor([distr.range_max]), grid)

        # 计算点质量项
        term_point_mass = (q_range_min - distr.range_min) ** 2 * distr.point_mass_range_min + (
            q_range_max - distr.range_max
        ) ** 2 * distr.point_mass_range_max
        res += term_point_mass

    return res
