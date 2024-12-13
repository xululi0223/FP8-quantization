#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
import copy
from enum import auto

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch import nn

from utils import to_numpy, BaseEnumOptions, MethodMap, ClassEnumOptions


class RangeEstimatorBase(nn.Module):
    """
    抽象基类，用于估计张量的数值范围（最小值和最大值）。
    它为具体的范围估计器类提供了基础结构和通用方法，确保所有子类具备统一的接口和行为。
    """
    def __init__(self, per_channel=False, quantizer=None, *args, **kwargs):
        """
        Args:
            per_channel: 是否逐通道估计，默认为False，表示全局估计。
            quantizer: 关联的量化器对象。
        """
        super().__init__(*args, **kwargs)
        self.register_buffer("current_xmin", None)      # 注册当前最小值缓冲区
        self.register_buffer("current_xmax", None)      # 注册当前最大值缓冲区
        self.per_channel = per_channel
        self.quantizer = quantizer

    def forward(self, x):
        """
        Accepts an input tensor, updates the current estimates of x_min and x_max
        and returns them.
        Parameters
        ----------
        x:  Input tensor

        Returns
        -------
        self.current_xmin: tensor

        self.current_xmax: tensor

        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset the range estimator.
        """
        # 重置当前的最小值和最大值
        self.current_xmin = None
        self.current_xmax = None

    def __repr__(self):
        """
        定制类的字符串表示形式。
        """
        # We overwrite this from nn.Module as we do not want to have submodules such as
        # self.quantizer in the reproduce. Otherwise it behaves as expected for an nn.Module.
        lines = self.extra_repr().split("\n")
        extra_str = lines[0] if len(lines) == 1 else "\n  " + "\n  ".join(lines) + "\n"

        return self._get_name() + "(" + extra_str + ")"     # 形成类似ClassName(extra information)的格式


class CurrentMinMaxEstimator(RangeEstimatorBase):
    """
    基于当前输入张量的最小值和最大值来估计数值范围。
    它支持按通道和全局两种模式，并可通过设置百分位数来进行鲁棒的范围估计，减少异常值的影响。
    """
    def __init__(self, percentile=None, *args, **kwargs):
        """
        Args:
            percentile: 百分位数，用于鲁棒的范围估计。
        """
        self.percentile = percentile
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        前向传播方法。
        更新并返回当前输入张量x的最小值和最大值。
        """
        # 如果是按通道模式，则将输入张量展平
        if self.per_channel:
            x = x.view(x.shape[0], -1)
        if self.percentile:
            # 使用百分位数来限制范围
            axis = -1 if self.per_channel else None     # 沿着通道维度或全局维度
            data_np = to_numpy(x)
            x_min, x_max = np.percentile(               # 计算百分位数
                data_np, (self.percentile, 100 - self.percentile), axis=axis
            )
            self.current_xmin = torch.tensor(x_min).to(x.device)    # 将最小值转换为张量
            self.current_xmax = torch.tensor(x_max).to(x.device)    # 将最大值转换为张量
        else:
            # 不使用百分位数，直接计算最小值和最大值
            self.current_xmin = x.min(-1)[0].detach() if self.per_channel else x.min().detach()     # 根据是否按通道计算最小值
            self.current_xmax = x.max(-1)[0].detach() if self.per_channel else x.max().detach()     # 根据是否按通道计算最大值

        return self.current_xmin, self.current_xmax


class AllMinMaxEstimator(RangeEstimatorBase):
    """
    通过遍历所有输入数据，累积并更新最小值和最大值。
    它支持按通道和全局两种模式，适用于需要在多个批次中持续更新范围的场景。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        更新并返回所有输入数据的最小值和最大值。
        """
        if self.per_channel:
            # Along 1st dim
            x_flattened = x.view(x.shape[0], -1)        # 展平张量
            x_min = x_flattened.min(-1)[0].detach()     # 计算最小值
            x_max = x_flattened.max(-1)[0].detach()     # 计算最大值
        else:
            # 全局模式
            x_min = torch.min(x).detach()
            x_max = torch.max(x).detach()

        # 更新当前最小值和最大值
        if self.current_xmin is None:
            self.current_xmin = x_min
            self.current_xmax = x_max
        else:
            self.current_xmin = torch.min(self.current_xmin, x_min)
            self.current_xmax = torch.max(self.current_xmax, x_max)

        return self.current_xmin, self.current_xmax


class RunningMinMaxEstimator(RangeEstimatorBase):
    """
    使用动量技术（Momentum）来平滑更新最小值和最大值。
    通过引入动量参数，可以在新的数据批次到来时，保持先前范围的影响，从而实现平滑和稳定的范围估计。
    """
    def __init__(self, momentum=0.9, *args, **kwargs):
        """
        Args:
            momentum: 动量参数，用于平滑更新最小值和最大值。
        """
        self.momentum = momentum
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        前向传播方法。
        更新并返回当前输入张量x的最小值和最大值。
        """
        if self.per_channel:
            # Along 1st dim
            x_flattened = x.view(x.shape[0], -1)            # 展平张量
            x_min = x_flattened.min(-1)[0].detach()         # 计算最小值
            x_max = x_flattened.max(-1)[0].detach()         # 计算最大值
        else:
            # 全局模式
            x_min = torch.min(x).detach()
            x_max = torch.max(x).detach()

        # 更新当前最小值和最大值
        if self.current_xmin is None:
            self.current_xmin = x_min
            self.current_xmax = x_max
        else:
            # 使用动量技术平滑更新最小值和最大值
            self.current_xmin = (1 - self.momentum) * x_min + self.momentum * self.current_xmin
            self.current_xmax = (1 - self.momentum) * x_max + self.momentum * self.current_xmax

        return self.current_xmin, self.current_xmax


class OptMethod(BaseEnumOptions):
    """
    枚举类，定义了范围估计器中可用的优化方法。
    """
    grid = auto()               # 网格搜索优化方法，使用auto()自动编号
    golden_section = auto()     # 黄金分割法优化方法，使用auto()自动编号


class LineSearchEstimator(RangeEstimatorBase):
    """
    通过线搜索方法（如网格搜索和黄金分割法）来优化并估计张量的数值范围（最小值和最大值）。
    该类在量化过程中使用，以最小化量化误差（如均方误差，MSE），从而确定最优的量化范围。
    """
    def __init__(
        self,
        num_candidates=1000,
        opt_method=OptMethod.grid,
        range_margin=0.5,
        expand_range=10.0,
        *args,
        **kwargs,
    ):
        """
        Args:
            num_candidates: 候选数量，用于网格搜索，默认值为1000。
            opt_method: 优化方法，可选值为OptMethod枚举类中的值，默认为网格搜索。
            range_margin: 范围边界，用于限制范围的边界，默认值为0.5。
            expand_range: 扩展范围，用于扩展范围的大小，默认值为10.0。
        """
        super().__init__(*args, **kwargs)
        assert opt_method in OptMethod          # 断言优化方法是OptMethod枚举类中的一个有效值
        self.opt_method = opt_method
        self.num_candidates = num_candidates
        self.expand_range = expand_range
        self.loss_array = None
        self.max_pos_thr = None
        self.max_neg_thr = None
        self.max_search_range = None
        self.one_sided_dist = None
        self.range_margin = range_margin
        if self.quantizer is None:              # 如果没有传入量化器，则抛出异常
            raise NotImplementedError(
                "A Quantizer must be given as an argument to the MSE Range" "Estimator"
            )
        self.max_int_skew = (2**self.quantizer.n_bits) // 4     # 用于不对称量化的偏斜量

    def loss_fx(self, data, neg_thr, pos_thr, per_channel_loss=False):
        """
        损失函数方法。
        
        Args:
            data: 输入数据张量。
            neg_thr: 负阈值。
            pos_thr: 正阈值。
            per_channel_loss: 是否逐通道计算损失，默认为False。
        """
        y = self.quantize(data, x_min=neg_thr, x_max=pos_thr)       # 量化输入数据
        temp_sum = torch.sum(((data - y) ** 2).view(len(data), -1), dim=1)  # 计算量化均方误差
        # if we want to return the MSE loss of each channel separately, speeds up the per-channel
        # grid search
        if per_channel_loss:                        # 如果逐通道计算损失，则直接返回损失
            return to_numpy(temp_sum)
        else:                                       # 否则，返回损失的总和
            return to_numpy(torch.sum(temp_sum))

    @property
    def step_size(self):
        """
        计算步长，即每个候选的范围增量。
        """
        if self.one_sided_dist is None:
            raise NoDataPassedError()

        return self.max_search_range / self.num_candidates

    @property
    def optimization_method(self):
        """
        根据优化方法和数据分布，返回相应的优化函数。
        """
        if self.one_sided_dist is None:
            raise NoDataPassedError()

        if self.opt_method == OptMethod.grid:
            # Grid search method
            if self.one_sided_dist or self.quantizer.symmetric:     # 若数据为单边分布或量化器对称量化
                # 1-D grid search
                return self._perform_1D_search                      # 返回1D网格搜索方法
            else:
                # 2-D grid_search
                return self._perform_2D_search                      # 否则，返回2D网格搜索方法
        elif self.opt_method == OptMethod.golden_section:
            # Golden section method
            if self.one_sided_dist or self.quantizer.symmetric:     # 若数据为单边分布或量化器对称量化
                return self._golden_section_symmetric               # 返回对称黄金分割法
            else:
                return self._golden_section_asymmetric              # 否则，返回不对称黄金分割法
        else:
            raise NotImplementedError("Optimization Method not Implemented")

    def quantize(self, x_float, x_min=None, x_max=None):
        """
        量化方法。
        使用临时量化器对象，将输入数据x_float进行量化，返回量化后的数据。
        
        Args:
            x_float: 需要量化的浮点数张量。
            x_min: 最小阈值，默认为None。
            x_max: 最大阈值，默认为None。
        """
        temp_q = copy.deepcopy(self.quantizer)          # 复制量化器对象
        # In the current implementation no optimization procedure requires temp quantizer for
        # loss_fx to be per-channel
        temp_q.per_channel = False                      # 设置量化器为全局模式
        if x_min or x_max:
            temp_q.set_quant_range(x_min, x_max)        # 若传入了最小值和最大值，则设置量化范围
        return temp_q(x_float)                          # 返回量化后的数据

    def _define_search_range(self, data):
        """
        定义搜索范围。
        根据数据分布和量化器对称性，定义搜索范围和初始化损失数组。
        
        Args:
            data: 输入数据张量。
        """
        self.channel_groups = len(data) if self.per_channel else 1                  # 计算通道组数
        self.current_xmax = torch.zeros(self.channel_groups, device=data.device)    # 初始化当前最大值
        self.current_xmin = torch.zeros(self.channel_groups, device=data.device)    # 初始化当前最小值

        if self.one_sided_dist or self.quantizer.symmetric:             # 若数据为单边分布或量化器对称量化
            # 1D search space
            self.loss_array = np.zeros(                                 # 初始化损失数组
                (self.channel_groups, self.num_candidates + 1)
            )  # 1D search space
            self.loss_array[:, 0] = np.inf  # exclude interval_start=interval_finish
            # Defining the search range for clipping thresholds
            self.max_pos_thr = max(abs(float(data.min())), float(data.max())) + self.range_margin   # 计算最大正阈值
            self.max_neg_thr = -self.max_pos_thr * self.expand_range                                # 计算最大负阈值
            self.max_search_range = self.max_pos_thr * self.expand_range                            # 计算最大搜索范围
        else:                                                           # 否则，为不对称量化
            # 2D search space (3rd and 4th index correspond to asymmetry where fourth
            # index represents whether the skew is positive (0) or negative (1))
            self.loss_array = np.zeros(                                                             # 初始化损失数组
                [self.channel_groups, self.num_candidates + 1, self.max_int_skew, 2]
            )  # 2D search space
            self.loss_array[:, 0, :, :] = np.inf  # exclude interval_start=interval_finish
            # Define the search range for clipping thresholds in asymmetric case
            self.max_pos_thr = float(data.max()) + self.range_margin                                # 计算最大正阈值
            self.max_neg_thr = float(data.min()) - self.range_margin                                # 计算最大负阈值
            self.max_search_range = max(abs(self.max_pos_thr), abs(self.max_neg_thr))               # 计算最大搜索范围

    def _perform_1D_search(self, data):
        """
        在1D搜索空间中进行网格搜索，遍历所有候选阈值组合，计算每个组合的损失。
        选择损失最小的阈值组合作为最佳范围。
        Grid search through all candidate quantizers in 1D to find the best
        The loss is accumulated over all batches without any momentum
        :param data: input tensor
        """
        for cand_index in range(1, self.num_candidates + 1):        # 遍历所有候选索引
            neg_thr = 0 if self.one_sided_dist else -self.step_size * cand_index    # 计算负阈值，若数据为单边分布则为0，否则为-cand_index * 步长
            pos_thr = self.step_size * cand_index                                   # 计算正阈值，为cand_index * 步长

            self.loss_array[:, cand_index] += self.loss_fx(                         # 计算损失，累积到损失数组中
                data, neg_thr, pos_thr, per_channel_loss=self.per_channel
            )

        min_cand = self.loss_array.argmin(axis=1)                   # 找到每个通道组最小损失对应的候选索引
        xmin = (                                                    # 计算最佳的最小值，若数据为单边分布则为0，否则为-min_cand * 步长
            np.zeros(self.channel_groups) if self.one_sided_dist else -self.step_size * min_cand
        ).astype(np.single)
        xmax = (self.step_size * min_cand).astype(np.single)        # 计算最佳的最大值，为min_cand * 步长
        self.current_xmax = torch.tensor(xmax).to(device=data.device)   # 将最大值转换为张量
        self.current_xmin = torch.tensor(xmin).to(device=data.device)   # 将最小值转换为张量

    def forward(self, data):
        """
        首次调用时，初始化搜索范围。
        根据优化方法执行相应的搜索/优化过程。
        返回最佳的最小值current_xmin和最大值current_xmax。
        
        Args:
            data: 输入数据张量。
        """
        if self.loss_array is None:
            # Initialize search range on first batch, and accumulate losses with subsequent calls

            # Decide whether input distribution is one-sided
            if self.one_sided_dist is None:
                self.one_sided_dist = bool((data.min() >= 0).item())    # 判断数据是否为单边分布

            # Define search
            self._define_search_range(data)                             # 首次调用时，初始化搜索范围

        # Perform Search/Optimization for Quantization Ranges
        self.optimization_method(data)                                  # 根据优化方法执行相应的搜索/优化过程

        return self.current_xmin, self.current_xmax

    def reset(self):
        """
        重置方法。
        重置范围估计器的状态，清除损失数组，使之可以重新开始估计过程。
        """
        super().reset()
        self.loss_array = None

    def extra_repr(self):
        """
        额外的字符串表示方法。
        """
        repr = "opt_method={}".format(self.opt_method.name)             # 优化方法
        if self.opt_method == OptMethod.grid:
            repr += " ,num_candidates={}".format(self.num_candidates)   # 若为网格搜索，则添加候选数量
        return repr


class FP_MSE_Estimator(RangeEstimatorBase):
    """
    通过浮点均方误差（FP MSE）的方法来估计张量的数值范围（最小值和最大值）。
    该类在量化过程中使用，以确定最佳的量化范围，使量化后的数据尽可能地接近原始数据，从而最小化量化误差。
    """
    def __init__(
        self, num_candidates=100, opt_method=OptMethod.grid, range_margin=0.5, *args, **kwargs
    ):
        """
        Args:
            num_candidates: 候选数量，用于网格搜索，默认值为100。
            opt_method: 优化方法，可选值为OptMethod枚举类中的值，默认为网格搜索。
            range_margin: 范围边界，用于限制范围的边界，默认值为0.5。
        """
        super(FP_MSE_Estimator, self).__init__(*args, **kwargs)
        assert opt_method == OptMethod.grid     # 断言优化方法是网格搜索

        self.num_candidates = num_candidates
        self.mses = self.search_grid = None

    def _define_search_range(self, x, mbit_list):
        """
        定义搜索范围。
        根据输入数据和尾数位数，定义搜索范围和初始化损失数组。
        生成用于量化的候选最大值和对应的均方误差。
        
        Args:
            x: 输入数据张量。
            mbit_list: 有效尾数位数的列表。
        """
        # 根据是否逐通道，展平输入数据
        if self.per_channel:
            x = x.view(x.shape[0], -1)
        else:
            x = x.view(1, -1)
        # 计算每个通道的最大绝对值
        mxs = [torch.max(torch.abs(xc.min()), torch.abs(xc.max())) for xc in x]

        if self.search_grid is None:                # 若搜索网格未定义
            assert self.mses is None                # 断言损失数组未定义

            lsp = [torch.linspace(0.1 * mx.item(), 1.2 * mx.item(), 111) for mx in mxs]     # 为每个通道生成一个线性空间

            # 111 x n_channels
            search_grid = torch.stack(lsp).to(x.device).transpose(0, 1)                     # 将线性空间堆叠为搜索网格，形状为[111, n_channels]

            # mbits x 111 x n_channels (or 1 in case not --per-channel)
            mses = torch.stack([torch.zeros_like(search_grid) for _ in range(len(mbit_list))])  # 初始化损失数组

            self.mses = mses
            self.search_grid = search_grid

        return self.search_grid, self.mses

    def forward(self, x):
        """
        前向传播方法。
        通过遍历不同的有效尾数位数和搜索网格，计算每种配置下的均方误差（MSE）。
        选择最优的尾数位数和量化范围，使得量化后的数据与原始数据的MSE最小。
        返回最佳的最小值和最大值，用于量化。
        
        Args:
            x: 输入数据张量。
        """
        # 获取有效尾数位数列表
        mbit_list = [float(self.quantizer.mantissa_bits)]

        # 若包含尾数位数，则定义尾数位数列表为[1, 2, ..., n_bits - sign_bits - 1]
        if self.quantizer.mse_include_mantissa_bits:
            # highest possible value is self.n_bits - self.sign_bits - 1
            mbit_list = [
                float(x) for x in range(1, self.quantizer.n_bits - self.quantizer.sign_bits)
            ]

        # 定义搜索范围并初始化MSE数组
        search_grid, mses = self._define_search_range(x, mbit_list)

        # 断言MSE数组的形状与search_grid匹配
        assert mses.shape[1:] == search_grid.shape, f"{mses.shape}, {search_grid.shape}"

        # Need to do this here too to get correct search range
        # 确定符号位
        sign_bits = int(torch.any(x < 0)) if self.quantizer.allow_unsigned else 1

        # 确定需要平均的维度
        meandims = list(torch.arange(len(x.shape)))
        if self.per_channel:
            meandims = meandims[1:]
            
        # 遍历不同的尾数位数和搜索网格，计算每种配置下的均方误差
        for m, mbits in enumerate(mbit_list):
            mbits = torch.Tensor([mbits]).to(x.device)      # 将尾数位数转换为张量
            self.quantizer.mantissa_bits = mbits            # 设置量化器的尾数位数
            for i, maxval in enumerate(search_grid):        # 遍历搜索网格
                x_min, x_max = sign_bits * -1.0 * maxval, maxval    # 计算最小值和最大值
                self.quantizer.set_quant_range(x_min, x_max)        # 设置量化范围
                xfp = self.quantizer(x)                     # 量化输入数据

                # get MSE per channel (mean over all non-channel dims)
                mse = ((x - xfp) ** 2).mean(meandims)       # 计算均方误差
                mses[m, i, :] += mse                        # 累积到MSE数组中

        # Find best mbits per channel
        # 选择最佳尾数位数
        best_mbits_per_channel = mses.min(1)[0].argmin(0)   # 获取每通道最小MSE对应的最佳尾数位数索引

        # Get plurality vote on mbits
        best_mbit_idx = torch.mode(best_mbits_per_channel).values.item()    # 通过多数投票确定全局最佳的尾数位数索引
        best_mbits = float(mbit_list[best_mbit_idx])                        # 获取全局最佳的尾数位数

        # then, find best per-channel scale for best mbit
        # first, get the MSES for the best mbit, then argmin over linspace dim to get best index per channel
        # 确定最佳量化范围
        mses = mses[best_mbit_idx].argmin(0)                                # 对于选定的最佳尾数位数，找到每个通道的最佳量化范围索引
        # then, for each channel, get the argmin MSE max value
        maxval = torch.tensor([search_grid[mses[i], i] for i in range(search_grid.shape[-1])]).to(  # 获取最佳的maxval
            x.device
        )

        self.quantizer.mantissa_bits = torch.tensor(best_mbits).to(         # 设置量化器的最佳尾数位数
            self.quantizer.mantissa_bits.device
        )

        maxval = maxval.to(self.quantizer.maxval.device)
        return sign_bits * -1.0 * maxval, maxval                            # 返回最佳的最小值和最大值


def estimate_range_line_search(W, quant, num_candidates=None):
    """
    用于通过线搜索方法（如网格搜索）估计张量 `W` 的数值范围（最小值和最大值）。
    该函数利用 `LineSearchEstimator` 类，根据给定的量化器 `quant` 和候选数量 `num_candidates` 来确定最佳的量化范围，
    以最小化量化误差（如均方误差，MSE）。

    Args:
        W: 输入数据张量。
        quant: 量化器对象。
        num_candidates: 候选数量，用于网格搜索，默认为None。
    """
    # 根据候选数量，创建LineSearchEstimator对象
    if num_candidates is None:
        est_fp = LineSearchEstimator(quantizer=quant)
    else:
        est_fp = LineSearchEstimator(quantizer=quant, num_candidates=num_candidates)

    mse_range_min_fp, mse_range_max_fp = est_fp.forward(W)      # 通过线搜索方法估计张量的数值范围
    return (mse_range_min_fp, mse_range_max_fp)


class NoDataPassedError(Exception):
    """
    自定义异常类，用于在范围估计器未接收到数据时抛出错误。
    Raised data has been passed into the Range Estimator
    """

    def __init__(self):
        super().__init__("Data must be pass through the range estimator to be initialized")


class RangeEstimators(ClassEnumOptions):
    """
    枚举类，定义和管理不同类型的范围估计器。
    通过MethodMap，将不同的范围估计器类映射为枚举成员，便于在代码中统一引用和调用。
    """
    current_minmax = MethodMap(CurrentMinMaxEstimator)
    allminmax = MethodMap(AllMinMaxEstimator)
    running_minmax = MethodMap(RunningMinMaxEstimator)
    MSE = MethodMap(FP_MSE_Estimator)
