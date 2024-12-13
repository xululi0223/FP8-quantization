# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import torch

from utils.distributions import ClippedGaussDistr, UniformDistr, ClippedStudentTDistr
from quantization.quant_error_estimator import (
    compute_expected_quant_mse,
    compute_expected_dot_prod_mse,
)
from quantization.quantizers.fp8_quantizer import FPQuantizer
from quantization.range_estimators import estimate_range_line_search
from quantization.quantizers.uniform_quantizers import SymmetricUniformQuantizer
from utils import seed_all


def compute_quant_error(distr, n_bits=8, n_samples=5000000, seed=10):
    """
    主要功能是评估不同浮点位数（FP8）量化方案下的量化误差。
    它通过对给定分布（`distr`）进行采样，应用不同的量化方法，计算量化后的均方误差（MSE）和信噪比（SQNR），以评估量化方法的性能。

    Args:
        distr: 分布对象，用于生成随机样本。
        n_bits: 量化位数，默认为8。
        n_samples: 采样数量，默认为5000000。
        seed: 随机种子，默认为10。
    """
    # 设置随机种子
    seed_all(seed)
    # 生成分布样本
    distr_sample = torch.tensor(distr.sample((n_samples,)))
    # 遍历不同的指数位数
    for exp_bits in [5, 4, 3, 2, 0]:
        # 计算尾数位数
        mantissa_bits = n_bits - 1 - exp_bits
        # 选择量化器
        if exp_bits > 0:
            quant = FPQuantizer(n_bits=8, mantissa_bits=mantissa_bits, set_maxval=True) # 浮点数量化器
        elif exp_bits == 0:
            quant = SymmetricUniformQuantizer(n_bits=n_bits)                            # 对称均匀量化器

        # 估计量化范围
        (quant_range_min, quant_range_max) = estimate_range_line_search(distr_sample, quant)
        # 计算量化后的期望均方误差（MSE）
        quant_expected_mse = compute_expected_quant_mse(
            distr, quant, quant_range_min, quant_range_max, n_samples
        )
        # 计算量化后的信噪比（SQNR）
        quant_sqnr = -10.0 * np.log10(quant_expected_mse)

        # 计算量化后的点积期望均方误差（Dot Product Expected MSE）
        dot_prod_expected_mse = compute_expected_dot_prod_mse(
            distr,
            distr,
            quant,
            quant,
            quant_range_min,
            quant_range_max,
            quant_range_min,
            quant_range_max,
        )
        # 计算量化后的点积信噪比（Dot Product SQNR）
        dot_prod_sqnr = -10.0 * np.log10(dot_prod_expected_mse)

        # 打印结果
        print(
            "FP8 {} E {} M Quantization: expected MSE {:.2e}".format(
                exp_bits, mantissa_bits, quant_expected_mse
            ),
            " SQNR ",
            "{:.2e}\n".format(quant_sqnr),
            "Dot product:".rjust(23),
            " expected MSE {:.2e}".format(dot_prod_expected_mse),
            " SQNR ",
            "{:.2e}".format(dot_prod_sqnr),
        )


if __name__ == "__main__":
    # 定义分布列表
    distr_list = [
        UniformDistr(range_min=-1.0, range_max=1.0, params_dict={}),
        ClippedGaussDistr(params_dict={"mu": 0.0, "sigma": 1.0}, range_min=-10.0, range_max=10.0),
        ClippedStudentTDistr(params_dict={"nu": 8.0}, range_min=-100.0, range_max=100.0),
    ]

    # 遍历分布并计算量化误差
    for distr in distr_list:
        print("*" * 80)
        distr.print()
        compute_quant_error(distr)
