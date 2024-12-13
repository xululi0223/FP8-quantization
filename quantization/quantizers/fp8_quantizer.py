#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
import torch.nn as nn
from quantization.quantizers.base_quantizers import QuantizerBase
import numpy as np
from itertools import product
from torch.autograd import Function
from quantization.quantizers.rounding_utils import round_ste_func


def generate_all_values_fp(
    num_total_bits: int = 8, num_exponent_bits: int = 4, bias: int = 8
) -> list:
    """
    生成所有可能的FP8（8位浮点数）值。
    通过遍历符号位、指数位和尾数位的所有组合，计算出对应的浮点数值，并将其排序后返回。
    
    Args:
        num_total_bits: 总位数，默认为8。
        num_exponent_bits: 指数位数，默认为4。
        bias: 偏置值，默认为8。
    """
    # 计算尾数位数
    num_fraction_bits = num_total_bits - 1 - num_exponent_bits

    all_values = []             # 初始化存储所有值的列表
    exp_lower = -bias           # 计算指数的下界
    for S in [-1.0, 1.0]:       # 遍历符号位
        for E_str_iter in product(*[[0, 1]] * num_exponent_bits):       # 遍历指数位
            for F_str_iter in product(*[[0, 1]] * num_fraction_bits):   # 遍历尾数位
                # 构建二进制字符串
                E_str = "".join(str(i) for i in E_str_iter)             # 将指数位的二进制数组合转换为字符串形式
                F_str = "".join(str(i) for i in F_str_iter)             # 将尾数位的二进制数组合转换为字符串形式

                # encoded exponent
                # 解码指数值
                E_enc = decode_binary_str(E_str)                        # 将二进制字符串转换为十进制数值
                E_eff = E_enc - bias                                    # 计算有效指数值
                # 判断是否为次正规数
                if E_eff == exp_lower:
                    is_subnormal = 1
                else:
                    is_subnormal = 0

                # 解码尾数值
                F_enc = decode_binary_str(F_str) * 2**-num_fraction_bits    # 将二进制字符串转换为十进制数值
                F_eff = F_enc + 1 - is_subnormal                        # 根据是否为次正规数调整有效尾数值

                fp8_val = S * 2.0 ** (E_enc - bias + is_subnormal) * F_eff  # 根据符号、指数和尾数计算最终的FP8值
                all_values.append(fp8_val)
    res = np.array(all_values)
    res = np.sort(res)
    return res


def generate_all_float_values_scaled(num_total_bits, num_exp_bits, exp_bias, range_limit_fp):
    """
    生成缩放后的所有FP8浮点数值。
    
    Args:
        num_total_bits: 总位数，默认为8。
        num_exp_bits: 指数位数，默认为4。
        exp_bias: 偏置值，默认为8。
        range_limit_fp: 浮点数值的范围限制。
    """
    grid = generate_all_values_fp(num_total_bits, num_exp_bits, exp_bias)   # 生成所有可能的FP8值
    float_max_abs_val = np.max(np.abs(grid))                                # 计算最大绝对值

    float_scale = float_max_abs_val / range_limit_fp                        # 计算缩放比例
    floats_all = grid / float_scale                                         # 缩放所有FP8值
    return floats_all


def decode_float8(S, E, F, bias=16):
    """
    将FP8的符号位、指数位和尾数位解码为对应的浮点数值。
    
    Args:
        S: 符号位，0代表正数，1代表负数。
        E: 指数位的二进制字符串。
        F: 尾数位的二进制字符串。
        bias: 偏置值，默认为16。
    """
    sign = -2 * int(S) + 1                                                  # 解码符号位
    exponent = int(E, 2) if E else 0                                        # 解码指数位
    # Normal FP8   : exponent > 0 : result = 2^(exponent-bias) * 1.F
    # Subnormal FP8: exponent == 0: result = 2^(-bias+1)       * 0.F
    # Lowest quantization bin: 2^(-bias+1)       * {0.0 ... 1 + (2^mantissa-1)/2^mantissa}
    # All other bins         : 2^(exponent-bias) * {1.0 ... 1 + (2^mantissa-1)/2^mantissa}; exponent > 0
    A = int(exponent != 0)                                                  # 判断是否为正规数，如果是则A=1，否则A=0
    fraction = A + sum([2 ** -(i + 1) * int(a) for i, a in enumerate(F)])   # 计算尾数值
    exponent += int(exponent == 0)                                          # 调整指数值
    return sign * fraction * 2.0 ** (exponent - bias)                       # 计算最终的浮点数值


def i(x):
    """
    辅助函数，将输入x转换为一个包含该值的32位整数NumPy数组。
    """
    return np.array([x]).astype(np.int32)


def gen(n_bits, exponent_bits, bias):
    """
    生成所有可能的FP8浮点数值，并按升序排列。
    
    Args:
        n_bits: 总位数。
        exponent_bits: 指数位数。
        bias: 偏置值。
    """
    all_values = []                                                         # 初始化存储所有值的列表
    for s in product(*[[0, 1]] * 1):                                        # 遍历符号位
        for e in product(*[[0, 1]] * exponent_bits):                        # 遍历指数位
            for m in product(*[[0, 1]] * (n_bits - 1 - exponent_bits)):     # 遍历尾数位
                # 构建符号位、指数位和尾数位的二进制字符串
                s = str(s[0])
                e = "".join(str(i) for i in e)
                m = "".join(str(i) for i in m)
                all_values.append(decode_float8(s, e, m, bias=bias))        # 解码并存储浮点数值
    return sorted(all_values)


def get_max_value(num_exponent_bits: int = 4, bias: int = 8):
    """
    计算给定指数位数和偏置值下的FP8可表示的最大绝对值。
    """
    num_fraction_bits = 7 - num_exponent_bits                               # 计算尾数位数
    scale = 2**-num_fraction_bits                                           # 计算尾数位的缩放比例
    max_frac = 1 - scale                                                    # 计算尾数位的最大值
    max_value = 2 ** (2**num_exponent_bits - 1 - bias) * (1 + max_frac)     # 计算最大绝对值

    return max_value


def quantize_to_fp8_ste_MM(
    x_float: torch.Tensor,
    n_bits: int,
    maxval: torch.Tensor,
    num_mantissa_bits: torch.Tensor,
    sign_bits: int,
) -> torch.Tensor:
    """
    简化的FP8量化函数，利用直通估计器（STE）进行舍入。
    它根据输入张量动态调整缩放因子，通过限制输入值的范围并进行舍入，实现FP8量化。
    Simpler FP8 quantizer that exploits the fact that FP quantization is just INT quantization with
    scales that depend on the input.

    This allows to define FP8 quantization using STE rounding functions and thus learn the bias

    Args:
        x_float: 输入的浮点数张量。
        n_bits: 总位数。
        maxval: 最大值张量，用于限制输入范围。
        num_mantissa_bits: 尾数位数。
        sign_bits: 符号位数。
    """
    M = torch.clamp(round_ste_func(num_mantissa_bits), 1, n_bits - sign_bits)   # 计算尾数位数
    E = n_bits - sign_bits - M                                                  # 计算指数位数

    if maxval.shape[0] != 1 and len(maxval.shape) != len(x_float.shape):        # 如果maxval的第一个维度不为1且与x_float的维度不同
        maxval = maxval.view([-1] + [1] * (len(x_float.shape) - 1))             # 重新调整maxval的维度，匹配通道数
    bias = 2**E - torch.log2(maxval) + torch.log2(2 - 2 ** (-M)) - 1            # 计算偏置值，用于调整量化网格

    minval = -maxval if sign_bits == 1 else torch.zeros_like(maxval)            # 如果符号位为1，则最小值为-maxval，否则为0
    xc = torch.min(torch.max(x_float, minval), maxval)                          # 限制输入范围

    """
    2 notes here:
    1: Shifting by bias to ensure data is aligned to the scaled grid in case bias not in Z.
       Recall that implicitly bias := bias' - log2(alpha), where bias' in Z. If we assume 
       alpha in (0.5, 1], then alpha contracts the grid, which is equivalent to translate the
       data 'to the right' relative to the grid, which is what the subtraction of log2(alpha) 
       (which is negative) accomplishes. 
    2: Ideally this procedure doesn't affect gradients wrt the input (we want to use the STE).
       We can achieve this by detaching log2 of the (absolute) input.

    """

    # log_scales = torch.max((torch.floor(torch.log2(torch.abs(xc)) + bias)).detach(), 1.)
    log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(xc)) + bias)).detach(), 1.0) # 计算对数缩放因子

    scales = 2.0 ** (log_scales - M - bias)                                     # 计算实际缩放因子

    result = round_ste_func(xc / scales) * scales                               # 将输入xc缩放并进行STE舍入，乘以缩放因子，实现量化和反量化
    return result


class FP8QuantizerFunc(Function):
    """
    自定义的自动微分函数，用于实现FP8量化的前向和反向传播。
    前向传播使用FP8量化函数，反向传播则通过直通估计器（STE）直接传递梯度。
    """
    @staticmethod
    def forward(ctx, x_float, bias, num_exponent_bits):
        """
        前向传播过程。
        """
        return quantize_to_fp8_ste_MM(x_float, bias, num_exponent_bits)         # 调用FP8量化函数

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播过程。
        """
        return grad_output, None, None                                          # 直接传递梯度


def decode_binary_str(F_str):
    """
    将二进制字符串转换为对应的十进制整数值。
    """
    F = sum([2 ** -(i + 1) * int(a) for i, a in enumerate(F_str)]) * 2 ** len(F_str)
    return F


class FPQuantizer(QuantizerBase):
    """
    8位浮点数量化器，继承自基类QuantizerBase。
    实现了FP8的量化和反量化过程，支持可学习的最大值和尾数位数，通过STE进行梯度传递，以优化量化参数。
    8-bit Floating Point Quantizer
    """

    def __init__(
        self,
        *args,
        scale_domain=None,
        mantissa_bits=4,
        maxval=3,
        set_maxval=False,
        learn_maxval=False,
        learn_mantissa_bits=False,
        mse_include_mantissa_bits=True,
        allow_unsigned=False,
        **kwargs,
    ):
        """
        Args:
            scale_domain: 缩放域，默认为None。
            mantissa_bits: 尾数位数，默认为4。
            maxval: 最大值，默认为3。
            set_maxval: 是否设置最大值，默认为False。
            learn_maxval: 是否学习最大值，默认为False。
            learn_mantissa_bits: 是否学习尾数位数，默认为False。
            mse_include_mantissa_bits: 是否在均方误差中包含尾数位数，默认为True。
            allow_unsigned: 是否允许无符号量化，默认为False。
        """
        super().__init__(*args, **kwargs)

        self.mantissa_bits = mantissa_bits

        # 计算指数位数和默认偏置
        self.ebits = self.n_bits - self.mantissa_bits - 1
        self.default_bias = 2 ** (self.ebits - 1)

        # assume signed, correct when range setting turns out to be unsigned
        # 计算默认最大值
        default_maxval = (2 - 2 ** (-self.mantissa_bits)) * 2 ** (
            2**self.ebits - 1 - self.default_bias
        )

        # 初始化最大值
        self.maxval = maxval if maxval is not None else default_maxval

        # 将最大值和尾数位数转换为张量
        self.maxval = torch.Tensor([self.maxval])
        self.mantissa_bits = torch.Tensor([float(self.mantissa_bits)])

        self.set_maxval = set_maxval
        self.learning_maxval = learn_maxval
        self.learning_mantissa_bits = learn_mantissa_bits
        self.mse_include_mantissa_bits = mse_include_mantissa_bits

        self.allow_unsigned = allow_unsigned
        self.sign_bits = 1          # 符号位数，默认为1

    def forward(self, x_float):
        """
        前向传播方法，对输入的浮点数张量x_float进行FP8量化，并返回量化后的张量。
        """
        # 确保maxval和mantissa_bits与x_float在同一设备上
        if self.maxval.device != x_float.device:
            self.maxval = self.maxval.to(x_float.device)
        if self.mantissa_bits.device != x_float.device:
            self.mantissa_bits = self.mantissa_bits.to(x_float.device)

        # 调用FP8量化函数，对x_float进行量化
        res = quantize_to_fp8_ste_MM(
            x_float, self.n_bits, self.maxval, self.mantissa_bits, self.sign_bits
        )

        # 计算指数位数
        ebits = self.n_bits - self.mantissa_bits - 1
        return res

    def is_initialized(self):
        """
        检查量化器是否已经初始化。
        """
        return True

    def symmetric(self):
        """
        指示量化是否是对称的。
        """
        return False

    def effective_bit_width(self):
        """
        计算有效位宽。
        """
        return None

    def _make_unsigned(self, x_min):
        """
        判断是否可以进行无符号量化。
        """
        if isinstance(x_min, torch.Tensor):
            return self.allow_unsigned and torch.all(x_min >= 0)
        else:
            return self.allow_unsigned and x_min >= 0

    def set_quant_range(self, x_min, x_max):
        """
        设置量化器的量化范围，根据输入的最小值和最大值调整量化参数。
        """
        # 根据是否为无符号量化调整符号位数
        if self._make_unsigned(x_min):
            self.sign_bits = 0

        # 调整最大值
        if self.set_maxval:
            # 将x_max、x_min转换为张量
            if not isinstance(x_max, torch.Tensor):
                x_max = torch.Tensor([x_max]).to(self.maxval.device)
                x_min = torch.Tensor([x_min]).to(self.maxval.device)
            # 将maxval和mantissa_bits移动到与x_max相同的设备上
            if self.maxval.device != x_max.device:
                self.maxval = self.maxval.to(x_max.device)
            if self.mantissa_bits.device != x_max.device:
                self.mantissa_bits = self.mantissa_bits.to(x_max.device)

            mx = torch.abs(torch.max(torch.abs(x_min), x_max))  # 计算最大绝对值
            self.maxval = mx

            # 如果maxval不是张量，则将其转换为张量
            if not isinstance(self.maxval, torch.Tensor) or len(self.maxval.shape) == 0:
                self.maxval = torch.Tensor([self.maxval])

    def make_range_trainable(self):
        """
        将量化范围设置为可训练。
        """
        # 如果maxval可学习，则将其设置为可训练
        if self.learning_maxval:
            self.learn_maxval()
        # 如果mantissa_bits可学习，则将其设置为可训练
        if self.learning_mantissa_bits:
            self.learn_mantissa_bits()

    def learn_maxval(self):
        """
        将最大值设置为可学习。
        """
        self.learning_maxval = True                     # 设置最大值为可学习
        self.maxval = torch.nn.Parameter(self.maxval)   # 将maxval转换为可训练的参数

    def learn_mantissa_bits(self):
        """
        将尾数位数设置为可学习。
        """
        self.learning_mantissa_bits = True              # 设置尾数位数为可学习
        self.mantissa_bits = torch.nn.Parameter(self.mantissa_bits)     # 将mantissa_bits转换为可训练的参数

    def fix_ranges(self):
        """
        将量化范围固定为固定值。
        """
        # 如果maxval是可学习的，则将其转换为固定值
        if isinstance(self.maxval, nn.Parameter):
            self.parameter_to_fixed("maxval")
        # 如果mantissa_bits是可学习的，则将其转换为固定值
        if isinstance(self.mantissa_bits, nn.Parameter):
            self.parameter_to_fixed("mantissa_bits")

    def extra_repr(self):
        """
        额外字符串表示方法。
        """
        maxval = self.maxval

        M = torch.clamp(torch.round(self.mantissa_bits), 1, 7)      # 计算尾数位数
        E = 7 - M                                                   # 计算指数位数
        maxval = 2**E - torch.log2(self.maxval) + torch.log2(2 - 2 ** (-M)) - 1     # 计算最大值
        # 设置偏置字符串
        if maxval.shape[0] > 1:                                     # 逐通道量化
            bstr = "[per_channel]"
        else:
            bstr = f"{maxval.item()}"
        return f"Exponent: {E.item()} bits; mode: ; bias: {bstr}"
