#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import inspect
import torch

from quantization.quantizers.rounding_utils import scale_grad_func, round_ste_func
from .utils import QuantizerNotInitializedError
from .base_quantizers import QuantizerBase


class AsymmetricUniformQuantizer(QuantizerBase):
    """
    继承自 `QuantizerBase`，实现场景下的非对称均匀量化（Asymmetric Uniform Quantization）并使用直通估计器（STE）策略。
    在前向传播中对输入进行量化处理，在反向传播中直接传递梯度，忽略量化过程。
    PyTorch Module that implements Asymmetric Uniform Quantization using STE.
    Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.

    Parameters
    ----------
    n_bits: int
        Number of bits for quantization.
    scale_domain: str ('log', 'linear) with default='linear'
        Domain of scale factor
    per_channel: bool
        If True: allows for per-channel quantization
    """

    def __init__(
        self,
        n_bits,
        scale_domain="linear",
        discretizer=round_ste_func,
        discretizer_args=tuple(),
        grad_scaling=False,
        eps=1e-8,
        **kwargs
    ):
        """
        Args:
            n_bits: 量化位数
            scale_domain: 缩放因子的域，支持'linear'和'log'两种选择，默认为'linear'
            discretizer: 离散化函数或类，默认为round_ste_func
            discretizer_args: 离散化函数的参数，默认为空元组
            grad_scaling: 是否对梯度进行缩放，默认为False
            eps: 量化过程中的一个小值，默认为1e-8
        """
        super().__init__(n_bits=n_bits, **kwargs)

        assert scale_domain in ("linear", "log")
        self.register_buffer("_delta", None)            # 量化缩放因子
        self.register_buffer("_zero_float", None)       # 量化零点

        if inspect.isclass(discretizer):                # 如果discretizer是一个类，则实例化该类
            self.discretizer = discretizer(*discretizer_args)
        else:                                           # 否则直接使用discretizer
            self.discretizer = discretizer

        self.scale_domain = scale_domain
        self.grad_scaling = grad_scaling
        self.eps = eps

    # A few useful properties
    @property
    def delta(self):
        """
        获取量化缩放因子。
        """
        if self._delta is not None:
            return self._delta
        else:
            raise QuantizerNotInitializedError()

    @property
    def zero_float(self):
        """
        获取量化零点。
        """
        if self._zero_float is not None:
            return self._zero_float
        else:
            raise QuantizerNotInitializedError()

    @property
    def is_initialized(self):
        """
        检查量化器是否已经初始化。
        """
        return self._delta is not None

    @property
    def symmetric(self):
        """
        指示量化是否是对称的。
        """
        return False

    @property
    def int_min(self):
        """
        获取整数网格的最小值。
        """
        # integer grid minimum
        return 0.0

    @property
    def int_max(self):
        """
        获取整数网格的最大值。
        """
        # integer grid maximum
        return 2.0**self.n_bits - 1

    @property
    def scale(self):
        """
        获取量化缩放因子。
        """
        if self.scale_domain == "linear":       # 线性域下，直接返回量化缩放因子
            return torch.clamp(self.delta, min=self.eps)
        elif self.scale_domain == "log":        # 对数域下，返回量化缩放因子的指数
            return torch.exp(self.delta)

    @property
    def zero_point(self):
        """
        获取量化零点。
        """
        zero_point = self.discretizer(self.zero_float)      # 离散化函数处理量化零点
        zero_point = torch.clamp(zero_point, self.int_min, self.int_max)
        return zero_point

    @property
    def x_max(self):
        """
        获取量化范围的最大值。
        """
        return self.scale * (self.int_max - self.zero_point)

    @property
    def x_min(self):
        """
        获取量化范围的最小值。
        """
        return self.scale * (self.int_min - self.zero_point)

    def to_integer_forward(self, x_float, *args, **kwargs):
        """
        将全精度输入张量x_float量化为整数表示x_int。
        Qunatized input to its integer representation
        Parameters
        ----------
        x_float: PyTorch Float Tensor
                Full-precision Tensor

        Returns
        -------
        x_int: PyTorch Float Tensor of integers
        """
        if self.grad_scaling:                                   # 如果对梯度进行缩放
            grad_scale = self.calculate_grad_scale(x_float)     # 计算梯度缩放因子
            scale = scale_grad_func(self.scale, grad_scale)     # 调整量化缩放因子
            zero_point = (                                      # 调整量化零点
                self.zero_point if self.symmetric else scale_grad_func(self.zero_point, grad_scale)
            )
        else:                                                   # 如果不对梯度进行缩放
            scale = self.scale                                  # 直接使用量化缩放因子
            zero_point = self.zero_point                        # 直接使用量化零点

        x_int = self.discretizer(x_float / scale) + zero_point  # 量化处理
        x_int = torch.clamp(x_int, self.int_min, self.int_max)  # 限制量化结果的范围

        return x_int

    def forward(self, x_float, *args, **kwargs):
        """
        前向传播方法，用于将输入张量x_float进行量化处理（量化为整数，然后再反量化为浮点数）。
        Quantizes (quantized to integer and the scales back to original domain)
        Parameters
        ----------
        x_float: PyTorch Float Tensor
                Full-precision Tensor

        Returns
        -------
        x_quant: PyTorch Float Tensor
                Quantized-Dequantized Tensor
        """
        if self.per_channel:                                    # 如果是逐通道量化，则调整量化参数以适应输入张量的通道
            self._adjust_params_per_channel(x_float)

        if self.grad_scaling:                                   # 如果对梯度进行缩放
            grad_scale = self.calculate_grad_scale(x_float)     # 计算梯度缩放因子
            scale = scale_grad_func(self.scale, grad_scale)     # 调整量化缩放因子
            zero_point = (                                      # 调整量化零点
                self.zero_point if self.symmetric else scale_grad_func(self.zero_point, grad_scale)
            )
        else:                                                   # 如果不对梯度进行缩放
            scale = self.scale                                  # 直接使用量化缩放因子
            zero_point = self.zero_point                        # 直接使用量化零点

        x_int = self.to_integer_forward(x_float, *args, **kwargs)   # 将输入张量量化为整数
        x_quant = scale * (x_int - zero_point)                  # 反量化为浮点数

        return x_quant

    def calculate_grad_scale(self, quant_tensor):
        """
        计算梯度缩放因子。
        """
        # 设置正级别的数量
        num_pos_levels = self.int_max  # Qp in LSQ paper
        # 计算元素总数
        num_elements = quant_tensor.numel()  # nfeatures or nweights in LSQ paper
        # 如果是逐通道量化，调整元素总数
        if self.per_channel:
            # In the per tensor case we do not sum the gradients over the output channel dimension
            num_elements /= quant_tensor.shape[0]

        return (num_pos_levels * num_elements) ** -0.5  # 1 / sqrt (Qn * nfeatures)

    def _adjust_params_per_channel(self, x):
        """
        根据输入张量x的每个通道调整量化参数。
        Adjusts the quantization parameter tensors (delta, zero_float)
        to the input tensor shape if they don't match
        Parameters
        ----------
        x: input tensor
        """
        if x.ndim != self.delta.ndim:                       # 如果输入张量的维度与量化参数的维度不匹配，则调整量化参数的维度
            new_shape = [-1] + [1] * (len(x.shape) - 1)     # 只保留通道维度，其余维度设置为1
            self._delta = self.delta.view(new_shape)        # 调整量化缩放因子的维度
            if self._zero_float is not None:                # 如果量化零点不为空，则调整量化零点的维度
                self._zero_float = self._zero_float.view(new_shape)

    def _tensorize_min_max(self, x_min, x_max):
        """
        将提供的最小值和最大值转换为张量，并进行验证和调整。
        Converts provided min max range into tensors
        Parameters
        ----------
        x_min: float or PyTorch 1D tensor
        x_max: float or PyTorch 1D tensor

        Returns
        -------
        x_min: PyTorch Tensor 0 or 1-D
        x_max: PyTorch Tensor 0 or 1-D
        """
        # Ensure a torch tensor
        # 如果x_min、x_max不是张量，则转换为浮点张量
        if not torch.is_tensor(x_min):
            x_min = torch.tensor(x_min).float()
            x_max = torch.tensor(x_max).float()

        # 如果x_min是多维，且长度大于1，且不是逐通道量化，则抛出异常
        if x_min.dim() > 0 and len(x_min) > 1 and not self.per_channel:
            print(x_min)
            print(self.per_channel)
            raise ValueError(
                "x_min and x_max must be a float or 1-D Tensor"
                " for per-tensor quantization (per_channel=False)"
            )
        # Ensure we always use zero and avoid division by zero
        # 确保x_min不超过0，x_max不低于eps，避免除以0
        x_min = torch.min(x_min, torch.zeros_like(x_min))
        x_max = torch.max(x_max, torch.ones_like(x_max) * self.eps)

        return x_min, x_max

    def set_quant_range(self, x_min, x_max):
        """
        根据提供的最小值和最大值初始化量化参数。
        Instantiates the quantization parameters based on the provided
        min and max range

        Parameters
        ----------
        x_min: tensor or float
                Quantization range minimum limit
        x_max: tensor of float
                Quantization range minimum limit
        """
        self.x_min_fp32, self.x_max_fp32 = x_min, x_max
        x_min, x_max = self._tensorize_min_max(x_min, x_max)    # 将最小值和最大值转换为张量
        self._delta = (x_max - x_min) / self.int_max            # 计算量化缩放因子
        self._zero_float = (-x_min / self.delta).detach()       # 计算量化零点

        # 如果量化缩放因子的域是对数域，则对量化缩放因子取对数
        if self.scale_domain == "log":
            self._delta = torch.log(self.delta)

        self._delta = self._delta.detach()

    def make_range_trainable(self):
        """
        将量化范围参数转换为可训练的参数。
        """
        # Converts trainable parameters to nn.Parameters
        if self.delta not in self.parameters():                 # 如果量化缩放因子不是参数，则转换为参数，并将量化零点也转换为参数
            self._delta = torch.nn.Parameter(self._delta)
            self._zero_float = torch.nn.Parameter(self._zero_float)

    def fix_ranges(self):
        """
        将可训练的量化参数转换回固定的缓冲区。
        """
        # Removes trainable quantization params from nn.Parameters
        if self.delta in self.parameters():                     # 如果量化缩放因子是参数，则转换为缓冲区，并将量化零点也转换为缓冲区
            _delta = self._delta.data
            _zero_float = self._zero_float.data
            del self._delta  # delete the parameter
            del self._zero_float
            self.register_buffer("_delta", _delta)
            self.register_buffer("_zero_float", _zero_float)


class SymmetricUniformQuantizer(AsymmetricUniformQuantizer):
    """
    继承自 `AsymmetricUniformQuantizer`，实现场景下的对称均匀量化（Symmetric Uniform Quantization）并使用直通估计器（STE）策略。
    在前向传播中对输入进行量化处理，在反向传播中直接传递梯度，忽略量化过程。
    PyTorch Module that implements Symmetric Uniform Quantization using STE.
    Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.

    Parameters
    ----------
    n_bits: int
        Number of bits for quantization.
    scale_domain: str ('log', 'linear) with default='linear'
        Domain of scale factor
    per_channel: bool
        If True: allows for per-channel quantization
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_signed", None)           # 注册缓冲区_signed，用于指示是否为有符号量化

    @property
    def signed(self):
        """
        获取是否为有符号量化。
        """
        if self._signed is not None:
            return self._signed.item()
        else:
            raise QuantizerNotInitializedError()

    @property
    def symmetric(self):
        """
        指示量化是否是对称的。
        """
        return True

    @property
    def int_min(self):
        """
        获取整数网格的最小值。
        """
        return -(2.0 ** (self.n_bits - 1)) if self.signed else 0

    @property
    def int_max(self):
        """
        获取整数网格的最大值。
        """
        pos_n_bits = self.n_bits - self.signed
        return 2.0**pos_n_bits - 1

    @property
    def zero_point(self):
        """
        获取量化零点。
        """
        return 0.0

    def set_quant_range(self, x_min, x_max):
        """
        设置量化范围，根据对称量化的要求调整量化参数。
        """
        self.x_min_fp32, self.x_max_fp32 = x_min, x_max
        x_min, x_max = self._tensorize_min_max(x_min, x_max)    # 将最小值和最大值转换为张量
        self._signed = x_min.min() < 0                          # 判断是否为有符号量化

        x_absmax = torch.max(x_min.abs(), x_max)                # 计算最大绝对值
        self._delta = x_absmax / self.int_max                   # 计算量化缩放因子

        # 如果量化缩放因子的域是对数域，则对量化缩放因子取对数
        if self.scale_domain == "log":
            self._delta = torch.log(self._delta)

        self._delta = self._delta.detach()

    def make_range_trainable(self):
        """
        将量化范围参数转换为可训练的参数。
        """
        # Converts trainable parameters to nn.Parameters
        if self.delta not in self.parameters():                 # 如果量化缩放因子不是参数，则转换为参数
            self._delta = torch.nn.Parameter(self._delta)

    def fix_ranges(self):
        """
        将可训练的量化参数转换回固定的缓冲区。
        """
        # Removes trainable quantization params from nn.Parameters
        if self.delta in self.parameters():                     # 如果量化缩放因子是参数，则转换为缓冲区
            _delta = self._delta.data
            del self._delta  # delete the parameter
            self.register_buffer("_delta", _delta)

    def generate_grid(self):
        """
        生成量化的整数网格映射回原始域的点
        """
        x_int_rng = torch.arange(self.int_min, self.int_max + 1)    # 创建整数范围
        grid = self.scale * (x_int_rng - self.zero_point)       # 映射回原始域
        return grid
