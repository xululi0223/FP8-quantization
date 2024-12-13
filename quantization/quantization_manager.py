#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

from enum import auto

from torch import nn
from quantization.quantizers import QuantizerBase
from quantization.quantizers.utils import QuantizerNotInitializedError
from quantization.range_estimators import RangeEstimators, RangeEstimatorBase
from utils import BaseEnumOptions

from quantization.quantizers.uniform_quantizers import (
    SymmetricUniformQuantizer,
    AsymmetricUniformQuantizer,
)
from quantization.quantizers.fp8_quantizer import FPQuantizer

from utils import ClassEnumOptions, MethodMap


class QMethods(ClassEnumOptions):
    """
    继承自 `ClassEnumOptions`，用于定义和管理不同的量化方法。
    通过 `MethodMap`，它将具体的量化器类（如 `SymmetricUniformQuantizer`、`AsymmetricUniformQuantizer`、`FPQuantizer`）映射为枚举成员，
    方便在代码中统一引用和实例化不同的量化方法。
    """
    symmetric_uniform = MethodMap(SymmetricUniformQuantizer)
    asymmetric_uniform = MethodMap(AsymmetricUniformQuantizer)
    fp_quantizer = MethodMap(FPQuantizer)


class QuantizationManager(nn.Module):
    """
    用于实现量化及其范围估计的管理。
    它负责初始化量化器、设置量化范围、管理量化状态（如估计范围、固定范围、学习范围等），
    以及在前向传播过程中应用量化操作。
    Implementation of Quantization and Quantization Range Estimation

    Parameters
    ----------
    n_bits: int
        Number of bits for the quantization.
    qmethod: QMethods member (Enum)
        The quantization scheme to use, e.g. symmetric_uniform, asymmetric_uniform,
        qmn_uniform etc.
    init: RangeEstimators member (Enum)
        Initialization method for the grid from
    per_channel: bool
        If true, will use a separate quantization grid for each kernel/channel.
    x_min: float or PyTorch Tensor
        The minimum value which needs to be represented.
    x_max: float or PyTorch Tensor
        The maximum value which needs to be represented.
    qparams: kwargs
        dictionary of quantization parameters to passed to the quantizer instantiation
    range_estim_params: kwargs
         dictionary of parameters to passed to the range estimator instantiation
    """

    def __init__(
        self,
        qmethod: QuantizerBase = QMethods.symmetric_uniform.cls,
        init: RangeEstimatorBase = RangeEstimators.current_minmax.cls,
        per_channel=False,
        x_min=None,
        x_max=None,
        qparams=None,
        range_estim_params=None,
    ):
        """
        Args:
            qmethod: 量化方法，默认为对称均匀量化器。
            init: 范围估计初始化方法，默认为当前最小最大值估计器。
            per_channel: 是否按通道进行量化，默认为 False。
            x_min: 最小值，用于初始化量化器的范围。
            x_max: 最大值，用于初始化量化器的范围。
            qparams: 量化器参数。
            range_estim_params: 范围估计器参数。
        """
        super().__init__()
        self.state = Qstates.estimate_ranges
        self.qmethod = qmethod
        self.init = init
        self.per_channel = per_channel
        self.qparams = qparams if qparams else {}
        self.range_estim_params = range_estim_params if range_estim_params else {}
        self.range_estimator = None

        # define quantizer
        # 实例化量化器
        self.quantizer = self.qmethod(per_channel=self.per_channel, **qparams)
        self.quantizer.state = self.state

        # define range estimation method for quantizer initialisation
        # 如果提供了 x_min 和 x_max，则直接设置量化器的范围并固定范围
        if x_min is not None and x_max is not None:
            self.set_quant_range(x_min, x_max)
            self.fix_ranges()
        else:
            # 如果没有提供 x_min 和 x_max，则实例化范围估计器
            # set up the collector function to set the ranges
            self.range_estimator = self.init(
                per_channel=self.per_channel, quantizer=self.quantizer, **self.range_estim_params
            )

    @property
    def n_bits(self):
        """
        返回量化器使用的位数。
        """
        return self.quantizer.n_bits

    def estimate_ranges(self):
        """
        量化状态管理方法，估计范围。
        设置量化状态为估计范围，表示当前正在估计量化范围。
        """
        self.state = Qstates.estimate_ranges
        self.quantizer.state = self.state

    def fix_ranges(self):
        """
        量化状态管理方法，固定范围。
        如果量化器已初始化，则设置量化状态为固定范围，表示当前固定量化范围。
        否则，抛出量化器未初始化异常。
        """
        if self.quantizer.is_initialized:
            self.state = Qstates.fix_ranges
            self.quantizer.state = self.state
        else:
            raise QuantizerNotInitializedError()

    def learn_ranges(self):
        """
        学习范围。
        使量化器的范围参数可训练，设置状态为学习范围。
        """
        self.quantizer.make_range_trainable()
        self.state = Qstates.learn_ranges
        self.quantizer.state = self.state

    def estimate_ranges_train(self):
        """
        训练时估计范围。
        设置状态为训练时估计范围，在训练期间估计量化范围。
        """
        self.state = Qstates.estimate_ranges_train
        self.quantizer.state = self.state

    def reset_ranges(self):
        """
        重置范围。
        重置范围估计器和量化器，并重新设置状态为估计范围。
        """
        self.range_estimator.reset()
        self.quantizer.reset()
        self.estimate_ranges()

    def forward(self, x):
        """
        前向传播方法。
        根据当前状态决定是否估计量化范围。
        如果当前状态为估计范围或训练时估计范围，则通过范围估计器计算当前输入的最小值和最大值，并设置量化器的范围。
        最后，应用量化操作，返回量化后的结果。
        """
        if self.state == Qstates.estimate_ranges or (
            self.state == Qstates.estimate_ranges_train and self.training
        ):
            # Note this can be per tensor or per channel
            cur_xmin, cur_xmax = self.range_estimator(x)    # 通过范围估计器计算当前输入的最小值和最大值
            self.set_quant_range(cur_xmin, cur_xmax)        # 设置量化器的范围

        return self.quantizer(x)                            # 应用量化操作

    def set_quant_range(self, x_min, x_max):
        """
        设置量化范围方法。
        将计算得到的最小值和最大值设置为量化器的范围。
        """
        self.quantizer.set_quant_range(x_min, x_max)

    def extra_repr(self):
        """
        额外字符串表示方法。
        """
        return "state={}".format(self.state.name)


class Qstates(BaseEnumOptions):
    """
    定义量化管理器的不同状态。
    这些状态决定了量化器在训练和评估过程中的行为，如是否正在估计量化范围、固定范围或学习范围等。
    """
    estimate_ranges = auto()  # ranges are updated in eval and train mode
    fix_ranges = auto()  # quantization ranges are fixed for train and eval
    learn_ranges = auto()  # quantization params are nn.Parameters
    estimate_ranges_train = auto()  # quantization ranges are updated during train and fixed for
    # eval
