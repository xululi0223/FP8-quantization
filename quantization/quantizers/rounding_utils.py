# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

from torch import nn
import torch
from torch.autograd import Function

# Functional
from utils import MethodMap, ClassEnumOptions


class RoundStraightThrough(Function):
    """
    自定义求导函数，用于实现直通估计器策略中的四舍五入操作。
    在前向传播中执行四舍五入操作，而在反向传播中直接传递梯度，实现梯度不变的效果。
    """
    @staticmethod
    def forward(ctx, x):
        """
        前向传播过程。
        """
        return torch.round(x)       # 执行四舍五入操作

    @staticmethod
    def backward(ctx, output_grad):
        """
        反向传播过程。
        """
        return output_grad          # 直接传递梯度


class StochasticRoundSTE(Function):
    """
    自定义求导函数，实现在前向传播中进行随机四舍五入，而在反向传播中同样采用直通估计器策略，直接传递梯度。
    """
    @staticmethod
    def forward(ctx, x):
        """
        前向传播过程。
        """
        # Sample noise between [0, 1)
        noise = torch.rand_like(x)          # 生成与输入张量x相同形状的随机噪声，范围在[0, 1)之间
        return torch.floor(x + noise)       # 执行随机四舍五入操作

    @staticmethod
    def backward(ctx, output_grad):
        """
        反向传播过程。
        """
        return output_grad                  # 直接传递梯度


class ScaleGradient(Function):
    """
    自定义求导函数，用于在反向传播中对梯度进行缩放，但在前向传播中保持输入不变。
    该功能在需要调整梯度流的情景中非常有用，例如在梯度惩罚或梯度裁剪中。
    """
    @staticmethod
    def forward(ctx, x, scale):
        """
        前向传播过程。
        """
        ctx.scale = scale                   # 保存缩放因子
        return x

    @staticmethod
    def backward(ctx, output_grad):
        """
        反向传播过程。
        """
        return output_grad * ctx.scale, None    # 对梯度进行缩放


class EWGSFunctional(Function):
    """
    自定义求导函数，实现了增强加权梯度缩放（EWGS）策略。
    在前向传播中对输入进行四舍五入，而在反向传播中对梯度进行缩放。
    x_in: float input
    scaling_factor: backward scaling factor
    x_out: discretized version of x_in within the range of [0,1]
    """

    @staticmethod
    def forward(ctx, x_in, scaling_factor):
        """
        前向传播过程。
        """
        x_int = torch.round(x_in)               # 执行四舍五入操作
        ctx._scaling_factor = scaling_factor    # 保存缩放因子
        ctx.save_for_backward(x_in - x_int)     # 计算量化前后的差值，保存以备反向传播使用
        return x_int

    @staticmethod
    def backward(ctx, g):
        """
        反向传播过程。
        """
        diff = ctx.saved_tensors[0]             # 获取量化前后的差值
        delta = ctx._scaling_factor             # 获取缩放因子
        scale = 1 + delta * torch.sign(g) * diff    # 计算梯度缩放因子：1 + delta * sign(g) * diff
        return g * scale, None, None            # 对梯度进行缩放


class StackSigmoidFunctional(Function):
    """
    自定义求导函数，结合了堆叠sigmoid函数的前向和反向传播逻辑。
    在前向传播中对输入进行四舍五入，并在反向传播中调整梯度，利用sigmoid函数对梯度进行平滑处理。
    """
    @staticmethod
    def forward(ctx, x, alpha):
        """
        前向传播过程。
        
        Args:
            x: 输入张量
            alpha: 控制sigmoid函数斜率的参数
        """
        # Apply round to nearest in the forward pass
        ctx.save_for_backward(x, alpha)         # 保存输入张量x和alpha参数
        return torch.round(x)                   # 执行四舍五入操作

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播过程。
        """
        x, alpha = ctx.saved_tensors
        sig_min = torch.sigmoid(alpha / 2)
        sig_scale = 1 - 2 * sig_min
        x_base = torch.floor(x).detach()        # 获取输入张量x的整数部分
        x_rest = x - x_base - 0.5
        stacked_sigmoid_grad = (                # 计算堆叠sigmoid的梯度
            torch.sigmoid(x_rest * -alpha)
            * (1 - torch.sigmoid(x_rest * -alpha))
            * -alpha
            / sig_scale
        )
        return stacked_sigmoid_grad * grad_output, None     # 对梯度进行调整


# Parametrized modules
class ParametrizedGradEstimatorBase(nn.Module):
    """
    抽象基类，用于定义可参数化的梯度估计器的基本接口和通用功能。
    该类提供了关于梯度参数是否可训练以及相关功能的方法，为具体的梯度估计器实现了统一的框架。
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._trainable = False                             # 默认梯度参数不可训练

    def make_grad_params_trainable(self):
        """
        将梯度参数设置为可训练的（即将缓冲区转换为参数）。
        """
        self._trainable = True                              # 设置梯度参数可训练
        for name, buf in self.named_buffers(recurse=False): # 遍历模型的缓冲区
            setattr(self, name, torch.nn.Parameter(buf))    # 将缓冲区转换为可训练的参数

    def make_grad_params_tensor(self):
        """
        将梯度参数设置为固定的张量（即将参数转换为缓冲区）。
        """
        self._trainable = False                             # 设置梯度参数不可训练
        for name, param in self.named_parameters(recurse=False):    # 遍历模型的参数
            cur_value = param.data                          # 获取参数的值
            delattr(self, name)                             # 删除参数
            self.register_buffer(name, cur_value)           # 将参数转换为缓冲区

    def forward(self, x):
        """
        前向传播过程。
        """
        raise NotImplementedError()


class StackedSigmoid(ParametrizedGradEstimatorBase):
    """
    实现堆叠sigmoid函数作为梯度估计器。
    在前向传播中应用堆叠sigmoid函数，而在反向传播中通过自定义方法调整梯度。
    Stacked sigmoid estimator based on a simulated sigmoid forward pass
    """

    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: 控制sigmoid函数斜率的参数
        """
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha))  # 注册缓冲区alpha

    def forward(self, x):
        """
        前向传播过程。
        """
        return stacked_sigmoid_func(x, self.alpha)          # 执行堆叠sigmoid函数

    def extra_repr(self):
        """
        额外的字符串表示方法。
        """
        return f"alpha={self.alpha.item()}"                 # 返回alpha参数的值


class EWGSDiscretizer(ParametrizedGradEstimatorBase):
    """
    实现在前向传播中应用EWGS策略的离散化功能。在反向传播中，根据缩放因子调整梯度。
    """
    def __init__(self, scaling_factor=0.2):
        """
        Args:
            scaling_factor: 缩放因子，用于调整反向传播中的梯度。
        """
        super().__init__()
        self.register_buffer("scaling_factor", torch.tensor(scaling_factor))    # 注册缓冲区scaling_factor

    def forward(self, x):
        """
        前向传播过程。
        """
        return ewgs_func(x, self.scaling_factor)            # 执行EWGS策略

    def extra_repr(self):
        """
        额外的字符串表示方法。
        """
        return f"scaling_factor={self.scaling_factor.item()}"   # 返回scaling_factor参数的值


class StochasticRounding(nn.Module):
    """
    实现了随机四舍五入策略。
    根据模型的训练状态，在训练阶段应用随机四舍五入，评估阶段则采用直通估计器策略中的四舍五入。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        前向传播过程。
        """
        if self.training:
            return stochastic_round_ste_func(x)
        else:
            return round_ste_func(x)


round_ste_func = RoundStraightThrough.apply
stacked_sigmoid_func = StackSigmoidFunctional.apply
scale_grad_func = ScaleGradient.apply
stochastic_round_ste_func = StochasticRoundSTE.apply
ewgs_func = EWGSFunctional.apply


class GradientEstimator(ClassEnumOptions):
    """
    枚举类，使用ClassEnumOptions来映射不同的梯度估计方法。
    定义了多种梯度估计策略，如直通估计器、随机四舍五入、EWGS、堆叠sigmoid函数。
    """
    ste = MethodMap(round_ste_func)
    stoch_round = MethodMap(StochasticRounding)
    ewgs = MethodMap(EWGSDiscretizer)
    stacked_sigmoid = MethodMap(StackedSigmoid)
