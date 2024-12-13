#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import collections
import os
import random
from collections import namedtuple
from enum import Flag, auto
from functools import partial

import click
import numpy as np
import torch
import torch.nn as nn


class DotDict(dict):
    """
    A dictionary that allows attribute-style access.
    允许通过属性访问字典的键值对。
    
    Examples
    --------
    >>> config = DotDict(a=None)
    >>> config.a = 42
    >>> config.b = 'egg'
    >>> config  # can be used as dict
    {'a': 42, 'b': 'egg'}
    """

    def __setattr__(self, key, value):
        """
        重载了设置属性的方法，使得可以通过属性的方式设置字典的键值对。
        """
        self.__setitem__(key, value)

    def __delattr__(self, key):
        """
        重载了删除属性的方法，使得可以通过属性的方式删除字典的键值对。
        """
        self.__delitem__(key)

    def __getattr__(self, key):
        """
        重载了获取属性的方法，使得可以通过属性的方式获取字典的键值对。
        """
        if key in self:
            return self.__getitem__(key)
        raise AttributeError(f"DotDict instance has no key '{key}' ({self.keys()})")


def relu(x):
    """
    实现了ReLU激活函数。
    
    Args:
        x: 输入的张量
    """
    x = np.array(x)
    return x * (x > 0)


def get_all_layer_names(model, subtypes=None):
    """
    获取模型中指定层的名称，如果未指定层类型，则获取所有层的名称。
    
    Args:
        model: 待获取层名称的模型
        subtypes: 指定的层类型，默认为None
    """
    if subtypes is None:
        return [name for name, module in model.named_modules()][1:]             # 跳过第一个模块（通常是模型本身）
    return [name for name, module in model.named_modules() if isinstance(module, subtypes)]     # 返回指定类型的层名称


def get_layer_name_to_module_dict(model):
    """
    获取模型中的所有层的名称到模块的映射字典。
    
    Args:
        model: 待获取层名称到模块的映射字典的模型
    """
    return {name: module for name, module in model.named_modules() if name}


def get_module_to_layer_name_dict(model):
    """
    获取模型中的所有模块到层名称的映射字典。
    
    Args:
        model: 待获取模块到层名称的映射字典的模型
    """
    modules_to_names = collections.OrderedDict()                    # 创建一个有序字典
    for name, module in model.named_modules():
        modules_to_names[module] = name                             # 将模块到层名称的映射添加到字典中
    return modules_to_names


def get_layer_name(model, layer):
    """
    根据给定的层对象，返回其在模型中的名称。
    
    Args:
        model: 模型
        layer: 层对象
    """
    for name, module in model.named_modules():
        if module == layer:
            return name
    return None                                                     # 如果未找到，则返回None


def get_layer_by_name(model, layer_name):
    """
    根据给定的层名称，返回其在模型中的对象。
    
    Args:
        model: 模型
        layer_name: 层名称
    """
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    return None                                                     # 如果未找到，则返回None


def create_conv_layer_list(cls, model: nn.Module) -> list:
    """
    Function finds all prunable layers in the provided model
    查找并返回模型中所有可剪枝的卷积层。

    Parameters
    ----------
    cls: SVD class
    model : torch.nn.Module
    A pytorch model.

    Returns
    -------
    conv_layer_list : list
    List of all prunable layers in the given model.

    """
    conv_layer_list = []

    def fill_list(mod):
        """
        内部函数，用于检查模块类型是否在支持的层类型中，如果是，则将其添加到conv_layer_list中。
        """
        if isinstance(mod, tuple(cls.supported_layer_types)):
            conv_layer_list.append(mod)

    model.apply(fill_list)                              # 对模型的所有模块应用fill_list函数
    return conv_layer_list


def create_linear_layer_list(cls, model: nn.Module) -> list:
    """
    Function finds all prunable layers in the provided model
    查找并返回模型中所有可剪枝的全连接层。

    Parameters
    ----------
    model : torch.nn.Module
        A pytorch model.

    Returns
    -------
    conv_layer_list : list
        List of all prunable layers in the given model.

    """
    conv_layer_list = []

    def fill_list(mod):
        """
        内部函数，用于检查模块类型是否在支持的层类型中，如果是，则将其添加到conv_layer_list中。
        """
        if isinstance(mod, tuple(cls.supported_layer_types)):
            conv_layer_list.append(mod)

    model.apply(fill_list)                              # 对模型的所有模块应用fill_list函数
    return conv_layer_list


def to_numpy(tensor):
    """
    Helper function that turns the given tensor into a numpy array
    用于将给定的张量转换为NumPy数组。

    Parameters
    ----------
    tensor : torch.Tensor

    Returns
    -------
    tensor : float or np.array

    """
    if isinstance(tensor, np.ndarray):                  # 如果输入是NumPy数组，则直接返回
        return tensor
    if hasattr(tensor, "is_cuda"):                      # 如果张量在GPU上，则将其转移到CPU上，然后返回其NumPy数组
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
    if hasattr(tensor, "detach"):                       # 如果张量有detach方法，则调用该方法，然后返回其NumPy数组
        return tensor.detach().numpy()
    if hasattr(tensor, "numpy"):                        # 如果张量有numpy方法，则调用该方法，然后返回其NumPy数组
        return tensor.numpy()

    return np.array(tensor)                             # 否则，直接返回输入的NumPy数组


def set_module_attr(model, layer_name, value):
    """
    在给定的模型中，根据层的名称设置对应模块的属性。
    
    Args:
        model: 模型
        layer_name: 层名称
        value: 属性值
    """
    split = layer_name.split(".")                       # 将层名称按'.'分割，得到每一级模块的名称列表

    this_module = model
    for mod_name in split[:-1]:
        if mod_name.isdigit():
            this_module = this_module[int(mod_name)]    # 如果模块名称是数字，则直接索引
        else:
            this_module = getattr(this_module, mod_name)    # 否则，通过getattr方法获取下一级模块

    last_mod_name = split[-1]                           # 获取最后一级模块的名称
    if last_mod_name.isdigit():
        this_module[int(last_mod_name)] = value         # 如果最后一级模块的名称是数字，则直接索引赋值
    else:
        setattr(this_module, last_mod_name, value)      # 否则，通过setattr方法设置属性值


def search_for_zero_planes(model: torch.nn.Module):
    """If list of modules to winnow is empty to start with, search through all modules to check
    if any
    planes have been zeroed out. Update self._list_of_modules_to_winnow with any findings.
    在给定的模型中搜索权重/偏置参数为0的层，并返回需要进行剪枝的模块列表。
    
    :param model: torch model to search through modules for zeroed parameters
    """

    list_of_modules_to_winnow = []                      # 初始化需要进行剪枝的模块列表
    for _, module in model.named_modules():             # 遍历模型的所有模块
        if isinstance(module, (torch.nn.Linear, torch.nn.modules.conv.Conv2d)):
            in_channels_to_winnow = _assess_weight_and_bias(module.weight, module.bias)     # 获取需要剪枝的输入通道索引列表
            if in_channels_to_winnow:                   # 如果需要剪枝的输入通道列表不为空，则将模块添加到需要剪枝的模块列表中
                list_of_modules_to_winnow.append((module, in_channels_to_winnow))
    return list_of_modules_to_winnow


def _assess_weight_and_bias(weight: torch.nn.Parameter, _bias: torch.nn.Parameter):
    """
    检查给定的权重和偏置，找出那些全为0的输入通道索引
    
    Args:
        4-dim weights [CH-out, CH-in, H, W] and 1-dim bias [CH-out]
    """
    if len(weight.shape) > 2:                           # 如果权重的维度大于2，则说明是卷积层
        input_channels_to_ignore = (weight.sum((0, 2, 3)) == 0).nonzero().squeeze().tolist()    # 在C_out、H、W维度上求和，然后找出全为0的输入通道索引
    else:                                               # 否则，说明是全连接层
        input_channels_to_ignore = (weight.sum(0) == 0).nonzero().squeeze().tolist()

    if type(input_channels_to_ignore) != list:          # 如果输入通道索引不是列表，则转换为列表
        input_channels_to_ignore = [input_channels_to_ignore]

    return input_channels_to_ignore


def seed_all(seed: int = 1029, deterministic: bool = False):
    """
    用于为所有已知的随机数生成器设置种子，以使实验具有可重复性。
    This is our attempt to make experiments reproducible by seeding all known RNGs and setting
    appropriate torch directives.
    For a general discussion of reproducibility in Pytorch and CUDA and a documentation of the
    options we are using see, e.g.,
    https://pytorch.org/docs/1.7.1/notes/randomness.html
    https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

    As of today (July 2021), even after seeding and setting some directives,
    there remain unfortunate contradictions:
    1. CUDNN
    - having CUDNN enabled leads to
      - non-determinism in Pytorch when using the GPU, cf. MORPH-10999.
    - having CUDNN disabled leads to
      - most regression tests in Qrunchy failing, cf. MORPH-11103
      - significantly increased execution time in some cases
      - performance degradation in some cases
    2. torch.set_deterministic(d)
    - setting d = True leads to errors for Pytorch algorithms that do not (yet) have a deterministic
      counterpart, e.g., the layer `adaptive_avg_pool2d_backward_cuda` in vgg16__torchvision.

    Thus, we leave the choice of enforcing determinism by disabling CUDNN and non-deterministic
    algorithms to the user. To keep it simple, we only have one switch for both.
    This situation could be re-evaluated upon updates of Pytorch, CUDA, CUDNN.
    """

    # 种子必须是整数，且不小于0
    assert isinstance(seed, int), f"RNG seed must be an integer ({seed})"
    assert seed >= 0, f"RNG seed must be a positive integer ({seed})"

    # Builtin RNGs
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numpy RNG
    np.random.seed(seed)

    # CUDNN determinism (setting those has not lead to errors so far)
    torch.backends.cudnn.benchmark = False              # 禁用CUDNN的自动优化
    torch.backends.cudnn.deterministic = True           # 启用确定性算法

    # Torch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Problematic settings, see docstring. Precaution: We do not mutate unless asked to do so
    if deterministic is True:
        torch.backends.cudnn.enabled = False

        torch.set_deterministic(True)  # Use torch.use_deterministic_algorithms(True) in torch 1.8.1
        # When using torch.set_deterministic(True), it is advised by Pytorch to set the
        # CUBLAS_WORKSPACE_CONFIG variable as follows, see
        # https://pytorch.org/docs/1.7.1/notes/randomness.html#avoiding-nondeterministic-algorithms
        # and the link to the CUDA homepage on that website.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def assert_allclose(actual, desired, *args, **kwargs):
    """
    对Numpy中assert_allclose的封装，用于比较两个数组是否元素级接近。
    这在测试和验证过程中非常有用，可以用于判断模型输出或参数是否与预期值一致。
    A more beautiful version of torch.all_close.
    
    Args:
        actual: 实际值
        desired: 期望值
    """
    np.testing.assert_allclose(to_numpy(actual), to_numpy(desired), *args, **kwargs)


def count_params(module):
    """
    计算一个Pytorch模型或模块中参数的总数量。
    这对于评估模型的大小、复杂度以及模型压缩等任务非常有用。
    
    Args:
        module: Pytorch模型或模块
    """
    return len(nn.utils.parameters_to_vector(module.parameters()))


class StopForwardException(Exception):
    """
    自定义异常类，用于在前向传播过程中中断计算。
    Used to throw and catch an exception to stop traversing the graph.
    """

    pass


class StopForwardHook:
    """
    实现了一个钩子类，用于在前向传播过程中抛出StopForwardException异常，从而中断计算。
    """
    def __call__(self, module, *args):
        raise StopForwardException


def sigmoid(x):
    """
    实现了Sigmoid激活函数。
    """
    return 1.0 / (1.0 + np.exp(-x))


class CosineTempDecay:
    """
    实现了基于余弦函数的温度衰减策略。
    常用于模拟退火算法或需要逐步降低某个参数的场景。
    通过余弦函数，可以实现平滑的温度衰减过程。
    """
    def __init__(self, t_max, temp_range=(20.0, 2.0), rel_decay_start=0):
        """
        Args:
            t_max: 总的时间步长或迭代次数，决定衰减的持续时间。
            temp_range: 温度的起始值和结束值。
            rel_decay_start: 相对衰减开始点。
        """
        self.t_max = t_max
        self.start_temp, self.end_temp = temp_range
        self.decay_start = rel_decay_start * t_max          # 计算衰减开始的绝对时间步长

    def __call__(self, t):
        """
        Args:
            t: 当前时间步长或迭代次数。
        """
        if t < self.decay_start:                            # 如果当前时间步长小于衰减开始时间，则返回起始温度
            return self.start_temp

        rel_t = (t - self.decay_start) / (self.t_max - self.decay_start)        # 计算相对于衰减开始后的时间进度
        return self.end_temp + 0.5 * (self.start_temp - self.end_temp) * (1 + np.cos(rel_t * np.pi))


class BaseEnumOptions(Flag):
    """
    基于enum.Flag的枚举类，用于创建具有名称列表和字符串表示的枚举选项。
    """
    def __str__(self):
        return self.name                                    # 返回枚举选项的名称

    @classmethod
    def list_names(cls):
        """
        用于获取所有枚举成员的名称列表。
        """
        return [m.name for m in cls]


class ClassEnumOptions(BaseEnumOptions):
    """
    继承自BaseEnumOptions，添加了与类相关的属性和方法，使得枚举成员可以与具体的类关联。
    它主要用于将枚举选项与实现类关联，方便根据选项动态创建对应的类实例。
    """
    @property
    def cls(self):
        """
        用于获取枚举成员关联的类。
        """
        return self.value.cls

    def __call__(self, *args, **kwargs):
        return self.value.cls(*args, **kwargs)              # 通过枚举成员关联的类创建实例

# 创建MethodMap的部分函数，用于创建具有名称和类属性的枚举选项
MethodMap = partial(namedtuple("MethodMap", ["value", "cls"]), auto())


def split_dict(src: dict, include=(), remove_prefix: str = ""):
    """
    将源字典按照指定的键列表`include`分割成两个字典，一个包含指定的键值对，另一个包含剩余的键值对。
    这个函数常用于从配置字典中提取部分参数，同时保留未使用的参数。
    Splits dictionary into a DotDict and a remainder.
    The arguments to be placed in the first DotDict are those listed in `include`.
    Parameters
    ----------
    src: dict
        The source dictionary.
    include:
        List of keys to be returned in the first DotDict.
    remove_suffix:
        remove prefix from key
    """
    result = DotDict()                                      # 创建一个空的DotDict实例，用于存放包含的键值对

    for arg in include:                                     # 遍历指定的键列表
        if remove_prefix:
            key = arg.replace(f"{remove_prefix}_", "", 1)   # 如果指定了前缀，则将前缀从键名中移除
        else:
            key = arg
        result[key] = src[arg]                              # 将指定的键值对添加到DotDict中
    remainder = {key: val for key, val in src.items() if key not in include}    # 将未包含的键值对添加到剩余的字典中
    return result, remainder


class ClickEnumOption(click.Choice):
    """
    对click.Choice类型进行了调整，适用于基于BaseEnumOptions枚举类的选项。
    它用于在命令行接口（CLI）中定义枚举类型的选项，支持与click库的集成，帮助验证用于输入并提供自动完成功能。
    Adjusted click.Choice type for BaseOption which is based on Enum
    """

    def __init__(self, enum_options, case_sensitive=True):
        """
        Args:
            enum_options: 枚举选项类
            case_sensitive: 是否区分大小写，默认为True
        """
        assert issubclass(enum_options, BaseEnumOptions)                    # 确保枚举选项类是BaseEnumOptions的子类
        self.base_option = enum_options
        super().__init__(self.base_option.list_names(), case_sensitive)

    def convert(self, value, param, ctx):
        """
        用于在命令行参数解析时，将字符串值转换为对应的枚举成员。
        
        Args:
            value: 用户输入的值，类型为字符串。
            param: 参数对象，可用于获取参数的上下文信息。
            ctx: 上下文对象，包含命令行解析的上下文信息。
        """
        # Exact match
        if value in self.choices:                                           # 如果用户输入的值在选项列表中，则直接返回对应的枚举成员
            return self.base_option[value]

        # Match through normalization and case sensitivity
        # first do token_normalize_func, then lowercase
        # preserve original `value` to produce an accurate message in
        # `self.fail`
        normed_value = value                                                # 初始化标准化后的值
        normed_choices = self.choices                                       # 初始化标准化后的选项列表

        if ctx is not None and ctx.token_normalize_func is not None:        # 如果提供了标准化函数，则对值和选项列表进行标准化
            normed_value = ctx.token_normalize_func(value)
            normed_choices = [ctx.token_normalize_func(choice) for choice in self.choices]

        if not self.case_sensitive:                                         # 如果不区分大小写，则将值和选项列表转换为小写
            normed_value = normed_value.lower()
            normed_choices = [choice.lower() for choice in normed_choices]

        if normed_value in normed_choices:                                  # 如果标准化后的值在标准化后的选项列表中，则返回对应的枚举成员
            return self.base_option[normed_value]

        self.fail(                                                          # 否则，抛出异常，提示用户输入的值无效
            "invalid choice: %s. (choose from %s)" % (value, ", ".join(self.choices)), param, ctx
        )
