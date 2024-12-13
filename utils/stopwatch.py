#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import sys
import time


class Stopwatch:
    """
    跨平台上下文管理器，用于计时。
    A simple cross-platform context-manager stopwatch.

    Examples
    --------
    >>> import time
    >>> with Stopwatch(verbose=True) as st:
    ...     time.sleep(0.101)  #doctest: +ELLIPSIS
    Elapsed time: 0.10... sec
    """

    def __init__(self, name=None, verbose=False):
        """
        Args:
            name: 用于标识计时器的名称。
            verbose: 是否在停止时打印经过时间。
        """
        self._name = name
        self._verbose = verbose

        self._start_time_point = 0.0        # 初始化开始时间点
        self._total_duration = 0.0          # 初始化总时间
        self._is_running = False            # 初始化是否正在计时标记位

        if sys.platform == "win32":         # Windows平台使用time.clock()作为计时函数
            # on Windows, the best timer is time.clock()
            self._timer_fn = time.clock
        else:                               # 其他平台使用time.time()作为计时函数
            # on most other platforms, the best timer is time.time()
            self._timer_fn = time.time

    def __enter__(self, verbose=False):
        """
        上下文管理器的进入方法。
        """
        return self.start()                 # 调用start()方法，开始计时，并返回self实例

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器的退出方法。
        """
        self.stop()                         # 调用stop()方法，结束计时
        if self._verbose:
            self.print()                    # 如果verbose为True，则打印经过时间

    def start(self):
        """
        开始计时方法。
        """
        if not self._is_running:            # 如果没有正在计时
            self._start_time_point = self._timer_fn()       # 记录当前时间点
            self._is_running = True         # 设置正在计时标记位为True
        return self

    def stop(self):
        """
        停止计时方法。
        """
        if self._is_running:                # 如果正在计时
            self._total_duration += self._timer_fn() - self._start_time_point   # 更新总时间
            self._is_running = False        # 设置正在计时标记位为False
        return self

    def reset(self):
        """
        重置计时器方法。
        """
        self._start_time_point = 0.0        # 重置开始时间点
        self._total_duration = 0.0          # 重置总时间
        self._is_running = False            # 设置正在计时标记位为False
        return self

    def _update_state(self):
        """
        内部状态更新方法。
        更新计时器的当前状态，特别是在计时器仍在运行时获取更新的经过时间。
        """
        now = self._timer_fn()              # 获取当前时间点
        self._total_duration += now - self._start_time_point        # 更新总时间
        self._start_time_point = now        # 更新开始时间点

    def _format(self):
        """
        内部格式化方法。
        格式化计数信息为字符串。
        """
        prefix = f"[{self._name}]" if self._name is not None else "Elapsed time"    # 如果有名称，则使用名称作为前缀，否则使用默认前缀
        info = f"{prefix}: {self._total_duration:.3f} sec"                          # 构建包含前缀和总时间的字符串
        return info

    def format(self):
        """
        格式化方法。
        用于获取当前计时信息的格式化字符串。
        """
        if self._is_running:                # 如果正在计时，则更新状态，并返回格式化字符串
            self._update_state()
        return self._format()

    def print(self):
        """
        打印方法。
        打印当前计时信息到控制台。
        """
        print(self.format())

    def get_total_duration(self):
        """
        获取总时间方法。
        获取总共经过的时间（以秒为单位）。
        """
        if self._is_running:                # 如果正在计时，则更新状态，并返回总时间
            self._update_state()
        return self._total_duration
