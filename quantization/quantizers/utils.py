#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.


class QuantizerNotInitializedError(Exception):
    """
    自定义异常类，用于在量化器未初始化时抛出异常。
    Raised when a quantizer has not been initialized
    """

    def __init__(self):
        super(QuantizerNotInitializedError, self).__init__(
            "Quantizer has  not been initialized yet"
        )
