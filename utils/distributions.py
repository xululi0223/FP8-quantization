#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import scipy.stats as stats
import scipy.integrate
from scipy import special


class DistrBase:
    """
    抽象基类，用于定义概率分布的接口。
    提供了一系列方法，以支持不同概率分布的计算和采样。
    """
    def __init__(self, params_dict, range_min, range_max, *args, **kwargs):
        """
        Args:
            params_dict: 概率分布的参数字典。
            range_min: 概率分布的下界。
            range_max: 概率分布的上界。
        """
        self.params_dict = params_dict
        assert range_max >= range_min
        self.range_min = range_min
        self.range_max = range_max

    def pdf(self, x):
        """
        返回在点x处的概率密度函数值。
        """
        raise NotImplementedError()

    def sample(self, shape):
        """
        根据分布参数生成指定形状的样本。
        """
        raise NotImplementedError()

    def eval_point_mass_range_min(self):
        """
        评估在`range_min`处的点质量。
        """
        raise NotImplementedError()

    def eval_point_mass_range_max(self):
        """
        评估在`range_max`处的点质量。
        """
        raise NotImplementedError()

    def integr_interv_p_sqr_r(self, a, b):
        """
        计算区间[a, b]上概率密度函数平方与参数r的积分。
        """
        raise NotImplementedError()

    def integr_interv_x_p_r_signed(self, a, b):
        """
        计算区间[a, b]上x * p(x)的积分，考虑符号。
        """
        raise NotImplementedError()

    def eval_p_sqr_r(self, x, grid):
        """
        评估x * p(r) 在点x和网格grid上的值。
        """
        raise NotImplementedError()

    def eval_x_p_r_signed(self, x, grid):
        """
        评估x * p(r) 在点x和网格grid上的值，考虑符号。
        """
        raise NotImplementedError()

    def eval_non_central_second_moment(self):
        """
        评估分布的非中心二阶矩。
        """
        raise NotImplementedError()

    def print(self):
        """
        打印分布的详细信息。
        """
        raise NotImplementedError()


class ClippedGaussDistr(DistrBase):
    """
    继承自DistrBase，定义了截断高斯分布的相关方法。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mu = self.params_dict["mu"]             # 提取高斯分布的均值
        sigma = self.params_dict["sigma"]       # 提取高斯分布的标准差
        self.point_mass_range_min = stats.norm.cdf(self.range_min, loc=mu, scale=sigma)     # 计算在range_min处的累积分布函数CDF值，即P(X <= range_min)
        self.point_mass_range_max = 1.0 - stats.norm.cdf(self.range_max, loc=mu, scale=sigma)   # 计算在range_max处的累积分布函数CDF值，即P(X >= range_max)

    def print(self):
        """
        打印高斯分布的详细信息。
        """
        print(
            "Gaussian distr ",
            ", mu = ",
            self.params_dict["mu"],         # 打印均值
            ", sigma = ",
            self.params_dict["sigma"],      # 打印标准差
            " clipped at [",
            self.range_min,                 # 打印裁剪范围的下界
            ",",
            self.range_max,                 # 打印裁剪范围的上界
            "]",
        )

    def cdf(self, x_np):
        """
        计算给定点x_np的累积分布函数值，即P(X <= x_np)。
        """
        p = stats.norm.cdf(x_np, self.params_dict["mu"], self.params_dict["sigma"])
        return p

    def pdf(self, x):
        """
        计算给定点x的概率密度函数值f(x)。
        """
        x_np = x.cpu().numpy()
        p = stats.norm.pdf(x_np, self.params_dict["mu"], self.params_dict["sigma"])
        return p

    def inverse_cdf(self, x):
        """
        计算给定概率x的逆累积分布函数值，即P(X <= res) = x的res值。
        """
        res = stats.norm.ppf(x, loc=self.params_dict["mu"], scale=self.params_dict["sigma"])
        return res

    def sample(self, shape):
        """
        根据分布参数生成符合被裁剪高斯分布的指定形状的样本。
        """
        r = np.random.normal(                           # 生成服从标准正态分布的样本
            loc=self.params_dict["mu"], scale=self.params_dict["sigma"], size=shape
        )
        r = np.clip(r, self.range_min, self.range_max)  # 将样本裁剪到指定范围内
        return r

    def integr_interv_p_sqr_r(self, a, b, u):
        """
        计算区间[a, b]上概率密度函数平方与参数r的积分，即∫[a, b] p(r)^2 dr。
        """
        assert b >= a
        mu = self.params_dict["mu"]
        sigma = self.params_dict["sigma"]
        root_half = np.sqrt(0.5)
        root_half_pi = np.sqrt(0.5 * np.pi)
        t1 = -sigma * (                                                     # 计算积分的第一部分
            np.exp((-0.5 * a**2 + 1.0 * a * mu - 0.5 * mu**2) / sigma**2)   # 计算指数部分
            * sigma
            * (-1.0 * a - 1.0 * mu + 2.0 * u)
            + (
                -root_half_pi * mu**2
                - root_half_pi * sigma**2
                + 2.0 * root_half_pi * mu * u
                - root_half_pi * u**2
            )
            * special.erf((-root_half * a + root_half * mu) / sigma)        # 计算误差函数部分
        )
        t2 = sigma * (                                                      # 计算积分的第二部分
            np.exp((-0.5 * b**2 + 1.0 * b * mu - 0.5 * mu**2) / sigma**2)
            * sigma
            * (-1.0 * b - 1.0 * mu + 2.0 * u)
            + (
                -root_half_pi * mu**2
                - root_half_pi * sigma**2
                + 2.0 * root_half_pi * mu * u
                - root_half_pi * u**2
            )
            * special.erf((-root_half * b + root_half * mu) / sigma)
        )
        const = 1 / sigma / np.sqrt(2 * np.pi)
        return (t1 + t2) * const

    def integr_interv_x_p_signed_r(self, a, b, x0):
        """
        计算区间[a, b]上x * p(x)的积分，考虑符号，即∫[a, b] x * p(x) dx。
        """
        assert b >= a
        mu = self.params_dict["mu"]
        sigma = self.params_dict["sigma"]
        root_half = np.sqrt(0.5)
        root_half_pi = np.sqrt(0.5 * np.pi)

        res = (
            x0
            * sigma
            * (
                np.exp(-((0.5 * mu**2) / sigma**2))
                * (
                    np.exp((a * (-0.5 * a + mu)) / sigma**2)
                    - np.exp((b * (-0.5 * b + mu)) / sigma**2)
                )
                * sigma
                - root_half_pi * mu * special.erf((root_half * a - root_half * mu) / sigma)
                + root_half_pi * mu * special.erf((root_half * b - root_half * mu) / sigma)
            )
            + sigma
            * (
                np.exp((-0.5 * a**2 + a * mu - 0.5 * mu**2) / sigma**2)
                * (-a * sigma - mu * sigma)
                + (-root_half_pi * mu**2 - root_half_pi * sigma**2)
                * special.erf((-root_half * a + root_half * mu) / sigma)
            )
            - sigma
            * (
                np.exp((-0.5 * b**2 + b * mu - 0.5 * mu**2) / sigma**2)
                * (-b * sigma - mu * sigma)
                + (-root_half_pi * mu**2 - root_half_pi * sigma**2)
                * special.erf((-root_half * b + root_half * mu) / sigma)
            )
        )

        const = 1 / sigma / np.sqrt(2 * np.pi)
        return res * const

    def integr_p_times_x(self, a, b):
        """
        计算区间[a, b]上p(r) * x的积分，即∫[a, b] p(r) * x dr。
        """
        assert b >= a
        mu = self.params_dict["mu"]
        sigma = self.params_dict["sigma"]

        root_half = np.sqrt(0.5)
        root_half_pi = np.sqrt(0.5 * np.pi)

        res = sigma * (
            np.exp(-((0.5 * mu**2) / sigma**2))
            * (
                np.exp((a * (-0.5 * a + mu)) / sigma**2)
                - np.exp((b * (-0.5 * b + mu)) / sigma**2)
            )
            * sigma
            - root_half_pi * mu * special.erf(root_half * (a - mu) / sigma)
            + root_half_pi * mu * special.erf(root_half * (b - mu) / sigma)
        )

        scale = 1 / (sigma * np.sqrt(2 * np.pi))
        return res * scale

    def eval_non_central_second_moment(self):
        """
        评估被裁剪高斯分布的非中心二阶矩，即 E[X^2]。
        """
        term_range_min = self.point_mass_range_min * self.range_min**2      # 计算range_min处点质量与range_min^2的乘积，表示左侧裁剪部分的贡献
        term_range_max = self.point_mass_range_max * self.range_max**2      # 计算range_max处点质量与range_max^2的乘积，表示右侧裁剪部分的贡献
        term_middle_intergral = self.integr_interv_p_sqr_r(self.range_min, self.range_max, 0.0) # 计算裁剪范围内的积分
        return term_range_min + term_middle_intergral + term_range_max


class ClippedStudentTDistr(DistrBase):
    """
    继承自DistrBase，定义了截断学生t分布的相关方法。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nu = self.params_dict["nu"]                                         # 提取学生t分布的自由度参数nu
        self.point_mass_range_min = stats.t.cdf(self.range_min, nu)         # 计算在range_min处的累积分布函数CDF值，即P(X <= range_min)
        self.point_mass_range_max = 1.0 - stats.t.cdf(self.range_max, nu)   # 计算在range_max处的累积分布函数CDF值，即P(X >= range_max)

    def print(self):
        """
        打印学生t分布的详细信息。
        """
        print(
            "Student's-t distr",
            ", nu = ",
            self.params_dict["nu"],     # 打印自由度参数nu
            " clipped at [",
            self.range_min,             # 打印裁剪范围的下界
            ",",
            self.range_max,             # 打印裁剪范围的上界
            "]",
        )

    def pdf(self, x):
        """
        计算给定点x的概率密度函数值f(x)。
        """
        x_np = x.cpu().numpy()
        p = stats.t.pdf(x_np, self.params_dict["nu"])
        return p

    def cdf(self, x):
        """
        计算给定点x的累积分布函数值，即P(X <= x)。
        """
        p = stats.t.cdf(x, self.params_dict["nu"])
        return p

    def inverse_cdf(self, x):
        """
        计算给定概率x的逆累积分布函数值，即P(X <= res) = x的res值。
        """
        res = stats.t.ppf(x, self.params_dict["nu"])
        return res

    def sample(self, shape):
        """
        根据分布参数生成符合被裁剪学生t分布的指定形状的样本。
        """
        r = np.random.standard_t(self.params_dict["nu"], size=shape)    # 生成服从自由度为nu的学生t分布的样本
        r = np.clip(r, self.range_min, self.range_max)                  # 将样本裁剪到指定范围内
        return r

    def integr_interv_p_sqr_r(self, a, b, u):
        """
        计算区间[a, b]上概率密度函数平方与参数r的积分，即∫[a, b] p(r)^2 dr。
        """
        assert b >= a
        nu = self.params_dict["nu"]

        first_term = (2.0 * nu * (-1.0 + ((a**2 + nu) / nu) ** (1.0 / 2.0 - nu / 2.0)) * u) / (
            1.0 - nu
        )
        second_term = -(2 * nu * (-1 + ((b**2 + nu) / nu) ** (1.0 / 2.0 - nu / 2)) * u) / (
            1.0 - nu
        )
        third_term = (
            -a
            * u**2
            * scipy.special.hyp2f1(1.0 / 2.0, (1.0 + nu) / 2.0, 3.0 / 2.0, -(a**2.0 / nu))
        )
        forth_term = (
            b
            * u**2.0
            * scipy.special.hyp2f1(1.0 / 2.0, (1.0 + nu) / 2.0, 3.0 / 2.0, -(b**2 / nu))
        )
        fifth_term = (
            -1.0
            / 3.0
            * a**3
            * scipy.special.hyp2f1(3.0 / 2.0, (1.0 + nu) / 2.0, 5.0 / 2.0, -(a**2 / nu))
        )
        sixth_term = (
            1.0
            / 3.0
            * b**3
            * scipy.special.hyp2f1(3.0 / 2.0, (1.0 + nu) / 2.0, 5.0 / 2.0, -(b**2 / nu))
        )
        res = first_term + second_term + third_term + forth_term + fifth_term + sixth_term

        const = (
            scipy.special.gamma(0.5 * (nu + 1.0))
            / np.sqrt(np.pi * nu)
            / scipy.special.gamma(0.5 * nu)
        )
        return res * const

    def integr_p_times_x(self, a, b):
        """
        计算区间[a, b]上p(r) * x的积分，即∫[a, b] p(r) * x dr。
        """
        assert b >= a
        nu = self.params_dict["nu"]
        res = 0.0

        const = (
            scipy.special.gamma(0.5 * (nu + 1.0))
            / np.sqrt(np.pi * nu)
            / scipy.special.gamma(0.5 * nu)
        )
        return res * const

    def scale(self):
        """
        计算并返回分布的缩放因子。
        """
        nu = self.params_dict["nu"]
        res = (
            scipy.special.gamma(0.5 * (nu + 1.0))
            / np.sqrt(np.pi * nu)
            / scipy.special.gamma(0.5 * nu)
        )
        return res

    def integr_cubic_root_p(self, a, b):
        """
        计算立方根概率密度函数的积分，即∫[a, b] p(r)^(1/3) dr。
        """
        assert b >= a
        nu = self.params_dict["nu"]

        common_mult = 1.0 / (nu - 2.0) * 3.0 * (a * b) ** (-nu / 3.0) * nu ** ((1.0 + nu) / 6.0)
        first_term = (
            scipy.special.hyp2f1(
                1 / 6.0 * (-2.0 + nu), (1.0 + nu) / 6, (4.0 + nu) / 6.0, -nu / a**2
            )
            * a ** (2.0 / 3.0)
            * b ** (nu / 3.0)
        )
        second_term = (
            -scipy.special.hyp2f1(
                1 / 6.0 * (-2.0 + nu), (1.0 + nu) / 6, (4.0 + nu) / 6.0, -nu / b**2
            )
            * b ** (2.0 / 3.0)
            * a ** (nu / 3.0)
        )

        return common_mult * (first_term + second_term)

    def integr_interv_u_t_times_p_no_constant(self, a, b, u):
        """
        计算不包含常数项的u * t与概率密度函数的积分，即∫[a, b] u * t * p(t) dt。
        """
        assert b >= a
        df = self.params_dict["nu"]
        res = (
            df ** ((1.0 + df) / 2.0)
            * (-((a**2 + df) ** (1.0 / 2.0 - df / 2.0)) + (b**2 + df) ** (1.0 / 2.0 - df / 2.0))
            * u
        ) / (1.0 - df)
        return res

    def integr_interv_x_p_signed_r(self, a, b, x0):
        """
        计算区间[a, b]上x * p(x)的积分，考虑符号，即∫[a, b] x * p(x) dx。
        """
        assert b >= a
        nu = self.params_dict["nu"]

        const = (
            scipy.special.gamma(0.5 * (nu + 1.0))
            / np.sqrt(np.pi * nu)
            / scipy.special.gamma(0.5 * nu)
        )
        r1 = self.integr_interv_u_t_times_p_no_constant(a, b, x0) * const
        r2 = self.integr_interv_p_sqr_r(a, b, 0.0)

        res = r1 - r2
        return res

    def eval_non_central_second_moment(self):
        """
        评估被裁剪学生t分布的非中心二阶矩，即 E[X^2]。
        """
        term_range_min = self.point_mass_range_min * self.range_min**2
        term_range_max = self.point_mass_range_max * self.range_max**2
        term_middle_intergral = self.integr_interv_p_sqr_r(self.range_min, self.range_max, 0.0)
        return term_range_min + term_middle_intergral + term_range_max


class UniformDistr(DistrBase):
    """
    继承自DistrBase，定义了均匀分布的相关方法。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = 1 / (self.range_max - self.range_min)      # 计算均匀分布的概率密度函数值

    def print(self):
        """
        计算均匀分布的详细信息。
        """
        print("Uniform distribution on [", self.range_min, ",", self.range_max, "]")

    def pdf(self, x):
        """
        计算给定点x的概率密度函数值f(x)。
        """
        return self.p

    def cdf(self, x):
        """
        计算给定点x的累积分布函数值，即P(X <= x)。
        """
        return (x - self.range_min) * self.p

    def sample(self, shape):
        """
        根据分布参数生成符合均匀分布的指定形状的样本。
        """
        return np.random.uniform(self.range_min, self.range_max, shape)

    def integr_interv_p_sqr_r(self, a, b, u):
        """
        计算区间[a, b]上概率密度函数平方与参数r的积分，即∫[a, b] p(r)^2 dr。
        """
        assert b >= a
        res = -(a**3 / 3.0) + b**3 / 3.0 + a**2 * u - b**2 * u - a * u**2 + b * u**2
        return res * self.p

    def eval_non_central_second_moment(self):
        """
        评估均匀分布的非中心二阶矩，即 E[X^2]。
        """
        if not isinstance(self, UniformDistr):
            term_range_min = self.point_mass_range_min() * self.range_min**2
            term_range_max = self.point_mass_range_max() * self.range_max**2
        else:
            term_range_min = 0.0
            term_range_max = 0.0
        term_middle_intergral = self.integr_interv_p_sqr_r(self.range_min, self.range_max, 0.0)

        return term_range_min + term_middle_intergral + term_range_max

    def integr_p_times_x(self, a, b):
        """
        计算区间[a, b]上p(r) * x的积分，即∫[a, b] p(r) * x dr。
        """
        return 0.5 * (b**2 - a**2) * self.p

    def integr_interv_x_p_signed_r(self, a, b, x0):
        """
        计算区间[a, b]上x * p(x)的积分，考虑符号，即∫[a, b] x * p(x) dx。
        """
        assert b >= a
        res = 0.5 * a**2 - 0.5 * b**2 + (b - a) * x0
        return res * self.p
