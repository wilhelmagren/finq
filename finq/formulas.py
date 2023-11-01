"""
MIT License

Copyright (c) 2023 Wilhelm Ågren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

File created: 2023-11-01
Last updated: 2023-11-01
"""

import numpy as np
from typing import Union


def k_moment(x: np.ndarray, k: int) -> float:
    """ """

    return ((x - x.mean()) ** k).sum() / x.shape[0]


def adjusted_fisher_pearson_skewness_coefficient(
    x: np.ndarray,
) -> float:
    """ """

    n = x.shape[0]
    coeff = np.sqrt(n * (n - 1)) / (n - 2)

    m2 = k_moment(x, 2)
    m3 = k_moment(x, 3)

    return coeff * (m3 / (m2 ** (3 / 2)))


def period_returns(x: np.ndarray, period: int = 1) -> np.ndarray:
    """ """

    return (x[:, period:] / x[:, :-period]) - 1


def sharpe_ratio(
    r: Union[float, np.ndarray],
    v: Union[float, np.ndarray],
    rfr: float,
) -> Union[float, np.ndarray]:
    """ """

    return (r - rfr) / v


def weighted_returns(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """ """

    return np.dot(w, x)


def weighted_variance(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """ """

    return np.dot(w.T, np.dot(cov, w))
