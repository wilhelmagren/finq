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

File created: 2023-10-18
Last updated: 2023-10-25
"""

import logging
import numpy as np
import pandas as pd
from typing import (
    Optional,
)

log = logging.getLogger(__name__)


class Asset(object):
    """ """

    def __init__(
        self,
        data: pd.Series,
        name: str,
        *,
        market: Optional[str] = None,
        index_name: Optional[str] = None,
        price_type: str = "Close",
    ):
        """ """

        self._data = data
        self._name = name
        self._market = market
        self._index_name = index_name
        self._price_type = price_type

    def period_returns(self, period: int = 1) -> pd.Series:
        """ """
        return self._data.pct_change(periods=period)

    def period_returns_mean(self, period: int = 1) -> np.float32:
        """ """
        return self.period_returns().mean(axis=0).as_numpy().astype(np.float32)

    def volatility(self, period: int = 1) -> np.float32:
        """ """
        return self.period_returns().std() * np.sqrt(period)

    def skewness(self) -> np.float32:
        """ """
        return self._data.skew()
