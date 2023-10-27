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
Last updated: 2023-10-28
"""

import numpy.typing
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

        log.debug("pre-computing some standard quantitives used for an Asset...")
        self._metrics = {}
        self._metrics["daily_returns"] = self.period_returns(period=1)
        self._metrics["daily_returns_mean"] = self.period_returns_mean(period=1)
        self._metrics["yearly_returns_mean"] = self.period_returns_mean(period=252)
        self._metrics["yearly_volatility"] = self.volatility(period=1, trading_days=252)
        self._metrics["skewness"] = self.skewness()
        log.debug("OK!")

    def __str__(self) -> str:
        """ """

        format = f"<{self.__class__.__name__} called `{self._name}`"
        if self._market:
            format += f" on {self._market}"
        if self._index_name:
            format += f" in {self._index_name}"

        format += f" (price type: {self._price_type})"
        format += f"\n-- num samples:\t\t\t{self._data.shape[0]}"
        format += (
            f"\n-- daily returns mean:\t\t{self._metrics['daily_returns_mean']:.5f}"
        )
        format += (
            f"\n-- yearly returns mean:\t\t{self._metrics['yearly_returns_mean']:.5f}"
        )
        format += f"\n-- yearly volatility:\t\t{self._metrics['yearly_volatility']:.5f}"
        format += f"\n-- unbiased skewness:\t\t{self._metrics['skewness']:.5f}"
        format += f"\nobject located at {hex(id(self))}>"

        return format

    def period_returns(self, period: int = 1) -> pd.Series:
        """ """
        return self._data.pct_change(periods=period)

    def period_returns_mean(self, period: int = 1) -> np.typing.DTypeLike:
        """ """
        return self.period_returns(period=period).mean(axis=0)

    def volatility(
        self, period: int = 1, trading_days: int = 252
    ) -> np.typing.DTypeLike:
        """ """
        return self.period_returns(period=period).std() * np.sqrt(trading_days)

    def skewness(self) -> np.float32:
        """
        Computes the skewness of the saved data. Uses the ``Adjusted Fisher-Pearson
        standardized moment coefficient`` formula without bias [1, 2]. Skewness is a
        measure of the asymmetry of the probability distribution for a real-valued
        random variable around its mean.

        Returns
        -------
        np.float32
            The skewness measure for the saved historical price data.

        References
        ----------
        [1] Skewness calculation on scipy.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
        [2] Moment calculation on scipy.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html

        """
        return self._data.skew().astype(np.float32)

    def historical_mean_return(self, trading_days: int = 252) -> np.typing.DTypeLike:
        """
        Compute the historical mean return for a given period.

        Parameters
        ----------
        trading_days : int
            The number of trading days to use when calculating the historical mean return.

        Returns
        -------
        np.typing.DTypeLike
            The historical mean return of the ``Asset``.

        """
        return self.period_returns_mean(period=1) * trading_days

    @property
    def data(self) -> pd.Series:
        """
        Return the saved data by accessing it as a property of the ``Asset`` object.

        Returns
        -------
        pd.Series
            A ``pd.Series`` copy of the saved data.

        """
        return self._data

    @data.setter
    def data(self, data: pd.Series):
        """
        Set the value of the data attribute for the ``Asset`` object.

        Parameters
        ----------
        data : pd.Series
            The new ``pd.Series`` to set as data attribute for the object.

        """
        self._data = data

    @property
    def name(self) -> str:
        """
        Get the name property of the ``Asset`` object.

        Returns
        -------
        str
            The name of the ``Asset``.

        """
        return self._name

    @name.setter
    def name(self, name: str):
        """
        Set the value of the name property for the ``Asset`` object.

        Parameters
        ----------
        name : str
            THe new ``str`` to set as name attribute for the object.

        """
        self._name = name

    def as_numpy(self, dtype: np.typing.DTypeLike = np.float32) -> np.ndarray:
        """
        Return the saved data as an numpy array. It will have the shape (n_sample, ).

        Parameters
        ----------
        dtype : np.typing.DTypeLike
            The data type to create the new ``np.ndarray`` as.
            Defaults to ``np.float32``.

        Returns
        -------
        np.ndarray
            A new ``np.ndarray`` from the ``pd.Series`` data.

        """
        return self._data.to_numpy().astype(dtype)
