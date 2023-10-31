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

File created: 2023-10-20
Last updated: 2023-10-29
"""

import logging
import pandas as pd
import numpy as np

from finq.asset import Asset
from finq.exceptions import (
    InvalidCombinationOfArgumentsError,
    PortfolioNotYetOptimizedError,
)

from typing import (
    Any,
    List,
    Dict,
    Union,
    Optional,
)

log = logging.getLogger(__name__)


class Portfolio(object):
    """ """

    def __init__(
        self,
        data: Union[List[Asset], np.ndarray, pd.DataFrame],
        *,
        weights: Optional[np.ndarray] = None,
        names: Optional[Union[Dict[str, str], List[str]]] = None,
        symbols: Optional[Union[Dict[str, str], List[str]]] = None,
        confidence_level: float = 0.95,
        risk_free_rate: float = 5e-3,
        n_trading_days: int = 252,
    ):
        """ """

        if not isinstance(data, list):
            if names is None and symbols is None and not isinstance(data, pd.DataFrame):
                raise InvalidCombinationOfArgumentsError(
                    "You need to provide the names and ticker symbols of each asset that you "
                    "want to include in your portfolio if the data you provided is neither a "
                    "`list` of `Asset` objects or a `pd.DataFrame`. You can also try "
                    "providing only one of the arguments `names` and `symbols`, but then as "
                    "a dictionary of the form `key=name` `value=symbol`."
                )

        if isinstance(data, list):
            symbols = [a.name for a in data]
            data = np.array([a.data for a in data])

        if isinstance(data, pd.DataFrame):
            symbols = data.columns
            data = data.to_numpy().T

        if isinstance(names, dict):
            symbols = list(names.values())
            names = list(names.keys())

        if isinstance(symbols, dict):
            names = list(symbols.keys())
            symbols = list(symbols.values())

        self._data = data
        self._weights = weights
        self._names = names
        self._symbols = symbols

        self._confidence_level = confidence_level
        self._risk_free_rate = risk_free_rate
        self._n_trading_days = n_trading_days

        self._div_fn = np.vectorize(self._divide)

    @staticmethod
    def _divide(u: np.ndarray, v: np.ndarray) -> Union[ZeroDivisionError, np.ndarray]:
        """ """
        if v == 0:
            if u == 0:
                return np.ones(u.shape).astype(u.dtype)
            raise ZeroDivisionError(
                f"Tried to perform the following division: {u}/{v}.",
            )

        return u / v

    def assets_period_returns(self, period: int = 1) -> np.ndarray:
        """ """
        return (
            self._div_fn(
                self._data[:, 1::period],
                self._data[:, :-1:period],
            )
            - 1
        )

    def assets_period_returns_mean(self, period: int = 1) -> np.ndarray:
        """ """
        return np.mean(self.assets_period_returns(period=period), axis=1)

    def assets_period_returns_cov(self, period: int = 1) -> np.ndarray:
        """ """
        return np.cov(self.assets_period_returns(period=period), rowvar=True)

    def assets_volatility(self, period: int = 1) -> np.ndarray:
        """ """
        return np.std(
            self.assets_period_returns(period=period),
            axis=1,
        ) * np.sqrt(self._n_trading_days)

    def period_returns(self, period: int = 1) -> np.ndarray:
        """ """

        if self._weights is None:
            raise PortfolioNotYetOptimizedError(
                "No portfolio weights available to calculate period returns with."
            )

        return np.dot(self._weights.T, self.assets_period_returns(period=period))

    def period_returns_mean(self, period: int = 1) -> np.ndarray:
        """ """

        if self._weights is None:
            raise PortfolioNotYetOptimizedError(
                "No portfolio weights available to calculate period returns mean with."
            )

        return np.dot(self._weights.T, self.assets_period_returns_mean(period=period))

    def volatility(
        self, *, period: int = 1, n_trading_days: Optional[int] = None
    ) -> float:
        """ """

        if self._weights is None:
            raise PortfolioNotYetOptimizedError(
                "No portfolio weights available to calculate volatility with."
            )

        if n_trading_days is None:
            n_trading_days = self._n_trading_days

        std = np.sqrt(
            np.dot(
                self._weights.T,
                np.dot(
                    self.assets_period_returns_cov(period=period),
                    self._weights,
                ),
            )
        )

        return std * np.sqrt(n_trading_days)

    @property
    def weights(self) -> Optional[np.ndarray]:
        """ """
        return self._weights

    @weights.setter
    def weights(self, weights: Union[List[float], np.ndarray]):
        """ """
        self._weights = weights

    def optimize(self, method: str, **kwargs: Dict[str, Any]):
        """ """
        raise NotImplementedError
