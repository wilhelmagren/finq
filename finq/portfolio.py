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
Last updated: 2023-11-01
"""

import logging
import pandas as pd
import numpy as np
from functools import wraps

from finq.asset import Asset
from finq.datasets import Dataset
from finq.exceptions import (
    FinqError,
    InvalidCombinationOfArgumentsError,
    InvalidPortfolioWeightsError,
    PortfolioNotYetOptimizedError,
)
from finq.formulas import (
    period_returns,
    sharpe_ratio,
    weighted_returns,
    weighted_variance,
)

from typing import (
    Any,
    Callable,
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
        data: Union[Dataset, List[Asset], np.ndarray, pd.DataFrame],
        *,
        weights: Optional[np.ndarray] = None,
        names: Optional[Union[Dict[str, str], List[str]]] = None,
        symbols: Optional[Union[Dict[str, str], List[str]]] = None,
        confidence_level: float = 0.95,
        risk_free_rate: float = 5e-3,
        n_trading_days: int = 252,
    ):
        """ """

        if isinstance(data, Dataset):
            assets = data.as_assets()
            data = list(assets.values())
            symbols = list(assets.keys())

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

    def weights_are_normalized(self) -> bool:
        """ """

        return self._weights.sum() == 1.0

    def check_valid_weights(func) -> Callable:
        """ """

        @wraps(func)
        def _check_weights(self, *args, **kwargs) -> Optional[FinqError]:
            """ """

            if self._weights is None:
                raise PortfolioNotYetOptimizedError(
                    "Portfolio weights are `None`. Perhaps you have not yet optimized it? "
                )

            if not self.weights_are_normalized():
                raise InvalidPortfolioWeightsError(
                    "Your portfolio weights are not normalized. Make sure to normalize them "
                    "(they sum to one) before calculating any analytical quantities. "
                )

            return func(*args, **kwargs)

        return _check_weights

    def daily_returns(self) -> np.ndarray:
        """ """

        return period_returns(self._data, period=1)

    def yearly_returns(self) -> np.ndarray:
        """ """

        return period_returns(self._data, period=self._n_trading_days)

    def period_returns(self, period: int) -> np.ndarray:
        """ """

        return period_returns(self._data, period=period)

    def daily_returns_mean(self) -> float:
        """ """

        return np.mean(period_returns(self._data, period=1), axis=1)

    def yearly_returns_mean(self) -> float:
        """ """

        return np.mean(period_returns(self._data, period=self._n_trading_days), axis=1)

    def period_returns_mean(self, period: int) -> float:
        """ """

        return np.mean(period_returns(self._data, period=period), axis=1)

    def daily_covariance(self) -> np.ndarray:
        """ """

        return np.cov(period_returns(self._data, period=1), rowvar=True)

    def yearly_covariance(self) -> np.ndarray:
        """ """

        return np.cov(
            period_returns(self._data, period=self._n_trading_days), rowvar=True
        )

    def period_covariance(self, period: int) -> np.ndarray:
        """ """

        return np.cov(period_returns(self._data, period=period), rowvar=True)

    @check_valid_weights
    def volatility(self) -> float:
        """ """

        return np.sqrt(
            weighted_variance(
                self._weights,
                self.daily_covariance(),
            )
        ) * np.sqrt(self._n_trading_days)

    @check_valid_weights
    def expected_returns(self) -> float:
        """ """

        return np.mean(
            weighted_returns(
                self._weights,
                self.daily_returns_mean() * self._n_trading_days,
            ),
            axis=1,
        )

    @check_valid_weights
    def sharpe_ratio(self) -> float:
        """ """

        r = self.expected_returns()
        v = self.volatility()
        return sharpe_ratio(r, v, self._risk_free_rate)

    @property
    def weights(self) -> Optional[np.ndarray]:
        """ """
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray):
        """ """
        self._weights = weights

    def optimize(self, method: str, **kwargs: Dict[str, Any]):
        """ """
        raise NotImplementedError
