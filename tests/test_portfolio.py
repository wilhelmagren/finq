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

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch

from finq import Portfolio
from finq.datasets import CustomDataset

from .datasets.mock_df import _random_df


class PortfolioTests(unittest.TestCase):
    """ """

    @patch("yfinance.Ticker.history")
    def test_constructor_dataset(self, mock_ticker_data):
        """ """

        df = _random_df(["Open", "High", "Low", "Close"], days=400)
        mock_ticker_data.return_value = df

        names = ["dummy", "stuff", "kebab", "pizza"]
        symbols = ["dummy.ST", "stuff.ST", "kebab.ST", "pizza.ST"]

        d = CustomDataset(
            names,
            symbols,
            market="OMX",
            save=False,
        )

        d = d.run("2y")

        p = Portfolio(
            d,
            confidence_level=0.90,
            risk_free_rate=1e-4,
            n_trading_days=240,
        )

        self.assertTrue(isinstance(p._data, np.ndarray))

