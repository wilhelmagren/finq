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

File created: 2023-10-25
Last updated: 2023-11-01
"""

import unittest
import pandas as pd
import numpy as np

from finq import Asset
from finq.formulas import adjusted_fisher_pearson_skewness_coefficient


def _assert_all_close(
    a,
    b,
    ctx: unittest.TestCase,
    rtol: float = 1e-3,
    atol: float = 0,
):
    """ """

    if not hasattr(a, "__iter__"):
        a = [a]

    if not hasattr(b, "__iter__"):
        b = [b]

    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)


class AssetTests(unittest.TestCase):
    """ """

    def test_equality(self):
        """ """

        a = Asset(
            pd.Series(np.random.uniform(10, 1000, size=(4000,))),
            "XD",
            market="kebab",
            index_name="holykebab",
            pre_compute=False,
        )

        b = Asset(
            pd.Series(np.random.normal(200, 10, size=(200,))),
            "cool",
            market="NASDAQ",
            pre_compute=True,
        )

        c = Asset(
            b.data,
            "cool",
            market="NASDAQ",
            pre_compute=True,
        )

        self.assertEqual(b, c)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, 40)
        self.assertNotEqual(a, "xd")
        self.assertNotEqual(c, b"0903910")

        b.data = pd.Series(np.random.normal(100, 10, size=(40,)))
        self.assertNotEqual(b, c)

    def test_period_returns(self):
        """ """

        a = Asset(
            pd.Series([1, 1, 1, 1, 1]),
            "cool-asset.st",
        )

        a_pr = a.period_returns(2)
        a_expected = pd.Series([np.nan, np.nan, 0, 0, 0])
        _assert_all_close(a_pr, a_expected, self)

        b = Asset(
            pd.Series([1, 2, 3, 4, 5, 6]),
            "verycool.st",
        )

        b_pr_one = b.period_returns(1)
        b_expected_one = pd.Series([np.nan, 1, 0.5, 0.333333, 0.25, 0.2])
        _assert_all_close(
            b_pr_one,
            b_expected_one,
            self,
        )

        b_pr_three = b.period_returns(period=3)
        b_expected_three = pd.Series([np.nan, np.nan, np.nan, 3, 1.5, 1])
        _assert_all_close(
            b_pr_three,
            b_expected_three,
            self,
        )

    def test_period_returns_mean(self):
        """ """

        a = Asset(
            pd.Series([1, 2, 3, 4, 5, 6, 7]),
            "verycool-asset.st",
            market="OMX",
            index_name="OMXS30",
            price_type="Open",
        )

        a_prm = a.period_returns_mean()
        a_expected = sum([1, 0.5, 0.3333, 0.25, 0.2, 0.166666667]) / 6
        _assert_all_close(a_expected, a_prm, self)

        a_prm_four = a.period_returns_mean(period=4)
        a_expected_four = sum([4, 2, 1.3333333]) / 3
        _assert_all_close(
            a_expected_four,
            a_prm_four,
            self,
        )

    def test_volatility(self):
        """ """

        a = Asset(
            pd.Series([-1, 1, -1, 1, -2, 2]),
            "volatile",
        )

        a_vol = a.volatility()
        a_expected = a.period_returns().std() * np.sqrt(252)
        _assert_all_close(
            a_expected,
            a_vol,
            self,
        )

        a_vol_four = a.volatility(trading_days=4)
        a_expected_four = a.period_returns().std() * np.sqrt(4)
        _assert_all_close(
            a_expected_four,
            a_vol_four,
            self,
        )

        a_vol_weekly_year = a.volatility(period=7)
        a_expected_weekly_year = a.period_returns(period=7).std() * np.sqrt(252)
        _assert_all_close(
            a_expected_weekly_year,
            a_vol_weekly_year,
            self,
        )

    def test_skewness(self):
        """ """

        a = Asset(
            pd.Series([-1, -0.5, 0.0, 1.0, 2.0]),
            "skew",
        )

        a_skew = a.skewness()
        a_expected = adjusted_fisher_pearson_skewness_coefficient(
            np.array([-1, -0.5, 0.0, 1.0, 2.0]),
        )
        _assert_all_close(
            a_expected,
            a_skew,
            self,
        )

        b = Asset(
            pd.Series([-100, -50, -50, -20, -10, 0, 2]),
            "skew-big",
        )

        b_skew = b.skewness()
        b_expected = adjusted_fisher_pearson_skewness_coefficient(
            np.array([-100, -50, -50, -20, -10, 0, 2]),
        )
        _assert_all_close(
            b_expected,
            b_skew,
            self,
        )
