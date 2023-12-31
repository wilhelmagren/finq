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

File created: 2023-10-16
Last updated: 2023-10-31
"""

import shutil
import unittest
from unittest.mock import patch

from .mock_df import _random_df
from finq.datasets import NDX
from finq.datautil import default_finq_save_path
from finq import Asset


class NDXTests(unittest.TestCase):
    """ """

    def setUp(self):
        """ """

        self._save_path = default_finq_save_path()
        self._dataset_name = "NDX"

    def tearDown(self):
        """ """

        path = self._save_path / self._dataset_name

        if path.is_dir():
            shutil.rmtree(path)

    @patch("yfinance.Ticker.get_info")
    @patch("yfinance.Ticker.history")
    def test_fetch_then_load(self, mock_ticker_data, mock_ticker_info):
        """ """

        mock_ticker_info.return_value = {
            "funny info about option": "yes very much",
        }

        df = _random_df(["Open", "High", "Low", "Close"])
        mock_ticker_data.return_value = df

        d = NDX(save=True)
        d = d.fetch_data_and_info("3mo")
        d.run("3mo")

        info_path = self._save_path / self._dataset_name / "info"
        data_path = self._save_path / self._dataset_name / "data"

        self.assertTrue(info_path.is_dir())
        self.assertTrue(data_path.is_dir())

        self.assertTrue((d._save_path / "info").is_dir())
        self.assertTrue((d._save_path / "data").is_dir())

        n = NDX(save=False)
        n.fetch_data("1y")

        self.assertEqual(
            d.get_tickers(),
            n.get_tickers(),
        )

        self.assertEqual(
            d.as_numpy("High").shape,
            n.as_numpy("High").shape,
        )

        self.assertEqual(
            len(d["AMD"].index.values),
            len(n.get_data()["AMD"].index.values),
        )

    @patch("yfinance.Ticker.get_info")
    @patch("yfinance.Ticker.history")
    def test_fetch_then_as_assets(self, mock_ticker_data, mock_ticker_info):
        """ """

        mock_ticker_info.return_value = {
            "funny info about option": "yes very much",
        }

        df = _random_df(["Open", "High", "Low", "Close"])
        mock_ticker_data.return_value = df

        d = NDX(n_requests=2, t_interval=2, separator=",")
        d.run("6m")

        assets = d.as_assets("Close")
        for t, a in assets.items():
            self.assertTrue(isinstance(a, Asset))
            self.assertTrue(t in d.get_tickers())

        self.assertEqual(len(d.get_data().values()), len(assets.values()))
