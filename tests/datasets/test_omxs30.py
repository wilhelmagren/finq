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

File created: 2023-10-12
Last updated: 2023-10-31
"""

import shutil
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch

from .mock_df import _random_df
from finq.datautil import default_finq_save_path
from finq.datasets import OMXS30


class OMXS30Tests(unittest.TestCase):
    """ """

    def setUp(self):
        """ """

        self._save_path = default_finq_save_path()
        self._dataset_name = "OMXS30"

    def tearDown(self):
        """ """

        path = self._save_path / self._dataset_name

        if path.is_dir():
            shutil.rmtree(path)

    @patch("yfinance.Ticker.get_info")
    @patch("yfinance.Ticker.history")
    def test_fetch_data_no_save(self, mock_ticker_data, mock_ticker_info):
        """ """

        mock_ticker_info.return_value = {
            "cool info about ticker": "i own 100% of this, super green asset",
        }

        df = _random_df(["Open", "High", "Close", "Low"])
        mock_ticker_data.return_value = df

        dataset = OMXS30(save=False)

        dataset = dataset.fetch_data("1y").fix_missing_data().verify_data()
        df_shb = dataset["SHB-A.ST"]

        self.assertTrue(isinstance(df_shb, pd.DataFrame))
        self.assertTrue(isinstance(dataset.as_numpy(), np.ndarray))

    @patch("yfinance.Ticker.get_info")
    @patch("yfinance.Ticker.history")
    def test_fetch_data_save(self, mock_ticker_data, mock_ticker_info):
        """ """

        mock_ticker_info.return_value = {
            "cool info about ticker": "i own 100% of this, super green asset",
        }

        df = _random_df(["Open", "High", "Close", "Low"])
        mock_ticker_data.return_value = df

        dataset = OMXS30(save_path=self._save_path, save=True)
        dataset = dataset.fetch_data("1y").fix_missing_data().verify_data()
        dataset = dataset.fetch_info()

        data_path = self._save_path / self._dataset_name / "data"
        info_path = self._save_path / self._dataset_name / "info"

        self.assertTrue(data_path.is_dir())
        self.assertTrue(info_path.is_dir())
        self.assertEquals(30, len(dataset._data.keys()))
        self.assertEquals(30, len(dataset.get_tickers()))
        self.assertEquals(30, dataset.as_numpy("Close").shape[0])

    @patch("yfinance.Ticker.get_info")
    @patch("yfinance.Ticker.history")
    def test_fetch_data_save_different_path(self, mock_ticker_data, mock_ticker_info):
        """ """

        mock_ticker_info.return_value = {
            "cool info about ticker": "i own 100% of this, super green asset",
        }

        df = _random_df(["Open", "High", "Close", "Low"])
        mock_ticker_data.return_value = df

        custom_path = Path(".lolhahatest")

        dataset = OMXS30(save_path=custom_path, save=True)
        dataset.run("6m")

        expected_custom_path = Path(".") / self._dataset_name
        self.assertTrue(dataset._save_path.is_dir())
        self.assertEqual(
            expected_custom_path,
            dataset._save_path,
        )

        shutil.rmtree(expected_custom_path)
