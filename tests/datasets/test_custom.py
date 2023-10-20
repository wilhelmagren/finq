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

File created: 2023-10-11
Last updated: 2023-10-21
"""

import os
import shutil
import unittest
import numpy as np
from unittest.mock import patch, PropertyMock
from pathlib import Path

from .mock_df import _random_df
from finq.datasets.custom import CustomDataset

SAVE_PATH = ".data/CUSTOM_COOL/"


class CustomDatasetTest(unittest.TestCase):
    """ """

    def setUp(self):
        """ """
        names = [
            "Alfa Laval",
            "Boliden",
            "Investor B",
            "SCA B",
            "SEB A",
            "Sv. Handelsbanken A",
        ]

        symbols = [
            "ALFA.ST",
            "BOL.ST",
            "INVE-B.ST",
            "SCA-B.ST",
            "SEB-A.ST",
            "SHB-A.ST",
        ]

        self._names = names
        self._symbols = symbols
        self._market = "OMX"
        self._save_path = SAVE_PATH

    def tearDown(self):
        """ """
        path = Path(SAVE_PATH)

        if path.is_dir():
            shutil.rmtree(SAVE_PATH)

    @patch("yfinance.Ticker.info", new_callable=PropertyMock)
    @patch("yfinance.Ticker.history")
    def test_fetch_visualize(self, mock_ticker_data, mock_ticker_info):
        """ """

        mock_ticker_info.return_value = {
            "funny info about option": "yes very much",
        }

        df = _random_df(["Open", "High", "Low", "Close"])
        mock_ticker_data.return_value = df

        dataset = CustomDataset(
            self._names,
            self._symbols,
            market=self._market,
            save_path=self._save_path,
            save=False,
        )

        self.assertEqual(dataset.get_tickers(), self._symbols)

        png_path = Path("customplot1984198.png")
        dataset = dataset.fetch_data("1y").fix_missing_data().verify_data()
        dataset.visualize(log_scale=True, save_path=png_path, show=False)

        self.assertTrue(png_path.exists())
        os.remove(png_path)

    @patch("yfinance.Ticker.info", new_callable=PropertyMock)
    @patch("yfinance.Ticker.history")
    def test_fetch_data_no_save(self, mock_ticker_data, mock_ticker_info):
        """ """

        mock_ticker_info.return_value = {
            "funny info about option": "yes very much",
        }

        df = _random_df(["Open", "High", "Low", "Close"])
        mock_ticker_data.return_value = df

        dataset = CustomDataset(
            self._names,
            self._symbols,
            market=self._market,
            save_path=self._save_path,
            save=False,
        )

        dataset = dataset.fetch_data("1y").fix_missing_data().verify_data()

        self.assertEqual(
            dataset.get_tickers(),
            self._symbols,
        )

        self.assertTrue(isinstance(dataset.as_numpy(), np.ndarray))

    @patch("yfinance.Ticker.info", new_callable=PropertyMock)
    @patch("yfinance.Ticker.history")
    def test_fetch_data_save(self, mock_ticker_data, mock_ticker_info):
        """ """

        mock_ticker_info.return_value = {
            "funny info about option": "yes very much",
        }

        df = _random_df(["Open", "High", "Low", "Close"])
        mock_ticker_data.return_value = df

        dataset = CustomDataset(
            self._names,
            self._symbols,
            market=self._market,
            save_path=self._save_path,
            save=True,
        )

        dataset = dataset.fetch_data("1y").fix_missing_data().verify_data()

        self.assertEqual(
            dataset.get_tickers(),
            self._symbols,
        )

        self.assertTrue(
            Path(dataset._save_path).is_dir(),
        )

    @patch("finq.datautil._fetch_names_and_symbols")
    @patch("yfinance.Ticker.info", new_callable=PropertyMock)
    @patch("yfinance.Ticker.history")
    def test_fetch_index_data(self, mock_ticker_data, mock_ticker_info, mock_get):
        """ """

        mock_get.return_value.names = ["VOLVO", "ATCO", "INVESTOR", "SEBANKEN"]
        mock_get.return_value.symbols = [
            "VOLV-B.ST",
            "ATCO-A.ST",
            "INVE-B.ST",
            "SEB-A.ST",
        ]

        mock_ticker_info.return_value = {
            "funny info about option": "yes very much",
        }

        df = _random_df(["Open", "High", "Low", "Close"])
        mock_ticker_data.return_value = df

        dataset = CustomDataset(
            index_name="OMXS30",
            market=self._market,
            save=False,
        )

        dataset.run("1y")

        self.assertEqual(
            dataset._save_path,
            Path(".data/OMXS30/"),
        )

        top_stocks = [
            "VOLV-B.ST",
            "ATCO-A.ST",
            "INVE-B.ST",
            "SEB-A.ST",
        ]

        stocks_found_in_index = all([t in dataset.get_tickers() for t in top_stocks])
        self.assertTrue(stocks_found_in_index)

    @patch("yfinance.Ticker.info", new_callable=PropertyMock)
    @patch("yfinance.Ticker.history")
    def test_fetch_then_load(self, mock_ticker_data, mock_ticker_info):
        """ """

        mock_ticker_info.return_value = {
            "funny info about option": "yes very much",
        }

        df = _random_df(["Open", "High", "Low", "Close"])
        mock_ticker_data.return_value = df

        dataset = CustomDataset(
            self._names,
            self._symbols,
            market=self._market,
            save_path=self._save_path,
            save=True,
        )

        dataset.run("1y")

        info_path = Path(self._save_path) / "info"
        data_path = Path(self._save_path) / "data"

        self.assertTrue(info_path.is_dir())
        self.assertTrue(data_path.is_dir())

        d = CustomDataset(
            self._names,
            self._symbols,
            market=self._market,
            save_path=self._save_path,
            save=False,
        )

        d.fetch_data("1y")

        self.assertEqual(
            d.get_tickers(),
            self._symbols,
        )

    def test_all_args_is_none(self):
        """ """
        self.assertRaises(ValueError, CustomDataset)
