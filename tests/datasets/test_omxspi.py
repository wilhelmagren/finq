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

File created: 2023-10-13
Last updated: 2023-10-19
"""

import os
import shutil
import unittest
from unittest.mock import patch, PropertyMock
from pathlib import Path

from .mock_df import _random_df
from finq.datasets.omxspi import OMXSPI

SAVE_PATH = ".data/OMXSPI/"


class OMXSPITest(unittest.TestCase):
    """ """

    def setUp(self):
        """ """
        self._save_path = SAVE_PATH

    def tearDown(self):
        """ """
        path = Path(self._save_path)

        if path.is_dir():
            shutil.rmtree(path)

    @patch("yfinance.Ticker.info", new_callable=PropertyMock)
    @patch("yfinance.Ticker.history")
    def test_fetch_data_save_visualize(self, mock_ticker_data, mock_ticker_info):
        """ """

        mock_ticker_info.return_value = {
            "OMXSPI funny info about option": "yes very much",
        }

        df = _random_df(["Open", "High", "Low", "Close"])
        mock_ticker_data.return_value = df

        dataset = OMXSPI(
            save_path=self._save_path,
            save=True,
        )

        png_path = Path("omxspiPLOT12984198.png")
        dataset = dataset.fetch_data("1y").fix_missing_data().verify_data()
        dataset.visualize(log_scale=False, save_path=png_path, show=False)

        self.assertTrue(
            Path(dataset._save_path).is_dir(),
        )
        self.assertTrue(png_path.exists())
        os.remove(png_path)
