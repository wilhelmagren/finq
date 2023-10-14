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
Last updated: 2023-10-11
"""

import shutil
import unittest
import logging
import numpy as np
from pathlib import Path

from finq.datasets.custom import CustomDataset

log = logging.getLogger(__name__)
SAVE_PATH = ".data/CUSTOM_COOL/"


class CustomDatasetTest(unittest.TestCase):
    """ """

    def setUp(self):
        """ """
        log.info(f"setting up `{self.__class__.__name__}`...")
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
        self._save_path = SAVE_PATH

    def tearDown(self):
        """ """
        path = Path(SAVE_PATH)

        if path.is_dir():
            log.info(f"deleting `{path}` recursively...")
            shutil.rmtree(SAVE_PATH)
            log.info("OK!")

    def test_fetch_data_no_save(self):
        """ """
        dataset = CustomDataset(
            self._names,
            self._symbols,
            save_path=self._save_path,
            save=False,
        )

        dataset = dataset.fetch_data("1y").fix_missing_data().verify_data()

        self.assertEqual(
            dataset.get_tickers(),
            self._symbols,
        )

        self.assertTrue(isinstance(dataset.as_numpy(), np.ndarray))

    def test_fetch_data_save(self):
        """ """
        dataset = CustomDataset(
            self._names,
            self._symbols,
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

    def test_fetch_index_data(self):
        """ """
        dataset = CustomDataset(
            nasdaq_index="OMXS30",
            save=False,
        )

        dataset.run("1y")

        self.assertEqual(
            dataset._save_path,
            ".data/OMXS30/",
        )

        top_stocks = [
            "VOLV-B.ST",
            "ATCO-A.ST",
            "INVE-B.ST",
            "SEB-A.ST",
        ]

        stocks_found_in_index = all([t in dataset.get_tickers() for t in top_stocks])
        self.assertTrue(stocks_found_in_index)

    def test_fetch_then_load(self):
        """ """
        dataset = CustomDataset(
            self._names,
            self._symbols,
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
