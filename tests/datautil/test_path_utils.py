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

File created: 2023-10-21
Last updated: 2023-10-21
"""

import unittest
import shutil

from pathlib import Path
from finq.datautil import (
    default_finq_cache_path,
    default_finq_save_path,
    all_tickers_saved,
)


class PathUtilsTests(unittest.TestCase):
    """ """

    def test_default_finq_cache_path(self):
        """ """

        path = default_finq_cache_path()
        self.assertEqual(
            Path.home() / ".finq" / "http_cache",
            path,
        )

    def test_default_finq_save_path(self):
        """ """

        path = default_finq_save_path()
        self.assertEqual(
            Path.home() / ".finq" / "data",
            path,
        )

    def test_all_tickers_saved(self):
        """ """

        test_path = Path.cwd() / ".dummytest"

        test_info_path = test_path / "info"
        test_data_path = test_path / "data"

        test_info_path.mkdir(parents=True)
        test_data_path.mkdir(parents=True)

        tickers = ["A", "B", "C", "D", "E"]

        for ticker in tickers:
            with open(test_info_path / f"{ticker}.json", "w") as f:
                f.write("dummytest")
            with open(test_data_path / f"{ticker}.csv", "w") as f:
                f.write("dummytest")

        self.assertTrue(all_tickers_saved(test_path, tickers))
        shutil.rmtree(test_path)
