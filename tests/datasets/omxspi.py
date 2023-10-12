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
Last updated: 2023-10-13
"""

import shutil
import unittest
import logging
from pathlib import Path


log = logging.getLogger(__name__)
SAVE_PATH = ".data/OMXSPI/"


class OMXSPITest(unittest.TestCase):
    """ """

    def setUp(self):
        """ """
        log.info(f"setting up `{self.__class__.__name__}`...")

        self._save_path = SAVE_PATH

    def tearDown(self):
        """ """
        path = Path(SAVE_PATH)

        if path.is_dir():
            log.info(f"deleting `{path}` recursively...")
            shutil.rmtree(SAVE_PATH)
            log.info("OK!")
