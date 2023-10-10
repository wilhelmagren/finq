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

File created: 2023-10-09
Last updated: 2023-10-10
"""

from finq import datasets
from finq import portfolio

import numpy as np
import logging

# Create logger and set up configuration
# Levels in decreasing order of verbosity:
#   - NOTSET         0
#   - DEBUG         10
#   - INFO          20
#   - WARNING       30
#   - ERROR         40
#   - CRITICAL      50
#
# To change the logging level after having imported the library,
# use the function set_logging_level with preferred logging level.

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s\t] %(message)s")

console_handler.setFormatter(formatter)
log.addHandler(console_handler)


def set_log_level(level: int):
    """ """
    log.setLevel(level)


def set_random_seed(seed: int):
    """ """
    np.random.seed(seed)
