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
Last updated: 2023-10-15
"""

from finq import datasets  # noqa
from finq import datautil  # noqa

import numpy as np
import os
import logging
from typing import (
    Union,
    Literal,
)

# Create logger and set up configuration accordingly.
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

from .log import get_module_log

logging.basicConfig(
    format="[%(asctime)s] [%(module_name)s] [%(levelname)s\t] %(message)s",
)

log = get_module_log(__name__)
log.setLevel(os.getenv("LOGLEVEL", logging.DEBUG))


def set_log_level(
    level: Union[int, Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]],
):
    """ """
    log.debug(f"changing log level to: `{level}`")
    log.setLevel(level)


def set_random_seed(seed: int):
    """ """
    log.debug(f"setting numpy random seed to: `{seed}`")
    np.random.seed(seed)
